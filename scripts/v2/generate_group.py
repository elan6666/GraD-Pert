"""Materialize independent v2 group configs from one explicitly frozen parent."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

GROUPS = ("B0", "H1", "H2", "P1", "L0", "L1", "L2", "A1", "A2", "A3", "A4", "S1", "S2")


def levels(group: str) -> list[tuple[str, dict]]:
    if group == "B0":
        return [
            ("prediction_only", {"lambda1": 0, "lambda2": 0}),
            ("joint_ssl", {"lambda1": 1, "lambda2": 0.1}),
        ]
    if group == "H1":
        return [("lr_1e3", {"learning_rate": 0.001}), ("lr_1e4", {"learning_rate": 0.0001})]
    if group == "H2":
        return [("wd_0", {"weight_decay": 0.0}), ("wd_1e5", {"weight_decay": 0.00001})]
    if group == "P1":
        return [(f"width_{width}", {"width": width}) for width in (256, 128)]
    if group == "L0":
        return [
            (f"ssl_{a}{b}", {"lambda1": a, "lambda2": b * 0.1})
            for a, b in itertools.product((0, 1), repeat=2)
        ]
    if group in ("L1", "L2"):
        stage = int(group[-1])
        names = ("condition", "node", "spread") if stage == 1 else ("dino", "ibot", "koleo")
        rows = []
        for bits in itertools.product((0, 1), repeat=3):
            values = {
                "lambda1": 1 if stage == 1 and any(bits) else 0,
                "lambda2": 0.1 if stage == 2 and any(bits) else 0,
            }
            values.update(
                {
                    f"ssl{stage}_{name}": bit * weight
                    for name, bit, weight in zip(names, bits, (0.8, 0.4, 0.1), strict=True)
                }
            )
            rows.append(("components_" + "".join(map(str, bits)), values))
        return rows
    if group == "S2":
        return [
            ("mixed_koleo_off", {"ssl2_koleo": 0}),
            ("single_koleo_off", {"max_conditions": 1, "ssl2_koleo": 0}),
            ("mixed_koleo_on", {"ssl2_koleo": 0.1}),
        ]
    key, values = {
        "A1": ("attention", ("hybrid", "per_gene")),
        "A2": ("attention", ("hybrid", "full_latent", "delta_full", "full")),
        "A3": ("streams", (4, 1)),
        "A4": ("gene_initialization", ("genept", "random")),
        "S1": ("ssl1_reduction", ("condition_mean", "row_mean")),
    }[group]
    return [(str(value), {key: value}) for value in values]


def generate(parent: Path, group: str, output: Path) -> dict:
    load_experiment_config(parent)
    raw = yaml.safe_load(parent.read_text())
    if raw["model_id"] != "gradpert_v2" or group not in GROUPS:
        raise ValueError("requires a v2 parent and a supported group")
    if group == "S2" and raw["model"]["parameters"]["max_conditions"]["value"] <= 1:
        raise ValueError("S2 needs a mixed-condition parent")
    if output.exists():
        raise FileExistsError("group output must be new; existing experiments are immutable")
    rows = []
    prepared = []
    for name, overrides in levels(group):
        config = copy.deepcopy(raw)
        for key, value in overrides.items():
            owner = (
                config["training"]
                if key in ("learning_rate", "weight_decay")
                else config["model"]["parameters"]
            )
            owner[key] = {
                "value": value,
                "source": "project_preregistered",
                "reference": "docs/design/GRADPERT_V2.md",
            }
        if "learning_rate" in overrides:
            schedule = config["training"]["scheduler"]["value"]
            schedule["max_lr"] = overrides["learning_rate"]
            schedule["min_lr"] = 0.2 * overrides["learning_rate"]
        prepared.append((name, config, overrides))
    output.mkdir(parents=True)
    for name, config, overrides in prepared:
        path = output / name / "gradpert_v2" / f"{raw['dataset_id']}.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_experiment_config(path)
        rows.append(
            {
                "name": name,
                "config": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "overrides": overrides,
            }
        )
    manifest = {
        "schema_version": "gradpert-v2-group-configs-1",
        "group": group,
        "parent_config": str(parent.resolve()),
        "parent_sha256": sha256_file(parent),
        "generator_sha256": sha256_file(Path(__file__)),
        "status": "configs_valid_pending_capacity_and_launch_preflight",
        "selection_rule": "validation prediction loss only; no within-group rolling parent",
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.parent, args.group, args.output)
    print(
        json.dumps({"group": args.group, "rows": len(result["rows"]), "status": result["status"]})
    )


if __name__ == "__main__":
    main()
