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

GROUPS = ("B0", "H1", "H2", "H3", "P1", "L0", "L1", "L2", "A1", "A2", "A3", "A4", "S1", "S2")


def levels(group: str, batch_levels: list[int] | None = None) -> list[tuple[str, dict]]:
    if group == "H3":
        if (
            batch_levels is None
            or len(batch_levels) not in (2, 3)
            or any(type(n) is not int or n < 1 for n in batch_levels)
            or batch_levels != sorted(set(batch_levels))
        ):
            raise ValueError("H3 requires two or three distinct measured batch levels")
        return [(f"batch_{n}", {"microbatch": n, "train_batch_size": n}) for n in batch_levels]
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


def row_configuration(raw: dict, overrides: dict) -> dict:
    config = copy.deepcopy(raw)
    for key, value in overrides.items():
        owner = (
            config["training"]
            if key in ("learning_rate", "weight_decay", "train_batch_size")
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
    return config


def measured_batches(parent: Path, probes: list[dict]) -> list[int]:
    from capacity_report import collect

    raw = yaml.safe_load(parent.read_text())
    batches = []
    for probe in probes:
        receipt, config = Path(probe["receipt"]), Path(probe["config"])
        if sha256_file(receipt) != probe["receipt_sha256"]:
            raise ValueError("capacity receipt changed")
        observed = collect(receipt, config)
        candidate = yaml.safe_load(config.read_text())
        if observed["world_size"] != 1 or observed["accumulation"] != 1:
            raise ValueError("H3 physical batch levels require single-rank unaccumulated probes")
        if candidate["dataset_id"] != raw["dataset_id"] or candidate["data"] != raw["data"]:
            raise ValueError("capacity probe dataset differs from parent")
        for key, value in raw["model"]["parameters"].items():
            if key != "microbatch" and candidate["model"]["parameters"].get(key) != value:
                raise ValueError("capacity model or loss profile differs from parent")
        batches.append(observed["microbatch"])
    levels("H3", batches)
    return batches


def generate(
    parent: Path, group: str, output: Path, *, batch_probes: list[dict] | None = None
) -> dict:
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
    batch_levels = measured_batches(parent, batch_probes or []) if group == "H3" else None
    if group != "H3" and batch_probes:
        raise ValueError("capacity batch probes apply only to H3")
    for name, overrides in levels(group, batch_levels):
        config = row_configuration(raw, overrides)
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
    if group == "H3":
        manifest["batch_probes"] = batch_probes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_group(manifest_path: Path, parent: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "gradpert-v2-group-configs-1":
        raise ValueError("unknown group manifest schema")
    if sha256_file(parent) != manifest["parent_sha256"]:
        raise ValueError("frozen group parent checksum mismatch")
    load_experiment_config(parent)
    raw = yaml.safe_load(parent.read_text())
    batch_levels = (
        measured_batches(parent, manifest.get("batch_probes", []))
        if manifest["group"] == "H3"
        else None
    )
    expected = dict(levels(manifest["group"], batch_levels))
    rows = manifest["rows"]
    if len(rows) != len(expected) or {r["name"] for r in rows} != set(expected):
        raise ValueError("group rows differ from the registered factor levels")
    root = manifest_path.parent.resolve()
    verified = []
    for row in rows:
        path = (root / row["config"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("row config escapes its group directory")
        if row["overrides"] != expected[row["name"]]:
            raise ValueError("row overrides differ from registered factor levels")
        if sha256_file(path) != row["sha256"]:
            raise ValueError("row config changed after group generation")
        load_experiment_config(path)
        if yaml.safe_load(path.read_text()) != row_configuration(raw, row["overrides"]):
            raise ValueError("row changes fields outside its declared factor")
        verified.append({"name": row["name"], "config": str(path), "sha256": row["sha256"]})
    return {
        "status": "group_config_verified_not_launch_preflight",
        "manifest_sha256": sha256_file(manifest_path),
        "group": manifest["group"],
        "parent_sha256": manifest["parent_sha256"],
        "rows": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--batch-probes", type=Path)
    args = parser.parse_args()
    if args.verify_manifest:
        if args.group or args.output:
            parser.error("verification does not accept generation arguments")
        print(json.dumps(verify_group(args.verify_manifest, args.parent), indent=2))
        return
    if not args.group or not args.output:
        parser.error("generation requires --group and --output")
    result = generate(
        args.parent,
        args.group,
        args.output,
        batch_probes=(json.loads(args.batch_probes.read_text()) if args.batch_probes else None),
    )
    print(
        json.dumps({"group": args.group, "rows": len(result["rows"]), "status": result["status"]})
    )


if __name__ == "__main__":
    main()
