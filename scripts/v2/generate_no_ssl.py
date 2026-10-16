"""Seal a prediction-only loss ablation on the unchanged FG+PO/node0 B0."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_fgpo_family import configs as anchor_configs

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "fgpo-no-ssl-medium-1"
REFERENCE = ".byte-os/plans/NO_SSL_D0_20261016.plan.md"
ARMS = {"D0": {"lambda1": 0.0, "lambda2": 0.0}}


def configs(source: Path) -> dict:
    value = copy.deepcopy(anchor_configs(source, microbatch=80)["B0"])
    for key, setting in ARMS["D0"].items():
        value["model"]["parameters"][key] = {
            "value": setting,
            "source": "user_locked",
            "reference": REFERENCE,
        }
    return {"D0": value}


def metadata() -> dict:
    return {
        "schema": SCHEMA,
        "family": "no_ssl_d0_20261016",
        "priority": ["D0"],
        "preflight_priority": ["D0"],
        "common_microbatch": 80,
        "world_size": 2,
        "accumulation": 2,
        "global_batch": 320,
        "epochs": 4,
        "seed": 1,
        "eval_batch": 128,
        "validation": "disabled",
        "selection": "final_epoch",
        "test_roles": ["last"],
        "preflight_updates": 10,
        "eval_capacity_reference_batch": 128,
        "analysis": "same medium population; inactive distillation heads stay uncollected",
        "rerun_old_BR_or_officials": False,
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata():
        raise ValueError("D0 prediction-only protocol changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("only the prioritized D0 arm is authorized")
    expected = configs(source)
    for row in value["rows"]:
        target = path.parent / row["name"] / "gradpert_v2/nadig_jurkat.yaml"
        if (
            row
            != {
                "name": row["name"],
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": ARMS[row["name"]],
            }
            or yaml.safe_load(target.read_text()) != expected[row["name"]]
        ):
            raise ValueError("D0 must change only the two distillation coefficients")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    for name, value in configs(source).items():
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        rows.append(
            dict(
                name=name,
                config=str(path.relative_to(source)),
                sha256=sha256_file(path),
                changes=ARMS[name],
            )
        )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({**metadata(), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2/no_ssl_d0_20261016"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
