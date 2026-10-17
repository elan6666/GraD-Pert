"""Seal two prioritized prototype/Teacher ablations on the FG+PO/node0 anchor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_fgpo_family import configs as anchor_configs

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "fgpo-dino-pair-medium-1"
REMAINING_SCHEMA = "fgpo-dino-dp2-medium-1"
REFERENCE = ".byte-os/plans/DINO_PAIR_20261016.plan.md"
ARMS = {
    "B0-DP": {"prototype_weight_norm": True, "prototype_freeze_epochs": 1},
    "DP2": {
        "prototype_weight_norm": True,
        "prototype_freeze_epochs": 1,
        "prototype_fixed_scale": True,
        "prototype_freeze_gradients": True,
        "teacher_start": 0.996,
        "teacher_temperature_start": 0.04,
        "teacher_temperature_end": 0.03,
        "teacher_temperature_warmup_epochs": 1,
    },
}


def configs(source: Path) -> dict:
    base = anchor_configs(source, microbatch=80)["B0"]
    result = {}
    for name, changes in ARMS.items():
        value = copy.deepcopy(base)
        # Fully resolved files, including unchanged temperature controls.
        settings = {
            "prototype_fixed_scale": False,
            "teacher_temperature_start": 0.04,
            "teacher_temperature_end": 0.04,
            "teacher_temperature_warmup_epochs": 0,
            **changes,
        }
        for key, setting in settings.items():
            value["model"]["parameters"][key] = {
                "value": setting,
                "source": "user_locked",
                "reference": REFERENCE,
            }
        result[name] = value
    return result


def metadata(*, remaining_only: bool = False) -> dict:
    priority = ["DP2"] if remaining_only else list(ARMS)
    return {
        "schema": REMAINING_SCHEMA if remaining_only else SCHEMA,
        "family": "dino_pair_20261016",
        "priority": priority,
        "preflight_priority": priority,
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
        "analysis": "same medium basic/deep/gradient lists; actual Teacher temperature",
        "rerun_old_BR_or_officials": False,
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    expected_metadata = metadata(remaining_only=value.get("schema") == REMAINING_SCHEMA)
    if {k: v for k, v in value.items() if k != "rows"} != expected_metadata:
        raise ValueError("two-arm distillation protocol changed")
    if [r["name"] for r in value["rows"]] != expected_metadata["priority"]:
        raise ValueError("authorized two-arm order changed")
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
            raise ValueError("distillation arm config or identity differs")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path, *, remaining_only: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    expected_metadata = metadata(remaining_only=remaining_only)
    values = configs(source)
    for name in expected_metadata["priority"]:
        value = values[name]
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        rows.append(
            {
                "name": name,
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": ARMS[name],
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({**expected_metadata, "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2/dino_pair_20261016"))
    parser.add_argument("--remaining-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(generate(root, root / args.output, remaining_only=args.remaining_only), indent=2)
    )
