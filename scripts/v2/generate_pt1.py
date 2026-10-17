"""Seal the approved two-epoch SSL then two-epoch MSE ablation on current B0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from generate_fgpo_family import configs as anchor_configs

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "fgpo-pt1-staged-medium-1"
ARMS = {
    "PT1": {
        "training_mode": "ssl_then_mse",
        "ssl_pretrain_epochs": 2,
        "finetune_learning_rate": 2e-4,
    }
}
REFERENCE = ".byte-os/plans/PT1_STAGED_VIEWS_20261017.plan.md"


def configuration(source: Path) -> dict:
    value = anchor_configs(source, microbatch=80)["B0"]
    for key, setting in ARMS["PT1"].items():
        value["model"]["parameters"][key] = {
            "value": setting,
            "source": "user_locked",
            "reference": REFERENCE,
        }
    return value


def metadata() -> dict:
    return dict(
        schema=SCHEMA,
        family="pt1_staged_20261017",
        priority=["PT1"],
        preflight_priority=["PT1"],
        common_microbatch=80,
        world_size=2,
        accumulation=2,
        global_batch=320,
        epochs=4,
        seed=1,
        eval_batch=128,
        validation="disabled",
        selection="final_epoch",
        test_roles=["last"],
        preflight_updates=10,
        eval_capacity_reference_batch=128,
        analysis="same medium population plus condition-stratified Global/Local diagnostics",
        rerun_old_BR_or_officials=False,
    )


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    value = json.loads(path.read_text())
    target = path.parent / "PT1/gradpert_v2/nadig_jurkat.yaml"
    if (
        not path.is_relative_to(source)
        or {k: v for k, v in value.items() if k != "rows"} != metadata()
    ):
        raise ValueError("PT1 scope or protocol changed")
    expected = {
        "name": "PT1",
        "config": str(target.relative_to(source)),
        "sha256": sha256_file(target),
        "changes": ARMS["PT1"],
    }
    if value["rows"] != [expected] or yaml.safe_load(target.read_text()) != configuration(source):
        raise ValueError("PT1 must retain its exact two-stage B0 configuration")
    load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("PT1 needs a fresh source-contained manifest")
    path = output / "PT1/gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(configuration(source), sort_keys=False, allow_unicode=True))
    value = {
        **metadata(),
        "rows": [
            {
                "name": "PT1",
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": ARMS["PT1"],
            }
        ],
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(value, indent=2) + "\n")
    return verify_manifest(source, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2/pt1_staged_20261017"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
