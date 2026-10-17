"""Seal 2 SSL epochs plus 4 MSE epochs with two explicit medium-analysis endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from generate_pt1 import configuration as pt1_configuration

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "fgpo-pt2-staged-dual-endpoint-medium-1"
REFERENCE = ".byte-os/plans/PT2_STAGED_2PLUS4_20261017.plan.md"
ARMS = {
    "PT2": {
        "training_mode": "ssl_then_mse",
        "ssl_pretrain_epochs": 2,
        "finetune_learning_rate": 2e-4,
        "weighted_gram_backend": "triton_io",
    }
}
ENDPOINTS = {
    "after_ssl2": {"role": "pretrain", "epoch": 2, "training_stage": "ssl_pretrain"},
    "after_finetune4": {"role": "last", "epoch": 6, "training_stage": "mse_finetune"},
}


def configuration(source: Path) -> dict:
    value = pt1_configuration(source)
    for key, setting in ARMS["PT2"].items():
        value["model"]["parameters"][key] = dict(
            value=setting, source="user_locked", reference=REFERENCE
        )
    value["training"]["formal_run_policy"] = "v2_fixed_6"
    value["training"]["max_epochs"] = dict(value=6, source="user_locked", reference=REFERENCE)
    return value


def metadata() -> dict:
    from generate_pt1 import metadata as pt1_metadata

    return {
        **pt1_metadata(),
        "schema": SCHEMA,
        "family": "pt2_staged_2plus4_20261017",
        "priority": ["PT2"],
        "preflight_priority": ["PT2"],
        "epochs": 6,
        "prediction_epochs": 4,
        "analysis_endpoints": ENDPOINTS,
        "analysis": "one physical frozen medium manifest shared by epoch2/pretrain and epoch6/last",
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    value = json.loads(path.read_text())
    target = path.parent / "PT2/gradpert_v2/nadig_jurkat.yaml"
    if (
        not path.is_relative_to(source)
        or {k: v for k, v in value.items() if k != "rows"} != metadata()
    ):
        raise ValueError("PT2 scope, stages or endpoint protocol changed")
    expected = dict(
        name="PT2",
        config=str(target.relative_to(source)),
        sha256=sha256_file(target),
        changes=ARMS["PT2"],
    )
    if value["rows"] != [expected] or yaml.safe_load(target.read_text()) != configuration(source):
        raise ValueError("PT2 must retain its exact two-plus-four configuration")
    load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("PT2 needs a fresh source-contained manifest")
    path = output / "PT2/gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(configuration(source), sort_keys=False, allow_unicode=True))
    value = {
        **metadata(),
        "rows": [
            dict(
                name="PT2",
                config=str(path.relative_to(source)),
                sha256=sha256_file(path),
                changes=ARMS["PT2"],
            )
        ],
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(value, indent=2) + "\n")
    return verify_manifest(source, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2/pt2_native_io_20261017"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
