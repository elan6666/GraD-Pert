"""Seal a graph-only normalization comparison on the existing R3-RI recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.config.training_defaults import with_training_throughput_defaults
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("configs/v2/training_speed_20261106/baseline/gradpert_v2/nadig_jurkat.yaml")
FAMILY = Path("configs/v2/graph_prenorm_20261108")
REFERENCE = "docs/experiments/GRAPH_PRENORM_20261108.md"


def generate(root: Path = ROOT) -> dict:
    parent = with_training_throughput_defaults(yaml.safe_load((root / PARENT).read_text()))
    rows = []
    for arm, norm in (("baseline", "historical"), ("graph_rmsnorm", "rmsnorm")):
        value = copy.deepcopy(parent)
        value["model"]["parameters"]["graph_input_norm"] = {
            "value": norm,
            "source": "user_locked",
            "reference": REFERENCE,
        }
        path = FAMILY / arm / "gradpert_v2/nadig_jurkat.yaml"
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        if destination.exists() and destination.read_text() != content:
            raise ValueError(f"refuse to overwrite changed experiment config: {path}")
        destination.write_text(content)
        config = load_experiment_config(destination)
        architecture, _ = V2Options.parse_parameters(config.model.parameters)
        assert architecture.graph_input_norm == norm
        rows.append({"arm": arm, "config": str(path), "sha256": sha256_file(destination)})
    manifest = {
        "schema": "gradpert-graph-prenorm-comparison-1",
        "parent_config": str(PARENT),
        "parent_sha256": sha256_file(root / PARENT),
        "changed_model_parameters": ["graph_input_norm"],
        "shared_execution_preset": ["relay_validate_once", "cache_kda_constants"],
        "epochs": 2,
        "checkpoint_role": "last",
        "evaluation_partition": "validation",
        "deg": False,
        "test_tuning": False,
        "rows": rows,
    }
    target = root / FAMILY / "manifest.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if target.exists() and target.read_text() != content:
        raise ValueError("refuse to overwrite changed experiment manifest")
    target.write_text(content)
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
