"""Seal only the Q-compression arm; existing baseline and step1 are reused."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("configs/v2/graph_prenorm_20261108/baseline/gradpert_v2/nadig_jurkat.yaml")
TARGET = Path("configs/v2/mla_query_compression_20261108/q_rank64/gradpert_v2/nadig_jurkat.yaml")


def generate(root: Path = ROOT) -> dict:
    value = copy.deepcopy(yaml.safe_load((root / PARENT).read_text()))
    value["model"]["parameters"]["mla_query_rank"] = {
        "value": 64,
        "source": "project_preregistered",
        "reference": "docs/experiments/MLA_QUERY_COMPRESSION_20261108.md",
    }
    path = root / TARGET
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    if path.exists() and path.read_text() != content:
        raise ValueError("refuse to overwrite changed Q-compression config")
    path.write_text(content)
    arch, _ = V2Options.parse_parameters(load_experiment_config(path).model.parameters)
    assert arch.mla_query_rank == 64 and arch.graph_input_norm == "historical"
    return {
        "parent_config": str(PARENT),
        "parent_sha256": sha256_file(root / PARENT),
        "config": str(TARGET),
        "sha256": sha256_file(path),
        "epochs": 2,
        "only_new_arm": "Q compression; no new baseline",
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
