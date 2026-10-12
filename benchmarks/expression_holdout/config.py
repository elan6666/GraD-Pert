"""Self-contained configs for officially trained expression-holdout baselines."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from gradpert.hashing import sha256_file

OFFICIAL = {
    "scouter": (
        "https://github.com/PancakeZoy/scouter",
        "0cfddd000e19b72ff033ba67c8315f7bc3304932",
    ),
    "scdfm": (
        "https://github.com/AI4Science-WestlakeU/scDFM",
        "8de47e7d3d443939f93f768f7902c66e0e82eda2",
    ),
    "gears": ("https://github.com/snap-stanford/GEARS", "f374e43e197b295016d80395d7a54ddb81cc6769"),
}


def load_config(path: Path) -> dict:
    value = (
        yaml.safe_load(path.read_text()) if path.suffix != ".json" else json.loads(path.read_text())
    )
    required = {
        "schema",
        "model_id",
        "seed",
        "source_code",
        "data",
        "model",
        "training",
        "evaluation",
        "artifacts",
        "adaptations",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("self-contained official baseline config fields differ")
    model = value["model_id"]
    if value["schema"] != "official-expression-holdout-v1" or model not in OFFICIAL:
        raise ValueError("unsupported official baseline schema/model")
    if (value["source_code"]["repository"], value["source_code"]["commit"]) != OFFICIAL[model]:
        raise ValueError("official source/config identity mismatch")
    if value["seed"] != 1 or not value["data"]["exclude_test_target_expression"]:
        raise ValueError("comparison fixes seed1 and expression exclusion")
    ev = value["evaluation"]
    if (
        ev["protocol"] != "v2"
        or ev["control_cells"] != 300
        or ev["checkpoint"] != "last"
        or ev["gene_groups"] != ["all", "seen_expression", "unseen_expression"]
        or ev["gene_sets"] != ["all", "deg"]
        or ev["query_count"] != 1000
        or type(ev["cell_batch"]) is not int
        or ev["cell_batch"] <= 0
    ):
        raise ValueError("common evaluation protocol changed")
    if value["artifacts"] != {"result_mode": "metrics_only", "zero_pkl": True}:
        raise ValueError("official baseline artifact contract changed")
    for key, digest in value["data"]["expected_hashes"].items():
        if not key or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("canonical identity must use explicit SHA256")
    for name in ["train_selection_path", "train_selection_sha256"]:
        if not value["data"].get(name):
            raise ValueError("cap40 selection requires explicit path/hash")
    return value


def require_data_identity(prepared, config: dict) -> None:
    for key, digest in config["data"]["expected_hashes"].items():
        if prepared.receipt.get(key) != digest:
            raise ValueError(f"canonical data identity differs: {key}")
    if (
        prepared.receipt["train_selection"]["manifest_sha256"]
        != config["data"]["train_selection_sha256"]
    ):
        raise ValueError("cap40 selection identity differs")


def validate_preflight(
    receipt: dict, *, config: Path, source_commit: str, official_commit: str, runtime_sha256: str
) -> None:
    if (
        receipt.get("schema") != "official-expression-holdout-preflight-v1"
        or receipt.get("status") != "passed"
        or receipt.get("config_sha256") != sha256_file(config)
        or receipt.get("source_commit") != source_commit
        or receipt.get("source_clean") is not True
        or receipt.get("official_commit") != official_commit
        or receipt.get("runtime_sha256") != runtime_sha256
        or receipt.get("optimizer_updates") != 10
        or receipt.get("restored_checkpoint_predictions_finite") is not True
        or receipt.get("controls_per_condition") != 300
        or receipt.get("zero_pkl") is not True
    ):
        raise ValueError("real official 10-update/checkpoint/common-inference preflight mismatch")
