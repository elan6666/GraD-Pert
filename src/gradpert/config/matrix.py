"""Completeness verification for the model-by-dataset config matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gradpert.config.loader import load_experiment_config
from gradpert.config.schema import DatasetId, ModelId
from gradpert.hashing import sha256_file

MODEL_IDS: tuple[ModelId, ...] = (
    "gradpert_b2",
    "gears",
    "txpert_public",
    "matched_control_mean",
    "global_train_delta",
    "general_train_delta",
)
DATASET_IDS: tuple[DatasetId, ...] = (
    "replogle_k562_essential",
    "replogle_rpe1_essential",
    "nadig_jurkat",
    "nadig_hepg2",
    "norman",
)


def expected_config_paths(root: str | Path) -> tuple[Path, ...]:
    base = Path(root)
    return tuple(base / model / f"{dataset}.yaml" for model in MODEL_IDS for dataset in DATASET_IDS)


def verify_config_matrix(root: str | Path) -> dict[str, Any]:
    """Validate an exact 6x5 config matrix and return its hash index."""

    base = Path(root)
    expected = set(expected_config_paths(base))
    discovered = set(base.glob("*/*.yaml")) | set(base.glob("*/*.yml"))
    missing = sorted(str(path) for path in expected - discovered)
    extra = sorted(str(path) for path in discovered - expected)
    if missing or extra:
        raise ValueError(f"Config matrix mismatch: missing={missing}, extra={extra}")

    entries: list[dict[str, Any]] = []
    for path in sorted(expected):
        config = load_experiment_config(path)
        entries.append(
            {
                "path": str(path),
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "sha256": sha256_file(path),
            }
        )
    return {"count": len(entries), "expected_count": 30, "entries": entries}
