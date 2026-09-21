"""Small metrics-only receipts, with immutable ordered populations stored once."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file


def compact_validation(result: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Keep exact row IDs once instead of embedding them in every epoch/checkpoint."""
    if result["split"] != "val":
        raise ValueError("validation history cannot contain test results")
    keys = (
        "condition_id",
        "control_row_ids",
        "control_row_ids_sha256",
        "truth_row_ids",
        "truth_row_ids_sha256",
    )
    population = {
        "schema_version": "gradpert-v2-validation-population-1",
        "control_manifest_sha256": result["control_manifest_sha256"],
        "reference_sha256": result["reference_sha256"],
        "query_recipe": result["query_recipe"],
        "conditions": [{key: row[key] for key in keys} for row in result["conditions"]],
    }
    path = root / "validation_population.json"
    if path.exists():
        if read_json(path) != population:
            raise ValueError("validation population or inference recipe changed across epochs")
    else:
        atomic_json(path, population)
    compact = copy.deepcopy(result)
    for row in compact["conditions"]:
        del row["control_row_ids"]
        del row["truth_row_ids"]
    compact["population_receipt"] = {"file": path.name, "sha256": sha256_file(path)}
    return compact
