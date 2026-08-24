"""Hash-pinned systems selection from matched native capacity receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json
from gradpert.data.registry import DATASET_IDS
from gradpert.hashing import sha256_file

CUDA_ALLOCATOR_CONFIG = "expandable_segments:True"
MINIMUM_EPOCH_SPEEDUP = 1.25
MINIMUM_MEMORY_HEADROOM_FRACTION = 0.10


def _load_passed_receipt(path: Path, *, expected_batch_size: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"capacity receipt must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "prototype-head-capacity-v2"
        or payload.get("status") != "development_capacity_passed"
        or payload.get("batch_size") != expected_batch_size
        or payload.get("pytorch_alloc_conf") != CUDA_ALLOCATOR_CONFIG
        or payload.get("selected_prototype_count") != 16384
        or payload.get("capacity_probe_steps") != 128
    ):
        raise ValueError(f"capacity receipt contract mismatch: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("speed-first capacity receipt must accept the 16384 candidate")
    candidate = candidates[0]
    if candidate.get("prototype_count") != 16384 or candidate.get("accepted") is not True:
        raise ValueError("capacity receipt did not accept prototype_count=16384")
    datasets = candidate.get("datasets")
    if not isinstance(datasets, list) or [item.get("dataset_id") for item in datasets] != list(
        DATASET_IDS
    ):
        raise ValueError("capacity receipt does not contain the ordered five datasets")
    threshold = payload.get("acceptance_threshold_bytes")
    if not isinstance(threshold, int) or threshold <= 0:
        raise ValueError("capacity receipt has an invalid memory threshold")
    for item in datasets:
        numeric_positive = (
            "observed_probe_cells",
            "probe_wall_seconds",
            "steps_per_second",
            "cells_per_second",
            "estimated_epoch_seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
        if (
            item.get("observed_probe_steps") != 128
            or item.get("accepted") is not True
            or item.get("failure") is not None
            or any(
                not isinstance(item.get(key), (int, float)) or item[key] <= 0
                for key in numeric_positive
            )
            or item["peak_reserved_bytes"] > threshold
        ):
            raise ValueError(f"invalid dataset capacity observation: {item.get('dataset_id')}")
    return payload


def compare_batch_capacity_receipts(
    *,
    batch64_path: str | Path,
    batch256_path: str | Path,
    source_commit: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select 64 or 256 from matched systems evidence without metric tuning."""

    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git hash")
    path64 = Path(batch64_path)
    path256 = Path(batch256_path)
    receipt64 = _load_passed_receipt(path64, expected_batch_size=64)
    receipt256 = _load_passed_receipt(path256, expected_batch_size=256)
    if (
        receipt64["device_name"] != receipt256["device_name"]
        or receipt64["device_total_bytes"] != receipt256["device_total_bytes"]
        or receipt64["run_seed"] != receipt256["run_seed"]
    ):
        raise ValueError("batch comparison receipts do not share hardware and run seed")

    observations64 = {item["dataset_id"]: item for item in receipt64["candidates"][0]["datasets"]}
    observations256 = {item["dataset_id"]: item for item in receipt256["candidates"][0]["datasets"]}
    rows = []
    for dataset_id in DATASET_IDS:
        item64 = observations64[dataset_id]
        item256 = observations256[dataset_id]
        threshold = int(receipt256["acceptance_threshold_bytes"])
        rows.append(
            {
                "dataset_id": dataset_id,
                "batch64_estimated_epoch_seconds": item64["estimated_epoch_seconds"],
                "batch256_estimated_epoch_seconds": item256["estimated_epoch_seconds"],
                "batch256_epoch_speedup": (
                    item64["estimated_epoch_seconds"] / item256["estimated_epoch_seconds"]
                ),
                "batch64_cells_per_second": item64["cells_per_second"],
                "batch256_cells_per_second": item256["cells_per_second"],
                "batch256_cell_throughput_ratio": (
                    item256["cells_per_second"] / item64["cells_per_second"]
                ),
                "batch64_peak_reserved_bytes": item64["peak_reserved_bytes"],
                "batch256_peak_reserved_bytes": item256["peak_reserved_bytes"],
                "batch256_memory_headroom_bytes": (threshold - item256["peak_reserved_bytes"]),
                "batch256_memory_headroom_fraction": (
                    (threshold - item256["peak_reserved_bytes"]) / threshold
                ),
            }
        )
    minimum_speedup = min(item["batch256_epoch_speedup"] for item in rows)
    minimum_headroom = min(item["batch256_memory_headroom_fraction"] for item in rows)
    selected_batch_size = (
        256
        if minimum_speedup >= MINIMUM_EPOCH_SPEEDUP
        and minimum_headroom >= MINIMUM_MEMORY_HEADROOM_FRACTION
        else 64
    )
    result: dict[str, Any] = {
        "schema_version": "batch-capacity-comparison-v1",
        "source_commit_attestation": source_commit,
        "selection_uses_test_metrics": False,
        "allocator_config": CUDA_ALLOCATOR_CONFIG,
        "prototype_count": 16384,
        "probe_steps_per_dataset": 128,
        "selection_rule": {
            "minimum_epoch_speedup_for_batch256": MINIMUM_EPOCH_SPEEDUP,
            "minimum_memory_headroom_fraction_for_batch256": (MINIMUM_MEMORY_HEADROOM_FRACTION),
        },
        "inputs": {
            "batch64_sha256": sha256_file(path64),
            "batch256_sha256": sha256_file(path256),
        },
        "minimum_batch256_epoch_speedup": minimum_speedup,
        "minimum_batch256_memory_headroom_fraction": minimum_headroom,
        "selected_batch_size": selected_batch_size,
        "datasets": rows,
    }
    atomic_json(output_path, result)
    return result
