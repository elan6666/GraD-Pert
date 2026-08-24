from __future__ import annotations

import json
from pathlib import Path

from gradpert.data.registry import DATASET_IDS
from gradpert.training.performance import compare_batch_capacity_receipts


def _receipt(batch_size: int, epoch_seconds: float, peak_reserved: int) -> dict[str, object]:
    datasets = [
        {
            "dataset_id": dataset_id,
            "observed_probe_steps": 128,
            "observed_probe_cells": 128 * batch_size,
            "probe_wall_seconds": 128.0,
            "steps_per_second": 1.0,
            "cells_per_second": float(batch_size),
            "estimated_epoch_seconds": epoch_seconds,
            "peak_allocated_bytes": peak_reserved - 100,
            "peak_reserved_bytes": peak_reserved,
            "accepted": True,
            "failure": None,
        }
        for dataset_id in DATASET_IDS
    ]
    return {
        "schema_version": "prototype-head-capacity-v2",
        "status": "development_capacity_passed",
        "batch_size": batch_size,
        "pytorch_alloc_conf": "expandable_segments:True",
        "selected_prototype_count": 16384,
        "capacity_probe_steps": 128,
        "acceptance_threshold_bytes": 1000,
        "device_name": "test GPU",
        "device_total_bytes": 2000,
        "run_seed": 1,
        "candidates": [{"prototype_count": 16384, "accepted": True, "datasets": datasets}],
    }


def test_batch_comparison_selects_256_only_with_speed_and_headroom(tmp_path: Path) -> None:
    path64 = tmp_path / "64.json"
    path256 = tmp_path / "256.json"
    path64.write_text(json.dumps(_receipt(64, 400.0, 700)), encoding="utf-8")
    path256.write_text(json.dumps(_receipt(256, 100.0, 800)), encoding="utf-8")

    result = compare_batch_capacity_receipts(
        batch64_path=path64,
        batch256_path=path256,
        source_commit="a" * 40,
        output_path=tmp_path / "comparison.json",
    )

    assert result["selected_batch_size"] == 256
    assert result["minimum_batch256_epoch_speedup"] == 4.0
    assert result["selection_uses_test_metrics"] is False


def test_batch_comparison_falls_back_to_64_when_headroom_is_small(tmp_path: Path) -> None:
    path64 = tmp_path / "64.json"
    path256 = tmp_path / "256.json"
    path64.write_text(json.dumps(_receipt(64, 400.0, 700)), encoding="utf-8")
    path256.write_text(json.dumps(_receipt(256, 100.0, 950)), encoding="utf-8")

    result = compare_batch_capacity_receipts(
        batch64_path=path64,
        batch256_path=path256,
        source_commit="b" * 40,
        output_path=tmp_path / "comparison.json",
    )

    assert result["selected_batch_size"] == 64


def test_tracked_development_comparison_selects_batch256() -> None:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "registry/capacity/batch_comparison.development.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["selected_batch_size"] == 256
    assert payload["minimum_batch256_epoch_speedup"] > 3.2
    assert payload["minimum_batch256_memory_headroom_fraction"] > 0.24
    assert payload["selection_uses_test_metrics"] is False
