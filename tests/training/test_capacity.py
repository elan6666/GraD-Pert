from __future__ import annotations

import json
from pathlib import Path

from gradpert.training.capacity import (
    CAPACITY_PROBE_STEPS,
    CUDA_ALLOCATOR_CONFIG,
    PROTOTYPE_CANDIDATES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prototype_candidates_are_frozen_largest_first() -> None:
    assert PROTOTYPE_CANDIDATES == (16384, 8192)
    assert CAPACITY_PROBE_STEPS == 128
    assert CUDA_ALLOCATOR_CONFIG == "expandable_segments:True"


def test_capacity_gate_persists_failure_evidence_before_raising() -> None:
    source = (PROJECT_ROOT / "src/gradpert/training/capacity.py").read_text(encoding="utf-8")
    failed_write = source.index('write_receipt("development_capacity_failed")')
    terminal_raise = source.index(
        'raise RuntimeError("no frozen prototype-head candidate passed the server fit gate")'
    )
    assert failed_write < terminal_raise


def test_capacity_gate_records_throughput_for_batch_comparison() -> None:
    source = (PROJECT_ROOT / "src/gradpert/training/capacity.py").read_text(encoding="utf-8")
    for field in (
        "observed_probe_cells",
        "probe_wall_seconds",
        "steps_per_second",
        "cells_per_second",
        "estimated_epoch_seconds",
    ):
        assert field in source


def test_development_capacity_receipt_covers_all_five_datasets() -> None:
    receipt = json.loads(
        (PROJECT_ROOT / "registry/capacity/prototype_head.development.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema_version"] == "prototype-head-capacity-summary-v2"
    assert receipt["formal_eligible"] is False
    assert receipt["full_server_receipt_sync_status"] == "pending_allowlisted_sync"
    assert receipt["capacity_probe_steps"] == CAPACITY_PROBE_STEPS
    assert receipt["selected_prototype_count"] == 16384
    selected = receipt["candidate_decisions"][-1]
    assert selected["prototype_count"] == 16384
    assert selected["accepted"] is True
    assert {item["dataset_id"] for item in selected["datasets"]} == {
        "replogle_k562_essential",
        "replogle_rpe1_essential",
        "nadig_jurkat",
        "nadig_hepg2",
        "norman",
    }
    assert all(
        item["observed_probe_steps"] == CAPACITY_PROBE_STEPS
        and item["peak_reserved_bytes"] <= receipt["acceptance_threshold_bytes"]
        for item in selected["datasets"]
    )
    assert [item["prototype_count"] for item in receipt["candidate_decisions"]] == [
        65536,
        32768,
        16384,
    ]
    assert all(not item["accepted"] for item in receipt["candidate_decisions"][:-1])
