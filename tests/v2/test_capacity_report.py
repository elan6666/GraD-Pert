import json
import runpy
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
collect = runpy.run_path(str(ROOT / "scripts/v2/capacity_report.py"))["collect"]
CONFIG = ROOT / "configs/v2/capacity/m8/gradpert_v2/nadig_jurkat.yaml"


def valid_receipt():
    return {
        "kind": "capacity_only",
        "status": "passed",
        "steps_completed": 128,
        "config_sha256": sha256_file(CONFIG),
        "data": {
            "training_expression_policy": {"exclude_test_target_expression": True},
            "loss_protocol": {
                "version": "unified-global-population-v1",
                "reduction": "row_mean",
                "koleo_population": "complete_global_effective_batch",
                "ibot": "masked_tokens_per_cell_then_valid_cell_population",
            },
        },
        "source": {
            "dirty": False,
            "commit": "a" * 40,
            "published_commit": "a" * 40,
            "formal_eligible": True,
        },
        "resume_checkpoint_sha256": "b" * 64,
        "inference_shape": [300, 5000],
        "measured_update_seconds": [2.0] * 120,
        "cells_per_second": 4.0,
        "gpu": "0",
        "peak_allocated_bytes": 8000,
        "peak_reserved_bytes": 10000,
    }


def test_report_preserves_microbatch_and_does_not_invent_missing_communication(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(valid_receipt()))
    row = collect(path, CONFIG)
    assert row["microbatch"] == row["effective_batch"] == 8
    assert row["gradient_reduction_mean_seconds"] is None
    assert row["receipt_sha256"] == sha256_file(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("kind", "integration_only"),
        ("steps_completed", 8),
        ("inference_shape", [2, 5000]),
        ("config_sha256", "wrong"),
        ("resume_checkpoint_sha256", None),
        ("measured_update_seconds", [2.0]),
        ("world_size", 2),
        ("data", {}),
        ("data", {"training_expression_policy": {"exclude_test_target_expression": False}}),
    ],
)
def test_report_rejects_incomplete_or_mismatched_evidence(tmp_path, field, value):
    data = valid_receipt()
    data[field] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collect(path, CONFIG)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reduction", "condition_mean"),
        ("koleo_population", "rank_local"),
        ("ibot", "all_masked_tokens"),
        ("version", "legacy"),
    ],
)
def test_report_rejects_changed_loss_population(tmp_path, field, value):
    data = valid_receipt()
    data["data"]["loss_protocol"][field] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unified loss protocol"):
        collect(path, CONFIG)
