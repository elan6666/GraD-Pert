"""Reject confounded/partial timing evidence and recompute rank maxima."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_benchmarks", Path(__file__).resolve().parents[2] / "scripts/v2/compare_benchmarks.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def receipt(seconds):
    ranks = []
    for rank in (0, 1):
        ranks.append(
            {
                "rank": rank,
                "physical_gpu": str(rank),
                "view_generator_after_sha256": "after",
                "update_seconds": [seconds - rank + i for i in (0, 2, 1)],
                "data_wait_seconds": [1 + rank] * 3,
                "profile": None,
                "peak_allocated_bytes": 1000,
                "peak_reserved_bytes": 2000,
            }
        )
    return {
        "status": "passed",
        "kind": "benchmark_only",
        "steps_requested": 5,
        "steps_completed": 5,
        "warmup_steps": 2,
        "world_size": 2,
        "gpu": "0,1",
        "profile_last_update": False,
        "profile_memory": False,
        "sync_phase_timing": False,
        "source": {"dirty": False, "formal_eligible": True, "commit": "a", "published_commit": "a"},
        "environment": {},
        "data": {},
        "batch_schedule_hashes": ["a", "b", "c", "d", "e"],
        "ordered_batch_schedule_sha256": "ordered",
        "view_generator_before_sha256": "before",
        "optimizer_routes": [],
        "hardware": {
            "affinity": [0, 1],
            "allocator": "expected",
            "torch_threads": 2,
            "torch_interop_threads": 64,
        },
        "rank_measurements": ranks,
        "measured_update_seconds": [seconds + i for i in (0, 2, 1)],
        "measured_step_seconds_including_data_wait": [seconds + i for i in (1, 3, 2)],
    }


def test_reports_two_replicates_and_recomputes_whole_step_rank_max():
    result = MODULE.summarize([receipt(10), receipt(8), receipt(8.5), receipt(10.5)])
    assert result["status"] == "comparable"
    assert result["both_B_faster_than_both_A"]
    assert result["total_time_speed_ratio_A_over_B"] == pytest.approx(12.25 / 10.25)
    assert result["rows"][0]["total_mean_seconds"] == 12
    assert len(result["pair_ratios"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("steps_completed", 4),
        ("kind", "profile_only"),
        ("sync_phase_timing", True),
        ("ordered_batch_schedule_sha256", "different"),
        ("measured_update_seconds", [1, 2, 3]),
    ],
)
def test_rejects_partial_confounded_or_inconsistent_receipts(field, value):
    rows = [receipt(10) for _ in range(4)]
    rows[1][field] = value
    with pytest.raises(ValueError):
        MODULE.summarize(rows)


def test_rejects_changed_rank_rng_or_nonfinite_duration():
    rows = [receipt(10) for _ in range(4)]
    changed = copy.deepcopy(rows)
    changed[2]["rank_measurements"][1]["view_generator_after_sha256"] = "wrong"
    with pytest.raises(ValueError, match="rank view_generator"):
        MODULE.summarize(changed)
    rows[1]["rank_measurements"][0]["update_seconds"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid timing"):
        MODULE.summarize(rows)


def test_cpu_prefetch_factor_requires_identical_configs_and_only_expected_flags():
    payloads = [{"model": {"parameters": {"width": {"value": 256}}}} for _ in range(2)]
    rows = [receipt(10) for _ in range(4)]
    for record, flag in zip(rows, (False, True, True, False), strict=True):
        record["cpu_prefetch_diagnostic_only"] = flag
    MODULE.validate_execution_factor(copy.deepcopy(payloads), rows, "cpu_prefetch")
    changed = copy.deepcopy(rows)
    changed[1]["sequence_checkpoint_disabled_diagnostic_only"] = True
    with pytest.raises(ValueError, match="checkpoint override"):
        MODULE.validate_execution_factor(copy.deepcopy(payloads), changed, "cpu_prefetch")
    changed = copy.deepcopy(rows)
    changed[2]["cpu_prefetch_diagnostic_only"] = False
    with pytest.raises(ValueError, match="prefetch execution flag"):
        MODULE.validate_execution_factor(copy.deepcopy(payloads), changed, "cpu_prefetch")
    payloads[1]["model"]["parameters"]["width"]["value"] = 128
    with pytest.raises(ValueError, match="configuration factor"):
        MODULE.validate_execution_factor(payloads, rows, "cpu_prefetch")


def test_fused_sinkhorn_rejects_hidden_prefetch_and_missing_activation():
    payloads = [{"model": {"parameters": {}}} for _ in range(2)]
    rows = [receipt(10) for _ in range(4)]
    for record, flag in zip(rows, (False, True, True, False), strict=True):
        record["fused_sinkhorn_diagnostic_only"] = flag
        record["fused_sinkhorn_module_count"] = 40 if flag else 0
    MODULE.validate_execution_factor(copy.deepcopy(payloads), rows, "fused_sinkhorn")
    rows[1]["cpu_prefetch_diagnostic_only"] = True
    with pytest.raises(ValueError, match="CPU prefetch"):
        MODULE.validate_execution_factor(copy.deepcopy(payloads), rows, "fused_sinkhorn")
    rows[1]["cpu_prefetch_diagnostic_only"] = False
    rows[2]["fused_sinkhorn_module_count"] = 0
    with pytest.raises(ValueError, match="no fused modules"):
        MODULE.validate_execution_factor(copy.deepcopy(payloads), rows, "fused_sinkhorn")
