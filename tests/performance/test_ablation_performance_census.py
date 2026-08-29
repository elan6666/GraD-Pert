from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/performance/ablation_performance_census.py"
MATRIX = PROJECT_ROOT / "configs/ablations/nadig_jurkat/matrix.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ablation_performance_census_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load ablation performance census script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def census() -> ModuleType:
    return _load_script()


@pytest.fixture(scope="module")
def matrix_sha256() -> str:
    return hashlib.sha256(MATRIX.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bindings(census: ModuleType, matrix_sha256: str):
    return census.bind_matrix_variants(
        MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
    )


def _batch(census: ModuleType, global_step: int, *, suffix: str = ""):
    batch_size = census.EXACT_TRAIN_BATCH_SIZE
    condition_ids = [f"condition-{index % 5}{suffix}" for index in range(batch_size)]
    return census.OrderedBatchIdentity.create(
        global_step=global_step,
        row_ids=[f"row-{global_step}-{index}{suffix}" for index in range(batch_size)],
        condition_ids=condition_ids,
        control_row_ids=[f"control-{global_step}-{index}{suffix}" for index in range(batch_size)],
        active_anchor_ids=[[f"anchor-{index % 7}{suffix}"] for index in range(batch_size)],
        actual_batch_size=batch_size,
        unique_condition_count=len(set(condition_ids)),
    )


def _batches(census: ModuleType, count: int):
    return tuple(_batch(census, index) for index in range(count))


def _training_only() -> dict[str, object]:
    return {
        "scope": "performance_training_only",
        "real_canonical_evaluation_constructor_count": 0,
        "validation_cache_materialized": False,
        "validation_callback_count": 0,
        "validation_accessed": False,
        "test_truth_accessed": False,
        "truth_access_attempts": [],
    }


def _stage_payload(
    census: ModuleType,
    binding,
    stage_id: str,
    expected_batches,
    *,
    status: str = "complete",
) -> dict[str, object]:
    protocol = census.STAGE_PROTOCOLS[stage_id]
    observed_count = protocol.total_steps if status == "complete" else 0
    timing_samples = (
        [float(100 + index) for index in range(protocol.measured_steps)]
        if protocol.timing_acceptance and status == "complete"
        else []
    )
    return {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "stage_id": stage_id,
        "protocol": protocol.payload(),
        "status": status,
        "training_only_evidence": _training_only(),
        "instrumentation": {
            "timing_acceptance": protocol.timing_acceptance,
            "heavy_capacity_instrumentation": protocol.heavy_capacity_instrumentation,
            "torch_profiler_enabled": protocol.torch_profiler_enabled,
        },
        "attempted_batch_count": observed_count,
        "completed_step_count": observed_count,
        "observed_step_count": observed_count,
        "batches": [batch.payload() for batch in expected_batches[:observed_count]],
        "timing_samples_ms": timing_samples,
        "torch_profiler_trace_sha256": ("a" * 64 if stage_id == "diagnostic_profile" else None),
        "torch_profiler_table_sha256": ("b" * 64 if stage_id == "diagnostic_profile" else None),
        "primary_failure": None if status == "complete" else {"type": "RuntimeError"},
        "teardown_failures": [],
    }


def _write_receipt(path: Path, payload: dict[str, object]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_stage_protocols_are_frozen_and_separate_timing_from_profile(census: ModuleType) -> None:
    protocols = census.STAGE_PROTOCOLS
    assert (protocols["p1_capacity"].warmup_steps, protocols["p1_capacity"].measured_steps) == (
        0,
        1,
    )
    assert protocols["p1_capacity"].heavy_capacity_instrumentation is True
    assert protocols["p1_capacity"].timing_acceptance is False
    assert (protocols["p2_timing"].warmup_steps, protocols["p2_timing"].measured_steps) == (
        5,
        20,
    )
    assert (protocols["p3_timing"].warmup_steps, protocols["p3_timing"].measured_steps) == (
        10,
        100,
    )
    assert protocols["diagnostic_profile"].profiler_schedule == {
        "wait": 1,
        "warmup": 1,
        "active": 3,
    }
    assert protocols["diagnostic_profile"].total_steps == 5
    assert protocols["diagnostic_profile"].timing_acceptance is False


def test_exact_matrix_binding_covers_25_rows_and_rejects_tampering(
    census: ModuleType,
    bindings,
    matrix_sha256: str,
) -> None:
    assert len(bindings) == 25
    assert bindings[0].variant_id == census.A0_VARIANT_ID
    assert [binding.matrix_row_index for binding in bindings] == list(range(25))
    assert all(binding.run_seed == 1 for binding in bindings)
    with pytest.raises(ValueError, match="matrix SHA-256 differs"):
        census.bind_matrix_variants(
            MATRIX,
            repository_root=PROJECT_ROOT,
            expected_matrix_sha256="0" * 64,
        )
    selected = census.bind_matrix_variant(
        MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
        variant_id="d2_control_transformer",
    )
    assert selected.semantic_factor == "decoder_mode"


def test_batch_identity_is_order_sensitive_and_prefix_bound(census: ModuleType) -> None:
    expected = _batches(census, 3)
    assert census.batch_sequence_sha256(expected) == census.batch_sequence_sha256(expected)
    census.require_batch_prefix(expected[:2], expected)
    changed = list(expected[:2])
    changed[1] = _batch(census, 1, suffix="-changed")
    with pytest.raises(ValueError, match="frozen prefix"):
        census.require_batch_prefix(changed, expected)
    with pytest.raises(ValueError, match="zero-based contiguous"):
        census.batch_sequence_sha256((expected[1],))


def test_training_only_evidence_fails_closed(census: ModuleType) -> None:
    census.require_training_only_evidence(_training_only())
    accessed = _training_only()
    accessed["validation_accessed"] = True
    with pytest.raises(ValueError, match="validation_accessed"):
        census.require_training_only_evidence(accessed)


def _stable_promotion(census: ModuleType, *, selected: bool = False):
    return census.decide_p3_promotion(
        variant_id="m1_single_string_gat",
        step_wall_ms=[100.0] * 20,
        reserved_gpu_bytes=[10 * census.GIB] * 20,
        free_gpu_bytes=[50 * census.GIB] * 20,
        total_gpu_bytes=80 * census.GIB,
        selected_implementation_target=selected,
    )


def test_promotion_is_stable_when_no_preregistered_trigger_fires(census: ModuleType) -> None:
    decision = _stable_promotion(census)
    assert decision.promoted is False
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("step_wall", "reserved", "free", "selected", "expected_reason"),
    [
        (
            [100.0, 130.0] * 10,
            [10] * 20,
            [50] * 20,
            False,
            "relative_mad_above_limit",
        ),
        (
            [100.0] * 8 + [130.0] * 2 + [100.0] * 8 + [130.0] * 2,
            [10] * 20,
            [50] * 20,
            False,
            "p95_over_p50_above_limit",
        ),
        (
            [100.0] * 10 + [115.0] * 10,
            [10] * 20,
            [50] * 20,
            False,
            "half_drift_above_limit",
        ),
        (
            [100.0] * 20,
            [10] * 15 + [11] * 5,
            [50] * 20,
            False,
            "reserved_memory_growth_above_limit",
        ),
        (
            [100.0] * 20,
            [10] * 20,
            [13] * 20,
            False,
            "near_absolute_gpu_headroom",
        ),
        (
            [100.0] * 20,
            [10] * 20,
            [50] * 20,
            True,
            "selected_implementation_target",
        ),
    ],
)
def test_every_preregistered_p3_promotion_trigger(
    census: ModuleType,
    step_wall: list[float],
    reserved: list[int],
    free: list[int],
    selected: bool,
    expected_reason: str,
) -> None:
    decision = census.decide_p3_promotion(
        variant_id="m1_single_string_gat",
        step_wall_ms=step_wall,
        reserved_gpu_bytes=[value * census.GIB for value in reserved],
        free_gpu_bytes=[value * census.GIB for value in free],
        total_gpu_bytes=80 * census.GIB,
        selected_implementation_target=selected,
    )
    assert decision.promoted is True
    assert expected_reason in decision.reasons


def test_non_a0_promotion_always_pairs_the_reference(census: ModuleType) -> None:
    a0 = census.PromotionDecision(
        variant_id=census.A0_VARIANT_ID,
        promoted=False,
        reasons=(),
        statistics={},
    )
    candidate = _stable_promotion(census, selected=True)
    paired = census.pair_a0_promotion((a0, candidate))
    assert paired[0].promoted is True
    assert "paired_reference_a0" in paired[0].reasons
    assert paired[1].promoted is True


def test_attempt_roots_are_fresh_numbered_and_fail_on_unexpected_entries(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    first = census.claim_fresh_attempt_root(
        tmp_path,
        variant_id=census.A0_VARIANT_ID,
        stage_id="p1_capacity",
    )
    second = census.claim_fresh_attempt_root(
        tmp_path,
        variant_id=census.A0_VARIANT_ID,
        stage_id="p1_capacity",
    )
    assert first.name == "attempt-001"
    assert second.name == "attempt-002"
    (second.parent / "manual-output.txt").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entry"):
        census.claim_fresh_attempt_root(
            tmp_path,
            variant_id=census.A0_VARIANT_ID,
            stage_id="p1_capacity",
        )


def test_atomic_failure_receipt_preserves_primary_and_teardown_errors(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "failure.json"

    def operation(observer):
        assert observer is not None
        observer.entered("student_local", {"reserved_gpu_bytes": 123})
        raise MemoryError("synthetic oom")

    def teardown() -> None:
        raise RuntimeError("synthetic teardown")

    with pytest.raises(MemoryError, match="synthetic oom"):
        census.execute_with_atomic_stage_receipt(
            receipt_path=receipt_path,
            base_receipt={"scope": "performance_training_only"},
            operation=operation,
            optional_step_observer_available=True,
            teardown=teardown,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["last_entered_stage"] == "student_local"
    assert receipt["last_completed_stage"] is None
    assert receipt["primary_failure"]["type"] == "MemoryError"
    assert receipt["teardown_failures"][0]["type"] == "RuntimeError"


def _records_with_a0_timing(
    census: ModuleType,
    bindings,
    expected_batches,
    tmp_path: Path,
    *,
    include_p3: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for binding in bindings:
        if binding.variant_id != census.A0_VARIANT_ID:
            records.append(
                {
                    "variant_id": binding.variant_id,
                    "state": "unavailable_preflight",
                    "disposition_reason": "synthetic unavailable",
                    "stages": {},
                }
            )
            continue
        stages: dict[str, object] = {}
        for stage_id in ("p1_capacity", "p2_timing"):
            stages[stage_id] = _write_receipt(
                tmp_path / binding.variant_id / f"{stage_id}.json",
                _stage_payload(census, binding, stage_id, expected_batches),
            )
        if include_p3:
            stages["p3_timing"] = _write_receipt(
                tmp_path / binding.variant_id / "p3_timing.json",
                _stage_payload(census, binding, "p3_timing", expected_batches),
            )
        records.append(
            {
                "variant_id": binding.variant_id,
                "state": "p3_complete" if include_p3 else "p2_complete",
                "stages": stages,
            }
        )
    return records


def test_aggregate_requires_exact_25_order_and_separates_20_and_100_panels(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    expected_batches = _batches(census, census.STAGE_PROTOCOLS["p3_timing"].total_steps)
    records = _records_with_a0_timing(
        census,
        bindings,
        expected_batches,
        tmp_path,
        include_p3=True,
    )
    report = census.aggregate_census_report(
        bindings=bindings,
        row_records=records,
        expected_batches=expected_batches,
    )
    assert report["status"] == "complete_with_preregistered_unavailable_or_capacity_failures"
    assert report["measured_20_row_count"] == 1
    assert report["measured_100_row_count"] == 1
    assert [row["variant_id"] for row in report["timing_panels"]["p2_20_measured_steps"]] == [
        census.A0_VARIANT_ID
    ]
    assert [row["variant_id"] for row in report["timing_panels"]["p3_100_measured_steps"]] == [
        census.A0_VARIANT_ID
    ]
    assert report["timing_panels_must_not_be_ranked_together"] is True
    with pytest.raises(ValueError, match="exact matrix order"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=list(reversed(records)),
            expected_batches=expected_batches,
        )


def test_aggregate_rejects_tampered_receipt_and_profiler_timing_mixing(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    expected_batches = _batches(census, census.STAGE_PROTOCOLS["p3_timing"].total_steps)
    records = _records_with_a0_timing(
        census,
        bindings,
        expected_batches,
        tmp_path,
        include_p3=False,
    )
    a0 = records[0]
    p2_pointer = a0["stages"]["p2_timing"]
    assert isinstance(p2_pointer, dict)
    receipt_path = Path(p2_pointer["receipt_path"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["torch_profiler_trace_sha256"] = "c" * 64
    receipt_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    p2_pointer["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cannot enter a timing-acceptance receipt"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=records,
            expected_batches=expected_batches,
        )
    p2_pointer["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=records,
            expected_batches=expected_batches,
        )
