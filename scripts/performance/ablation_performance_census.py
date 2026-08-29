#!/usr/bin/env python3
"""Frozen protocol and receipt tooling for the 25-row performance census.

This module does not implement or launch model training.  It binds the exact
schema-v2 matrix, defines the only accepted bounded measurement stages, seals
ordered training-batch identities, and aggregates already-produced worker
receipts without mixing profiler samples into timing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeVar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.execution.ablation_matrix import (  # noqa: E402
    SUCCESSOR_V2_CONTRACT,
    SUCCESSOR_V2_MATRIX_ID,
)
from gradpert.hashing import sha256_file  # noqa: E402

GIB = 1024**3
A0_VARIANT_ID = "a0_ratio_ring_half"
MATRIX_SCHEMA_VERSION = "2"
MATRIX_ROW_COUNT = 25
EXACT_TRAIN_BATCH_SIZE = 256
EXACT_EVAL_BATCH_SIZE = 256
EXACT_PROTOTYPE_COUNT = 16384

StageId = Literal["p1_capacity", "p2_timing", "p3_timing", "diagnostic_profile"]
RowState = Literal[
    "unavailable_preflight",
    "blocked_missing_prerequisite",
    "deferred_resource_busy",
    "capacity_failed",
    "execution_failed",
    "p1_pass",
    "p2_complete",
    "p3_complete",
]

_T = TypeVar("_T")


@dataclass(frozen=True)
class StageProtocol:
    stage_id: StageId
    warmup_steps: int
    measured_steps: int
    timing_acceptance: bool
    heavy_capacity_instrumentation: bool
    torch_profiler_enabled: bool
    profiler_schedule: dict[str, int] | None = None

    @property
    def total_steps(self) -> int:
        if self.profiler_schedule is not None:
            return sum(self.profiler_schedule.values())
        return self.warmup_steps + self.measured_steps

    def payload(self) -> dict[str, object]:
        return {**asdict(self), "total_steps": self.total_steps}


STAGE_PROTOCOLS: dict[StageId, StageProtocol] = {
    "p1_capacity": StageProtocol(
        stage_id="p1_capacity",
        warmup_steps=0,
        measured_steps=1,
        timing_acceptance=False,
        heavy_capacity_instrumentation=True,
        torch_profiler_enabled=False,
    ),
    "p2_timing": StageProtocol(
        stage_id="p2_timing",
        warmup_steps=5,
        measured_steps=20,
        timing_acceptance=True,
        heavy_capacity_instrumentation=False,
        torch_profiler_enabled=False,
    ),
    "p3_timing": StageProtocol(
        stage_id="p3_timing",
        warmup_steps=10,
        measured_steps=100,
        timing_acceptance=True,
        heavy_capacity_instrumentation=False,
        torch_profiler_enabled=False,
    ),
    "diagnostic_profile": StageProtocol(
        stage_id="diagnostic_profile",
        warmup_steps=1,
        measured_steps=3,
        timing_acceptance=False,
        heavy_capacity_instrumentation=False,
        torch_profiler_enabled=True,
        profiler_schedule={"wait": 1, "warmup": 1, "active": 3},
    ),
}


@dataclass(frozen=True)
class FrozenVariantBinding:
    matrix_path: str
    matrix_sha256: str
    matrix_id: str
    matrix_schema_version: str
    matrix_row_index: int
    variant_id: str
    config_path: str
    config_sha256: str
    run_seed: int
    semantic_factor: str | None
    declared_parameter_diffs: tuple[str, ...]
    genept_preflight_required: bool

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrderedBatchIdentity:
    global_step: int
    row_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    control_row_ids: tuple[str, ...]
    active_anchor_ids: tuple[tuple[str, ...], ...]
    actual_batch_size: int
    unique_condition_count: int

    @classmethod
    def create(
        cls,
        *,
        global_step: int,
        row_ids: Sequence[str],
        condition_ids: Sequence[str],
        control_row_ids: Sequence[str],
        active_anchor_ids: Sequence[Sequence[str]],
        actual_batch_size: int,
        unique_condition_count: int,
    ) -> OrderedBatchIdentity:
        identity = cls(
            global_step=global_step,
            row_ids=tuple(row_ids),
            condition_ids=tuple(condition_ids),
            control_row_ids=tuple(control_row_ids),
            active_anchor_ids=tuple(tuple(anchors) for anchors in active_anchor_ids),
            actual_batch_size=actual_batch_size,
            unique_condition_count=unique_condition_count,
        )
        identity.validate()
        return identity

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> OrderedBatchIdentity:
        anchors = payload.get("active_anchor_ids")
        if not isinstance(anchors, list) or any(not isinstance(value, list) for value in anchors):
            raise ValueError("batch active-anchor identities are malformed")
        string_fields: dict[str, tuple[str, ...]] = {}
        for name in ("row_ids", "condition_ids", "control_row_ids"):
            raw = payload.get(name)
            if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
                raise ValueError(f"batch {name} are malformed")
            string_fields[name] = tuple(raw)
        if any(any(not isinstance(value, str) for value in group) for group in anchors):
            raise ValueError("batch anchor IDs must be strings")
        global_step = payload.get("global_step")
        actual_batch_size = payload.get("actual_batch_size")
        unique_condition_count = payload.get("unique_condition_count")
        if not all(
            isinstance(value, int)
            for value in (global_step, actual_batch_size, unique_condition_count)
        ):
            raise ValueError("batch numeric identity fields are malformed")
        return cls.create(
            global_step=global_step,
            row_ids=string_fields["row_ids"],
            condition_ids=string_fields["condition_ids"],
            control_row_ids=string_fields["control_row_ids"],
            active_anchor_ids=anchors,
            actual_batch_size=actual_batch_size,
            unique_condition_count=unique_condition_count,
        )

    def validate(self) -> None:
        if self.global_step < 0:
            raise ValueError("batch global step must be nonnegative")
        if self.actual_batch_size <= 0:
            raise ValueError("actual batch size must be positive")
        fields = (
            self.row_ids,
            self.condition_ids,
            self.control_row_ids,
            self.active_anchor_ids,
        )
        if any(len(field) != self.actual_batch_size for field in fields):
            raise ValueError("ordered batch identity lengths differ from actual batch size")
        if any(not value for field in self.active_anchor_ids for value in field):
            raise ValueError("active anchor IDs must be nonempty strings")
        if any(not value for field in fields[:3] for value in field):
            raise ValueError("batch row/condition/control IDs must be nonempty strings")
        if self.unique_condition_count != len(set(self.condition_ids)):
            raise ValueError("unique condition count differs from ordered condition IDs")

    def payload(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "row_ids": list(self.row_ids),
            "condition_ids": list(self.condition_ids),
            "control_row_ids": list(self.control_row_ids),
            "active_anchor_ids": [list(values) for values in self.active_anchor_ids],
            "actual_batch_size": self.actual_batch_size,
            "unique_condition_count": self.unique_condition_count,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())


@dataclass(frozen=True)
class PromotionThresholds:
    relative_mad: float = 0.10
    p95_over_p50: float = 1.25
    half_drift: float = 0.10
    reserved_growth: float = 0.05
    minimum_free_bytes: int = 4 * GIB
    minimum_free_fraction: float = 0.15
    near_headroom_bytes: int = 2 * GIB
    near_headroom_fraction_points: float = 0.05


DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds()


@dataclass(frozen=True)
class PromotionDecision:
    variant_id: str
    promoted: bool
    reasons: tuple[str, ...]
    statistics: dict[str, float]


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def bind_matrix_variants(
    matrix_path: str | Path,
    *,
    repository_root: str | Path,
    expected_matrix_sha256: str,
) -> tuple[FrozenVariantBinding, ...]:
    """Bind the exact schema-v2 matrix and every immutable row/config identity."""

    _validate_sha256(expected_matrix_sha256, field="expected matrix SHA-256")
    root = Path(repository_root).resolve(strict=True)
    matrix = Path(matrix_path).resolve(strict=True)
    if sha256_file(matrix) != expected_matrix_sha256:
        raise ValueError("performance-census matrix SHA-256 differs")
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != MATRIX_SCHEMA_VERSION
        or payload.get("matrix_id") != SUCCESSOR_V2_MATRIX_ID
        or payload.get("row_count") != MATRIX_ROW_COUNT
    ):
        raise ValueError("performance census requires the exact schema-v2 25-row matrix")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != MATRIX_ROW_COUNT:
        raise ValueError("performance census matrix row payload is malformed")
    raw_variant_ids = [raw.get("variant_id") if isinstance(raw, dict) else None for raw in raw_rows]
    if (
        any(not isinstance(variant_id, str) for variant_id in raw_variant_ids)
        or len(set(raw_variant_ids)) != MATRIX_ROW_COUNT
        or set(raw_variant_ids) != set(SUCCESSOR_V2_CONTRACT)
    ):
        raise ValueError("performance census matrix variant set differs")
    baseline_relative = (
        "configs/ablations/nadig_jurkat/a0_ratio_ring_half/gradpert_b2/nadig_jurkat.yaml"
    )
    baseline_config = load_experiment_config((root / baseline_relative).resolve(strict=True))
    baseline_parameters = {
        name: parameter.value for name, parameter in baseline_config.model.parameters.items()
    }
    bindings: list[FrozenVariantBinding] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError("performance census matrix row is malformed")
        variant_id = raw.get("variant_id")
        relative_config = raw.get("config_path")
        config_sha256 = raw.get("config_sha256")
        semantic_factor = raw.get("semantic_factor")
        declared_diffs = raw.get("declared_parameter_diffs")
        requires_genept = raw.get("genept_preflight_required")
        if (
            not isinstance(variant_id, str)
            or not isinstance(relative_config, str)
            or relative_config
            != f"configs/ablations/nadig_jurkat/{variant_id}/gradpert_b2/nadig_jurkat.yaml"
            or not isinstance(config_sha256, str)
            or semantic_factor != SUCCESSOR_V2_CONTRACT[variant_id][0]
            or not isinstance(declared_diffs, list)
            or any(not isinstance(name, str) for name in declared_diffs)
            or declared_diffs != sorted(set(declared_diffs))
            or not isinstance(requires_genept, bool)
        ):
            raise ValueError("performance census matrix row identity is malformed")
        config_path = (root / relative_config).resolve(strict=True)
        if (
            not config_path.is_relative_to(root)
            or config_path.is_symlink()
            or sha256_file(config_path) != config_sha256
        ):
            raise ValueError("performance census config path/SHA identity differs")
        config = load_experiment_config(config_path)
        prototype = config.model.parameters.get("prototype_count")
        if (
            raw.get("dataset_id") != "nadig_jurkat"
            or raw.get("split_policy") != "frozen_canonical"
            or raw.get("run_seed") != 1
            or raw.get("max_epochs") != 10
            or raw.get("result_mode") != "metrics_only"
            or config.model.model_id != "gradpert_b2"
            or config.dataset_id != "nadig_jurkat"
            or config.training.max_epochs.value != 10
            or config.training.run_seeds != [1]
            or config.training.early_stopping
            or config.artifacts.result_mode != "metrics_only"
            or config.training.train_batch_size.value != EXACT_TRAIN_BATCH_SIZE
            or config.training.eval_batch_size.value != EXACT_EVAL_BATCH_SIZE
            or prototype is None
            or prototype.value != EXACT_PROTOTYPE_COUNT
        ):
            raise ValueError("performance census config batch/prototype identity differs")
        resolved_variant = config.model.parameters.get("performance_pilot_variant")
        if resolved_variant is None or resolved_variant.value != f"vnext_{variant_id}":
            raise ValueError("performance census config variant identity differs")
        observed_parameters = {
            name: parameter.value for name, parameter in config.model.parameters.items()
        }
        observed_diffs = {
            name
            for name in set(baseline_parameters) | set(observed_parameters)
            if baseline_parameters.get(name, "<missing>")
            != observed_parameters.get(name, "<missing>")
        }
        observed_diffs.discard("performance_pilot_variant")
        if observed_diffs != set(declared_diffs):
            raise ValueError("performance census resolved semantic diff differs")
        bindings.append(
            FrozenVariantBinding(
                matrix_path=str(matrix),
                matrix_sha256=expected_matrix_sha256,
                matrix_id=SUCCESSOR_V2_MATRIX_ID,
                matrix_schema_version=MATRIX_SCHEMA_VERSION,
                matrix_row_index=index,
                variant_id=variant_id,
                config_path=str(config_path),
                config_sha256=config_sha256,
                run_seed=1,
                semantic_factor=semantic_factor,
                declared_parameter_diffs=tuple(declared_diffs),
                genept_preflight_required=requires_genept,
            )
        )
    return tuple(bindings)


def bind_matrix_variant(
    matrix_path: str | Path,
    *,
    repository_root: str | Path,
    expected_matrix_sha256: str,
    variant_id: str,
) -> FrozenVariantBinding:
    bindings = bind_matrix_variants(
        matrix_path,
        repository_root=repository_root,
        expected_matrix_sha256=expected_matrix_sha256,
    )
    matches = [binding for binding in bindings if binding.variant_id == variant_id]
    if len(matches) != 1:
        raise ValueError("performance-census variant is not one exact matrix row")
    return matches[0]


def batch_sequence_sha256(batches: Sequence[OrderedBatchIdentity]) -> str:
    if any(batch.global_step != index for index, batch in enumerate(batches)):
        raise ValueError("batch identities must form a zero-based contiguous step sequence")
    return _sha256_json([batch.sha256 for batch in batches])


def require_batch_prefix(
    observed: Sequence[OrderedBatchIdentity],
    expected: Sequence[OrderedBatchIdentity],
) -> None:
    if len(observed) > len(expected):
        raise ValueError("observed batch sequence is longer than the frozen manifest")
    observed_hashes = [batch.sha256 for batch in observed]
    expected_hashes = [batch.sha256 for batch in expected[: len(observed)]]
    if observed_hashes != expected_hashes:
        raise ValueError("observed ordered batch identities differ from the frozen prefix")


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing samples must be finite positive values")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def decide_p3_promotion(
    *,
    variant_id: str,
    step_wall_ms: Sequence[float],
    reserved_gpu_bytes: Sequence[int],
    free_gpu_bytes: Sequence[int],
    total_gpu_bytes: int,
    selected_implementation_target: bool,
    thresholds: PromotionThresholds = DEFAULT_PROMOTION_THRESHOLDS,
) -> PromotionDecision:
    """Apply only preregistered operational reasons for a P3 timing extension."""

    if len(step_wall_ms) != STAGE_PROTOCOLS["p2_timing"].measured_steps:
        raise ValueError("P3 promotion requires exactly 20 measured P2 steps")
    if len(reserved_gpu_bytes) != len(step_wall_ms) or len(free_gpu_bytes) != len(step_wall_ms):
        raise ValueError("P3 promotion memory telemetry must align with P2 timing samples")
    if total_gpu_bytes <= 0 or any(value < 0 for value in (*reserved_gpu_bytes, *free_gpu_bytes)):
        raise ValueError("P3 promotion GPU telemetry is malformed")
    summary = timing_summary(step_wall_ms)
    median = summary["p50"]
    mad = statistics.median(abs(value - median) for value in step_wall_ms)
    relative_mad = mad / median
    p95_over_p50 = summary["p95"] / median
    midpoint = len(step_wall_ms) // 2
    first_median = statistics.median(step_wall_ms[:midpoint])
    second_median = statistics.median(step_wall_ms[midpoint:])
    half_drift = abs(second_median / first_median - 1.0)
    quarter = max(1, len(reserved_gpu_bytes) // 4)
    first_reserved = max(reserved_gpu_bytes[:quarter])
    last_reserved = max(reserved_gpu_bytes[-quarter:])
    reserved_growth = (
        math.inf
        if first_reserved == 0 and last_reserved > 0
        else 0.0
        if first_reserved == 0
        else max(0.0, last_reserved / first_reserved - 1.0)
    )
    minimum_free = min(free_gpu_bytes)
    minimum_required = max(
        thresholds.minimum_free_bytes,
        math.ceil(total_gpu_bytes * thresholds.minimum_free_fraction),
    )
    free_fraction = minimum_free / total_gpu_bytes
    near_absolute = minimum_free <= minimum_required + thresholds.near_headroom_bytes
    near_fraction = free_fraction <= (
        thresholds.minimum_free_fraction + thresholds.near_headroom_fraction_points
    )
    reasons: list[str] = []
    if relative_mad > thresholds.relative_mad:
        reasons.append("relative_mad_above_limit")
    if p95_over_p50 > thresholds.p95_over_p50:
        reasons.append("p95_over_p50_above_limit")
    if half_drift > thresholds.half_drift:
        reasons.append("half_drift_above_limit")
    if reserved_growth > thresholds.reserved_growth:
        reasons.append("reserved_memory_growth_above_limit")
    if near_absolute:
        reasons.append("near_absolute_gpu_headroom")
    if near_fraction:
        reasons.append("near_fractional_gpu_headroom")
    if selected_implementation_target:
        reasons.append("selected_implementation_target")
    statistics_payload = {
        **summary,
        "relative_mad": relative_mad,
        "p95_over_p50": p95_over_p50,
        "half_drift": half_drift,
        "reserved_growth": reserved_growth,
        "minimum_free_gpu_bytes": float(minimum_free),
        "minimum_free_gpu_fraction": free_fraction,
    }
    if any(not math.isfinite(value) for value in statistics_payload.values()):
        raise ValueError("P3 promotion statistics must be finite")
    return PromotionDecision(
        variant_id=variant_id,
        promoted=bool(reasons),
        reasons=tuple(reasons),
        statistics=statistics_payload,
    )


def pair_a0_promotion(
    decisions: Sequence[PromotionDecision],
) -> tuple[PromotionDecision, ...]:
    """Promote A0 whenever any non-A0 comparator receives a P3 extension."""

    by_id = {decision.variant_id: decision for decision in decisions}
    if len(by_id) != len(decisions) or A0_VARIANT_ID not in by_id:
        raise ValueError("paired P3 decisions require unique variants including A0")
    if any(decision.promoted for decision in decisions if decision.variant_id != A0_VARIANT_ID):
        a0 = by_id[A0_VARIANT_ID]
        reasons = tuple(dict.fromkeys((*a0.reasons, "paired_reference_a0")))
        by_id[A0_VARIANT_ID] = PromotionDecision(
            variant_id=a0.variant_id,
            promoted=True,
            reasons=reasons,
            statistics=a0.statistics,
        )
    return tuple(by_id[decision.variant_id] for decision in decisions)


def claim_fresh_attempt_root(
    census_root: str | Path,
    *,
    variant_id: str,
    stage_id: StageId,
) -> Path:
    """Atomically claim the next immutable attempt-N root for one row/stage."""

    if not variant_id or "/" in variant_id or variant_id in {".", ".."}:
        raise ValueError("variant ID is unsafe for an attempt root")
    stage_root = Path(census_root).resolve() / variant_id / stage_id
    if stage_root.is_symlink():
        raise ValueError("attempt stage root cannot be a symlink")
    stage_root.mkdir(parents=True, exist_ok=True)
    attempts: list[int] = []
    for path in stage_root.iterdir():
        if path.is_symlink() or not path.is_dir() or not path.name.startswith("attempt-"):
            raise ValueError("attempt stage root contains an unexpected entry")
        suffix = path.name.removeprefix("attempt-")
        if len(suffix) != 3 or not suffix.isdigit():
            raise ValueError("attempt directory name is malformed")
        attempts.append(int(suffix))
    next_attempt = max(attempts, default=0) + 1
    if next_attempt > 999:
        raise ValueError("attempt count exceeds the frozen three-digit namespace")
    destination = stage_root / f"attempt-{next_attempt:03d}"
    destination.mkdir(exist_ok=False)
    return destination


def require_training_only_evidence(payload: Mapping[str, object]) -> None:
    if payload.get("scope") != "performance_training_only":
        raise ValueError("census receipt lacks performance_training_only scope")
    expected = {
        "real_canonical_evaluation_constructor_count": 0,
        "validation_cache_materialized": False,
        "validation_callback_count": 0,
        "validation_accessed": False,
        "test_truth_accessed": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"training-only evidence rejected field: {name}")
    if payload.get("truth_access_attempts") != []:
        raise ValueError("training-only evidence contains a truth-access attempt")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class AtomicStageObserver:
    """Optional worker observer that preserves the last entered/completed stage."""

    def __init__(self, receipt_path: str | Path, base_receipt: Mapping[str, object]) -> None:
        self.receipt_path = Path(receipt_path)
        self.receipt: dict[str, object] = {
            "schema_version": "nadig-vnext-performance-stage-progress-v1",
            **dict(base_receipt),
            "status": "running",
            "last_entered_stage": None,
            "last_completed_stage": None,
            "stage_events": [],
            "primary_failure": None,
            "teardown_failures": [],
        }
        self._write()

    def _write(self) -> None:
        _atomic_json(self.receipt_path, self.receipt)

    def entered(self, stage: str, telemetry: Mapping[str, object] | None = None) -> None:
        event = {"event": "entered", "stage": stage, "telemetry": dict(telemetry or {})}
        events = self.receipt["stage_events"]
        assert isinstance(events, list)
        events.append(event)
        self.receipt["last_entered_stage"] = stage
        self._write()

    def completed(self, stage: str, telemetry: Mapping[str, object] | None = None) -> None:
        if self.receipt["last_entered_stage"] != stage:
            raise ValueError("cannot complete a stage that was not the last entered stage")
        event = {"event": "completed", "stage": stage, "telemetry": dict(telemetry or {})}
        events = self.receipt["stage_events"]
        assert isinstance(events, list)
        events.append(event)
        self.receipt["last_completed_stage"] = stage
        self._write()

    def finalize(
        self,
        *,
        result: Mapping[str, object] | None,
        primary_failure: BaseException | None,
        teardown_failures: Sequence[Mapping[str, str]],
    ) -> None:
        self.receipt["status"] = (
            "complete" if primary_failure is None and not teardown_failures else "failed"
        )
        self.receipt["result"] = None if result is None else dict(result)
        self.receipt["primary_failure"] = (
            None
            if primary_failure is None
            else {"type": type(primary_failure).__name__, "message": str(primary_failure)}
        )
        self.receipt["teardown_failures"] = [dict(value) for value in teardown_failures]
        self._write()


def execute_with_atomic_stage_receipt(
    *,
    receipt_path: str | Path,
    base_receipt: Mapping[str, object],
    operation: Callable[[AtomicStageObserver | None], Mapping[str, object]],
    optional_step_observer_available: bool,
    teardown: Callable[[], None] | None = None,
) -> Mapping[str, object]:
    """Execute one worker operation and preserve primary and teardown failures."""

    observer = AtomicStageObserver(receipt_path, base_receipt)
    primary_failure: BaseException | None = None
    result: Mapping[str, object] | None = None
    teardown_failures: list[dict[str, str]] = []
    try:
        result = operation(observer if optional_step_observer_available else None)
    except BaseException as error:
        primary_failure = error
    finally:
        if teardown is not None:
            try:
                teardown()
            except BaseException as error:
                teardown_failures.append(
                    {"stage": "teardown", "type": type(error).__name__, "message": str(error)}
                )
        observer.finalize(
            result=result,
            primary_failure=primary_failure,
            teardown_failures=teardown_failures,
        )
    if primary_failure is not None:
        raise primary_failure
    if teardown_failures:
        raise RuntimeError("census worker teardown failed")
    assert result is not None
    return result


def _load_hashed_json(path: str | Path, expected_sha256: str) -> dict[str, object]:
    _validate_sha256(expected_sha256, field="receipt SHA-256")
    resolved = Path(path).resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise ValueError("census stage receipt SHA-256 differs")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("census stage receipt must be a JSON object")
    return payload


def _validate_stage_receipt(
    payload: Mapping[str, object],
    *,
    binding: FrozenVariantBinding,
    stage_id: StageId,
    expected_batches: Sequence[OrderedBatchIdentity],
) -> dict[str, object]:
    protocol = STAGE_PROTOCOLS[stage_id]
    if (
        payload.get("schema_version") != "nadig-vnext-performance-stage-v1"
        or payload.get("evidence_class") != "performance_training_only"
        or payload.get("scientific_completion") is not False
        or payload.get("variant_id") != binding.variant_id
        or payload.get("config_sha256") != binding.config_sha256
        or payload.get("matrix_sha256") != binding.matrix_sha256
        or payload.get("stage_id") != stage_id
        or payload.get("protocol") != protocol.payload()
    ):
        raise ValueError("census stage receipt identity differs")
    training_only = payload.get("training_only_evidence")
    if not isinstance(training_only, dict):
        raise ValueError("census stage receipt lacks training-only evidence")
    require_training_only_evidence(training_only)
    instrumentation = payload.get("instrumentation")
    if not isinstance(instrumentation, dict):
        raise ValueError("census stage receipt lacks instrumentation identity")
    expected_instrumentation = {
        "timing_acceptance": protocol.timing_acceptance,
        "heavy_capacity_instrumentation": protocol.heavy_capacity_instrumentation,
        "torch_profiler_enabled": protocol.torch_profiler_enabled,
    }
    if any(instrumentation.get(name) != value for name, value in expected_instrumentation.items()):
        raise ValueError("census stage instrumentation differs from its frozen protocol")
    if protocol.timing_acceptance and (
        payload.get("torch_profiler_trace_sha256") is not None
        or payload.get("torch_profiler_table_sha256") is not None
    ):
        raise ValueError("profiler artifacts cannot enter a timing-acceptance receipt")
    status = payload.get("status")
    if status not in {"complete", "failed"}:
        raise ValueError("census stage status is malformed")
    batches_payload = payload.get("batches")
    if not isinstance(batches_payload, list):
        raise ValueError("census stage receipt lacks ordered batches")
    batches = tuple(
        OrderedBatchIdentity.from_payload(value)
        for value in batches_payload
        if isinstance(value, dict)
    )
    if len(batches) != len(batches_payload):
        raise ValueError("census batch receipt entries are malformed")
    if any(batch.actual_batch_size != EXACT_TRAIN_BATCH_SIZE for batch in batches):
        raise ValueError("census stage batch size differs from the frozen batch 256")
    require_batch_prefix(batches, expected_batches)
    attempted_batches = payload.get("attempted_batch_count")
    completed_steps = payload.get("completed_step_count")
    observed_steps = payload.get("observed_step_count")
    if (
        not isinstance(attempted_batches, int)
        or attempted_batches != len(batches)
        or not isinstance(completed_steps, int)
        or completed_steps != observed_steps
        or completed_steps < 0
        or attempted_batches < completed_steps
        or attempted_batches > completed_steps + 1
    ):
        raise ValueError("census attempted/completed step evidence differs from batches")
    if status == "complete" and (
        completed_steps != protocol.total_steps or attempted_batches != protocol.total_steps
    ):
        raise ValueError("complete census stage has the wrong step count")
    if status == "failed" and attempted_batches > protocol.total_steps:
        raise ValueError("failed census stage exceeds its bounded step count")
    timing_values = payload.get("timing_samples_ms")
    if not isinstance(timing_values, list) or any(
        not isinstance(value, (int, float)) for value in timing_values
    ):
        raise ValueError("census timing samples are malformed")
    if protocol.timing_acceptance:
        expected_count = (
            protocol.measured_steps
            if status == "complete"
            else min(
                protocol.measured_steps,
                max(0, completed_steps - protocol.warmup_steps),
            )
        )
        if len(timing_values) != expected_count:
            raise ValueError("census timing sample count differs from its frozen protocol")
        summary = (
            timing_summary([float(value) for value in timing_values]) if timing_values else None
        )
    else:
        if timing_values:
            raise ValueError("capacity/profile receipt cannot expose timing-acceptance samples")
        summary = None
    return {
        "status": status,
        "attempted_batch_count": attempted_batches,
        "completed_step_count": completed_steps,
        "observed_step_count": completed_steps,
        "batch_sequence_sha256": batch_sequence_sha256(batches),
        "timing_summary_ms": summary,
        "receipt_primary_failure": payload.get("primary_failure"),
    }


def aggregate_census_report(
    *,
    bindings: Sequence[FrozenVariantBinding],
    row_records: Sequence[Mapping[str, object]],
    expected_batches: Sequence[OrderedBatchIdentity],
) -> dict[str, object]:
    """Aggregate exactly 25 rows while keeping P2 and P3 panels separate."""

    if len(bindings) != MATRIX_ROW_COUNT:
        raise ValueError("census aggregation requires exactly 25 frozen bindings")
    expected_order = [binding.variant_id for binding in bindings]
    observed_order = [record.get("variant_id") for record in row_records]
    if observed_order != expected_order:
        raise ValueError("census row records must match the exact matrix order")
    if len(set(observed_order)) != MATRIX_ROW_COUNT:
        raise ValueError("census row records are missing or duplicated")
    if len(expected_batches) < STAGE_PROTOCOLS["p3_timing"].total_steps:
        raise ValueError("frozen batch manifest must cover the full P3 prefix")
    if any(batch.actual_batch_size != EXACT_TRAIN_BATCH_SIZE for batch in expected_batches):
        raise ValueError("frozen batch manifest differs from the exact batch 256")

    allowed_states: set[str] = {
        "unavailable_preflight",
        "blocked_missing_prerequisite",
        "deferred_resource_busy",
        "capacity_failed",
        "execution_failed",
        "p1_pass",
        "p2_complete",
        "p3_complete",
    }
    report_rows: list[dict[str, object]] = []
    panel_20: list[dict[str, object]] = []
    panel_100: list[dict[str, object]] = []
    for binding, record in zip(bindings, row_records, strict=True):
        state = record.get("state")
        if state not in allowed_states:
            raise ValueError(f"census row state is malformed: {binding.variant_id}")
        raw_stages = record.get("stages", {})
        if not isinstance(raw_stages, dict):
            raise ValueError("census row stages must be an object")
        stage_summaries: dict[str, object] = {}
        for raw_stage_id, evidence in raw_stages.items():
            if raw_stage_id not in STAGE_PROTOCOLS or not isinstance(evidence, dict):
                raise ValueError("census row contains an unknown stage receipt")
            path = evidence.get("receipt_path")
            expected_sha = evidence.get("receipt_sha256")
            if not isinstance(path, str) or not isinstance(expected_sha, str):
                raise ValueError("census stage receipt pointer is malformed")
            payload = _load_hashed_json(path, expected_sha)
            summary = _validate_stage_receipt(
                payload,
                binding=binding,
                stage_id=raw_stage_id,
                expected_batches=expected_batches,
            )
            stage_summaries[raw_stage_id] = {
                "receipt_path": str(Path(path).resolve()),
                "receipt_sha256": expected_sha,
                **summary,
            }
        required_by_state: dict[str, tuple[StageId, ...]] = {
            "p1_pass": ("p1_capacity",),
            "p2_complete": ("p1_capacity", "p2_timing"),
            "p3_complete": ("p1_capacity", "p2_timing", "p3_timing"),
        }
        for required in required_by_state.get(str(state), ()):
            summary = stage_summaries.get(required)
            if not isinstance(summary, dict) or summary.get("status") != "complete":
                raise ValueError(f"census state lacks complete required stage: {required}")
        if state in {"capacity_failed", "execution_failed"} and not stage_summaries:
            raise ValueError("failed census row must preserve at least one stage receipt")
        if state in {"p2_complete", "p3_complete"}:
            summary_20 = stage_summaries["p2_timing"]
            assert isinstance(summary_20, dict)
            panel_20.append(
                {
                    "variant_id": binding.variant_id,
                    "receipt_sha256": summary_20["receipt_sha256"],
                    "timing_summary_ms": summary_20["timing_summary_ms"],
                }
            )
        if state == "p3_complete":
            summary_100 = stage_summaries["p3_timing"]
            assert isinstance(summary_100, dict)
            panel_100.append(
                {
                    "variant_id": binding.variant_id,
                    "receipt_sha256": summary_100["receipt_sha256"],
                    "timing_summary_ms": summary_100["timing_summary_ms"],
                }
            )
        report_rows.append(
            {
                "variant_binding": binding.payload(),
                "state": state,
                "stages": stage_summaries,
                "disposition_reason": record.get("disposition_reason"),
            }
        )

    counts = Counter(str(record["state"]) for record in row_records)
    blocking = sum(
        counts[state]
        for state in (
            "blocked_missing_prerequisite",
            "deferred_resource_busy",
            "execution_failed",
            "p1_pass",
        )
    )
    measured_20 = len(panel_20)
    if blocking:
        status = "partial_blocked"
    elif measured_20 == MATRIX_ROW_COUNT:
        status = "complete_all_25_measured"
    else:
        status = "complete_with_preregistered_unavailable_or_capacity_failures"
    disposition_sections = {
        state: [str(record["variant_id"]) for record in row_records if record["state"] == state]
        for state in (
            "unavailable_preflight",
            "blocked_missing_prerequisite",
            "deferred_resource_busy",
            "capacity_failed",
            "execution_failed",
        )
    }
    return {
        "schema_version": "nadig-vnext-performance-census-report-v1",
        "status": status,
        "matrix_id": bindings[0].matrix_id,
        "matrix_sha256": bindings[0].matrix_sha256,
        "row_count": MATRIX_ROW_COUNT,
        "stage_protocols": {name: value.payload() for name, value in STAGE_PROTOCOLS.items()},
        "frozen_batch_manifest": {
            "batch_count": len(expected_batches),
            "batch_sequence_sha256": batch_sequence_sha256(expected_batches),
        },
        "state_counts": dict(sorted(counts.items())),
        "disposition_sections": disposition_sections,
        "measured_20_row_count": measured_20,
        "measured_100_row_count": len(panel_100),
        "rows": report_rows,
        "timing_panels": {
            "p2_20_measured_steps": panel_20,
            "p3_100_measured_steps": panel_100,
        },
        "timing_panels_must_not_be_ranked_together": True,
    }


def _load_batch_manifest(path: Path) -> tuple[OrderedBatchIdentity, ...]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "nadig-vnext-performance-batch-manifest-v1"
    ):
        raise ValueError("frozen batch manifest schema differs")
    batches = payload.get("batches")
    if not isinstance(batches, list) or any(not isinstance(value, dict) for value in batches):
        raise ValueError("frozen batch manifest rows are malformed")
    identities = tuple(OrderedBatchIdentity.from_payload(value) for value in batches)
    batch_sequence_sha256(identities)
    return identities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the frozen 25-row census plan")
    plan.add_argument("--json", action="store_true", dest="as_json")
    aggregate = subparsers.add_parser("aggregate", help="aggregate sealed worker receipts")
    aggregate.add_argument("--row-records", type=Path, required=True)
    aggregate.add_argument("--batch-manifest", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bindings = bind_matrix_variants(
        args.matrix,
        repository_root=args.repository_root,
        expected_matrix_sha256=args.expected_matrix_sha256,
    )
    if args.command == "plan":
        payload = {
            "schema_version": "nadig-vnext-performance-census-plan-v1",
            "matrix_sha256": args.expected_matrix_sha256,
            "stage_protocols": {name: value.payload() for name, value in STAGE_PROTOCOLS.items()},
            "rows": [binding.payload() for binding in bindings],
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    row_records = json.loads(args.row_records.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(row_records, list) or any(
        not isinstance(value, dict) for value in row_records
    ):
        raise ValueError("census row records must be a JSON list of objects")
    expected_batches = _load_batch_manifest(args.batch_manifest)
    report = aggregate_census_report(
        bindings=bindings,
        row_records=row_records,
        expected_batches=expected_batches,
    )
    _atomic_json(args.output.resolve(), report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
