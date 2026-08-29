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

from gradpert.config import ExperimentConfig, load_experiment_config  # noqa: E402
from gradpert.execution.ablation_matrix import (  # noqa: E402
    SUCCESSOR_V2_CONTRACT,
    SUCCESSOR_V2_MATRIX_ID,
)
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402
from gradpert.pilots import load_vnext_graph_topology  # noqa: E402
from gradpert.training.data import CanonicalTrainingData  # noqa: E402

GIB = 1024**3
A0_VARIANT_ID = "a0_ratio_ring_half"
MATRIX_SCHEMA_VERSION = "2"
MATRIX_ROW_COUNT = 25
EXACT_TRAIN_BATCH_SIZE = 256
EXACT_EVAL_BATCH_SIZE = 256
EXACT_PROTOTYPE_COUNT = 16384
EXACT_MAX_UNIQUE_CONDITIONS = 8
EXACT_STEPS_PER_EPOCH = 582
EXACT_FROZEN_BATCH_COUNT = 110
EXACT_BATCH_ORDER_POLICY = "condition_limited_seeded_v1"
EXACT_CONTROL_PAIRING_POLICY = "same_context_sha256_pcg64_per_epoch_v1"
EXACT_NATIVE_IDENTITY_FILES = frozenset(
    {
        "config.resolved.yaml",
        "source_identity.json",
        "environment.json",
        "resolved_local_view_contract.json",
        "training_data.json",
        "run_meta.json",
    }
)
EXACT_GENEPT_NATIVE_IDENTITY_FILES = frozenset({"genept_preflight.json", "genept_feature.json"})

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
            isinstance(value, int) and not isinstance(value, bool)
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
        numeric_fields = (
            self.global_step,
            self.actual_batch_size,
            self.unique_condition_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in numeric_fields):
            raise ValueError("batch numeric identity fields must be plain integers")
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
        if any(not anchors for anchors in self.active_anchor_ids):
            raise ValueError("active anchor groups must be nonempty")
        if any(not value for field in self.active_anchor_ids for value in field):
            raise ValueError("active anchor IDs must be nonempty strings")
        if any(not value for field in fields[:3] for value in field):
            raise ValueError("batch row/condition/control IDs must be nonempty strings")
        if self.unique_condition_count != len(set(self.condition_ids)):
            raise ValueError("unique condition count differs from ordered condition IDs")
        if not 1 <= self.unique_condition_count <= EXACT_MAX_UNIQUE_CONDITIONS:
            raise ValueError("unique condition count is outside the frozen census limit")

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
class FrozenBatchManifest:
    path: str
    sha256: str
    matrix_path: str
    matrix_sha256: str
    config_path: str
    config_sha256: str
    dataset_id: str
    protocol_id: str
    run_seed: int
    epoch: int
    batch_size: int
    max_unique_conditions: int
    epoch_step_count: int
    frozen_prefix_count: int
    batch_order_policy: str
    control_pairing_policy: str
    canonical_data_sha256: str
    observation_order_sha256: str
    split_content_sha256: str
    ordered_training_row_ids_sha256: str
    ordered_control_pools_sha256: str
    runtime_graph_root: str
    runtime_graph_manifest_path: str
    runtime_graph_manifest_sha256: str
    runtime_graph_gene_order_sha256: str
    batch_sequence_sha256: str
    batches: tuple[OrderedBatchIdentity, ...]


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


def _parameter_value(config: ExperimentConfig, name: str) -> object:
    parameter = config.model.parameters.get(name)
    if parameter is None:
        raise ValueError(f"A0 config lacks required parameter: {name}")
    return parameter.value


def freeze_batch_manifest(
    *,
    matrix_path: str | Path,
    repository_root: str | Path,
    expected_matrix_sha256: str,
    data_root: str | Path,
) -> dict[str, object]:
    """Freeze the exact expression-free 110-step schedule shared by all rows."""

    torch_module_loaded_before = "torch" in sys.modules
    if torch_module_loaded_before:
        raise ValueError("batch freeze must start before importing the CUDA runtime surface")
    expression_array_reads = 0

    binding = bind_matrix_variant(
        matrix_path,
        repository_root=repository_root,
        expected_matrix_sha256=expected_matrix_sha256,
        variant_id=A0_VARIANT_ID,
    )
    config = load_experiment_config(binding.config_path)
    runtime_graph_root = _parameter_value(config, "runtime_graph_root")
    max_unique_conditions = _parameter_value(config, "max_unique_conditions_per_batch")
    graph_hvg_count = _parameter_value(config, "graph_hvg_count")
    if (
        not isinstance(runtime_graph_root, str)
        or not isinstance(max_unique_conditions, int)
        or max_unique_conditions != EXACT_MAX_UNIQUE_CONDITIONS
        or graph_hvg_count != 512
        or config.training.train_batch_size.value != EXACT_TRAIN_BATCH_SIZE
    ):
        raise ValueError("A0 batch-freeze config differs from the exact census contract")
    relative_graph_root = Path(runtime_graph_root)
    if relative_graph_root.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_graph_root.parts
    ):
        raise ValueError("A0 runtime graph root must be a safe relative path")
    resolved_data_root = Path(data_root).resolve(strict=True)
    graph_root = resolved_data_root.joinpath(*relative_graph_root.parts)
    topology, graph_manifest = load_vnext_graph_topology(graph_root)
    graph_manifest_path = graph_root / "manifest.json"
    if (
        graph_manifest.requested_hvg_count != 512
        or graph_manifest.graph_gene_count != len(topology.gene_ids)
        or graph_manifest.graph_gene_order_sha256 != sha256_json(list(topology.gene_ids))
    ):
        raise ValueError("A0 runtime graph differs from the frozen HVG512 identity")

    with CanonicalTrainingData(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=resolved_data_root,
        run_seed=binding.run_seed,
        graph_gene_ids_override=topology.gene_ids,
        graph_manifest_path_override=graph_manifest_path,
    ) as training_data:
        training_data.require_experiment_data_contract(
            registry_version=config.data.registry_version,
            split_policy=config.data.split_policy,
        )
        original_expression_reader = training_data._read_expression_indices

        def reject_expression_read(*_args: object, **_kwargs: object) -> object:
            nonlocal expression_array_reads
            expression_array_reads += 1
            raise RuntimeError("batch identity freeze attempted to read expression arrays")

        training_data._read_expression_indices = reject_expression_read  # type: ignore[method-assign]
        try:
            identity_specs = training_data.training_batch_identity_specs(
                epoch=0,
                batch_size=EXACT_TRAIN_BATCH_SIZE,
                max_unique_conditions=EXACT_MAX_UNIQUE_CONDITIONS,
            )
        finally:
            training_data._read_expression_indices = original_expression_reader  # type: ignore[method-assign]
        if len(identity_specs) != EXACT_STEPS_PER_EPOCH:
            raise ValueError("Nadig Jurkat epoch step count differs from the frozen 582")
        selected_specs = identity_specs[:EXACT_FROZEN_BATCH_COUNT]
        identities = tuple(
            OrderedBatchIdentity.create(
                global_step=global_step,
                row_ids=spec.perturbed_row_ids,
                condition_ids=spec.condition_ids,
                control_row_ids=spec.control_row_ids,
                active_anchor_ids=[
                    spec.anchor_gene_ids_by_condition[condition] for condition in spec.condition_ids
                ],
                actual_batch_size=len(spec.condition_ids),
                unique_condition_count=len(set(spec.condition_ids)),
            )
            for global_step, spec in enumerate(selected_specs)
        )
        if len(identities) != EXACT_FROZEN_BATCH_COUNT or any(
            identity.actual_batch_size != EXACT_TRAIN_BATCH_SIZE for identity in identities
        ):
            raise ValueError("frozen census prefix must contain 110 full batch-256 identities")
        train_row_ids = tuple(
            training_data.row_ids[index] for index in training_data.train_row_indices
        )
        control_pool_payload = {
            context: list(row_ids)
            for context, row_ids in sorted(training_data.control_pools.items())
        }
        canonical_data_sha256 = training_data.manifest.canonical_adata_sha256
        observation_order_sha256 = training_data.manifest.observation_order_sha256
        split_content_sha256 = training_data.split.split_content_sha256

    torch_module_loaded_after = "torch" in sys.modules
    forbidden_runtime = {
        "cuda_imported_or_initialized": (torch_module_loaded_before or torch_module_loaded_after),
        "model_constructed": False,
        "optimizer_constructed": False,
        "expression_array_reads": expression_array_reads,
        "validation_object_constructed": False,
        "test_object_constructed": False,
    }
    if forbidden_runtime["cuda_imported_or_initialized"] or expression_array_reads:
        raise ValueError("batch freeze touched a forbidden runtime surface")

    return {
        "schema_version": "nadig-vnext-performance-batch-manifest-v2",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "matrix_path": binding.matrix_path,
        "matrix_sha256": binding.matrix_sha256,
        "matrix_id": binding.matrix_id,
        "a0_config_path": binding.config_path,
        "a0_config_sha256": binding.config_sha256,
        "dataset_id": config.dataset_id,
        "protocol_id": config.data.protocol_id,
        "run_seed": binding.run_seed,
        "epoch": 0,
        "batch_size": EXACT_TRAIN_BATCH_SIZE,
        "max_unique_conditions": EXACT_MAX_UNIQUE_CONDITIONS,
        "epoch_step_count": EXACT_STEPS_PER_EPOCH,
        "frozen_prefix_count": EXACT_FROZEN_BATCH_COUNT,
        "batch_order_policy": EXACT_BATCH_ORDER_POLICY,
        "control_pairing_policy": EXACT_CONTROL_PAIRING_POLICY,
        "canonical_data_sha256": canonical_data_sha256,
        "observation_order_sha256": observation_order_sha256,
        "split_content_sha256": split_content_sha256,
        "ordered_training_row_ids_sha256": sha256_json(list(train_row_ids)),
        "ordered_control_pools_sha256": sha256_json(control_pool_payload),
        "runtime_graph_root": runtime_graph_root,
        "runtime_graph_manifest_path": str(graph_manifest_path.resolve()),
        "runtime_graph_manifest_sha256": sha256_file(graph_manifest_path),
        "runtime_graph_gene_order_sha256": sha256_json(list(topology.gene_ids)),
        "forbidden_runtime": forbidden_runtime,
        "batch_sequence_sha256": batch_sequence_sha256(identities),
        "batches": [
            {**identity.payload(), "batch_identity_sha256": identity.sha256}
            for identity in identities
        ],
    }


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


def _claim_json_output(path: Path, payload: object) -> Path:
    """Atomically reserve a new evidence path before any expensive inspection."""

    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
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
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


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


_REPOSITORY_IDENTITY_FIELDS = (
    "repository_root",
    "declared_development_commit",
    "head_commit",
    "head_tree",
    "source_tree_sha256",
    "remote_url",
    "remote_ref",
    "published_commit",
    "formal_eligible",
    "status_porcelain_sha256",
)
_REPOSITORY_PREDICATES = frozenset(
    {
        "head_equals_development_commit",
        "worktree_clean",
        "formal_source_eligible",
        "published_commit_equals_development_commit",
        "remote_ref_equals_p0",
        "source_content_tree_equals_p0",
        "remote_url_equals_p0",
    }
)


def _validate_repository_identity(
    payload: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"complete census stage lacks {label} repository identity")
    predicates = payload.get("predicates")
    if (
        payload.get("schema_version") != "nadig-vnext-performance-repository-identity-v1"
        or not isinstance(predicates, dict)
        or set(predicates) != _REPOSITORY_PREDICATES
        or not all(value is True for value in predicates.values())
        or payload.get("formal_eligible") is not True
        or payload.get("status_porcelain") != ""
    ):
        raise ValueError(f"complete census stage {label} repository identity is not clean")
    return payload


def _validate_immutable_input_evidence(
    payload: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "file_count",
        "files",
        "ordered_file_bindings_sha256",
    }:
        raise ValueError(f"complete census stage {label} immutable-input evidence is malformed")
    files = payload.get("files")
    file_count = payload.get("file_count")
    if (
        payload.get("schema_version") != "nadig-vnext-performance-immutable-input-audit-v1"
        or not isinstance(files, list)
        or not files
        or not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count != len(files)
    ):
        raise ValueError(f"complete census stage {label} immutable-input evidence is malformed")
    for evidence in files:
        if not isinstance(evidence, dict) or set(evidence) != {
            "label",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"complete census stage {label} immutable file is malformed")
        file_label = evidence.get("label")
        path = evidence.get("path")
        sha256 = evidence.get("sha256")
        size_bytes = evidence.get("size_bytes")
        if (
            not isinstance(file_label, str)
            or not file_label
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(sha256, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise ValueError(f"complete census stage {label} immutable file is malformed")
        _validate_sha256(sha256, field=f"{label} immutable-file SHA-256")
    if payload.get("ordered_file_bindings_sha256") != _sha256_json(files):
        raise ValueError(f"complete census stage {label} immutable-input digest differs")
    return payload


def _validate_terminal_fallback_stage_receipt(
    payload: Mapping[str, object],
    *,
    protocol: StageProtocol,
) -> dict[str, object]:
    primary_failure = payload.get("primary_failure")
    construction_failure = payload.get("receipt_construction_failure")
    attempted_batches = payload.get("attempted_batch_count")
    completed_steps = payload.get("completed_step_count")
    if (
        payload.get("status") != "failed"
        or payload.get("running_receipt_replaced") is not True
        or not isinstance(primary_failure, dict)
        or set(primary_failure) != {"type", "message"}
        or not all(isinstance(primary_failure.get(name), str) for name in ("type", "message"))
        or not primary_failure.get("type")
        or not isinstance(construction_failure, dict)
        or set(construction_failure) != {"type", "message"}
        or not all(isinstance(construction_failure.get(name), str) for name in ("type", "message"))
        or not construction_failure.get("type")
        or not isinstance(attempted_batches, int)
        or isinstance(attempted_batches, bool)
        or not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or not 0 <= completed_steps <= attempted_batches <= protocol.total_steps
        or not isinstance(payload.get("teardown_failures"), list)
    ):
        raise ValueError("census fallback stage receipt is malformed")
    forbidden_measurement_fields = {
        "batch_sequence_sha256",
        "batches",
        "steps",
        "timing_samples_ms",
        "timing_summary_ms",
        "torch_profiler_trace_sha256",
        "torch_profiler_table_sha256",
    }
    if forbidden_measurement_fields.intersection(payload):
        raise ValueError("census fallback stage receipt cannot expose measurement evidence")
    return {
        "status": "failed",
        "terminal_receipt_kind": "construction_fallback",
        "attempted_batch_count": attempted_batches,
        "completed_step_count": completed_steps,
        "observed_step_count": completed_steps,
        "batch_sequence_sha256": None,
        "timing_summary_ms": None,
        "receipt_primary_failure": primary_failure,
        "physical_gpu_uuid": None,
        "stage_prerequisite": None,
    }


def _validate_stage_receipt(
    payload: Mapping[str, object],
    *,
    binding: FrozenVariantBinding,
    stage_id: StageId,
    batch_manifest: FrozenBatchManifest,
    p0_preflight_sha256: str,
) -> dict[str, object]:
    protocol = STAGE_PROTOCOLS[stage_id]
    expected_batches = batch_manifest.batches
    if (
        payload.get("schema_version") != "nadig-vnext-performance-stage-v1"
        or payload.get("evidence_class") != "performance_training_only"
        or payload.get("scientific_completion") is not False
        or payload.get("variant_id") != binding.variant_id
        or payload.get("config_sha256") != binding.config_sha256
        or payload.get("matrix_sha256") != binding.matrix_sha256
        or _sha256_json(payload.get("binding")) != _sha256_json(binding.payload())
        or payload.get("stage_id") != stage_id
        or payload.get("protocol") != protocol.payload()
    ):
        raise ValueError("census stage receipt identity differs")
    if payload.get("running_receipt_replaced") is True:
        return _validate_terminal_fallback_stage_receipt(payload, protocol=protocol)
    if "running_receipt_replaced" in payload:
        raise ValueError("census stage receipt has a malformed fallback marker")
    p0_binding = payload.get("p0_preflight")
    manifest_binding = payload.get("frozen_batch_manifest")
    if (
        not isinstance(p0_binding, dict)
        or p0_binding.get("receipt_sha256") != p0_preflight_sha256
        or not isinstance(manifest_binding, dict)
        or manifest_binding.get("receipt_sha256") != batch_manifest.sha256
        or manifest_binding.get("expected_batch_count") != batch_manifest.frozen_prefix_count
        or manifest_binding.get("expected_sequence_sha256") != batch_manifest.batch_sequence_sha256
    ):
        raise ValueError("census stage prerequisite binding differs")
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
    observed_prefix_sha256 = batch_sequence_sha256(batches)
    if (
        payload.get("batch_sequence_sha256") != observed_prefix_sha256
        or manifest_binding.get("observed_prefix_count") != len(batches)
        or manifest_binding.get("observed_prefix_sha256") != observed_prefix_sha256
        or manifest_binding.get("expected_prefix_sha256") != observed_prefix_sha256
        or manifest_binding.get("prefix_matches") is not True
    ):
        raise ValueError("census declared batch prefix differs from observed batches")
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
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != completed_steps:
        raise ValueError("census stage step evidence count differs")
    for index, step in enumerate(steps):
        expected_phase = "warmup" if index < protocol.warmup_steps else "measured"
        if (
            not isinstance(step, dict)
            or step.get("global_step") != index
            or step.get("batch_identity_sha256") != batches[index].sha256
            or step.get("phase") != expected_phase
        ):
            raise ValueError("census stage per-step batch evidence differs")
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
    repository = payload.get("repository_identity")
    final_repository = payload.get("final_repository_identity")
    final_immutable_inputs = payload.get("final_immutable_input_evidence")
    resource = payload.get("resource_preflight")
    capacity = payload.get("capacity_evidence")
    persistent_pkl = payload.get("persistent_pkl_scan")
    native_identity = payload.get("native_identity_receipts")
    if not all(
        isinstance(value, dict)
        for value in (repository, resource, capacity, persistent_pkl, native_identity)
    ):
        raise ValueError("census stage lacks repository/resource/PKL evidence")
    native_files = native_identity.get("files")
    if not isinstance(native_files, list) or any(
        not isinstance(value, dict) for value in native_files
    ):
        raise ValueError("census native identity bindings are malformed")
    native_names = {
        Path(str(value.get("relative_path"))).name
        for value in native_files
        if isinstance(value.get("relative_path"), str)
    }
    required_native_names = set(EXACT_NATIVE_IDENTITY_FILES)
    if binding.genept_preflight_required:
        required_native_names.update(EXACT_GENEPT_NATIVE_IDENTITY_FILES)
    if not required_native_names <= native_names:
        raise ValueError("census stage native identity bindings are incomplete")
    selected_gpu = resource.get("selected_physical_gpu")
    physical_gpu_uuid = selected_gpu.get("uuid") if isinstance(selected_gpu, dict) else None
    if not isinstance(physical_gpu_uuid, str) or not physical_gpu_uuid:
        raise ValueError("census stage lacks physical GPU UUID")
    if status == "complete":
        initial_repository = _validate_repository_identity(repository, label="initial")
        terminal_repository = _validate_repository_identity(final_repository, label="final")
        if any(
            initial_repository.get(name) != terminal_repository.get(name)
            for name in _REPOSITORY_IDENTITY_FIELDS
        ):
            raise ValueError("complete census stage repository identity changed during execution")
        initial_immutable_inputs = _validate_immutable_input_evidence(
            p0_binding.get("preclaim_immutable_input_evidence"),
            label="initial",
        )
        terminal_immutable_inputs = _validate_immutable_input_evidence(
            final_immutable_inputs,
            label="final",
        )
        if _sha256_json(initial_immutable_inputs) != _sha256_json(terminal_immutable_inputs):
            raise ValueError("complete census stage immutable inputs changed during execution")
        predicate_groups = (
            repository.get("predicates"),
            resource.get("predicates"),
            capacity.get("predicates"),
        )
        if any(
            not isinstance(group, dict)
            or not group
            or not all(bool(value) for value in group.values())
            for group in predicate_groups
        ):
            raise ValueError("complete census stage has a failed safety predicate")
        if (
            persistent_pkl.get("passed") is not True
            or persistent_pkl.get("persistent_pkl_count") != 0
            or payload.get("batch_gate_failure") is not None
            or payload.get("primary_failure") is not None
            or payload.get("teardown_failures") != []
        ):
            raise ValueError("complete census stage has failure or persistent-PKL evidence")
    return {
        "status": status,
        "attempted_batch_count": attempted_batches,
        "completed_step_count": completed_steps,
        "observed_step_count": completed_steps,
        "batch_sequence_sha256": observed_prefix_sha256,
        "timing_summary_ms": summary,
        "receipt_primary_failure": payload.get("primary_failure"),
        "physical_gpu_uuid": physical_gpu_uuid,
        "stage_prerequisite": payload.get("stage_prerequisite"),
    }


def aggregate_census_report(
    *,
    bindings: Sequence[FrozenVariantBinding],
    row_records: Sequence[Mapping[str, object]],
    batch_manifest: FrozenBatchManifest,
    p0_preflight_sha256: str,
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
    _validate_sha256(p0_preflight_sha256, field="P0 preflight SHA-256")
    expected_batches = batch_manifest.batches
    if len(expected_batches) != EXACT_FROZEN_BATCH_COUNT:
        raise ValueError("frozen batch manifest must contain exactly 110 steps")
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
                batch_manifest=batch_manifest,
                p0_preflight_sha256=p0_preflight_sha256,
            )
            stage_summaries[raw_stage_id] = {
                "receipt_path": str(Path(path).resolve()),
                "receipt_sha256": expected_sha,
                **summary,
            }
        p1_summary = stage_summaries.get("p1_capacity")
        for dependent_stage in ("p2_timing", "p3_timing", "diagnostic_profile"):
            dependent = stage_summaries.get(dependent_stage)
            if dependent is None:
                continue
            if not isinstance(p1_summary, dict) or p1_summary.get("status") != "complete":
                raise ValueError(f"{dependent_stage} lacks a complete P1 prerequisite")
            prerequisite = dependent.get("stage_prerequisite")
            if (
                not isinstance(prerequisite, dict)
                or prerequisite.get("receipt_sha256") != p1_summary.get("receipt_sha256")
                or prerequisite.get("physical_gpu_uuid") != p1_summary.get("physical_gpu_uuid")
                or dependent.get("physical_gpu_uuid") != p1_summary.get("physical_gpu_uuid")
            ):
                raise ValueError(f"{dependent_stage} P1/GPU binding differs")
        required_by_state: dict[str, tuple[StageId, ...]] = {
            "p1_pass": ("p1_capacity",),
            "p2_complete": ("p1_capacity", "p2_timing"),
            "p3_complete": ("p1_capacity", "p2_timing", "p3_timing"),
        }
        for required in required_by_state.get(str(state), ()):
            summary = stage_summaries.get(required)
            if not isinstance(summary, dict) or summary.get("status") != "complete":
                raise ValueError(f"census state lacks complete required stage: {required}")
        if state in {"capacity_failed", "execution_failed"} and (
            not stage_summaries
            or not any(
                isinstance(summary, dict) and summary.get("status") == "failed"
                for summary in stage_summaries.values()
            )
        ):
            raise ValueError("failed census row must preserve at least one failed stage receipt")
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
            "receipt_path": batch_manifest.path,
            "receipt_sha256": batch_manifest.sha256,
            "batch_count": len(expected_batches),
            "batch_sequence_sha256": batch_sequence_sha256(expected_batches),
        },
        "p0_preflight_sha256": p0_preflight_sha256,
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


def load_frozen_batch_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_matrix_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> FrozenBatchManifest:
    resolved = Path(path).resolve(strict=True)
    observed_sha256 = sha256_file(resolved)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, field="expected batch-manifest SHA-256")
        if observed_sha256 != expected_sha256:
            raise ValueError("frozen batch manifest SHA-256 differs")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "nadig-vnext-performance-batch-manifest-v2"
        or payload.get("evidence_class") != "performance_training_only"
        or payload.get("scientific_completion") is not False
        or payload.get("matrix_id") != SUCCESSOR_V2_MATRIX_ID
        or payload.get("dataset_id") != "nadig_jurkat"
        or payload.get("protocol_id") != "within_cell_unseen_single"
        or payload.get("run_seed") != 1
        or payload.get("epoch") != 0
        or payload.get("batch_size") != EXACT_TRAIN_BATCH_SIZE
        or payload.get("max_unique_conditions") != EXACT_MAX_UNIQUE_CONDITIONS
        or payload.get("epoch_step_count") != EXACT_STEPS_PER_EPOCH
        or payload.get("frozen_prefix_count") != EXACT_FROZEN_BATCH_COUNT
        or payload.get("batch_order_policy") != EXACT_BATCH_ORDER_POLICY
        or payload.get("control_pairing_policy") != EXACT_CONTROL_PAIRING_POLICY
    ):
        raise ValueError("frozen batch manifest identity differs")
    matrix_sha256 = payload.get("matrix_sha256")
    config_sha256 = payload.get("a0_config_sha256")
    if not isinstance(matrix_sha256, str) or not isinstance(config_sha256, str):
        raise ValueError("frozen batch manifest lacks matrix/config identities")
    _validate_sha256(matrix_sha256, field="batch-manifest matrix SHA-256")
    _validate_sha256(config_sha256, field="batch-manifest A0 config SHA-256")
    if expected_matrix_sha256 is not None and matrix_sha256 != expected_matrix_sha256:
        raise ValueError("frozen batch manifest matrix SHA-256 differs")
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise ValueError("frozen batch manifest A0 config SHA-256 differs")
    sealed_paths: dict[str, Path] = {}
    for name, label in (
        ("matrix_path", "matrix"),
        ("a0_config_path", "A0 config"),
        ("runtime_graph_manifest_path", "runtime-graph manifest"),
    ):
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"frozen batch manifest lacks {label} path")
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f"frozen batch manifest {label} path must be absolute")
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_file():
            raise ValueError(f"frozen batch manifest {label} path is not a regular file")
        sealed_paths[name] = resolved_candidate
    runtime_graph_root = payload.get("runtime_graph_root")
    if not isinstance(runtime_graph_root, str) or not runtime_graph_root:
        raise ValueError("frozen batch manifest lacks runtime-graph root")
    relative_graph_root = Path(runtime_graph_root)
    if relative_graph_root.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_graph_root.parts
    ):
        raise ValueError("frozen batch manifest runtime-graph root is unsafe")
    graph_suffix = (*relative_graph_root.parts, "manifest.json")
    observed_graph_suffix = tuple(
        sealed_paths["runtime_graph_manifest_path"].parts[-len(graph_suffix) :]
    )
    if observed_graph_suffix != graph_suffix:
        raise ValueError("frozen batch manifest runtime-graph path/root binding differs")
    forbidden_runtime = payload.get("forbidden_runtime")
    if not isinstance(forbidden_runtime, dict) or forbidden_runtime != {
        "cuda_imported_or_initialized": False,
        "expression_array_reads": 0,
        "model_constructed": False,
        "optimizer_constructed": False,
        "test_object_constructed": False,
        "validation_object_constructed": False,
    }:
        raise ValueError("frozen batch manifest used a forbidden runtime surface")
    batches = payload.get("batches")
    if (
        not isinstance(batches, list)
        or len(batches) != EXACT_FROZEN_BATCH_COUNT
        or any(not isinstance(value, dict) for value in batches)
    ):
        raise ValueError("frozen batch manifest rows are malformed")
    batch_identities = tuple(OrderedBatchIdentity.from_payload(value) for value in batches)
    for raw, identity in zip(batches, batch_identities, strict=True):
        if raw.get("batch_identity_sha256") != identity.sha256:
            raise ValueError("frozen batch identity SHA-256 differs")
    sequence_sha256 = batch_sequence_sha256(batch_identities)
    if payload.get("batch_sequence_sha256") != sequence_sha256 or any(
        identity.actual_batch_size != EXACT_TRAIN_BATCH_SIZE for identity in batch_identities
    ):
        raise ValueError("frozen batch manifest sequence differs")
    sha_fields = {
        "canonical_data_sha256": "canonical",
        "observation_order_sha256": "observation-order",
        "split_content_sha256": "split",
        "ordered_training_row_ids_sha256": "training-row-order",
        "ordered_control_pools_sha256": "control-pool-order",
        "runtime_graph_manifest_sha256": "runtime-graph manifest",
        "runtime_graph_gene_order_sha256": "runtime-graph gene-order",
    }
    identity_hashes: dict[str, str] = {}
    for name, label in sha_fields.items():
        value = payload.get(name)
        if not isinstance(value, str):
            raise ValueError(f"frozen batch manifest lacks {label} identity")
        _validate_sha256(value, field=f"batch-manifest {label} SHA-256")
        identity_hashes[name] = value
    if sha256_file(sealed_paths["matrix_path"]) != matrix_sha256:
        raise ValueError("frozen batch manifest matrix path/hash binding differs")
    if sha256_file(sealed_paths["a0_config_path"]) != config_sha256:
        raise ValueError("frozen batch manifest A0 config path/hash binding differs")
    if (
        sha256_file(sealed_paths["runtime_graph_manifest_path"])
        != identity_hashes["runtime_graph_manifest_sha256"]
    ):
        raise ValueError("frozen batch manifest runtime-graph path/hash binding differs")
    return FrozenBatchManifest(
        path=str(resolved),
        sha256=observed_sha256,
        matrix_path=str(sealed_paths["matrix_path"]),
        matrix_sha256=matrix_sha256,
        config_path=str(sealed_paths["a0_config_path"]),
        config_sha256=config_sha256,
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        run_seed=1,
        epoch=0,
        batch_size=EXACT_TRAIN_BATCH_SIZE,
        max_unique_conditions=EXACT_MAX_UNIQUE_CONDITIONS,
        epoch_step_count=EXACT_STEPS_PER_EPOCH,
        frozen_prefix_count=EXACT_FROZEN_BATCH_COUNT,
        batch_order_policy=EXACT_BATCH_ORDER_POLICY,
        control_pairing_policy=EXACT_CONTROL_PAIRING_POLICY,
        canonical_data_sha256=identity_hashes["canonical_data_sha256"],
        observation_order_sha256=identity_hashes["observation_order_sha256"],
        split_content_sha256=identity_hashes["split_content_sha256"],
        ordered_training_row_ids_sha256=identity_hashes["ordered_training_row_ids_sha256"],
        ordered_control_pools_sha256=identity_hashes["ordered_control_pools_sha256"],
        runtime_graph_root=runtime_graph_root,
        runtime_graph_manifest_path=str(sealed_paths["runtime_graph_manifest_path"]),
        runtime_graph_manifest_sha256=identity_hashes["runtime_graph_manifest_sha256"],
        runtime_graph_gene_order_sha256=identity_hashes["runtime_graph_gene_order_sha256"],
        batch_sequence_sha256=sequence_sha256,
        batches=batch_identities,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the frozen 25-row census plan")
    plan.add_argument("--json", action="store_true", dest="as_json")
    freeze = subparsers.add_parser(
        "freeze-batches",
        help="write the exact CPU-only 110-step training-batch manifest",
    )
    freeze.add_argument("--data-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate", help="aggregate sealed worker receipts")
    aggregate.add_argument("--row-records", type=Path, required=True)
    aggregate.add_argument("--batch-manifest", type=Path, required=True)
    aggregate.add_argument("--batch-manifest-sha256", required=True)
    aggregate.add_argument("--p0-preflight-receipt", type=Path, required=True)
    aggregate.add_argument("--p0-preflight-receipt-sha256", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    claimed_output: Path | None = None
    if args.command != "plan":
        claim_schema = (
            "nadig-vnext-performance-batch-manifest-claim-v1"
            if args.command == "freeze-batches"
            else "nadig-vnext-performance-census-report-claim-v1"
        )
        claimed_output = _claim_json_output(
            args.output,
            {
                "schema_version": claim_schema,
                "status": "claimed",
                "evidence_class": "performance_training_only",
                "scientific_completion": False,
            },
        )
    try:
        bindings = bind_matrix_variants(
            args.matrix,
            repository_root=args.repository_root,
            expected_matrix_sha256=args.expected_matrix_sha256,
        )
    except BaseException as error:
        if claimed_output is not None:
            _atomic_json(
                claimed_output,
                {
                    "schema_version": "nadig-vnext-performance-evidence-failure-v1",
                    "status": "failed",
                    "evidence_class": "performance_training_only",
                    "scientific_completion": False,
                    "primary_failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
            )
        raise
    if args.command == "plan":
        payload = {
            "schema_version": "nadig-vnext-performance-census-plan-v1",
            "matrix_sha256": args.expected_matrix_sha256,
            "stage_protocols": {name: value.payload() for name, value in STAGE_PROTOCOLS.items()},
            "rows": [binding.payload() for binding in bindings],
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "freeze-batches":
        assert claimed_output is not None
        try:
            payload = freeze_batch_manifest(
                matrix_path=args.matrix,
                repository_root=args.repository_root,
                expected_matrix_sha256=args.expected_matrix_sha256,
                data_root=args.data_root,
            )
        except BaseException as error:
            _atomic_json(
                claimed_output,
                {
                    "schema_version": "nadig-vnext-performance-batch-manifest-v2",
                    "status": "failed",
                    "evidence_class": "performance_training_only",
                    "scientific_completion": False,
                    "primary_failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
            )
            raise
        _atomic_json(claimed_output, payload)
        print(claimed_output)
        return 0
    row_records = json.loads(args.row_records.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(row_records, list) or any(
        not isinstance(value, dict) for value in row_records
    ):
        raise ValueError("census row records must be a JSON list of objects")
    batch_manifest = load_frozen_batch_manifest(
        args.batch_manifest,
        expected_sha256=args.batch_manifest_sha256,
        expected_matrix_sha256=args.expected_matrix_sha256,
        expected_config_sha256=bindings[0].config_sha256,
    )
    p0_preflight = _load_hashed_json(
        args.p0_preflight_receipt,
        args.p0_preflight_receipt_sha256,
    )
    if (
        p0_preflight.get("schema_version") != "nadig-vnext-performance-p0-preflight-v1"
        or p0_preflight.get("status") != "passed"
        or p0_preflight.get("matrix_sha256") != args.expected_matrix_sha256
    ):
        raise ValueError("aggregate P0 preflight identity differs")
    report = aggregate_census_report(
        bindings=bindings,
        row_records=row_records,
        batch_manifest=batch_manifest,
        p0_preflight_sha256=args.p0_preflight_receipt_sha256,
    )
    assert claimed_output is not None
    _atomic_json(claimed_output, report)
    print(claimed_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
