"""Hash-linked contracts for data, splits, runs, predictions, and evaluation."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from gradpert.config.schema import DatasetId, ModelId
from gradpert.contracts.base import GitCommit, NonEmpty, Sha256, StrictManifest
from gradpert.hashing import sha256_json


def _require_unique(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate IDs")


class SourceManifest(StrictManifest):
    schema_version: Literal["dataset-source-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    registry_version: NonEmpty
    source_url: NonEmpty
    filename: NonEmpty
    source_sha256: Sha256
    size_bytes: int = Field(gt=0)
    license_id: NonEmpty
    source_semantics: Literal["raw", "upstream_processed"]


class CanonicalDataManifest(StrictManifest):
    schema_version: Literal["canonical-data-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    state: Literal["canonical_ready"]
    canonical_adata_path: NonEmpty
    canonical_adata_sha256: Sha256
    source_manifest_sha256: Sha256
    preprocessing_manifest_sha256: Sha256
    qc_manifest_sha256: Sha256
    split_manifest_sha256: Sha256
    split_content_sha256: Sha256
    evaluation_controls_sha256: Sha256
    expression_gene_order_sha256: Sha256
    graph_gene_order_sha256: Sha256
    observation_order_sha256: Sha256
    n_cells: int = Field(gt=0)
    n_expression_genes: int = Field(gt=0)
    n_graph_genes: int = Field(gt=0)
    n_conditions: int = Field(gt=0)
    n_controls: int = Field(gt=0)

    @model_validator(mode="after")
    def enforce_gene_axes(self) -> CanonicalDataManifest:
        if self.n_graph_genes < self.n_expression_genes:
            raise ValueError("graph gene axis cannot be smaller than expression gene axis")
        return self


class SplitManifest(StrictManifest):
    schema_version: Literal["split-manifest-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    policy_id: NonEmpty
    split_seed: int
    control_condition_id: NonEmpty
    train_conditions: list[NonEmpty]
    val_conditions: list[NonEmpty]
    test_conditions: list[NonEmpty]
    split_content_sha256: Sha256

    def content_payload(self) -> dict[str, object]:
        """Return the exact fields committed by ``split_content_sha256``."""

        return {
            "dataset_id": self.dataset_id,
            "protocol_id": self.protocol_id,
            "policy_id": self.policy_id,
            "split_seed": self.split_seed,
            "control_condition_id": self.control_condition_id,
            "train_conditions": self.train_conditions,
            "val_conditions": self.val_conditions,
            "test_conditions": self.test_conditions,
        }

    @model_validator(mode="after")
    def enforce_partition(self) -> SplitManifest:
        for field_name, values in (
            ("train_conditions", self.train_conditions),
            ("val_conditions", self.val_conditions),
            ("test_conditions", self.test_conditions),
        ):
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            _require_unique(values, field_name)
            if self.control_condition_id in values:
                raise ValueError("control condition cannot be a perturbation split member")
        train = set(self.train_conditions)
        val = set(self.val_conditions)
        test = set(self.test_conditions)
        if train & val or train & test or val & test:
            raise ValueError("train/val/test condition sets must be disjoint")
        if sha256_json(self.content_payload()) != self.split_content_sha256:
            raise ValueError("split_content_sha256 does not match split fields")
        return self


class ControlDraw(StrictManifest):
    condition_id: NonEmpty
    context_policy: Literal["truth_cell_context_resampling"]
    source_pool_sha256: Sha256
    ordered_context_ids: list[NonEmpty]
    ordered_context_ids_sha256: Sha256
    ordered_row_ids: list[NonEmpty]
    ordered_row_ids_sha256: Sha256

    @model_validator(mode="after")
    def enforce_draw(self) -> ControlDraw:
        if len(self.ordered_row_ids) != 300:
            raise ValueError("each evaluation control draw must contain exactly 300 row IDs")
        if len(self.ordered_context_ids) != 300:
            raise ValueError("each evaluation control draw must contain exactly 300 context IDs")
        if sha256_json(self.ordered_context_ids) != self.ordered_context_ids_sha256:
            raise ValueError("ordered_context_ids_sha256 does not match ordered_context_ids")
        if sha256_json(self.ordered_row_ids) != self.ordered_row_ids_sha256:
            raise ValueError("ordered_row_ids_sha256 does not match ordered_row_ids")
        return self


class EvaluationControlManifest(StrictManifest):
    schema_version: Literal["evaluation-controls-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    split_name: Literal["val", "test"]
    split_content_sha256: Sha256
    evaluation_seed: Literal[20260824]
    rng: Literal["numpy_pcg64"]
    sample_with_replacement: Literal[True]
    context_policy: Literal["truth_cell_context_resampling"]
    n_controls_per_condition: Literal[300]
    draws: list[ControlDraw]

    @model_validator(mode="after")
    def enforce_conditions(self) -> EvaluationControlManifest:
        if not self.draws:
            raise ValueError("evaluation control manifest must contain draws")
        _require_unique([draw.condition_id for draw in self.draws], "draw conditions")
        return self


class GraphSourceArtifact(StrictManifest):
    source_name: Literal["go", "string"]
    upstream_relative_path: NonEmpty
    upstream_file_sha256: Sha256
    upstream_size_bytes: int = Field(gt=0)
    dropped_empty_endpoint_row_count: int = Field(ge=0)
    filtered_nonself_edge_count: int = Field(ge=0)
    pruned_nonself_edge_count: int = Field(ge=0)
    covered_gene_count: int = Field(ge=0)
    artifact_path: NonEmpty
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def enforce_edge_counts(self) -> GraphSourceArtifact:
        if self.pruned_nonself_edge_count > self.filtered_nonself_edge_count:
            raise ValueError("pruned graph cannot contain more edges than its filtered source")
        return self


class DatasetGraphManifest(StrictManifest):
    schema_version: Literal["dataset-graph-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    state: Literal["graph_ready"]
    source_registry_sha256: Sha256
    source_repository: NonEmpty
    source_commit: GitCommit
    canonical_data_sha256: Sha256
    graph_gene_order_sha256: Sha256
    graph_gene_count: int = Field(gt=0)
    candidate_target_count: int = Field(gt=0)
    both_sources_missing_target_count: int = Field(ge=0)
    topology_content_sha256: Sha256
    sources: list[GraphSourceArtifact]
    coverage_report_path: NonEmpty
    coverage_report_sha256: Sha256
    missing_genes_path: NonEmpty
    missing_genes_sha256: Sha256
    isolated_genes_path: NonEmpty
    isolated_genes_sha256: Sha256

    @model_validator(mode="after")
    def require_two_sources(self) -> DatasetGraphManifest:
        if len(self.sources) != 2 or {item.source_name for item in self.sources} != {
            "go",
            "string",
        }:
            raise ValueError("dataset graph requires exactly separate GO and STRING artifacts")
        if any(item.covered_gene_count > self.graph_gene_count for item in self.sources):
            raise ValueError("source coverage exceeds graph gene universe")
        if self.both_sources_missing_target_count > self.candidate_target_count:
            raise ValueError("missing target count exceeds candidate target count")
        return self


class EvaluationStateManifest(StrictManifest):
    schema_version: Literal["evaluation-state-v1"]
    dataset_id: DatasetId
    protocol_id: NonEmpty
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    expression_gene_order_sha256: Sha256
    condition_ids: list[NonEmpty]
    condition_ids_sha256: Sha256
    de_gene_indices: dict[NonEmpty, list[int]]
    de_gene_indices_sha256: Sha256
    top_de_gene_indices: dict[NonEmpty, list[int]]
    top_de_gene_indices_sha256: Sha256
    de_unavailable_reasons: dict[NonEmpty, NonEmpty]
    de_unavailable_reasons_sha256: Sha256
    de_method: Literal["scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"]
    de_reference: Literal["ctrl"]
    de_source_commit: GitCommit
    systema_reference_condition_ids: list[NonEmpty]
    systema_reference_condition_ids_sha256: Sha256
    arrays_path: NonEmpty
    arrays_sha256: Sha256
    systema_reference_content_sha256: Sha256
    metric_control_means_content_sha256: Sha256

    @model_validator(mode="after")
    def enforce_evaluation_state(self) -> EvaluationStateManifest:
        _require_unique(self.condition_ids, "evaluation-state conditions")
        _require_unique(
            self.systema_reference_condition_ids,
            "Systema reference conditions",
        )
        if sha256_json(self.condition_ids) != self.condition_ids_sha256:
            raise ValueError("evaluation-state condition hash mismatch")
        if sha256_json(self.systema_reference_condition_ids) != (
            self.systema_reference_condition_ids_sha256
        ):
            raise ValueError("Systema reference condition hash mismatch")
        expected = set(self.condition_ids)
        if set(self.de_gene_indices) != expected or set(self.top_de_gene_indices) != expected:
            raise ValueError("evaluation-state DE condition sets differ")
        if not set(self.de_unavailable_reasons).issubset(expected):
            raise ValueError("evaluation-state unavailable reason has an unknown condition")
        if sha256_json(self.de_unavailable_reasons) != self.de_unavailable_reasons_sha256:
            raise ValueError("evaluation-state unavailable reason hash mismatch")
        for field, values_by_condition, observed_hash in (
            (
                "de_gene_indices",
                self.de_gene_indices,
                self.de_gene_indices_sha256,
            ),
            (
                "top_de_gene_indices",
                self.top_de_gene_indices,
                self.top_de_gene_indices_sha256,
            ),
        ):
            for condition_id, values in values_by_condition.items():
                if len(values) > 20 or len(values) != len(set(values)):
                    raise ValueError(f"{field}[{condition_id}] must contain 0--20 unique IDs")
                if values and min(values) < 0:
                    raise ValueError(f"{field}[{condition_id}] contains a negative ID")
                unavailable = condition_id in self.de_unavailable_reasons
                if unavailable == bool(values):
                    raise ValueError(f"{field}[{condition_id}] availability and reason disagree")
            if sha256_json(values_by_condition) != observed_hash:
                raise ValueError(f"{field} content hash mismatch")
        return self


class ArrayReference(StrictManifest):
    """A server-only array address plus integrity/shape metadata."""

    artifact_path: NonEmpty
    key: NonEmpty
    dtype: NonEmpty
    shape: list[int]
    content_sha256: Sha256

    @model_validator(mode="after")
    def enforce_shape(self) -> ArrayReference:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("array shape dimensions must be positive")
        return self


class PredictionCondition(StrictManifest):
    condition_id: NonEmpty
    input_control_row_ids: list[NonEmpty]
    input_control_row_ids_sha256: Sha256
    prediction: ArrayReference
    input_control: ArrayReference

    @model_validator(mode="after")
    def enforce_common_shape(self) -> PredictionCondition:
        if len(self.input_control_row_ids) != 300:
            raise ValueError("prediction condition requires 300 input controls")
        if sha256_json(self.input_control_row_ids) != self.input_control_row_ids_sha256:
            raise ValueError("input control row hash mismatch")
        if len(self.prediction.shape) != 2 or len(self.input_control.shape) != 2:
            raise ValueError("prediction and input control arrays must be rank two")
        if self.prediction.shape != self.input_control.shape:
            raise ValueError("prediction and input control shapes must match")
        if self.prediction.shape[0] != 300:
            raise ValueError("prediction arrays must preserve 300 rows")
        return self


class PredictionArtifactManifest(StrictManifest):
    schema_version: Literal["prediction-artifact-v1"]
    model_id: ModelId
    dataset_id: DatasetId
    protocol_id: NonEmpty
    run_id: NonEmpty
    run_seed: int
    source_commit: GitCommit
    source_dirty: bool
    formal_eligible: bool
    config_sha256: Sha256
    environment_sha256: Sha256
    canonical_data_sha256: Sha256
    gene_order_sha256: Sha256
    split_content_sha256: Sha256
    control_manifest_sha256: Sha256
    checkpoint_sha256: Sha256 | None
    truth_included: Literal[False]
    prediction_kind: Literal["per_control_population"]
    conditions: list[PredictionCondition]

    @model_validator(mode="after")
    def enforce_artifact(self) -> PredictionArtifactManifest:
        if not self.conditions:
            raise ValueError("prediction artifact must contain conditions")
        _require_unique([item.condition_id for item in self.conditions], "prediction conditions")
        learned = self.model_id in {"gradpert_b2", "gears", "txpert_public"}
        if learned != (self.checkpoint_sha256 is not None):
            raise ValueError(
                "learned artifacts require a checkpoint; nonlearned artifacts forbid it"
            )
        if self.formal_eligible and self.source_dirty:
            raise ValueError("formal prediction artifacts require a clean source worktree")
        return self


class MetricAvailability(StrictManifest):
    metric_id: Literal[
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    ]
    available: bool
    macro_mean: float | None
    reason: str | None
    finite_condition_count: int = Field(ge=0)
    total_condition_count: int = Field(gt=0)

    @model_validator(mode="after")
    def enforce_reason(self) -> MetricAvailability:
        if self.available:
            if self.reason is not None or self.macro_mean is None:
                raise ValueError("available metric requires a mean and no unavailable reason")
            if not math.isfinite(self.macro_mean) or self.finite_condition_count == 0:
                raise ValueError("available metric requires finite condition values")
        elif not self.reason or self.macro_mean is not None or self.finite_condition_count != 0:
            raise ValueError("unavailable metric requires reason, no mean, and zero finite values")
        if self.finite_condition_count > self.total_condition_count:
            raise ValueError("finite condition count exceeds total")
        return self


class ConditionMetricValue(StrictManifest):
    metric_id: Literal[
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    ]
    value: float | None
    reason: str | None
    gene_count: int = Field(ge=0)

    @model_validator(mode="after")
    def enforce_value(self) -> ConditionMetricValue:
        if self.value is None:
            if not self.reason:
                raise ValueError("undefined condition metric requires a reason")
        elif self.reason is not None or not math.isfinite(self.value):
            raise ValueError("defined condition metric must be finite and reason-free")
        return self


class EvaluationCondition(StrictManifest):
    condition_id: NonEmpty
    truth_row_ids: list[NonEmpty]
    truth_row_ids_sha256: Sha256
    truth: ArrayReference
    metric_control_pool_mean: ArrayReference
    de_gene_indices: list[int]
    de_gene_indices_sha256: Sha256
    top_de_gene_indices: list[int]
    top_de_gene_indices_sha256: Sha256
    de_unavailable_reason: str | None
    metrics: list[ConditionMetricValue]

    @model_validator(mode="after")
    def enforce_condition(self) -> EvaluationCondition:
        if not self.truth_row_ids or len(self.truth_row_ids) != self.truth.shape[0]:
            raise ValueError("truth row IDs must match the non-empty truth population")
        _require_unique(self.truth_row_ids, "truth row IDs")
        if sha256_json(self.truth_row_ids) != self.truth_row_ids_sha256:
            raise ValueError("truth row hash mismatch")
        if len(self.truth.shape) != 2 or len(self.metric_control_pool_mean.shape) != 1:
            raise ValueError("truth must be rank two and metric control mean rank one")
        if self.truth.shape[1] != self.metric_control_pool_mean.shape[0]:
            raise ValueError("truth and metric control gene dimensions differ")
        for values, observed_hash, field in (
            (self.de_gene_indices, self.de_gene_indices_sha256, "de_gene_indices"),
            (
                self.top_de_gene_indices,
                self.top_de_gene_indices_sha256,
                "top_de_gene_indices",
            ),
        ):
            if len(values) != len(set(values)) or (values and min(values) < 0):
                raise ValueError(f"{field} must contain unique nonnegative indices")
            if values and max(values) >= self.truth.shape[1]:
                raise ValueError(f"{field} contains an out-of-range index")
            if sha256_json(values) != observed_hash:
                raise ValueError(f"{field} hash mismatch")
            if (self.de_unavailable_reason is not None) == bool(values):
                raise ValueError(f"{field} availability and reason disagree")
        expected = {
            "txpert_macro_pearson_delta",
            "trishift_pearson_delta",
            "systema_pearson",
        }
        if {metric.metric_id for metric in self.metrics} != expected or len(self.metrics) != 3:
            raise ValueError("evaluation condition requires exactly the three headline metrics")
        return self


class EvaluationBundleManifest(StrictManifest):
    schema_version: Literal["evaluation-bundle-v1"]
    prediction_manifest_sha256: Sha256
    prediction_artifact_file_sha256: Sha256
    gene_order_sha256: Sha256
    evaluation_state_manifest_file_sha256: Sha256
    evaluation_state_arrays_sha256: Sha256
    evaluation_state_condition_ids_sha256: Sha256
    de_gene_indices_sha256: Sha256
    top_de_gene_indices_sha256: Sha256
    de_unavailable_reasons_sha256: Sha256
    de_method: Literal["scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"]
    de_reference: Literal["ctrl"]
    de_source_commit: GitCommit
    systema_reference_condition_ids: list[NonEmpty]
    systema_reference_condition_ids_sha256: Sha256
    metric_control_means_content_sha256: Sha256
    systema_reference: ArrayReference
    conditions: list[EvaluationCondition]
    metric_registry_version: Literal["gradpert-metrics-v1"]
    metrics: list[MetricAvailability]

    @model_validator(mode="after")
    def enforce_metric_registry(self) -> EvaluationBundleManifest:
        if not self.conditions:
            raise ValueError("evaluation bundle requires conditions")
        _require_unique([item.condition_id for item in self.conditions], "evaluation conditions")
        _require_unique(
            self.systema_reference_condition_ids,
            "evaluation bundle Systema reference conditions",
        )
        if sha256_json(self.systema_reference_condition_ids) != (
            self.systema_reference_condition_ids_sha256
        ):
            raise ValueError("evaluation bundle Systema reference condition hash mismatch")
        gene_count = self.systema_reference.shape
        if len(gene_count) != 1:
            raise ValueError("Systema reference must be a vector")
        if any(condition.truth.shape[1] != gene_count[0] for condition in self.conditions):
            raise ValueError("evaluation condition gene dimensions differ")
        expected = {
            "txpert_macro_pearson_delta",
            "trishift_pearson_delta",
            "systema_pearson",
        }
        observed = {metric.metric_id for metric in self.metrics}
        if observed != expected or len(self.metrics) != 3:
            raise ValueError("evaluation bundle requires exactly the three headline metrics")
        if any(metric.total_condition_count != len(self.conditions) for metric in self.metrics):
            raise ValueError("metric summary denominator differs from condition count")
        return self


class RunManifest(StrictManifest):
    schema_version: Literal["run-manifest-v1"]
    run_id: NonEmpty
    model_id: ModelId
    dataset_id: DatasetId
    protocol_id: NonEmpty
    run_seed: int
    source_commit: GitCommit
    source_dirty: bool
    formal_eligible: bool
    config_sha256: Sha256
    environment_sha256: Sha256
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    control_manifest_sha256: Sha256
    status: Literal["started", "trained", "predicted", "evaluated", "failed"]
    best_checkpoint_sha256: Sha256 | None
    test_evaluations: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def enforce_formal_source(self) -> RunManifest:
        if self.formal_eligible and self.source_dirty:
            raise ValueError("formal runs require a clean source worktree")
        return self


class ServerArtifactPointer(StrictManifest):
    schema_version: Literal["server-artifact-pointer-v1"]
    run_id: NonEmpty
    source_commit: GitCommit
    server_root: NonEmpty
    prediction_manifest_path: NonEmpty
    prediction_manifest_sha256: Sha256
    evaluation_manifest_path: NonEmpty
    evaluation_manifest_sha256: Sha256
    synchronized_large_artifacts: Literal[False]


class ResultCatalogEntry(StrictManifest):
    """One explicit, hash-pinned evaluated run exposed to analysis clients."""

    run_id: NonEmpty
    model_id: ModelId
    dataset_id: DatasetId
    run_manifest_path: NonEmpty
    run_manifest_sha256: Sha256
    server_pointer_path: NonEmpty
    server_pointer_sha256: Sha256
    metrics_path: NonEmpty
    metrics_sha256: Sha256


class ResultCatalogManifest(StrictManifest):
    """Notebook-facing index that never discovers runs by directory or mtime."""

    schema_version: Literal["result-catalog-v1"]
    catalog_id: NonEmpty
    entries: list[ResultCatalogEntry]

    @model_validator(mode="after")
    def enforce_unique_runs(self) -> ResultCatalogManifest:
        if not self.entries:
            raise ValueError("result catalog requires at least one evaluated run")
        _require_unique([entry.run_id for entry in self.entries], "catalog run IDs")
        return self
