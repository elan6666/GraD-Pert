"""Evaluator-only truth join, metric computation, and sealed bundle loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from gradpert.artifacts._io import atomic_pickle, load_hash_pinned_pickle, sha256_file
from gradpert.artifacts.prediction import PredictionArtifact, sha256_array
from gradpert.contracts import (
    ArrayReference,
    ConditionMetricValue,
    EvaluationBundleManifest,
    EvaluationCondition,
    MetricAvailability,
    PredictionArtifactManifest,
)
from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.evaluation.metrics import (
    ConditionMetrics,
    compute_condition_metrics,
    macro_summarize,
)
from gradpert.evaluation.state import LoadedEvaluationState
from gradpert.hashing import sha256_json

LEGACY_EVALUATION_PKL_SCHEMA = "evaluation-pkl-v1"
EVALUATION_PKL_SCHEMA = "result-pkl-v1"
_LEGACY_CONDITION_KEYS = {
    "condition_id",
    "Pred",
    "InputCtrl",
    "InputCtrlRowIds",
    "Truth",
    "TruthRowIds",
    "MetricCtrlPoolMean",
    "DE_idx",
    "TopDE_idx",
    "DEUnavailableReason",
    "gene_ids",
    "metrics",
}
_CONDITION_KEYS = (_LEGACY_CONDITION_KEYS - {"InputCtrl"}) | {"InputCtrlIndices"}


@dataclass(frozen=True)
class EvaluationConditionInput:
    condition_id: str
    truth: np.ndarray[Any, Any]
    truth_row_ids: tuple[str, ...]
    metric_control_pool_mean: np.ndarray[Any, Any]
    de_gene_indices: tuple[int, ...]
    top_de_gene_indices: tuple[int, ...]
    de_unavailable_reason: str | None = None


@dataclass(frozen=True)
class EvaluationProvenance:
    """Exact frozen evaluator state used for every reported metric."""

    state_manifest_file_sha256: str
    state_arrays_sha256: str
    state_condition_ids_sha256: str
    de_gene_indices_sha256: str
    top_de_gene_indices_sha256: str
    de_unavailable_reasons_sha256: str
    de_method: Literal["scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"]
    de_reference: Literal["ctrl"]
    de_source_commit: str
    systema_reference_condition_ids: tuple[str, ...]
    systema_reference_condition_ids_sha256: str
    metric_control_means_content_sha256: str


@dataclass(frozen=True)
class EvaluationConditionArrays:
    condition_id: str
    prediction: np.ndarray[Any, Any]
    input_control: np.ndarray[Any, Any]
    input_control_row_ids: tuple[str, ...]
    truth: np.ndarray[Any, Any]
    truth_row_ids: tuple[str, ...]
    metric_control_pool_mean: np.ndarray[Any, Any]
    de_gene_indices: tuple[int, ...]
    top_de_gene_indices: tuple[int, ...]
    de_unavailable_reason: str | None
    metrics: ConditionMetrics


@dataclass(frozen=True)
class EvaluationBundle:
    manifest: EvaluationBundleManifest
    prediction_manifest: PredictionArtifactManifest
    gene_ids: tuple[str, ...]
    systema_reference: np.ndarray[Any, Any]
    conditions: Mapping[str, EvaluationConditionArrays]
    file_sha256: str


def _numeric_array(
    value: np.ndarray[Any, Any],
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.ascontiguousarray(array)


def _array_reference(path: Path, key: str, value: np.ndarray[Any, Any]) -> ArrayReference:
    return ArrayReference(
        artifact_path=str(path),
        key=key,
        dtype=str(value.dtype),
        shape=list(value.shape),
        content_sha256=sha256_array(value),
    )


def _metric_contracts(metrics: ConditionMetrics) -> list[ConditionMetricValue]:
    return [
        ConditionMetricValue(
            metric_id=result.metric_id,
            value=result.value,
            reason=result.reason,
            gene_count=result.gene_count,
        )
        for result in metrics.results
    ]


def _availability(metrics: Sequence[ConditionMetrics]) -> list[MetricAvailability]:
    contracts: list[MetricAvailability] = []
    for summary in macro_summarize(metrics):
        available = summary.macro_mean is not None
        reason = None
        if not available:
            unique_reasons = sorted(set(summary.unavailable_reasons))
            reason = ";".join(unique_reasons) or "no_finite_condition_values"
        contracts.append(
            MetricAvailability(
                metric_id=summary.metric_id,
                available=available,
                macro_mean=summary.macro_mean,
                reason=reason,
                finite_condition_count=summary.finite_condition_count,
                total_condition_count=summary.total_condition_count,
            )
        )
    return contracts


def _prepare_condition(
    *,
    artifact_path: Path,
    prediction: PredictionArtifact,
    source: EvaluationConditionInput,
    systema_reference: np.ndarray[Any, Any],
) -> tuple[dict[str, object], EvaluationCondition, ConditionMetrics]:
    if source.condition_id not in prediction.conditions:
        raise ValueError(f"evaluation condition is absent from prediction: {source.condition_id}")
    predicted = prediction.conditions[source.condition_id]
    gene_count = len(prediction.gene_ids)
    truth = _numeric_array(source.truth, name=f"Truth[{source.condition_id}]")
    if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] != gene_count:
        raise ValueError(f"Truth[{source.condition_id}] must have shape [N>0,{gene_count}]")
    truth_row_ids = tuple(str(value) for value in source.truth_row_ids)
    if len(truth_row_ids) != truth.shape[0] or len(truth_row_ids) != len(set(truth_row_ids)):
        raise ValueError("truth row IDs must be unique and match the truth population")
    metric_control = _numeric_array(
        source.metric_control_pool_mean,
        name=f"MetricCtrlPoolMean[{source.condition_id}]",
        shape=(gene_count,),
    )
    de = tuple(int(value) for value in source.de_gene_indices)
    top_de = tuple(int(value) for value in source.top_de_gene_indices)
    metrics = compute_condition_metrics(
        condition_id=source.condition_id,
        prediction=predicted.prediction,
        input_control=predicted.input_control,
        truth=truth,
        metric_control_pool_mean=metric_control,
        de_gene_indices=de,
        top_de_gene_indices=top_de,
        systema_reference=systema_reference,
        de_unavailable_reason=source.de_unavailable_reason,
    )
    key_root = f"conditions/{source.condition_id}"
    contract = EvaluationCondition(
        condition_id=source.condition_id,
        truth_row_ids=list(truth_row_ids),
        truth_row_ids_sha256=sha256_json(list(truth_row_ids)),
        truth=_array_reference(artifact_path, f"{key_root}/Truth", truth),
        metric_control_pool_mean=_array_reference(
            artifact_path,
            f"{key_root}/MetricCtrlPoolMean",
            metric_control,
        ),
        de_gene_indices=list(de),
        de_gene_indices_sha256=sha256_json(list(de)),
        top_de_gene_indices=list(top_de),
        top_de_gene_indices_sha256=sha256_json(list(top_de)),
        de_unavailable_reason=source.de_unavailable_reason,
        metrics=_metric_contracts(metrics),
    )
    payload: dict[str, object] = {
        "condition_id": source.condition_id,
        "Pred": predicted.prediction,
        "InputCtrl": predicted.input_control,
        "InputCtrlRowIds": predicted.input_control_row_ids,
        "Truth": truth,
        "TruthRowIds": truth_row_ids,
        "MetricCtrlPoolMean": metric_control,
        "DE_idx": de,
        "TopDE_idx": top_de,
        "DEUnavailableReason": source.de_unavailable_reason,
        "gene_ids": prediction.gene_ids,
        "metrics": [item.model_dump(mode="json") for item in contract.metrics],
    }
    return payload, contract, metrics


def _deduplicate_input_controls(
    payloads: Mapping[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Store each selected control row once while preserving every draw and its order."""

    row_to_index: dict[str, int] = {}
    pooled_ids: list[str] = []
    pooled_rows: list[np.ndarray[Any, Any]] = []
    compact: dict[str, dict[str, object]] = {}
    for condition_id in sorted(payloads):
        payload = dict(payloads[condition_id])
        row_ids = tuple(str(value) for value in cast(Sequence[object], payload["InputCtrlRowIds"]))
        values = np.asarray(payload.pop("InputCtrl"))
        if values.ndim != 2 or values.shape[0] != len(row_ids):
            raise ValueError(f"input controls and row IDs differ: {condition_id}")
        indices: list[int] = []
        for row_id, row in zip(row_ids, values, strict=True):
            observed = np.ascontiguousarray(row)
            existing = row_to_index.get(row_id)
            if existing is None:
                existing = len(pooled_ids)
                row_to_index[row_id] = existing
                pooled_ids.append(row_id)
                pooled_rows.append(observed)
            elif not np.array_equal(pooled_rows[existing], observed, equal_nan=True):
                raise ValueError(f"control row ID maps to conflicting expression: {row_id}")
            indices.append(existing)
        payload["InputCtrlIndices"] = np.asarray(indices, dtype=np.int64)
        compact[condition_id] = payload
    if not pooled_rows:
        raise ValueError("result bundle requires selected input controls")
    return (
        {
            "row_ids": tuple(pooled_ids),
            "expression": np.ascontiguousarray(np.stack(pooled_rows, axis=0)),
        },
        compact,
    )


def seal_evaluation_bundle(
    path: str | Path,
    *,
    prediction: PredictionArtifact,
    conditions: Sequence[EvaluationConditionInput],
    systema_reference: np.ndarray[Any, Any],
    provenance: EvaluationProvenance,
) -> EvaluationBundle:
    """Join evaluator-only truth and atomically seal a standalone condition bundle."""

    artifact_path = Path(path).resolve()
    reference = _numeric_array(
        systema_reference,
        name="systema_reference",
        shape=(len(prediction.gene_ids),),
    )
    if not conditions:
        raise ValueError("evaluation bundle requires conditions")
    if len({item.condition_id for item in conditions}) != len(conditions):
        raise ValueError("evaluation bundle contains duplicate condition IDs")
    if {item.condition_id for item in conditions} != set(prediction.conditions):
        raise ValueError("evaluation and prediction condition sets differ")

    payloads: dict[str, dict[str, object]] = {}
    contracts: list[EvaluationCondition] = []
    metric_rows: list[ConditionMetrics] = []
    for source in sorted(conditions, key=lambda item: item.condition_id):
        payload, contract, metrics = _prepare_condition(
            artifact_path=artifact_path,
            prediction=prediction,
            source=source,
            systema_reference=reference,
        )
        payloads[source.condition_id] = payload
        contracts.append(contract)
        metric_rows.append(metrics)
    manifest = EvaluationBundleManifest(
        schema_version="evaluation-bundle-v1",
        prediction_manifest_sha256=sha256_json(prediction.manifest.model_dump(mode="json")),
        prediction_artifact_file_sha256=prediction.file_sha256,
        gene_order_sha256=prediction.manifest.gene_order_sha256,
        evaluation_state_manifest_file_sha256=provenance.state_manifest_file_sha256,
        evaluation_state_arrays_sha256=provenance.state_arrays_sha256,
        evaluation_state_condition_ids_sha256=provenance.state_condition_ids_sha256,
        de_gene_indices_sha256=provenance.de_gene_indices_sha256,
        top_de_gene_indices_sha256=provenance.top_de_gene_indices_sha256,
        de_unavailable_reasons_sha256=provenance.de_unavailable_reasons_sha256,
        de_method=provenance.de_method,
        de_reference=provenance.de_reference,
        de_source_commit=provenance.de_source_commit,
        systema_reference_condition_ids=list(provenance.systema_reference_condition_ids),
        systema_reference_condition_ids_sha256=(provenance.systema_reference_condition_ids_sha256),
        metric_control_means_content_sha256=(provenance.metric_control_means_content_sha256),
        systema_reference=_array_reference(
            artifact_path,
            "systema_reference",
            reference,
        ),
        conditions=contracts,
        metric_registry_version="gradpert-metrics-v1",
        metrics=_availability(metric_rows),
    )
    shared_controls, compact_payloads = _deduplicate_input_controls(payloads)
    package = {
        "schema_version": EVALUATION_PKL_SCHEMA,
        "manifest": manifest.model_dump(mode="json"),
        "prediction_manifest": prediction.manifest.model_dump(mode="json"),
        "gene_ids": prediction.gene_ids,
        "systema_reference": reference,
        "shared_controls": shared_controls,
        "conditions": compact_payloads,
    }
    atomic_pickle(artifact_path, package)
    return load_evaluation_bundle(
        artifact_path,
        expected_file_sha256=sha256_file(artifact_path),
        trusted_root=artifact_path.parent,
    )


def seal_frozen_evaluation_bundle(
    path: str | Path,
    *,
    prediction: PredictionArtifact,
    data: CanonicalEvaluationData,
    state: LoadedEvaluationState,
) -> EvaluationBundle:
    """Join one prediction artifact to canonical truth through frozen evaluator state."""

    manifest = state.manifest
    prediction_manifest = prediction.manifest
    if (manifest.dataset_id, manifest.protocol_id) != (
        prediction_manifest.dataset_id,
        prediction_manifest.protocol_id,
    ):
        raise ValueError("evaluation state and prediction identities differ")
    if (data.manifest.dataset_id, data.manifest.protocol_id) != (
        prediction_manifest.dataset_id,
        prediction_manifest.protocol_id,
    ):
        raise ValueError("canonical evaluator and prediction identities differ")
    if manifest.canonical_data_sha256 != prediction_manifest.canonical_data_sha256:
        raise ValueError("evaluation state and prediction canonical hashes differ")
    if manifest.split_content_sha256 != prediction_manifest.split_content_sha256:
        raise ValueError("evaluation state and prediction split hashes differ")
    if data.control_manifest_file_sha256 != prediction_manifest.control_manifest_sha256:
        raise ValueError("canonical evaluator and prediction control-manifest hashes differ")
    if manifest.expression_gene_order_sha256 != prediction_manifest.gene_order_sha256:
        raise ValueError("evaluation state and prediction gene-order hashes differ")
    if data.expression_gene_ids != prediction.gene_ids:
        raise ValueError("canonical evaluator and prediction gene IDs differ")

    condition_index = {
        condition_id: index for index, condition_id in enumerate(manifest.condition_ids)
    }
    expected_conditions = tuple(draw.condition_id for draw in data.control_manifest.draws)
    if set(prediction.conditions) != set(expected_conditions):
        raise ValueError("prediction conditions differ from canonical evaluator split")
    inputs: list[EvaluationConditionInput] = []
    for condition_id in expected_conditions:
        if condition_id not in condition_index:
            raise ValueError(f"frozen evaluation state lacks condition: {condition_id}")
        truth = data.load_truth_rows(condition_id)
        inputs.append(
            EvaluationConditionInput(
                condition_id=condition_id,
                truth=truth.expression,
                truth_row_ids=truth.ordered_row_ids,
                metric_control_pool_mean=state.metric_control_means[condition_index[condition_id]],
                de_gene_indices=tuple(manifest.de_gene_indices[condition_id]),
                top_de_gene_indices=tuple(manifest.top_de_gene_indices[condition_id]),
                de_unavailable_reason=manifest.de_unavailable_reasons.get(condition_id),
            )
        )
    provenance = EvaluationProvenance(
        state_manifest_file_sha256=state.manifest_file_sha256,
        state_arrays_sha256=manifest.arrays_sha256,
        state_condition_ids_sha256=manifest.condition_ids_sha256,
        de_gene_indices_sha256=manifest.de_gene_indices_sha256,
        top_de_gene_indices_sha256=manifest.top_de_gene_indices_sha256,
        de_unavailable_reasons_sha256=manifest.de_unavailable_reasons_sha256,
        de_method=manifest.de_method,
        de_reference=manifest.de_reference,
        de_source_commit=manifest.de_source_commit,
        systema_reference_condition_ids=tuple(manifest.systema_reference_condition_ids),
        systema_reference_condition_ids_sha256=(manifest.systema_reference_condition_ids_sha256),
        metric_control_means_content_sha256=(manifest.metric_control_means_content_sha256),
    )
    return seal_evaluation_bundle(
        path,
        prediction=prediction,
        conditions=inputs,
        systema_reference=state.systema_reference,
        provenance=provenance,
    )


def _same_metric(observed: ConditionMetricValue, recomputed: Any) -> bool:
    if observed.metric_id != recomputed.metric_id or observed.reason != recomputed.reason:
        return False
    if observed.gene_count != recomputed.gene_count:
        return False
    if observed.value is None or recomputed.value is None:
        return observed.value is None and recomputed.value is None
    return bool(np.isclose(observed.value, recomputed.value, rtol=0.0, atol=1e-12))


def load_evaluation_bundle(
    path: str | Path,
    *,
    expected_file_sha256: str,
    trusted_root: str | Path,
) -> EvaluationBundle:
    """Hash-gate a bundle before pickle load, then recompute every metric."""

    artifact_path, package = load_hash_pinned_pickle(
        path,
        expected_file_sha256=expected_file_sha256,
        trusted_root=trusted_root,
    )
    if not isinstance(package, dict):
        raise ValueError("evaluation PKL package shape is invalid")
    schema_version = package.get("schema_version")
    expected_keys = {
        "schema_version",
        "manifest",
        "prediction_manifest",
        "gene_ids",
        "systema_reference",
        "conditions",
    }
    if schema_version == EVALUATION_PKL_SCHEMA:
        expected_keys.add("shared_controls")
    elif schema_version != LEGACY_EVALUATION_PKL_SCHEMA:
        raise ValueError("evaluation PKL schema version is unsupported")
    if set(package) != expected_keys:
        raise ValueError("evaluation PKL package shape is invalid")
    manifest = EvaluationBundleManifest.model_validate(package["manifest"])
    prediction_manifest = PredictionArtifactManifest.model_validate(package["prediction_manifest"])
    if sha256_json(prediction_manifest.model_dump(mode="json")) != (
        manifest.prediction_manifest_sha256
    ):
        raise ValueError("embedded prediction manifest hash mismatch")
    gene_ids = tuple(str(value) for value in package["gene_ids"])
    if (
        not gene_ids
        or len(gene_ids) != len(set(gene_ids))
        or sha256_json(list(gene_ids)) != manifest.gene_order_sha256
    ):
        raise ValueError("evaluation bundle gene order/hash mismatch")
    reference = _numeric_array(
        package["systema_reference"],
        name="systema_reference",
        shape=(len(gene_ids),),
    )
    if sha256_array(reference) != manifest.systema_reference.content_sha256:
        raise ValueError("Systema reference hash mismatch")
    if Path(manifest.systema_reference.artifact_path).resolve() != artifact_path:
        raise ValueError("evaluation manifest artifact path mismatch")

    raw_conditions = package["conditions"]
    if not isinstance(raw_conditions, dict):
        raise ValueError("evaluation PKL conditions must be a mapping")
    contracts = {item.condition_id: item for item in manifest.conditions}
    prediction_contracts = {item.condition_id: item for item in prediction_manifest.conditions}
    if set(raw_conditions) != set(contracts) or set(raw_conditions) != set(prediction_contracts):
        raise ValueError("evaluation/prediction manifest condition IDs differ")
    loaded: dict[str, EvaluationConditionArrays] = {}
    metric_rows: list[ConditionMetrics] = []
    shared_control_ids: tuple[str, ...] = ()
    shared_control_expression: np.ndarray[Any, Any] | None = None
    if schema_version == EVALUATION_PKL_SCHEMA:
        shared = package["shared_controls"]
        if not isinstance(shared, dict) or set(shared) != {"row_ids", "expression"}:
            raise ValueError("shared control pool shape is invalid")
        shared_control_ids = tuple(str(value) for value in shared["row_ids"])
        if not shared_control_ids or len(shared_control_ids) != len(set(shared_control_ids)):
            raise ValueError("shared control row IDs must be non-empty and unique")
        shared_control_expression = _numeric_array(
            shared["expression"], name="shared_controls/expression"
        )
        if shared_control_expression.shape != (len(shared_control_ids), len(gene_ids)):
            raise ValueError("shared control expression shape is invalid")
    for condition_id in sorted(raw_conditions):
        payload = raw_conditions[condition_id]
        condition_keys = (
            _CONDITION_KEYS if schema_version == EVALUATION_PKL_SCHEMA else _LEGACY_CONDITION_KEYS
        )
        if not isinstance(payload, dict) or set(payload) != condition_keys:
            raise ValueError(f"evaluation condition payload is invalid: {condition_id}")
        if payload["condition_id"] != condition_id or tuple(payload["gene_ids"]) != gene_ids:
            raise ValueError(f"evaluation condition identity/gene mismatch: {condition_id}")
        prediction = _numeric_array(
            payload["Pred"],
            name=f"Pred[{condition_id}]",
            shape=(300, len(gene_ids)),
        )
        if schema_version == EVALUATION_PKL_SCHEMA:
            indices = np.asarray(payload["InputCtrlIndices"])
            if (
                indices.shape != (300,)
                or not np.issubdtype(indices.dtype, np.integer)
                or int(indices.min()) < 0
                or shared_control_expression is None
                or int(indices.max()) >= shared_control_expression.shape[0]
            ):
                raise ValueError(f"input control indices are invalid: {condition_id}")
            input_control = np.ascontiguousarray(shared_control_expression[indices])
        else:
            indices = None
            input_control = _numeric_array(
                payload["InputCtrl"],
                name=f"InputCtrl[{condition_id}]",
                shape=(300, len(gene_ids)),
            )
        truth = _numeric_array(payload["Truth"], name=f"Truth[{condition_id}]")
        metric_control = _numeric_array(
            payload["MetricCtrlPoolMean"],
            name=f"MetricCtrlPoolMean[{condition_id}]",
            shape=(len(gene_ids),),
        )
        contract = contracts[condition_id]
        prediction_contract = prediction_contracts[condition_id]
        input_row_ids = tuple(str(value) for value in payload["InputCtrlRowIds"])
        if indices is not None and input_row_ids != tuple(
            shared_control_ids[int(index)] for index in indices
        ):
            raise ValueError(f"input control indices/row IDs differ: {condition_id}")
        truth_row_ids = tuple(str(value) for value in payload["TruthRowIds"])
        de = tuple(int(value) for value in payload["DE_idx"])
        top_de = tuple(int(value) for value in payload["TopDE_idx"])
        de_unavailable_reason = payload["DEUnavailableReason"]
        if de_unavailable_reason is not None and not isinstance(de_unavailable_reason, str):
            raise ValueError(f"evaluation DE unavailable reason is invalid: {condition_id}")
        if (
            sha256_array(prediction) != prediction_contract.prediction.content_sha256
            or sha256_array(input_control) != prediction_contract.input_control.content_sha256
            or list(input_row_ids) != prediction_contract.input_control_row_ids
            or sha256_array(truth) != contract.truth.content_sha256
            or list(truth_row_ids) != contract.truth_row_ids
            or sha256_array(metric_control) != contract.metric_control_pool_mean.content_sha256
            or list(de) != contract.de_gene_indices
            or list(top_de) != contract.top_de_gene_indices
            or de_unavailable_reason != contract.de_unavailable_reason
        ):
            raise ValueError(f"evaluation condition content/hash mismatch: {condition_id}")
        recomputed = compute_condition_metrics(
            condition_id=condition_id,
            prediction=prediction,
            input_control=input_control,
            truth=truth,
            metric_control_pool_mean=metric_control,
            de_gene_indices=de,
            top_de_gene_indices=top_de,
            systema_reference=reference,
            de_unavailable_reason=de_unavailable_reason,
        )
        observed_metrics = [
            ConditionMetricValue.model_validate(item) for item in payload["metrics"]
        ]
        if len(observed_metrics) != 3 or any(
            not _same_metric(observed, expected)
            for observed, expected in zip(observed_metrics, recomputed.results, strict=True)
        ):
            raise ValueError(f"evaluation condition metric mismatch: {condition_id}")
        metric_rows.append(recomputed)
        loaded[condition_id] = EvaluationConditionArrays(
            condition_id=condition_id,
            prediction=prediction,
            input_control=input_control,
            input_control_row_ids=input_row_ids,
            truth=truth,
            truth_row_ids=truth_row_ids,
            metric_control_pool_mean=metric_control,
            de_gene_indices=de,
            top_de_gene_indices=top_de,
            de_unavailable_reason=de_unavailable_reason,
            metrics=recomputed,
        )
    if _availability(metric_rows) != manifest.metrics:
        raise ValueError("evaluation summary metric mismatch")
    return EvaluationBundle(
        manifest=manifest,
        prediction_manifest=prediction_manifest,
        gene_ids=gene_ids,
        systema_reference=reference,
        conditions=loaded,
        file_sha256=sha256_file(artifact_path),
    )
