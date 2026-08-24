"""Distinct TxPert-delta, TriShift-delta, and Systema Pearson metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

MetricId = Literal[
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
]


@dataclass(frozen=True)
class ConditionMetricResult:
    metric_id: MetricId
    value: float | None
    reason: str | None
    gene_count: int


@dataclass(frozen=True)
class ConditionMetrics:
    condition_id: str
    results: tuple[ConditionMetricResult, ConditionMetricResult, ConditionMetricResult]


@dataclass(frozen=True)
class MetricSummary:
    metric_id: MetricId
    macro_mean: float | None
    finite_condition_count: int
    total_condition_count: int
    unavailable_reasons: tuple[str, ...]


def _vector(value: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    return array


def _population(value: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-two population")
    return array


def pearson_correlation(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> tuple[float | None, str | None]:
    x = _vector(left, "left")
    y = _vector(right, "right")
    if x.shape != y.shape:
        raise ValueError("Pearson vectors have different shapes")
    if x.size < 2:
        return None, "fewer_than_two_genes"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "non_finite_input"
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator == 0:
        return None, "constant_vector"
    return float(np.dot(x_centered, y_centered) / denominator), None


def _indices(indices: Sequence[int], gene_count: int, name: str) -> np.ndarray[Any, Any]:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must contain at least one gene index")
    if len(values) != len(set(int(index) for index in values)):
        raise ValueError(f"{name} contains duplicate indices")
    if int(values.min()) < 0 or int(values.max()) >= gene_count:
        raise ValueError(f"{name} contains an out-of-range index")
    return values


def _result(
    metric_id: MetricId,
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> ConditionMetricResult:
    value, reason = pearson_correlation(left, right)
    return ConditionMetricResult(
        metric_id=metric_id,
        value=value,
        reason=reason,
        gene_count=int(np.asarray(left).size),
    )


def compute_condition_metrics(
    *,
    condition_id: str,
    prediction: np.ndarray[Any, Any],
    input_control: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    metric_control_pool_mean: np.ndarray[Any, Any],
    de_gene_indices: Sequence[int],
    top_de_gene_indices: Sequence[int],
    systema_reference: np.ndarray[Any, Any],
    de_unavailable_reason: str | None = None,
) -> ConditionMetrics:
    if not condition_id:
        raise ValueError("condition_id must be non-empty")
    pred = _population(prediction, "prediction")
    control = _population(input_control, "input_control")
    observed = _population(truth, "truth")
    if pred.shape[0] != 300 or control.shape[0] != 300:
        raise ValueError("prediction and input_control must each preserve 300 rows")
    if pred.shape != control.shape or pred.shape[1] != observed.shape[1]:
        raise ValueError("prediction/control/truth gene dimensions differ")
    gene_count = pred.shape[1]
    metric_control = _vector(metric_control_pool_mean, "metric_control_pool_mean")
    reference = _vector(systema_reference, "systema_reference")
    if metric_control.shape != (gene_count,) or reference.shape != (gene_count,):
        raise ValueError("metric reference gene dimension differs")
    if de_unavailable_reason is None:
        de = _indices(de_gene_indices, gene_count, "de_gene_indices")
        top_de = _indices(top_de_gene_indices, gene_count, "top_de_gene_indices")
    else:
        if de_gene_indices or top_de_gene_indices:
            raise ValueError("unavailable DE reason requires empty DE index sets")
        de = np.asarray([], dtype=np.int64)
        top_de = np.asarray([], dtype=np.int64)

    pred_mean = pred.mean(axis=0)
    truth_mean = observed.mean(axis=0)
    input_control_mean = control.mean(axis=0)
    txpert = _result(
        "txpert_macro_pearson_delta",
        pred_mean - input_control_mean,
        truth_mean - input_control_mean,
    )
    if de_unavailable_reason is None:
        trishift = _result(
            "trishift_pearson_delta",
            (pred_mean - metric_control)[de],
            (truth_mean - metric_control)[de],
        )
        systema = _result(
            "systema_pearson",
            (pred_mean - reference)[top_de],
            (truth_mean - reference)[top_de],
        )
    else:
        reason = f"de_unavailable:{de_unavailable_reason}"
        trishift = ConditionMetricResult(
            metric_id="trishift_pearson_delta",
            value=None,
            reason=reason,
            gene_count=0,
        )
        systema = ConditionMetricResult(
            metric_id="systema_pearson",
            value=None,
            reason=reason,
            gene_count=0,
        )
    return ConditionMetrics(condition_id=condition_id, results=(txpert, trishift, systema))


def build_systema_reference(
    train_validation_noncontrol_populations: Mapping[str, np.ndarray[Any, Any]],
) -> np.ndarray[Any, Any]:
    """Equal-weight condition centroids; never cell-count-weight conditions."""

    if not train_validation_noncontrol_populations:
        raise ValueError("Systema reference requires train/validation non-control conditions")
    centroids = []
    gene_count: int | None = None
    for condition_id in sorted(train_validation_noncontrol_populations):
        if condition_id == "ctrl":
            raise ValueError("Systema reference input must exclude control")
        population = _population(
            train_validation_noncontrol_populations[condition_id],
            f"population[{condition_id}]",
        )
        if gene_count is None:
            gene_count = population.shape[1]
        elif population.shape[1] != gene_count:
            raise ValueError("Systema reference populations have different gene dimensions")
        centroids.append(population.mean(axis=0))
    return cast(np.ndarray[Any, Any], np.stack(centroids, axis=0).mean(axis=0))


def macro_summarize(condition_metrics: Sequence[ConditionMetrics]) -> tuple[MetricSummary, ...]:
    if not condition_metrics:
        raise ValueError("macro summary requires condition metrics")
    expected: tuple[MetricId, ...] = (
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    )
    summaries = []
    for result_index, metric_id in enumerate(expected):
        results = [condition.results[result_index] for condition in condition_metrics]
        if any(result.metric_id != metric_id for result in results):
            raise ValueError("condition metric registry/order mismatch")
        finite = [result.value for result in results if result.value is not None]
        reasons = tuple(sorted(result.reason for result in results if result.reason is not None))
        summaries.append(
            MetricSummary(
                metric_id=metric_id,
                macro_mean=float(np.mean(finite)) if finite else None,
                finite_condition_count=len(finite),
                total_condition_count=len(results),
                unavailable_reasons=reasons,
            )
        )
    return tuple(summaries)
