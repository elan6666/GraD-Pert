"""Pearson metrics on predeclared expression-gene partitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import (
    ConditionMetricResult,
    ConditionMetrics,
    MetricId,
    pearson_correlation,
)


def _score(metric_id: MetricId, prediction: np.ndarray, truth: np.ndarray) -> ConditionMetricResult:
    if len(prediction) < 2:
        return ConditionMetricResult(metric_id, None, "fewer_than_two_genes", len(prediction))
    value, reason = pearson_correlation(prediction, truth)
    return ConditionMetricResult(metric_id, value, reason, len(prediction))


def grouped_condition_metrics(
    *,
    condition_id: str,
    prediction: np.ndarray[Any, Any],
    input_control: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    metric_control_pool_mean: np.ndarray[Any, Any],
    de_gene_indices: list[int],
    top_de_gene_indices: list[int],
    systema_reference: np.ndarray[Any, Any],
    gene_indices: tuple[int, ...],
    de_unavailable_reason: str | None = None,
) -> ConditionMetrics:
    """Keep the original reference vectors and intersect each metric's gene axis.

    TxPert uses the complete selected group. TriShift and Systema use only
    genes in both the frozen per-condition DE set and this group.
    """
    n = prediction.shape[1]
    if (
        prediction.shape != input_control.shape
        or truth.ndim != 2
        or truth.shape[1] != n
        or prediction.shape[0] != 300
        or len(gene_indices) != len(set(gene_indices))
        or any(i < 0 or i >= n for i in gene_indices)
        or len(metric_control_pool_mean) != n
        or len(systema_reference) != n
    ):
        raise ValueError("grouped metric inputs are not aligned with the frozen evaluation axis")
    chosen = np.asarray(gene_indices, dtype=np.int64)
    pred_mean = prediction.astype(np.float64).mean(axis=0)
    truth_mean = truth.astype(np.float64).mean(axis=0)
    input_mean = input_control.astype(np.float64).mean(axis=0)
    txpert = _score(
        "txpert_macro_pearson_delta",
        (pred_mean - input_mean)[chosen],
        (truth_mean - input_mean)[chosen],
    )
    if de_unavailable_reason is not None:
        reason = f"de_unavailable:{de_unavailable_reason}"
        tri = ConditionMetricResult("trishift_pearson_delta", None, reason, 0)
        sys = ConditionMetricResult("systema_pearson", None, reason, 0)
    else:
        selected = set(gene_indices)
        de = np.asarray([i for i in de_gene_indices if i in selected], dtype=np.int64)
        top = np.asarray([i for i in top_de_gene_indices if i in selected], dtype=np.int64)
        tri = _score(
            "trishift_pearson_delta",
            (pred_mean - metric_control_pool_mean)[de],
            (truth_mean - metric_control_pool_mean)[de],
        )
        sys = _score(
            "systema_pearson",
            (pred_mean - systema_reference)[top],
            (truth_mean - systema_reference)[top],
        )
    return ConditionMetrics(condition_id, (txpert, tri, sys))
