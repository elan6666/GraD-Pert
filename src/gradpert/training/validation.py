"""Validation-only model selection under the frozen 300-control protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gradpert.evaluation.metrics import (
    compute_condition_metrics,
    macro_summarize,
    pearson_correlation,
)
from gradpert.evaluation.state import LoadedEvaluationState
from gradpert.graphs import GraphTopology, build_prediction_graph_view

if TYPE_CHECKING:
    import torch

    from gradpert.evaluation import CanonicalEvaluationData
    from gradpert.graphs import GraphView
    from gradpert.modeling import GraDPertJointModel


@dataclass(frozen=True)
class ValidationMetricResult:
    txpert_macro_pearson_delta: float
    finite_condition_count: int
    total_condition_count: int
    unavailable_condition_count: int
    prediction_loss: float = 0.0
    trishift_pearson_delta: float | None = None
    systema_pearson: float | None = None
    metric_details: tuple[dict[str, Any], ...] = ()
    reference_sha256: str | None = None


def mean_expression_mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """All-gene MSE of condition means, without pairing unpaired cells."""
    if prediction.ndim != 2 or truth.ndim != 2 or prediction.shape[1] != truth.shape[1]:
        raise ValueError("prediction/truth must share a two-dimensional gene axis")
    if not prediction.size or not truth.size:
        raise ValueError("empty validation expression")
    difference = prediction.mean(axis=0, dtype=np.float64) - truth.mean(axis=0, dtype=np.float64)
    loss = float(np.mean(np.square(difference)))
    if not np.isfinite(loss):
        raise RuntimeError("validation prediction loss is not finite")
    return loss


def evaluate_validation_macro_delta(
    *,
    model: GraDPertJointModel,
    topology: GraphTopology,
    data: CanonicalEvaluationData,
    anchors_by_condition: dict[str, tuple[int, ...]],
    device: torch.device,
    decode_batch_size: int,
    prediction_view: GraphView | None = None,
    evaluation_state: LoadedEvaluationState | None = None,
) -> ValidationMetricResult:
    """Stream validation conditions and never accept a test evaluator."""

    from gradpert.training.inference import iter_frozen_control_predictions

    if data.split_name != "val" or data.control_manifest.split_name != "val":
        raise ValueError("checkpoint selection accepts validation data only")
    expected = tuple(draw.condition_id for draw in data.control_manifest.draws)
    if set(anchors_by_condition) != set(expected):
        raise ValueError("validation anchors differ from the frozen condition set")
    values: list[float] = []
    losses: list[float] = []
    condition_metrics = []
    if evaluation_state is not None:
        if tuple(evaluation_state.manifest.condition_ids) != expected:
            raise ValueError(
                "validation metric state must contain only ordered validation conditions"
            )
        if evaluation_state.manifest.systema_reference_condition_ids != list(
            data.split.train_conditions
        ):
            raise ValueError("validation reference must be training only")
    observed: list[str] = []
    for prediction in iter_frozen_control_predictions(
        model=model,
        prediction_view=prediction_view or build_prediction_graph_view(topology),
        control_manifest=data.control_manifest,
        anchors_by_condition=anchors_by_condition,
        load_control_rows=data.load_control_rows,
        device=device,
        decode_batch_size=decode_batch_size,
    ):
        truth = data.load_truth_rows(prediction.condition_id)
        if evaluation_state is not None:
            state = evaluation_state
            condition = prediction.condition_id
            condition_metrics.append(
                compute_condition_metrics(
                    condition_id=condition,
                    prediction=prediction.prediction,
                    input_control=prediction.input_control,
                    truth=truth.expression,
                    metric_control_pool_mean=state.metric_control_means[len(observed)],
                    de_gene_indices=state.manifest.de_gene_indices[condition],
                    top_de_gene_indices=state.manifest.top_de_gene_indices[condition],
                    systema_reference=state.systema_reference,
                    de_unavailable_reason=state.manifest.de_unavailable_reasons.get(condition),
                )
            )
        losses.append(mean_expression_mse(prediction.prediction, truth.expression))
        control_mean = prediction.input_control.mean(axis=0)
        value, _ = pearson_correlation(
            prediction.prediction.mean(axis=0) - control_mean,
            truth.expression.mean(axis=0) - control_mean,
        )
        observed.append(prediction.condition_id)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if tuple(observed) != expected:
        raise RuntimeError("validation prediction order/count differs from its manifest")
    if not values:
        raise RuntimeError("validation macro Pearson delta has no finite conditions")
    summaries = macro_summarize(condition_metrics) if condition_metrics else ()
    by_metric = {s.metric_id: s.macro_mean for s in summaries}
    return ValidationMetricResult(
        txpert_macro_pearson_delta=float(np.mean(values)),
        finite_condition_count=len(values),
        total_condition_count=len(expected),
        unavailable_condition_count=len(expected) - len(values),
        prediction_loss=float(np.mean(losses)),
        trishift_pearson_delta=by_metric.get("trishift_pearson_delta"),
        systema_pearson=by_metric.get("systema_pearson"),
        metric_details=tuple(asdict(s) for s in summaries),
        reference_sha256=evaluation_state.manifest_file_sha256 if evaluation_state else None,
    )
