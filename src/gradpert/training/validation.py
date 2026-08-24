"""Validation-only model selection under the frozen 300-control protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gradpert.evaluation.metrics import pearson_correlation
from gradpert.graphs import GraphTopology, build_prediction_graph_view

if TYPE_CHECKING:
    import torch

    from gradpert.evaluation import CanonicalEvaluationData
    from gradpert.modeling import GraDPertJointModel


@dataclass(frozen=True)
class ValidationMetricResult:
    txpert_macro_pearson_delta: float
    finite_condition_count: int
    total_condition_count: int
    unavailable_condition_count: int


def evaluate_validation_macro_delta(
    *,
    model: GraDPertJointModel,
    topology: GraphTopology,
    data: CanonicalEvaluationData,
    anchors_by_condition: dict[str, tuple[int, ...]],
    device: torch.device,
    decode_batch_size: int,
) -> ValidationMetricResult:
    """Stream validation conditions and never accept a test evaluator."""

    from gradpert.training.inference import iter_frozen_control_predictions

    if data.split_name != "val" or data.control_manifest.split_name != "val":
        raise ValueError("checkpoint selection accepts validation data only")
    expected = tuple(draw.condition_id for draw in data.control_manifest.draws)
    if set(anchors_by_condition) != set(expected):
        raise ValueError("validation anchors differ from the frozen condition set")
    values: list[float] = []
    observed: list[str] = []
    for prediction in iter_frozen_control_predictions(
        model=model,
        prediction_view=build_prediction_graph_view(topology),
        control_manifest=data.control_manifest,
        anchors_by_condition=anchors_by_condition,
        load_control_rows=data.load_control_rows,
        device=device,
        decode_batch_size=decode_batch_size,
    ):
        truth = data.load_truth_rows(prediction.condition_id)
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
    return ValidationMetricResult(
        txpert_macro_pearson_delta=float(np.mean(values)),
        finite_condition_count=len(values),
        total_condition_count=len(expected),
        unavailable_condition_count=len(expected) - len(values),
    )
