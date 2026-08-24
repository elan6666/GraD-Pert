"""Truth-free native inference over an exact frozen 300-control manifest."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from gradpert.artifacts import PredictionConditionArrays
from gradpert.contracts import EvaluationControlManifest
from gradpert.graphs import GraphView
from gradpert.modeling import GraDPertJointModel


@dataclass(frozen=True)
class LoadedControlRows:
    ordered_row_ids: tuple[str, ...]
    expression: NDArray[np.floating]


ControlRowLoader = Callable[[tuple[str, ...]], LoadedControlRows]


def iter_frozen_control_predictions(
    *,
    model: GraDPertJointModel,
    prediction_view: GraphView,
    control_manifest: EvaluationControlManifest,
    anchors_by_condition: Mapping[str, tuple[int, ...]],
    load_control_rows: ControlRowLoader,
    device: torch.device,
    decode_batch_size: int,
) -> Iterator[PredictionConditionArrays]:
    """Yield each manifest prediction in order without retaining every condition."""

    if decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive")
    model_parameter = next(model.parameters())
    if model_parameter.device != device:
        raise ValueError("model parameters and requested inference device differ")
    manifest_conditions = tuple(draw.condition_id for draw in control_manifest.draws)
    if set(anchors_by_condition) != set(manifest_conditions):
        raise ValueError("active-anchor mapping and control-manifest condition sets differ")
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ordered_conditions, condition_states = model.prediction_condition_states(
                prediction_view,
                anchors_by_condition,
            )
            state_by_condition = {
                condition_id: condition_states[index]
                for index, condition_id in enumerate(ordered_conditions)
            }
            for draw in control_manifest.draws:
                expected_row_ids = tuple(draw.ordered_row_ids)
                loaded = load_control_rows(expected_row_ids)
                if loaded.ordered_row_ids != expected_row_ids:
                    raise ValueError(
                        f"control loader changed row order for condition {draw.condition_id}"
                    )
                controls = np.asarray(loaded.expression)
                if controls.shape != (300, model.expression_gene_count):
                    raise ValueError("loaded controls must be [300, expression_gene_count]")
                if not np.issubdtype(controls.dtype, np.number) or not np.isfinite(controls).all():
                    raise ValueError("loaded controls must contain finite numeric values")
                predicted_chunks = []
                condition_state = state_by_condition[draw.condition_id]
                for start in range(0, 300, decode_batch_size):
                    control_chunk = torch.as_tensor(
                        controls[start : start + decode_batch_size],
                        device=device,
                        dtype=model_parameter.dtype,
                    )
                    perturbation = condition_state.unsqueeze(0).expand(control_chunk.shape[0], -1)
                    predicted_chunks.append(
                        model.decode_expression(control_chunk, perturbation).cpu().numpy()
                    )
                prediction = np.concatenate(predicted_chunks, axis=0)
                yield PredictionConditionArrays(
                    condition_id=draw.condition_id,
                    prediction=prediction,
                    input_control=np.ascontiguousarray(controls),
                    input_control_row_ids=expected_row_ids,
                )
    finally:
        model.train(previous_training)


def predict_frozen_controls(
    *,
    model: GraDPertJointModel,
    prediction_view: GraphView,
    control_manifest: EvaluationControlManifest,
    anchors_by_condition: Mapping[str, tuple[int, ...]],
    load_control_rows: ControlRowLoader,
    device: torch.device,
    decode_batch_size: int,
) -> tuple[PredictionConditionArrays, ...]:
    """Materialize all yielded conditions for a final server-side artifact."""

    return tuple(
        iter_frozen_control_predictions(
            model=model,
            prediction_view=prediction_view,
            control_manifest=control_manifest,
            anchors_by_condition=anchors_by_condition,
            load_control_rows=load_control_rows,
            device=device,
            decode_batch_size=decode_batch_size,
        )
    )
