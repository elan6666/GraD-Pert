"""Separate basal/response CLS sensitivity and response-token intervention diagnostics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gradpert.modeling.v2 import GraDPertV2


@torch.no_grad()
def response_diagnostics(
    model: GraDPertV2,
    gene: Tensor,
    control: Tensor,
    alternative_control: Tensor,
    condition: Tensor,
    alternative_condition: Tensor,
    *,
    truth: Tensor | None = None,
    cell_batch: int = 2,
) -> dict[str, Any]:
    """Compare fixed query embeddings; callers supply frozen population identities.

    Changes are RMS distances between population-mean features, not paired-cell
    effects. Absolute prediction and predicted delta are reported separately so
    the raw-control residual cannot masquerade as learned perturbation sensitivity.
    Only baseline versus CLS-blocked predictions share the supplied truth label.
    """
    if control.shape != alternative_control.shape or condition.shape != alternative_condition.shape:
        raise ValueError("diagnostic populations must have matching shapes")
    if condition.shape != (len(control), model.options.width) or not control.numel():
        raise ValueError("diagnostic condition embeddings must align with control rows")
    if any(
        not torch.isfinite(x).all()
        for x in (gene, control, alternative_control, condition, alternative_condition)
    ):
        raise ValueError("diagnostic inputs must be finite")
    if truth is not None and (
        truth.ndim != 2
        or truth.shape[1] != control.shape[1]
        or not len(truth)
        or not torch.isfinite(truth).all()
    ):
        raise ValueError("diagnostic truth must be finite on the query gene axis")
    if cell_batch < 1:
        raise ValueError("diagnostic cell batch must be positive")
    training = model.training
    model.eval()
    try:

        def population_mean(
            cells: Tensor, perturbation: Tensor, *, blocked: bool = False
        ) -> dict[str, Tensor]:
            totals: dict[str, Tensor] = {}
            for start in range(0, len(cells), cell_batch):
                output = model.encode_response(
                    gene,
                    cells[start : start + cell_batch],
                    perturbation[start : start + cell_batch],
                    block_response_cls_to_gene=blocked,
                )
                for key in ("prediction", "delta", "control_cls", "response_cls"):
                    value = output[key].double().sum(0, keepdim=True)
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("nonfinite diagnostic model output")
                    totals[key] = value if key not in totals else totals[key] + value
            return {key: value / len(cells) for key, value in totals.items()}

        baseline = population_mean(control, condition)
        variants = {
            "change_control": population_mean(alternative_control, condition),
            "change_perturbation": population_mean(control, alternative_condition),
            "block_response_cls_to_gene": population_mean(control, condition, blocked=True),
        }

        def distance(left: Tensor, right: Tensor) -> float:
            return float((left.double().mean(0) - right.double().mean(0)).square().mean().sqrt())

        changes = {
            name: {
                key: distance(output[key], baseline[key])
                for key in ("prediction", "delta", "control_cls", "response_cls")
            }
            for name, output in variants.items()
        }
        result: dict[str, Any] = {
            "schema_version": "gradpert-v2-response-diagnostics-1",
            "distance": "rms_difference_of_population_means",
            "changes": changes,
        }
        if truth is not None:
            result["fixed_truth_prediction_mse"] = {
                "baseline": distance(baseline["prediction"], truth) ** 2,
                "block_response_cls_to_gene": distance(
                    variants["block_response_cls_to_gene"]["prediction"], truth
                )
                ** 2,
            }
        return result
    finally:
        model.train(training)
