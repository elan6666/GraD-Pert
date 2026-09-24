"""Separate basal/response CLS sensitivity and response-token intervention diagnostics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.modeling.v2 import GraDPertV2

from .views import NeighborhoodIndex


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


@torch.no_grad()
def evaluate_response_diagnostics(
    model: GraDPertV2,
    index: NeighborhoodIndex,
    data: CanonicalEvaluationData,
    *,
    query_gene_ids: tuple[str, ...],
    condition_id: str,
    alternative_condition_id: str,
    expected_split: str,
    device: torch.device,
    cell_batch: int,
) -> dict[str, Any]:
    """Bind the three interventions to frozen canonical control/truth populations."""
    import numpy as np

    from gradpert.hashing import sha256_json

    if data.split_name != expected_split or data.control_manifest.split_name != expected_split:
        raise ValueError("diagnostic split mismatch")
    genes = tuple(data.expression_gene_ids)
    if (
        index.gene_ids[: len(genes)] != genes
        or not query_gene_ids
        or len(set(query_gene_ids)) != len(query_gene_ids)
    ):
        raise ValueError("diagnostic query gene identity mismatch")
    queries = np.array([genes.index(g) for g in query_gene_ids], dtype=np.int64)
    draws = {draw.condition_id: draw for draw in data.control_manifest.draws}
    if condition_id == alternative_condition_id or any(
        c not in draws for c in (condition_id, alternative_condition_id)
    ):
        raise ValueError(
            "diagnostics require two distinct conditions in the requested frozen split"
        )
    loaded = [
        data.load_control_rows(tuple(draws[c].ordered_row_ids))
        for c in (condition_id, alternative_condition_id)
    ]
    for condition, loaded_controls in zip(
        (condition_id, alternative_condition_id), loaded, strict=True
    ):
        if tuple(loaded_controls.ordered_row_ids) != tuple(draws[condition].ordered_row_ids):
            raise ValueError("diagnostic ordered controls changed")
    gene_positions = {g: i for i, g in enumerate(index.gene_ids)}
    targets = [
        tuple(gene_positions[g] for g in c.split("+") if g != data.split.control_condition_id)
        for c in (condition_id, alternative_condition_id)
    ]
    ids = np.unique(np.concatenate((queries, np.asarray(targets[0]), np.asarray(targets[1]))))
    view = index.view(ids, targets, rng=np.random.default_rng(0), device=device, induced=False)
    training = model.training
    model.eval()
    try:
        graph = model.graph(
            view.ids, view.neighbors, view.valid, view.sources, context=view.context
        )
        conditions = model.aggregate_targets(graph, view.target_positions, view.target_valid)
        gene = graph[torch.tensor(np.searchsorted(ids, queries), device=device)]
        controls = [
            torch.from_numpy(np.ascontiguousarray(rows.expression[:, queries])).to(device)
            for rows in loaded
        ]
        truth = data.load_truth_rows(condition_id)
        result = response_diagnostics(
            model,
            gene,
            controls[0],
            controls[1],
            conditions[0].expand(len(controls[0]), -1),
            conditions[1].expand(len(controls[1]), -1),
            truth=torch.from_numpy(np.ascontiguousarray(truth.expression[:, queries])).to(device),
            cell_batch=cell_batch,
        )
        result.update(
            {
                "split": expected_split,
                "condition_id": condition_id,
                "alternative_condition_id": alternative_condition_id,
                "query_gene_ids": list(query_gene_ids),
                "control_manifest_sha256": data.control_manifest_file_sha256,
                "control_populations": [
                    {
                        "ordered_row_ids": list(rows.ordered_row_ids),
                        "sha256": sha256_json(list(rows.ordered_row_ids)),
                    }
                    for rows in loaded
                ],
                "truth_row_ids": list(truth.ordered_row_ids),
                "truth_row_ids_sha256": sha256_json(list(truth.ordered_row_ids)),
            }
        )
        return result
    finally:
        model.train(training)
