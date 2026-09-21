"""Ordered v2 inference through the existing frozen evaluation protocol."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import torch

from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.evaluation.metrics import compute_condition_metrics, macro_summarize
from gradpert.evaluation.state import LoadedEvaluationState
from gradpert.hashing import sha256_json
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.validation import mean_expression_mse

from .views import NeighborhoodIndex


def context_queries(
    evaluation_ids: tuple[int, ...], *, gene_count: int, budget: int, seed: int
) -> np.ndarray[Any, Any]:
    """Nested deterministic contexts containing one frozen evaluation gene set."""
    if (
        not evaluation_ids
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or any(type(g) is not int or g < 0 or g >= gene_count for g in evaluation_ids)
        or not len(evaluation_ids) <= budget <= gene_count
    ):
        raise ValueError("evaluation IDs must be unique, in-axis and fit the context budget")
    remaining = np.array(sorted(set(range(gene_count)) - set(evaluation_ids)), dtype=np.int64)
    np.random.default_rng(seed).shuffle(remaining)
    return np.sort(
        np.concatenate((np.asarray(evaluation_ids), remaining[: budget - len(evaluation_ids)]))
    )


@torch.no_grad()
def predict_query_set(
    model: GraDPertV2,
    index: NeighborhoodIndex,
    controls: np.ndarray[Any, Any],
    targets: tuple[int, ...],
    queries: np.ndarray[Any, Any],
    *,
    device: torch.device,
    cell_batch: int,
    block_response_cls_to_gene: bool = False,
) -> np.ndarray[Any, Any]:
    """Predict explicitly ordered expression columns in one shared context.

    Outputs follow queries exactly. Callers compare the same evaluation IDs by
    selecting their positions in each context, never by comparing different axes.
    """
    if (
        controls.ndim != 2
        or not controls.size
        or not np.isfinite(controls).all()
        or controls.shape[1] > index.n_nodes
        or cell_batch < 1
    ):
        raise ValueError("finite aligned control matrix and positive cell batch required")
    if (
        queries.ndim != 1
        or not len(queries)
        or queries.dtype.kind not in "iu"
        or (queries < 0).any()
        or (queries >= controls.shape[1]).any()
        or (np.diff(queries.astype(np.int64)) <= 0).any()
    ):
        raise ValueError("context query IDs must be unique increasing expression-axis indices")
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(t < 0 or t >= index.n_nodes for t in targets)
    ):
        raise ValueError("condition requires distinct in-graph target genes")
    model.eval()
    ids = np.union1d(queries, targets)
    view = index.view(ids, [targets], rng=np.random.default_rng(0), device=device, induced=False)
    gene = model.graph(view.ids, view.neighbors, view.valid, view.sources)
    condition = model.aggregate_targets(gene, view.target_positions, view.target_valid)
    positions = torch.tensor(np.searchsorted(ids, queries), device=device)
    result = np.empty((len(controls), len(queries)), dtype=np.float32)
    for row in range(0, len(controls), cell_batch):
        chunk = torch.from_numpy(
            np.ascontiguousarray(controls[row : row + cell_batch, queries])
        ).to(device)
        output = model.encode_response(
            gene[positions],
            chunk,
            condition.expand(len(chunk), -1),
            block_response_cls_to_gene=block_response_cls_to_gene,
        )
        result[row : row + len(chunk)] = output["prediction"].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite v2 query prediction")
    return result


@torch.no_grad()
def predict_controls(
    model: GraDPertV2,
    index: NeighborhoodIndex,
    controls: np.ndarray[Any, Any],
    targets: tuple[int, ...],
    *,
    device: torch.device,
    cell_batch: int,
    query_count: int,
    block_response_cls_to_gene: bool = False,
) -> np.ndarray[Any, Any]:
    """Partition the complete expression axis into fixed ordered query blocks.

    query_count>=axis length means a single full-context prediction. Smaller
    contexts are an explicit inference recipe, never an implicit missing-gene
    fill. Each output gene is predicted exactly once; targets remain graph queries.
    """
    if controls.ndim != 2 or not controls.size or cell_batch < 1 or query_count < 1:
        raise ValueError("nonempty aligned controls and positive inference budgets required")
    if controls.shape[1] > index.n_nodes or not np.isfinite(controls).all():
        raise ValueError("controls must be finite and contained in the graph axis")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("condition requires distinct target genes")
    if any(t < 0 or t >= index.n_nodes for t in targets):
        raise ValueError("target gene is outside graph axis")
    model.eval()
    prediction = np.empty_like(controls, dtype=np.float32)
    for start in range(0, controls.shape[1], query_count):
        stop = min(start + query_count, controls.shape[1])
        queries = np.arange(start, stop)
        prediction[:, start:stop] = predict_query_set(
            model,
            index,
            controls,
            targets,
            queries,
            device=device,
            cell_batch=cell_batch,
            block_response_cls_to_gene=block_response_cls_to_gene,
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite v2 prediction")
    return prediction


def evaluate(
    model: GraDPertV2,
    index: NeighborhoodIndex,
    data: CanonicalEvaluationData,
    reference: LoadedEvaluationState,
    *,
    expected_split: Literal["val", "test"],
    device: torch.device,
    cell_batch: int,
    query_count: int,
    block_response_cls_to_gene: bool = False,
) -> dict[str, Any]:
    if data.split_name != expected_split or data.control_manifest.split_name != expected_split:
        raise ValueError("evaluation split differs from requested lifecycle stage")
    gene_index = {g: i for i, g in enumerate(index.gene_ids)}
    if index.gene_ids[: len(data.expression_gene_ids)] != tuple(data.expression_gene_ids):
        raise ValueError("prediction expression axis differs from evaluator")
    if len(gene_index) != index.n_nodes:
        raise ValueError("evaluation graph axis size mismatch")
    reference_positions = {c: i for i, c in enumerate(reference.manifest.condition_ids)}
    if expected_split == "val":
        if list(reference.manifest.condition_ids) != list(data.split.val_conditions):
            raise ValueError("validation cannot use a combined test reference")
        if reference.manifest.systema_reference_condition_ids != list(data.split.train_conditions):
            raise ValueError("validation reference must be train-only")
    losses, metrics, rows = [], [], []
    for draw in data.control_manifest.draws:
        condition = draw.condition_id
        ids = tuple(draw.ordered_row_ids)
        controls = data.load_control_rows(ids)
        if controls.ordered_row_ids != ids:
            raise ValueError("inference control order changed")
        targets = tuple(
            gene_index[g] for g in condition.split("+") if g != data.split.control_condition_id
        )
        prediction = predict_controls(
            model,
            index,
            controls.expression,
            targets,
            device=device,
            cell_batch=cell_batch,
            query_count=query_count,
            block_response_cls_to_gene=block_response_cls_to_gene,
        )
        truth = data.load_truth_rows(condition)
        metric = compute_condition_metrics(
            condition_id=condition,
            prediction=prediction,
            input_control=controls.expression,
            truth=truth.expression,
            metric_control_pool_mean=reference.metric_control_means[reference_positions[condition]],
            de_gene_indices=reference.manifest.de_gene_indices[condition],
            top_de_gene_indices=reference.manifest.top_de_gene_indices[condition],
            systema_reference=reference.systema_reference,
            de_unavailable_reason=reference.manifest.de_unavailable_reasons.get(condition),
        )
        loss = mean_expression_mse(prediction, truth.expression)
        metrics.append(metric)
        losses.append(loss)
        rows.append(
            {
                "condition_id": condition,
                "loss": loss,
                "metrics": asdict(metric),
                "control_row_ids": list(ids),
                "control_row_ids_sha256": sha256_json(list(ids)),
                "truth_row_ids": list(truth.ordered_row_ids),
                "truth_row_ids_sha256": sha256_json(list(truth.ordered_row_ids)),
            }
        )
    if not losses:
        raise ValueError("evaluation split is empty")
    return {
        "split": expected_split,
        "prediction_loss": float(np.mean(losses)),
        "metrics": [asdict(m) for m in macro_summarize(metrics)],
        "conditions": rows,
        "control_manifest_sha256": data.control_manifest_file_sha256,
        "reference_sha256": reference.manifest_file_sha256,
        "query_recipe": {
            "name": "ordered_disjoint_blocks",
            "query_count": query_count,
            "expression_gene_count": len(data.expression_gene_ids),
            "block_response_cls_to_gene": block_response_cls_to_gene,
        },
    }
