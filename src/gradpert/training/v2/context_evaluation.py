"""Fixed-expression-axis context-size evaluation using canonical frozen controls."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.hashing import sha256_json
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.validation import mean_expression_mse

from .evaluation import context_queries, predict_query_set
from .views import NeighborhoodIndex


def evaluate_contexts(
    model: GraDPertV2,
    index: NeighborhoodIndex,
    data: CanonicalEvaluationData,
    *,
    evaluation_ids: tuple[int, ...],
    budgets: tuple[int, ...],
    context_seed: int,
    expected_split: Literal["val", "test"],
    device: torch.device,
    cell_batch: int,
) -> dict[str, Any]:
    """Score the same genes and rows at every budget; never serialize predictions.

    This targeted MSE is a separate G1 metric, not the full-axis benchmark or
    any of its three named Pearson metrics. Contexts are sampled from gene IDs
    alone, before control/truth expression is loaded.
    """
    if data.split_name != expected_split or data.control_manifest.split_name != expected_split:
        raise ValueError("context evaluation split mismatch")
    genes = tuple(data.expression_gene_ids)
    if index.gene_ids[: len(genes)] != genes:
        raise ValueError("context evaluation expression axis mismatch")
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("context budgets must be unique increasing values")
    contexts = {
        budget: context_queries(
            evaluation_ids, gene_count=len(genes), budget=budget, seed=context_seed
        )
        for budget in budgets
    }
    positions = {gene: i for i, gene in enumerate(index.gene_ids)}
    rows: list[dict[str, Any]] = []
    for draw in data.control_manifest.draws:
        controls = data.load_control_rows(tuple(draw.ordered_row_ids))
        if tuple(controls.ordered_row_ids) != tuple(draw.ordered_row_ids):
            raise ValueError("context evaluation control order changed")
        targets = tuple(
            positions[gene]
            for gene in draw.condition_id.split("+")
            if gene != data.split.control_condition_id
        )
        # Truth is used only after prediction, on the fixed evaluation columns.
        scores = []
        for budget, queries in contexts.items():
            prediction = predict_query_set(
                model,
                index,
                controls.expression,
                targets,
                queries,
                device=device,
                cell_batch=cell_batch,
            )
            selected = prediction[:, np.searchsorted(queries, evaluation_ids)]
            truth = data.load_truth_rows(draw.condition_id)
            scores.append(
                {
                    "context_budget": budget,
                    "fixed_axis_prediction_mse": mean_expression_mse(
                        selected, truth.expression[:, evaluation_ids]
                    ),
                }
            )
        rows.append(
            {
                "condition_id": draw.condition_id,
                "control_row_ids_sha256": sha256_json(list(controls.ordered_row_ids)),
                "truth_row_ids_sha256": sha256_json(list(truth.ordered_row_ids)),
                "scores": scores,
            }
        )
    if not rows:
        raise ValueError("context evaluation split is empty")
    return {
        "schema_version": "gradpert-v2-context-evaluation-1",
        "split": expected_split,
        "control_manifest_sha256": data.control_manifest_file_sha256,
        "evaluation_gene_ids": [genes[i] for i in evaluation_ids],
        "expression_gene_order_sha256": sha256_json(list(genes)),
        "context_seed": context_seed,
        "contexts": [
            {"budget": budget, "gene_ids": [genes[i] for i in queries]}
            for budget, queries in contexts.items()
        ],
        "conditions": rows,
        "macro": [
            {
                "context_budget": budget,
                "fixed_axis_prediction_mse": float(
                    np.mean([row["scores"][i]["fixed_axis_prediction_mse"] for row in rows])
                ),
            }
            for i, budget in enumerate(budgets)
        ],
    }
