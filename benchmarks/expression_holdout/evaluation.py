"""Public evaluator: identical controls, frozen references, DEG and 18 Pearson metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize
from gradpert.hashing import sha256_json


def evaluate(
    data: Any,
    reference: Any,
    predict: Callable,
    *,
    seen_indices: tuple[int, ...],
    cell_batch: int,
    query_count: int,
    seed: int,
    progress: Callable | None = None,
) -> dict:
    if data.split_name != "test" or reference.manifest.schema_version != "evaluation-state-v2":
        raise ValueError("official comparison requires frozen v2 test evaluation")
    genes = tuple(data.expression_gene_ids)
    if (
        not seen_indices
        or tuple(sorted(set(seen_indices))) != seen_indices
        or any(i < 0 or i >= len(genes) for i in seen_indices)
    ):
        raise ValueError("actual training exposure must bind the expression gene axis")
    unseen = tuple(i for i in range(len(genes)) if i not in set(seen_indices))
    groups = {"seen_expression": seen_indices, "unseen_expression": unseen}
    reference_positions = {c: i for i, c in enumerate(reference.manifest.condition_ids)}
    all_metrics = []
    grouped: dict[str, list] = {k: [] for k in groups}
    rows = []
    draws = data.control_manifest.draws
    if tuple(d.condition_id for d in draws) != tuple(data.split.test_conditions):
        raise ValueError("test condition coverage differs from frozen split")
    for number, draw in enumerate(draws, 1):
        ids = tuple(draw.ordered_row_ids)
        controls = data.load_control_rows(ids)
        if controls.ordered_row_ids != ids or len(ids) != 300:
            raise ValueError("ordered 300-control manifest changed")
        targets = tuple(
            g for g in draw.condition_id.split("+") if g != data.split.control_condition_id
        )
        condition_seed = int(sha256_json([seed, draw.condition_id])[:8], 16)
        prediction = np.asarray(
            predict(
                controls.expression,
                targets,
                genes,
                cell_batch=cell_batch,
                query_count=query_count,
                seed=condition_seed,
            ),
            dtype=np.float32,
        )
        if prediction.shape != controls.expression.shape or not np.isfinite(prediction).all():
            raise ValueError("official prediction must cover the complete finite [300,G] axis")
        truth = data.load_truth_rows(draw.condition_id)
        args = dict(
            condition_id=draw.condition_id,
            prediction=prediction,
            input_control=controls.expression,
            truth=truth.expression,
            metric_control_pool_mean=reference.metric_control_means[
                reference_positions[draw.condition_id]
            ],
            de_gene_indices=reference.manifest.de_gene_indices[draw.condition_id],
            systema_reference=reference.systema_reference,
            de_unavailable_reason=reference.manifest.de_unavailable_reasons.get(draw.condition_id),
        )
        metric = compute_condition_metrics_v2(**args)
        all_metrics.append(metric)
        grouped_rows = {}
        for name, axis in groups.items():
            if not axis:
                grouped_rows[name] = None
                continue
            item = compute_condition_metrics_v2(**args, gene_indices=axis)
            grouped[name].append(item)
            grouped_rows[name] = asdict(item)
        rows.append(
            {
                "condition_id": draw.condition_id,
                "metrics": asdict(metric),
                "gene_groups": grouped_rows,
                "ordered_control_row_ids_sha256": sha256_json(list(ids)),
                "ordered_truth_row_ids_sha256": sha256_json(list(truth.ordered_row_ids)),
                "control_count": len(ids),
                "truth_count": len(truth.ordered_row_ids),
            }
        )
        if progress:
            progress(number, len(draws), draw.condition_id)
    return {
        "schema": "official-expression-holdout-evaluation-v1",
        "summary": [asdict(item) for item in macro_summarize(all_metrics)],
        "gene_group_summaries": {
            k: [asdict(item) for item in macro_summarize(v)] if v else None
            for k, v in grouped.items()
        },
        "condition_metrics": rows,
        "gene_groups": {
            k: {"count": len(v), "gene_ids_sha256": sha256_json([genes[i] for i in v])}
            for k, v in groups.items()
        },
        "canonical_data_sha256": data.manifest.canonical_adata_sha256,
        "split_content_sha256": data.split.split_content_sha256,
        "expression_gene_order_sha256": sha256_json(list(genes)),
        "control_manifest_sha256": data.control_manifest_file_sha256,
        "evaluation_state_sha256": reference.manifest_file_sha256,
        "systema_reference_sha256": reference.manifest.systema_reference_content_sha256,
        "evaluation_recipe": {
            "controls": 300,
            "aggregation": "mean_then_pearson_condition_equal",
            "cell_batch": cell_batch,
            "query_count": query_count,
            "seed": seed,
        },
    }
