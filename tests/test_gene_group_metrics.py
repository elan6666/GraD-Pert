from __future__ import annotations

import numpy as np

from gradpert.evaluation.gene_groups import grouped_condition_metrics
from gradpert.evaluation.metrics import compute_condition_metrics


def test_grouped_metrics_preserve_original_references_and_de_intersection() -> None:
    prediction = np.tile(np.array([2.0, 4.0, 3.0, 8.0]), (300, 1))
    control = np.tile(np.array([1.0, 2.0, 1.0, 2.0]), (300, 1))
    truth = np.tile(np.array([3.0, 3.0, 5.0, 9.0]), (5, 1))
    reference = np.array([0.5, 0.5, 0.5, 0.5])
    kwargs = dict(
        condition_id="A",
        prediction=prediction,
        input_control=control,
        truth=truth,
        metric_control_pool_mean=reference,
        de_gene_indices=[0, 1, 2],
        top_de_gene_indices=[0, 1, 2],
        systema_reference=reference,
    )
    original = compute_condition_metrics(**kwargs)
    all_group = grouped_condition_metrics(**kwargs, gene_indices=(0, 1, 2, 3))
    assert all_group == original
    subset = grouped_condition_metrics(**kwargs, gene_indices=(1, 2, 3))
    assert subset.results[0].gene_count == 3
    assert subset.results[1].gene_count == 2
    assert subset.results[2].gene_count == 2
    sparse = grouped_condition_metrics(**kwargs, gene_indices=(3,))
    assert all(result.reason == "fewer_than_two_genes" for result in sparse.results)
    assert [result.gene_count for result in sparse.results] == [1, 0, 0]
