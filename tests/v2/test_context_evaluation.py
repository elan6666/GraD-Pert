from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_components import fixture
from test_evaluation import graph_index

from gradpert.training.v2.context_evaluation import evaluate_contexts


def data_fixture():
    controls = np.arange(12, dtype=np.float32).reshape(3, 4)
    truth = controls[:2].copy()
    return SimpleNamespace(
        split_name="val",
        expression_gene_ids=graph_index().gene_ids[:4],
        control_manifest=SimpleNamespace(
            split_name="val",
            draws=[SimpleNamespace(condition_id="g5+ctrl", ordered_row_ids=("c2", "c0", "c1"))],
        ),
        control_manifest_file_sha256="sealed-controls",
        split=SimpleNamespace(control_condition_id="ctrl"),
        load_control_rows=lambda ids: SimpleNamespace(ordered_row_ids=ids, expression=controls),
        load_truth_rows=lambda condition: SimpleNamespace(
            ordered_row_ids=("t0", "t1"), expression=truth
        ),
    ), truth


def test_context_comparison_scores_only_fixed_gene_axis():
    model, _ = fixture()
    for parameter in model.prediction.parameters():
        parameter.data.zero_()
    data, truth = data_fixture()
    kwargs = dict(
        evaluation_ids=(3, 0),
        budgets=(2, 4),
        context_seed=5,
        expected_split="val",
        device=torch.device("cpu"),
        cell_batch=2,
    )
    before = evaluate_contexts(model, graph_index(), data, **kwargs)
    truth[:, (1, 2)] = 1e8
    after = evaluate_contexts(model, graph_index(), data, **kwargs)
    assert before == after
    assert before["evaluation_gene_ids"] == ["g3", "g0"]
    assert [r["fixed_axis_prediction_mse"] for r in before["macro"]] == [4.0, 4.0]
    assert "prediction" not in before


@pytest.mark.parametrize("budgets", [(4, 2), (2, 2), (), (1, 4)])
def test_invalid_context_budgets_rejected_before_inference(budgets):
    model, _ = fixture()
    data, _ = data_fixture()
    with pytest.raises(ValueError):
        evaluate_contexts(
            model,
            graph_index(),
            data,
            evaluation_ids=(0, 3),
            budgets=budgets,
            context_seed=5,
            expected_split="val",
            device=torch.device("cpu"),
            cell_batch=2,
        )
