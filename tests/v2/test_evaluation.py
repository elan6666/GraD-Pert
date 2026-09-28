from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.graphs import GraphTopology
from gradpert.graphs.pruning import PrunedSourceGraph
from gradpert.training.v2.evaluation import evaluate, predict_controls
from gradpert.training.v2.views import NeighborhoodIndex


def graph_index():
    genes = tuple(f"g{i}" for i in range(6))
    return NeighborhoodIndex(
        GraphTopology(
            genes, {name: PrunedSourceGraph(name, 6, genes, (), 20) for name in ("go", "string")}
        ),
        degree=0,
        seed=1,
    )


@pytest.mark.parametrize("query_count", [1, 3, 4, 10])
def test_inference_covers_every_gene_and_preserves_control_order(query_count):
    model, _ = fixture()
    index = graph_index()
    controls = np.random.default_rng(2).normal(size=(5, 4)).astype(np.float32)
    # A zero decoder is the exact identity through the raw-control residual.
    for parameter in model.prediction.parameters():
        parameter.data.zero_()
    result = predict_controls(
        model,
        index,
        controls,
        (5,),
        device=torch.device("cpu"),
        cell_batch=2,
        query_count=query_count,
    )
    np.testing.assert_array_equal(result, controls)


def test_cell_batch_does_not_change_full_context_predictions():
    model, _ = fixture()
    controls = np.random.default_rng(3).normal(size=(5, 4)).astype(np.float32)
    kwargs = dict(device=torch.device("cpu"), query_count=4)
    a = predict_controls(model, graph_index(), controls, (0, 5), cell_batch=1, **kwargs)
    b = predict_controls(model, graph_index(), controls, (0, 5), cell_batch=5, **kwargs)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("targets", [(), (-1,), (6,), (1, 1)])
def test_invalid_target_identity_rejected(targets):
    model, _ = fixture()
    with pytest.raises(ValueError):
        predict_controls(
            model,
            graph_index(),
            np.zeros((2, 4), dtype=np.float32),
            targets,
            device=torch.device("cpu"),
            cell_batch=1,
            query_count=4,
        )


def test_validation_rejects_reference_including_test_conditions():
    model, _ = fixture()
    data = SimpleNamespace(
        split_name="val",
        control_manifest=SimpleNamespace(split_name="val"),
        expression_gene_ids=graph_index().gene_ids[:4],
        split=SimpleNamespace(val_conditions=["g0"], train_conditions=["g1"]),
    )
    reference = SimpleNamespace(manifest=SimpleNamespace(condition_ids=["g0", "g2"]))
    with pytest.raises(ValueError, match="combined test reference"):
        evaluate(
            model,
            graph_index(),
            data,
            reference,
            expected_split="val",
            device=torch.device("cpu"),
            cell_batch=2,
            query_count=4,
        )


def test_test_metrics_use_one_prediction_for_all_three_gene_axes(monkeypatch):
    from gradpert.training.v2 import evaluation as module

    index = graph_index()
    ids = tuple(f"control-{row}" for row in range(300))
    controls = np.zeros((300, 4), dtype=np.float32)
    truth = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (5, 1))
    calls = []

    def predict(*args, **kwargs):
        calls.append(1)
        return np.tile(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (300, 1))

    monkeypatch.setattr(module, "predict_controls", predict)
    data = SimpleNamespace(
        split_name="test",
        control_manifest=SimpleNamespace(
            split_name="test", draws=[SimpleNamespace(condition_id="g0", ordered_row_ids=ids)]
        ),
        expression_gene_ids=index.gene_ids[:4],
        split=SimpleNamespace(control_condition_id="ctrl"),
        load_control_rows=lambda ordered: SimpleNamespace(
            ordered_row_ids=ordered, expression=controls
        ),
        load_truth_rows=lambda condition: SimpleNamespace(
            ordered_row_ids=tuple(f"truth-{row}" for row in range(5)), expression=truth
        ),
        control_manifest_file_sha256="control-hash",
    )
    reference = SimpleNamespace(
        manifest=SimpleNamespace(
            condition_ids=["g0"],
            de_gene_indices={"g0": [0, 1, 2, 3]},
            top_de_gene_indices={"g0": [0, 1, 2, 3]},
            de_unavailable_reasons={},
        ),
        metric_control_means=[np.zeros(4)],
        systema_reference=np.zeros(4),
        manifest_file_sha256="reference-hash",
    )
    result = evaluate(
        fixture()[0],
        index,
        data,
        reference,
        expected_split="test",
        device=torch.device("cpu"),
        cell_batch=2,
        query_count=4,
        metric_gene_groups={
            "seen_expression": (0, 1),
            "unseen_expression": (2, 3),
            "empty_probe": (),
        },
    )
    assert calls == [1]
    assert len(result["metrics"]) == 3
    assert result["metric_gene_groups"]["seen_expression"]["gene_count"] == 2
    assert result["metric_gene_groups"]["unseen_expression"]["gene_count"] == 2
    assert len(result["metric_gene_groups"]["seen_expression"]["metrics"]) == 3
    assert result["metric_gene_groups"]["empty_probe"]["unavailable_reason"] == (
        "empty_expression_group"
    )


def test_context_budgets_are_nested_and_keep_evaluation_axis_fixed():
    from gradpert.training.v2.evaluation import context_queries

    evaluation = (2, 8)
    contexts = [context_queries(evaluation, gene_count=10, budget=n, seed=17) for n in (3, 6, 10)]
    assert set(contexts[0]) < set(contexts[1]) < set(contexts[2])
    for context in contexts:
        np.testing.assert_array_equal(context[np.searchsorted(context, evaluation)], evaluation)
    np.testing.assert_array_equal(contexts[-1], np.arange(10))
    with pytest.raises(ValueError):
        context_queries(evaluation, gene_count=10, budget=1, seed=17)


def test_explicit_noncontiguous_queries_preserve_gene_identity():
    from gradpert.training.v2.evaluation import predict_query_set

    model, _ = fixture()
    for parameter in model.prediction.parameters():
        parameter.data.zero_()
    controls = np.arange(20, dtype=np.float32).reshape(5, 4)
    queries = np.array([0, 3])
    result = predict_query_set(
        model, graph_index(), controls, (5,), queries, device=torch.device("cpu"), cell_batch=2
    )
    np.testing.assert_array_equal(result, controls[:, queries])
    with pytest.raises(ValueError, match="increasing"):
        predict_query_set(
            model,
            graph_index(),
            controls,
            (5,),
            np.array([3, 0]),
            device=torch.device("cpu"),
            cell_batch=2,
        )
