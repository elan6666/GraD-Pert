import torch
from test_components import fixture

from gradpert.training.v2.diagnostics import response_diagnostics


def test_diagnostics_separate_control_cls_and_raw_residual():
    model, batch = fixture()
    gene = torch.randn(4, model.options.width)
    condition = torch.randn(2, model.options.width)
    model.train()
    for parameter in model.prediction.parameters():
        parameter.data.zero_()
    result = response_diagnostics(
        model,
        gene,
        batch.control,
        batch.control + 1,
        condition,
        condition + 2,
        truth=batch.control + 0.2,
    )
    assert model.training
    changes = result["changes"]
    assert changes["change_control"]["prediction"] > 0.99
    assert changes["change_control"]["delta"] == 0
    assert changes["change_control"]["control_cls"] > 0
    assert changes["change_perturbation"]["control_cls"] == 0
    assert changes["change_perturbation"]["response_cls"] > 0
    assert changes["block_response_cls_to_gene"]["control_cls"] == 0
    assert abs(result["fixed_truth_prediction_mse"]["baseline"] - 0.04) < 1e-6
    assert all(parameter.grad is None for parameter in model.parameters())


def test_identical_inputs_have_zero_swap_sensitivity():
    model, batch = fixture()
    gene = torch.randn(4, model.options.width)
    condition = torch.randn(2, model.options.width)
    result = response_diagnostics(model, gene, batch.control, batch.control, condition, condition)
    for name in ("change_control", "change_perturbation"):
        assert all(value == 0 for value in result["changes"][name].values())
    assert "fixed_truth_prediction_mse" not in result


def test_chunked_diagnostics_match_full_population_with_unequal_last_chunk():
    model, batch = fixture()
    gene = torch.randn(4, model.options.width)
    control = torch.cat((batch.control, batch.control[:1]), dim=0)
    condition = torch.randn(3, model.options.width)
    args = (model, gene, control, control + 1, condition, condition + 2)
    whole = response_diagnostics(*args, cell_batch=3)
    chunked = response_diagnostics(*args, cell_batch=2)
    for name in whole["changes"]:
        for field, value in whole["changes"][name].items():
            assert abs(value - chunked["changes"][name][field]) < 2e-6


def test_frozen_population_adapter_keeps_condition_and_row_identities():
    from types import SimpleNamespace

    from test_evaluation import graph_index

    from gradpert.training.v2.diagnostics import evaluate_response_diagnostics

    model, batch = fixture()
    draws = [
        SimpleNamespace(condition_id=c, ordered_row_ids=ids)
        for c, ids in (("g0+ctrl", ("c0", "c1")), ("g5+ctrl", ("c2", "c3")))
    ]
    data = SimpleNamespace(
        split_name="val",
        control_manifest=SimpleNamespace(split_name="val", draws=draws),
        expression_gene_ids=graph_index().gene_ids[:4],
        split=SimpleNamespace(control_condition_id="ctrl"),
        control_manifest_file_sha256="controls",
        load_control_rows=lambda ids: SimpleNamespace(
            ordered_row_ids=ids, expression=batch.control.numpy() + (1 if ids[0] == "c2" else 0)
        ),
        load_truth_rows=lambda _: SimpleNamespace(
            ordered_row_ids=("t0", "t1"), expression=batch.truth.numpy()
        ),
    )
    result = evaluate_response_diagnostics(
        model,
        graph_index(),
        data,
        query_gene_ids=("g3", "g0"),
        condition_id="g0+ctrl",
        alternative_condition_id="g5+ctrl",
        expected_split="val",
        device=torch.device("cpu"),
        cell_batch=1,
    )
    assert result["control_populations"][0]["ordered_row_ids"] == ["c0", "c1"]
    assert result["control_populations"][1]["ordered_row_ids"] == ["c2", "c3"]
    assert result["query_gene_ids"] == ["g3", "g0"]
    assert result["changes"]["change_perturbation"]["control_cls"] == 0
