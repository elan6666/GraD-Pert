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
