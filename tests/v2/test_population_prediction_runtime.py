"""Synthetic native-path group loss, without expression files or GPU work."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_response_anchor_integration import integrated

from gradpert.training.v2.response_anchor_runtime import population_prediction_loss


def test_fp32_mean_before_square_and_fixed_target():
    values = torch.tensor([[1.0, 3.0], [3.0, 1.0]], requires_grad=True)
    target = torch.tensor([2.0, 2.0])
    loss = population_prediction_loss(values, target)
    assert loss.item() == 0
    assert (values - target).square().mean().item() == 1
    loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))
    assert population_prediction_loss(values.to(torch.bfloat16), target).dtype == torch.float32
    with pytest.raises(ValueError):
        population_prediction_loss(values, target.requires_grad_())


def test_native_prediction_head_receives_gradient_without_new_head_or_teacher():
    objective, batch, _, controller = integrated()
    del objective.student.response_anchor_readout
    del objective.teacher.response_anchor_readout
    controller.runtime.options.response_anchor_loss = "population_prediction"
    controller.population_targets = np.zeros((1322, 4), np.float32)
    before = copy.deepcopy(objective.teacher.state_dict())
    rng = torch.get_rng_state().clone()
    loss = controller.group_loss(batch, 0, 0)
    changed = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    torch.testing.assert_close(loss, controller.group_loss(changed, 0, 0), rtol=0, atol=0)
    with controller.group_rng(0, 0):
        loss.backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert any(
        p.grad is not None and p.grad.abs().sum() for p in objective.student.prediction.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() for p in objective.student.response.parameters()
    )
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert not objective.pending
    for key, value in before.items():
        torch.testing.assert_close(value, objective.teacher.state_dict()[key], rtol=0, atol=0)


def test_population_fields_are_strict_opt_in():
    from pathlib import Path

    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / "configs/v2/response_anchor_20261028/R14-FRGA/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    with pytest.raises(ValueError):
        replace(options, response_anchor_loss="population_prediction")
    fields = dict(
        response_anchor_population_path="/data/yilangliu/population.json",
        response_anchor_population_sha256="a" * 64,
        response_anchor_population_arrays_sha256="b" * 64,
    )
    with pytest.raises(ValueError):
        replace(options, **fields)
    assert (
        replace(
            options, response_anchor_loss="population_prediction", **fields
        ).response_anchor_weight
        == options.response_anchor_weight
    )


def test_evaluation_identity_does_not_open_train_sidecar(monkeypatch):
    from types import SimpleNamespace

    from gradpert.training.v2 import response_anchor_runtime
    from gradpert.training.v2.runtime import _configure_response_anchor

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must never load TRAIN bank")

    monkeypatch.setattr(response_anchor_runtime, "load_response_bank", forbidden)
    options = SimpleNamespace(
        response_anchor_loss="population_prediction",
        response_anchor_weight=0.1,
        response_anchor_seed=20261028,
        response_anchor_sha256="a" * 64,
        response_anchor_arrays_sha256="b" * 64,
        response_anchor_population_sha256="c" * 64,
        response_anchor_population_arrays_sha256="d" * 64,
    )
    runtime = SimpleNamespace(
        options=options,
        objective=SimpleNamespace(),
        identity={},
        data=None,
        purpose="evaluation",
        allowed_expression_ids=None,
    )
    _configure_response_anchor(runtime)
    identity = runtime.identity["response_anchor"]
    assert identity["loss"] == "population_prediction"
    assert identity["teacher_readout"] == "absent_no_additional_forward"
    assert not hasattr(runtime.objective, "response_anchor_backward")


def test_population_configuration_generator_roundtrip():
    from pathlib import Path

    from gradpert.config.v2 import V2Options
    from scripts.v2.generate_response_anchor import configuration

    config = configuration(
        Path(__file__).resolve().parents[2],
        bank_path="/data/yilangliu/bank.json",
        manifest_sha256="a" * 64,
        arrays_sha256="b" * 64,
        weight=0.1,
        loss_mode="population_prediction",
        population_path="/data/yilangliu/population.json",
        population_sha256="c" * 64,
        population_arrays_sha256="d" * 64,
    )
    from gradpert.config.schema import ExperimentConfig

    parsed = ExperimentConfig.model_validate(config)
    _, options = V2Options.parse_parameters(parsed.model.parameters)
    assert options.response_anchor_loss == "population_prediction"
    assert options.response_anchor_population_sha256 == "c" * 64
