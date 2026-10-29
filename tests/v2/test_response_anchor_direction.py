"""Synthetic CPU direction-loss, native branch and identity compatibility checks."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.training.v2.response_anchor import group_mean_loss
from gradpert.training.v2.response_anchor_runtime import ResponseAnchorController
from gradpert.training.v2.runtime import _configure_response_anchor


def identity_head(width=2, *, dtype=torch.float32):
    head = nn.Linear(width, width, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(width, dtype=dtype))
        head.bias.zero_()
    return head


@pytest.mark.parametrize("target,expected", [([1, 0], 0), ([0, 1], 1), ([-1, 0], 2)])
def test_direction_aligned_orthogonal_opposed(target, expected):
    z = torch.tensor([[2.0, 0.0]] * 40)
    fixed = torch.tensor(target, dtype=torch.float32, requires_grad=True)
    loss = group_mean_loss(z, fixed, identity_head(), "cosine_direction")
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert fixed.grad is None


def test_nonparallel_direction_has_actual_chain_rule_gradients():
    z = torch.tensor([[1.0, 0.4], [2.0, 1.0], [0.0, 2.0], [1.0, 1.8]], requires_grad=True)
    target = torch.tensor([0.4, -0.9], requires_grad=True)
    head = identity_head()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.3], [-0.4, 1.0]]))
        head.bias.copy_(torch.tensor([0.2, -0.1]))
    mean = z.detach().mean(0)
    u = head(mean).detach()
    unit, fixed = u / u.norm(), target.detach() / target.detach().norm()
    du = -(fixed - torch.dot(unit, fixed) * unit) / u.norm()
    loss = group_mean_loss(z, target, head, "cosine_direction")
    loss.backward()
    assert z.grad.norm() > 0 and head.bias.grad.norm() > 0
    torch.testing.assert_close(z.grad, (head.weight.detach().T @ du / 4).expand_as(z))
    torch.testing.assert_close(head.weight.grad, torch.outer(du, mean))
    torch.testing.assert_close(head.bias.grad, du)
    assert target.grad is None


def test_complete_group_mean_precedes_one_linear_and_normalization():
    z = torch.tensor([[100.0, 0.0], [100.0, 0.0], [-100.0, 0.0], [-96.0, 4.0]])
    head = identity_head()
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.5, -0.2]))
    target = torch.tensor([1.0, 0.0])
    calls = []
    hook = head.register_forward_pre_hook(lambda module, args: calls.append(args[0].detach()))
    try:
        actual = group_mean_loss(z, target, head, "cosine_direction")
    finally:
        hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], z.mean(0), rtol=0, atol=0)
    expected = 1 - F.cosine_similarity(head(z.mean(0)), target, dim=0, eps=1e-6)
    per_cell = (1 - F.cosine_similarity(head(z), target.expand_as(z), dim=1)).mean()
    normalize_before_mean = 1 - F.cosine_similarity(
        head(F.normalize(z, dim=1).mean(0)), target, dim=0
    )
    torch.testing.assert_close(actual, expected)
    assert abs(actual - per_cell) > 0.1
    assert abs(actual - normalize_before_mean) > 0.01


def test_head_keeps_bf16_autocast_but_direction_computation_is_fp32():
    head = identity_head()
    z = torch.tensor([[0.133, 0.271]] * 4)
    target = torch.tensor([0.51, 0.13])
    observed = []
    hook = head.register_forward_hook(lambda module, args, output: observed.append(output))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = group_mean_loss(z, target, head, "cosine_direction")
    finally:
        hook.remove()
    assert len(observed) == 1 and observed[0].dtype == torch.bfloat16
    output = observed[0].float()
    expected = 1 - torch.dot(output / output.norm(), target / target.norm())
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1e-6])
def test_zero_or_small_target_explicitly_fails(magnitude):
    with pytest.raises(ValueError, match="target norm"):
        group_mean_loss(
            torch.ones(4, 2), torch.tensor([magnitude, 0.0]), identity_head(), "cosine_direction"
        )


def test_fp32_epsilon_clamps_head_without_substituting_target_or_dropping_group():
    z = torch.tensor([[1e-8, 0.0]] * 4, requires_grad=True)
    head, diagnostics = identity_head(), {}
    loss = group_mean_loss(
        z, torch.tensor([1.0, 0.0]), head, "cosine_direction", diagnostics=diagnostics
    )
    assert loss.item() == pytest.approx(0.99, abs=1e-7)
    loss.backward()
    torch.testing.assert_close(z.grad[:, 0], torch.full((4,), -250000.0))
    assert diagnostics == {
        "head_observation_count": 1,
        "head_norm_min": pytest.approx(1e-8),
        "head_norm_max": pytest.approx(1e-8),
        "head_clamped_count": 1,
    }


def test_old_mse_default_is_exact_including_gradients_rng_and_no_diagnostics():
    generator = torch.Generator().manual_seed(25)
    z = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, generator=generator, dtype=torch.float64, requires_grad=True)
    head = nn.Linear(3, 3, dtype=torch.float64)
    other = copy.deepcopy(head)
    ref_z = z.detach().clone().requires_grad_()
    diagnostics = {}
    rng = torch.get_rng_state().clone()
    actual = group_mean_loss(z, target, head, diagnostics=diagnostics)
    reference = F.mse_loss(other(ref_z.mean(0)), target.detach())
    assert torch.equal(actual, reference)
    actual.backward()
    reference.backward()
    for a, b in zip((z, *head.parameters()), (ref_z, *other.parameters()), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert target.grad is None and diagnostics == {}


@pytest.mark.parametrize("mode", ["cosine", "direction", "", None, True])
def test_loss_and_config_reject_unknown_mode(mode):
    with pytest.raises(ValueError, match="mse or cosine_direction"):
        group_mean_loss(torch.ones(4, 2), torch.ones(2), identity_head(), mode)
    with pytest.raises(ValueError, match="mse or cosine_direction"):
        replace(options(), response_anchor_loss=mode)


def options():
    config = load_experiment_config(
        Path(__file__).parents[2]
        / "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    _, value = V2Options.parse_parameters(config.model.parameters)
    return replace(
        value,
        response_anchor_weight=0.1,
        response_anchor_path="/data/yilangliu/bank/manifest.json",
        response_anchor_sha256="a" * 64,
        response_anchor_arrays_sha256="b" * 64,
    )


def test_new_loss_field_parses_while_omitted_field_retains_mse_default():
    config = load_experiment_config(
        Path(__file__).parents[2]
        / "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    parameters = dict(config.model.parameters)
    _, original = V2Options.parse_parameters(parameters)
    assert original.response_anchor_loss == "mse"
    parameters["response_anchor_loss"] = SimpleNamespace(value="cosine_direction")
    _, changed = V2Options.parse_parameters(parameters)
    assert changed.response_anchor_loss == "cosine_direction"
    assert replace(changed, response_anchor_loss="mse") == original


@pytest.mark.parametrize("mode", ["mse", "cosine_direction"])
def test_evaluation_identity_does_not_read_or_construct_train_bank(monkeypatch, mode):
    from gradpert.training.v2 import response_anchor_runtime as module

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation attempted a numerical TRAIN bank read")

    monkeypatch.setattr(module, "load_response_bank", forbidden)
    monkeypatch.setattr(module, "ResponseAnchorController", forbidden)
    value = replace(options(), response_anchor_loss=mode)
    runtime = SimpleNamespace(
        options=value,
        objective=SimpleNamespace(),
        identity={},
        data=None,
        purpose="evaluation",
        allowed_expression_ids=None,
    )
    _configure_response_anchor(runtime)
    expected = {
        "manifest_sha256": "a" * 64,
        "arrays_sha256": "b" * 64,
        "weight": 0.1,
        "seed": 20261028,
        "head": "Linear256_256_default_random_bias_true_private_rng",
        "teacher_readout": "unused_ema_mirror_no_forward_or_target_values",
        "schedule": "hash_permutation_step_i_mod_T_rank_floor_i_div_T_mod_world",
    }
    if mode == "cosine_direction":
        expected.update(
            loss=mode,
            direction_epsilon=1e-6,
            direction_formula="1-dot(normalize_fp32(Linear(mean_group_cls)),normalize_fp32_fixed_target)",
        )
    assert runtime.identity["response_anchor"] == expected
    assert runtime.objective.response_anchor_enabled is True
    assert not hasattr(runtime.objective, "response_anchor_backward")


def native_direction_controller():
    from test_response_anchor_integration import integrated

    objective, batch, optimizer, original = integrated()
    original.runtime.options.response_anchor_loss = "cosine_direction"
    controller = ResponseAnchorController(original.runtime, original.bank)
    return objective, batch, optimizer, controller


@pytest.mark.parametrize("value", [0.0, 1e-8, float("nan")])
def test_all_1322_targets_audited_even_unselected_last_condition(value):
    _, _, _, controller = native_direction_controller()
    targets = controller.bank.condition_targets.copy()
    targets[-1] = value
    controller.bank.condition_targets = targets
    with pytest.raises(ValueError, match="bank target norms"):
        ResponseAnchorController(controller.runtime, controller.bank)
    controller.runtime.options.response_anchor_loss = "mse"
    legacy = ResponseAnchorController(controller.runtime, controller.bank)
    assert legacy.direction_summary() == {}


def test_native_direction_path_observes_actual_readout_without_decoder_or_updates():
    objective, batch, optimizer, controller = native_direction_controller()
    before = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    seen = []
    hook = objective.student.response_anchor_readout.register_forward_hook(
        lambda module, args, output: seen.append(output.detach())
    )
    try:
        loss = controller.group_loss(batch, 0, 0)
    finally:
        hook.remove()
    loss.backward()
    assert len(seen) == 1
    summary = controller.direction_summary()
    assert summary["response_anchor_loss"] == "cosine_direction"
    assert summary["eps"] == 1e-6
    assert summary["target_count"] == 1322 and summary["target_clamped_count"] == 0
    assert summary["head_observation_count"] == 1 and summary["head_clamped_count"] == 0
    assert summary["head_norm_min"] == pytest.approx(seen[0].float().norm().item())
    json.dumps(summary, allow_nan=False)
    assert optimizer.steps == 0 and not objective.pending
    assert all(p.grad is None for p in objective.student.prediction.parameters())
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in objective.student.response.parameters()
    )
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        torch.testing.assert_close(value, objective.state_dict()[key], rtol=0, atol=0)
    assert np.isfinite(loss.item())
