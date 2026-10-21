"""Explicit additive response injection with historical checkpoint compatibility."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_relay_method import small_architecture
from torch import nn

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.model import RelayResponseEncoder
from gradpert.training.v2.objective import JointObjective


class PassThrough(nn.Module):
    def forward(self, x, **kwargs):
        return x


def isolated_encoder(*, residual=True, mode="per_layer"):
    encoder = RelayResponseEncoder(
        replace(
            small_architecture(),
            streams=4,
            response_injection_residual=residual,
            perturbation_injection=mode,
        )
    )
    encoder.self_layers = nn.ModuleList(PassThrough() for _ in range(3))
    encoder.cross_layers = nn.ModuleList(PassThrough() for _ in range(3))
    encoder.ffn_layers = nn.ModuleList(PassThrough() for _ in range(3))
    encoder.norm = nn.Identity()
    return encoder


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_residual_injection_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="response_injection_residual must be boolean"):
        replace(small_architecture(), response_injection_residual=value)


def test_nonrelay_and_unknown_controls_fail_closed():
    with pytest.raises(ValueError, match="require relay attention"):
        V2Architecture(response_injection_residual=True)
    with pytest.raises(ValueError, match="unknown v2 architecture fields"):
        V2Architecture.parse({"response_injection_residul": True})


def test_default_payload_and_old_config_are_unchanged():
    arch = small_architecture()
    explicit = replace(arch, response_injection_residual=False)
    assert not arch.response_injection_residual
    assert "response_injection_residual" not in arch.payload()
    assert arch.payload() == explicit.payload()
    assert V2Architecture.parse(arch.payload()) == arch
    enabled = replace(arch, response_injection_residual=True)
    assert enabled.payload() == {**arch.payload(), "response_injection_residual": True}
    assert V2Architecture.parse(enabled.payload()) == enabled
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/relay_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    parameters = dict(load_experiment_config(path).model.parameters)
    parsed, _ = V2Options.parse_parameters(parameters)
    assert not parsed.response_injection_residual
    parameters["response_injection_residual"] = parameters["checkpoint_layers"].model_copy(
        update={"value": True}
    )
    enabled_config, _ = V2Options.parse_parameters(parameters)
    assert enabled_config.response_injection_residual


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_zero_branch_preserves_identity_and_gradient_only_in_residual_mode(residual, index):
    encoder = isolated_encoder(residual=residual)
    with torch.no_grad():
        for parameter in encoder.injections.parameters():
            parameter.zero_()
    # Four independent streams and the final CLS position all use the same rule.
    x = torch.randn(2, 5, 4, 8, requires_grad=True)
    output = encoder._layer(
        x, torch.randn(2, 4, 8), torch.randn(2, 8), torch.arange(4).expand(2, -1), index, False
    )
    expected = x if residual else torch.zeros_like(x)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(output.sum(), x)[0]
    torch.testing.assert_close(
        gradient, torch.ones_like(x) if residual else torch.zeros_like(x), rtol=0, atol=0
    )


@pytest.mark.parametrize("mode,count", [("per_layer", 3), ("entry", 1)])
def test_each_active_injection_adds_to_all_tokens_including_cls(mode, count):
    encoder = isolated_encoder(mode=mode)
    with torch.no_grad():
        for injection in encoder.injections:
            for parameter in injection.parameters():
                parameter.zero_()
            injection[-1].bias.fill_(1)
    x = torch.randn(2, 5, 8)
    result = encoder(
        x, torch.randn(2, 4, 8), torch.randn(2, 8), (torch.arange(4).expand(2, -1),) * 3
    )
    torch.testing.assert_close(result, x + count)


def test_default_output_and_gradients_match_historical_assignment():
    torch.manual_seed(28)
    encoder = RelayResponseEncoder(replace(small_architecture(), streams=2))
    reference = copy.deepcopy(encoder)
    x = torch.randn(2, 5, 2, 8, requires_grad=True)
    memory = torch.randn(2, 4, 8, requires_grad=True)
    condition = torch.randn(2, 8, requires_grad=True)
    copies = [v.detach().clone().requires_grad_() for v in (x, memory, condition)]
    order = torch.arange(4).expand(2, -1)
    result = x
    expected = copies[0]
    for index in range(3):
        result = encoder._layer(result, memory, condition, order, index, False)
        p = copies[2][:, None, None, :].expand_as(expected)
        expected = reference.injections[index](torch.cat((expected, p), dim=-1))
        expected = reference.self_layers[index](expected, order=order, block_cls_to_gene=False)
        expected = reference.cross_layers[index](expected, memory=copies[1], order=order)
        expected = reference.ffn_layers[index](expected)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    weight = torch.randn_like(result)
    actual_grad = torch.autograd.grad((result * weight).sum(), (x, memory, condition))
    reference_grad = torch.autograd.grad((expected * weight).sum(), copies)
    for actual, old in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)


def test_initialization_parameter_keys_and_teacher_are_unchanged():
    seeds = torch.arange(72, dtype=torch.float32).reshape(9, 8) / 72
    torch.manual_seed(41)
    old = GraDPertV2(seeds, small_architecture())
    torch.manual_seed(41)
    residual = GraDPertV2(seeds, replace(small_architecture(), response_injection_residual=True))
    assert old.state_dict().keys() == residual.state_dict().keys()
    assert [(n, p.shape) for n, p in old.named_parameters()] == [
        (n, p.shape) for n, p in residual.named_parameters()
    ]
    for name, value in old.state_dict().items():
        assert torch.equal(value, residual.state_dict()[name]), name
    residual.load_state_dict(old.state_dict(), strict=True)
    objective = JointObjective(residual)
    assert objective.teacher.options.response_injection_residual
    assert objective.teacher.response.injection_residual
    assert not any(p.requires_grad for p in objective.teacher.parameters())
    assert objective.student.state_dict().keys() == objective.teacher.state_dict().keys()
    for name, value in objective.student.state_dict().items():
        assert torch.equal(value, objective.teacher.state_dict()[name]), name
    inputs = (
        torch.randn(2, 5, 8),
        torch.randn(2, 4, 8),
        torch.randn(2, 8),
        (torch.arange(4).expand(2, -1),) * 3,
    )
    objective.student.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            objective.student.response(*inputs),
            objective.teacher.response(*inputs),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("residual", [False, True])
def test_recompute_matches_forward_input_and_parameter_gradients(residual):
    torch.manual_seed(72)
    plain = RelayResponseEncoder(
        replace(small_architecture(), streams=2, dropout=0.1, response_injection_residual=residual)
    ).train()
    recomputed = copy.deepcopy(plain)
    recomputed.checkpoint_layers = True
    inputs = (torch.randn(2, 5, 8), torch.randn(2, 4, 8), torch.randn(2, 8))
    weight = torch.randn(2, 5, 8)

    def run(encoder):
        args = [v.detach().clone().requires_grad_() for v in inputs]
        torch.manual_seed(91)
        output = encoder(*args, (torch.arange(4).expand(2, -1),) * 3)
        (output * weight).sum().backward()
        return output, [v.grad for v in args], {n: p.grad for n, p in encoder.named_parameters()}

    actual, actual_inputs, actual_params = run(recomputed)
    expected, expected_inputs, expected_params = run(plain)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name in actual_params:
        if expected_params[name] is None:
            assert actual_params[name] is None
        else:
            torch.testing.assert_close(actual_params[name], expected_params[name], rtol=0, atol=0)
