"""Nonlinearity, parameter sharing and optimizer-resume invariants."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gradpert.modeling.v2.mlp import UnifiedMLP


def _formula(layer, x):
    if isinstance(layer.norm, nn.RMSNorm):
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * layer.norm.weight
    gate = F.linear(x, layer.gate.weight, layer.gate.bias).clamp(max=10)
    up = F.linear(x, layer.up.weight, layer.up.bias).clamp(min=-10, max=10)
    return F.linear(F.silu(gate) * up, layer.down.weight, layer.down.bias)


@pytest.mark.parametrize("normalize", [True, False])
def test_output_and_all_gradients_match_explicit_formula(normalize):
    torch.manual_seed(17)
    layer = UnifiedMLP(4, 7, 3, normalize=normalize).double()
    reference = copy.deepcopy(layer)
    if normalize:
        with torch.no_grad():
            layer.norm.weight.copy_(torch.tensor([0.7, 1.4, 0.9, 1.8]))
            reference.norm.weight.copy_(layer.norm.weight)
    x = (torch.randn(2, 5, 4, dtype=torch.float64) * 4).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    actual = layer(x)
    expected = _formula(reference, reference_x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
    for parameter, other in zip(layer.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-12, rtol=1e-12)


def test_clipping_is_upper_only_for_gate_and_symmetric_for_up():
    layer = UnifiedMLP(2, 2, 2, normalize=False).double()
    with torch.no_grad():
        layer.gate.weight.zero_()
        layer.up.weight.zero_()
        layer.gate.bias.copy_(torch.tensor([20.0, -20.0]))
        layer.up.bias.copy_(torch.tensor([20.0, -20.0]))
        layer.down.weight.copy_(torch.eye(2))
        layer.down.bias.zero_()
    result = layer(torch.zeros(3, 2, dtype=torch.float64))
    expected = F.silu(torch.tensor([10.0, -20.0], dtype=torch.float64)) * torch.tensor(
        [10.0, -10.0], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected.expand(3, -1), atol=0, rtol=0)
    symmetric_gate_result = F.silu(torch.tensor(-10.0, dtype=torch.float64)) * -10
    assert result[0, 1] != symmetric_gate_result


@pytest.mark.parametrize("shape", [(5, 4), (2, 7, 4), (2, 3, 5, 4)])
def test_shapes_biases_and_parameter_count(shape):
    layer = UnifiedMLP(4, 11, 6)
    assert layer(torch.randn(shape)).shape == (*shape[:-1], 6)
    assert layer.norm.eps == 1e-6
    assert all(linear.bias is not None for linear in (layer.gate, layer.up, layer.down))
    expected = 4 + 2 * (4 * 11 + 11) + 11 * 6 + 6
    assert sum(parameter.numel() for parameter in layer.parameters()) == expected


def test_normalized_scalar_input_is_rejected_but_scalar_linear_input_is_supported():
    with pytest.raises(ValueError, match="Lift scalar expression"):
        UnifiedMLP(1, 8, 4)
    layer = UnifiedMLP(1, 8, 4, normalize=False)
    assert isinstance(layer.norm, nn.Identity)
    assert layer(torch.tensor([[1.0], [2.0]])).shape == (2, 4)


@pytest.mark.parametrize("widths", [(0, 4, 2), (3, -1, 2), (3, 4, 0)])
def test_invalid_widths_are_rejected(widths):
    with pytest.raises(ValueError, match="widths must be positive"):
        UnifiedMLP(*widths)


def test_zero_output_initialization_has_trainable_output_then_hidden_gradients():
    torch.manual_seed(31)
    layer = UnifiedMLP(4, 9, 3, zero_output=True)
    x = torch.randn(6, 4)
    truth = torch.randn(6, 3)
    initial_gate = layer.gate.weight.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.03)
    assert torch.equal(layer(x), torch.zeros_like(truth))
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(layer(x), truth).backward()
        assert layer.down.weight.grad.abs().sum() > 0
        if update == 0:
            assert torch.count_nonzero(layer.gate.weight.grad) == 0
        else:
            assert layer.gate.weight.grad.abs().sum() > 0
        optimizer.step()
    assert torch.count_nonzero(layer.down.weight) > 0
    assert not torch.equal(initial_gate, layer.gate.weight)


def test_shared_rows_are_permutation_equivariant():
    torch.manual_seed(41)
    layer = UnifiedMLP(4, 7, 3).eval()
    x = torch.randn(2, 6, 4)
    permutation = torch.tensor([4, 0, 2, 5, 1, 3])
    torch.testing.assert_close(layer(x[:, permutation]), layer(x)[:, permutation])


def test_dropout_placement_matches_gated_product_and_output():
    torch.manual_seed(43)
    layer = UnifiedMLP(4, 7, 3, dropout=0.4).train()
    x = torch.randn(2, 5, 4)
    rng = torch.get_rng_state()
    actual = layer(x)
    torch.set_rng_state(rng)
    normalized = layer.norm(x)
    product = F.silu(layer.gate(normalized).clamp(max=10)) * layer.up(normalized).clamp(
        min=-10, max=10
    )
    expected = F.dropout(layer.down(F.dropout(product, p=0.4, training=True)), p=0.4, training=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    layer.eval()
    torch.testing.assert_close(layer(x), _formula(layer, x))


def _update(layer, optimizer, x, truth):
    optimizer.zero_grad(set_to_none=True)
    prediction = layer(x)
    loss = F.mse_loss(prediction, truth)
    loss.backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    return prediction.detach().clone(), loss.detach().clone()


def test_nonzero_lr_multistep_resume_restores_optimizer_and_dropout_rng():
    torch.manual_seed(53)
    layer = UnifiedMLP(4, 9, 3, dropout=0.2).train()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.003, weight_decay=0.01)
    x = torch.randn(2, 5, 4)
    truth = torch.randn(2, 5, 3)
    initial = copy.deepcopy(layer.state_dict())
    for _ in range(2):
        _update(layer, optimizer, x, truth)
    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": layer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        checkpoint,
    )
    continued = [_update(layer, optimizer, x, truth) for _ in range(3)]

    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = UnifiedMLP(4, 9, 3, dropout=0.2).train()
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.003, weight_decay=0.01)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    replayed = [_update(resumed, resumed_optimizer, x, truth) for _ in range(3)]

    assert any(not torch.equal(initial[name], value) for name, value in layer.state_dict().items())
    for actual, expected in zip(continued, replayed, strict=True):
        for tensor, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(tensor, reference, atol=0, rtol=0)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    for parameter, reference in zip(layer.parameters(), resumed.parameters(), strict=True):
        actual_state = optimizer.state[parameter]
        resumed_state = resumed_optimizer.state[reference]
        assert actual_state.keys() == resumed_state.keys()
        for name, value in actual_state.items():
            torch.testing.assert_close(value, resumed_state[name], atol=0, rtol=0)
