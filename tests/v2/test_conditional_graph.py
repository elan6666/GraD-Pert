"""CG1 source conditioning preserves legal edges and zero-start graph behavior."""

from copy import deepcopy

import pytest
import torch
from torch import Tensor

from gradpert.modeling.v2.conditional_graph import (
    ControlGraphModulation,
    ControlSummary,
    conditional_sparse_read,
)
from gradpert.modeling.v2.model import SparseRead


def _fixture() -> tuple[SparseRead, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(31)
    layer = SparseRead(8, 2, 0.0, rank=4, source_key_gate=True)
    with torch.no_grad():
        layer.source_key_gate.normal_(std=0.15)
        layer.source_bias.normal_(std=0.1)
    query = torch.randn(2, 8)
    memory = torch.randn(5, 8)
    neighbors = torch.tensor([[0, 2, -1], [2, 3, 4]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    sources = torch.tensor(
        [[[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]],
        dtype=torch.bool,
    )
    return layer, query, memory, neighbors, valid, sources


def _expanded_reference(
    layer: SparseRead,
    query: Tensor,
    memory: Tensor,
    neighbors: Tensor,
    valid: Tensor,
    sources: Tensor,
    eta: Tensor,
    directions: Tensor,
) -> Tensor:
    """Small independent oracle; batch-expanded keys are only used in tests."""
    n, k = neighbors.shape
    q = layer.query(layer.norm1(query)).reshape(n, layer.heads, layer.head_width)
    selected = memory[neighbors.clamp_min(0)]
    assert layer.compress is not None and layer.latent_norm is not None
    selected = layer.latent_norm(layer.compress(selected))
    key = layer.key(selected).reshape(n, k, layer.heads, layer.head_width)
    value = layer.value(selected).reshape(n, k, layer.heads, layer.head_width)
    assert layer.source_key_gate is not None
    gate = 1 + torch.einsum("nks,shd->nkhd", sources.float(), layer.source_key_gate)
    gate = gate.unsqueeze(0) + torch.einsum("nks,bhs,shd->bnkhd", sources.float(), eta, directions)
    score = torch.einsum("nhd,bnkhd->bnhk", q.float(), key.float()[None] * gate.float())
    score = score / layer.head_width**0.5
    bias = sources.to(layer.source_bias.dtype) @ layer.source_bias
    score = score + bias.permute(0, 2, 1)[None]
    weights = score.masked_fill(~valid[None, :, None, :], float("-inf")).softmax(-1)
    read = torch.einsum("bnhk,nkhd->bnhd", weights.to(value.dtype), value).flatten(-2)
    x = query[None] + layer.dropout(layer.output(read))
    return x + layer.dropout(layer.ffn(layer.norm2(x)))


def test_zero_modulation_matches_static_output_and_gradients() -> None:
    layer, query, memory, neighbors, valid, sources = _fixture()
    baseline = deepcopy(layer)
    modulation = ControlGraphModulation(8, 2)
    summary = torch.randn(3, 8)
    eta = modulation(summary)
    torch.testing.assert_close(eta, torch.zeros_like(eta), rtol=0, atol=0)
    actual_query, actual_memory = query.clone().requires_grad_(), memory.clone().requires_grad_()
    expected_query, expected_memory = (
        query.clone().requires_grad_(),
        memory.clone().requires_grad_(),
    )
    actual = conditional_sparse_read(
        layer, actual_query, actual_memory, neighbors, valid, sources, eta, modulation.directions
    )
    expected = torch.stack(
        [baseline(expected_query, expected_memory, neighbors, valid, sources)] * len(summary)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual_query.grad, expected_query.grad, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(actual_memory.grad, expected_memory.grad, rtol=1e-5, atol=3e-6)
    for (_, actual_parameter), (_, expected_parameter) in zip(
        layer.named_parameters(), baseline.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=3e-6
        )
    assert modulation.projection.weight.grad is not None
    assert modulation.projection.weight.grad.abs().sum() > 0
    assert modulation.projection.bias.grad is not None
    assert modulation.projection.bias.grad.abs().sum() > 0
    assert modulation.directions.grad is not None
    torch.testing.assert_close(
        modulation.directions.grad, torch.zeros_like(modulation.directions), rtol=0, atol=0
    )


def test_factorized_dynamic_score_matches_expanded_formula_and_gradients() -> None:
    layer, query, memory, neighbors, valid, sources = _fixture()
    reference = deepcopy(layer)
    actual_eta = torch.randn(3, 2, 4).tanh().requires_grad_()
    expected_eta = actual_eta.detach().clone().requires_grad_()
    actual_directions = torch.randn(4, 2, 4).requires_grad_()
    expected_directions = actual_directions.detach().clone().requires_grad_()
    actual_query, actual_memory = query.clone().requires_grad_(), memory.clone().requires_grad_()
    expected_query, expected_memory = (
        query.clone().requires_grad_(),
        memory.clone().requires_grad_(),
    )
    actual = conditional_sparse_read(
        layer, actual_query, actual_memory, neighbors, valid, sources, actual_eta, actual_directions
    )
    expected = _expanded_reference(
        reference,
        expected_query,
        expected_memory,
        neighbors,
        valid,
        sources,
        expected_eta,
        expected_directions,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for actual_gradient, expected_gradient in (
        (actual_eta.grad, expected_eta.grad),
        (actual_directions.grad, expected_directions.grad),
        (actual_query.grad, expected_query.grad),
        (actual_memory.grad, expected_memory.grad),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=5e-6)
    for (_, actual_parameter), (_, expected_parameter) in zip(
        layer.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=2e-5, atol=5e-6
        )


def test_controls_change_attention_without_new_memory_edges() -> None:
    layer, query, memory, neighbors, valid, sources = _fixture()
    modulation = ControlGraphModulation(8, 2)
    with torch.no_grad():
        modulation.projection.weight.normal_(std=0.5)
        modulation.directions.normal_(std=0.5)
    summary = torch.stack((torch.ones(8), -torch.ones(8)))
    eta = modulation(summary)
    output = conditional_sparse_read(
        layer, query, memory, neighbors, valid, sources, eta, modulation.directions
    )
    assert not torch.allclose(output[0], output[1])
    changed_memory = memory.clone()
    changed_memory[1] += 100  # Gene 1 is outside every query's legal neighborhood.
    padded_neighbors = neighbors.clone()
    padded_neighbors[~valid] = 9999
    padded_sources = sources.clone()
    padded_sources[~valid] = True
    changed_output = conditional_sparse_read(
        layer,
        query,
        changed_memory,
        padded_neighbors,
        valid,
        padded_sources,
        eta,
        modulation.directions,
    )
    torch.testing.assert_close(output, changed_output, rtol=0, atol=0)
    changed_sources = sources.clone()
    changed_sources[0, 1] = torch.tensor([False, False, True, False])
    source_output = conditional_sparse_read(
        layer, query, memory, neighbors, valid, changed_sources, eta, modulation.directions
    )
    assert not torch.allclose(output[:, 0], source_output[:, 0])
    torch.testing.assert_close(output[:, 1], source_output[:, 1], rtol=0, atol=0)


def test_control_summary_is_weighted_sum_and_permutation_invariant() -> None:
    torch.manual_seed(17)
    layer = ControlSummary(8)
    identity, expression = torch.randn(5, 8), torch.randn(3, 5, 8)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    output = layer(identity, expression)
    reordered = layer(identity[permutation], expression[:, permutation])
    torch.testing.assert_close(output, reordered, rtol=2e-6, atol=2e-6)
    pooled = []
    handle = layer.summary.register_forward_pre_hook(
        lambda _module, inputs: pooled.append(inputs[0].detach().clone())
    )
    layer(identity, expression)
    layer(torch.cat((identity, identity)), torch.cat((expression, expression), dim=1))
    handle.remove()
    torch.testing.assert_close(pooled[1], 2 * pooled[0], rtol=2e-6, atol=2e-6)
    changed_expression = expression.clone()
    changed_expression[:, 0] = 0  # A masked embedding supplied by the caller.
    assert not torch.allclose(output, layer(identity, changed_expression))


@pytest.mark.parametrize("bad", ["empty_edge", "wrong_source", "retention"])
def test_invalid_sparse_layer_contract_is_rejected(bad: str) -> None:
    layer, query, memory, neighbors, valid, sources = _fixture()
    modulation = ControlGraphModulation(8, 2)
    if bad == "empty_edge":
        valid[0] = False
    elif bad == "wrong_source":
        sources = sources[..., :3]
    else:
        layer.retention = True
    with pytest.raises(ValueError):
        conditional_sparse_read(
            layer,
            query,
            memory,
            neighbors,
            valid,
            sources,
            modulation(torch.randn(3, 8)),
            modulation.directions,
        )


def test_summary_and_modulation_reject_misaligned_input_shapes() -> None:
    summary = ControlSummary(8)
    with pytest.raises(ValueError, match="aligned gene axis"):
        summary(torch.randn(4, 8), torch.randn(2, 5, 8))
    with pytest.raises(ValueError, match="at least one"):
        summary(torch.empty(0, 8), torch.empty(2, 0, 8))
    with pytest.raises(ValueError, match="divisible"):
        ControlGraphModulation(8, 3)
    with pytest.raises(ValueError, match="summary"):
        ControlGraphModulation(8, 2)(torch.randn(2, 7))
