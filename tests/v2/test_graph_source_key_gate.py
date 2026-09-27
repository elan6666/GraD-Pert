"""The fourth graph layer adds source-dependent matching, not new edges."""

from dataclasses import replace

import torch

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2.model import GeneGraph, SparseRead


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(2, 8)
    memory = torch.randn(5, 8)
    neighbors = torch.tensor([[0, 1, 4], [2, 3, 0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    sources = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    return query, memory, neighbors, valid, sources


def test_zero_initialized_gate_preserves_existing_forward_and_gradients() -> None:
    torch.manual_seed(37)
    original = SparseRead(8, 2, 0, rank=4, ffn_type="swiglu").eval()
    gated = SparseRead(8, 2, 0, rank=4, ffn_type="swiglu", source_key_gate=True).eval()
    missing, unexpected = gated.load_state_dict(original.state_dict(), strict=False)
    assert missing == ["source_key_gate"] and not unexpected
    assert gated.source_key_gate is not None
    assert torch.count_nonzero(gated.source_key_gate) == 0

    query, memory, neighbors, valid, sources = _inputs()
    old_query, old_memory = query.clone().requires_grad_(), memory.clone().requires_grad_()
    new_query, new_memory = query.clone().requires_grad_(), memory.clone().requires_grad_()
    expected = original(old_query, old_memory, neighbors, valid, sources)
    actual = gated(new_query, new_memory, neighbors, valid, sources)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(new_query.grad, old_query.grad, atol=0, rtol=0)
    torch.testing.assert_close(new_memory.grad, old_memory.grad, atol=0, rtol=0)
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(
            dict(gated.named_parameters())[name].grad, parameter.grad, atol=0, rtol=0
        )
    assert gated.source_key_gate.grad is not None
    assert gated.source_key_gate.grad.abs().sum() > 0


def test_multi_source_gate_changes_only_legal_neighbor_scores() -> None:
    torch.manual_seed(41)
    layer = SparseRead(8, 2, 0, rank=4, source_key_gate=True).eval()
    assert layer.source_key_gate is not None
    with torch.no_grad():
        layer.source_key_gate[0, 0, 0] = 2.0
        layer.source_key_gate[1, 0, 0] = -0.5
    query, memory, neighbors, valid, sources = _inputs()
    output = layer(query, memory, neighbors, valid, sources)
    selected = layer.latent_norm(layer.compress(memory[neighbors]))
    q = layer.query(layer.norm1(query)).reshape(2, 2, 4)
    k = layer.key(selected).reshape(2, 3, 2, 4)
    v = layer.value(selected).reshape(2, 3, 2, 4)
    gate = 1 + torch.einsum("nks,shd->nkhd", sources, layer.source_key_gate)
    assert gate[0, 0, 0, 0] == 2.5  # GO and STRING are both present.
    logits = torch.einsum("nhd,nkhd->nhk", q, k * gate) / 2
    logits += (sources @ layer.source_bias).permute(0, 2, 1)
    weights = logits.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
    read = torch.einsum("nhk,nkhd->nhd", weights, v).flatten(-2)
    expected = query + layer.output(read)
    expected = expected + layer.ffn(layer.norm2(expected))
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)

    changed_memory = memory.clone()
    changed_memory[4] += 100  # Only an invalid padded edge names this node.
    torch.testing.assert_close(
        output, layer(query, changed_memory, neighbors, valid, sources), atol=0, rtol=0
    )


def test_gate_is_optional_and_limited_to_fourth_relay_graph_layer() -> None:
    options = V2Architecture(
        width=8,
        heads=2,
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        graph_source_key_gate=True,
    )
    graph = GeneGraph(torch.randn(9, 8), options)
    assert len(graph.layers) == 4
    assert isinstance(graph.layers[3], SparseRead)
    assert graph.layers[3].source_key_gate is not None
    assert graph.layers[3].source_key_gate.shape == (4, 2, 4)
    old_graph = GeneGraph(torch.randn(9, 8), replace(options, graph_source_key_gate=False))
    assert isinstance(old_graph.layers[3], SparseRead)
    assert old_graph.layers[3].source_key_gate is None
