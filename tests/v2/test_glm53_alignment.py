"""Task-adapted sparse MLA and clipped SwiGLU invariants."""

import copy

import torch

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2.operators import GatedFeedForward, IndexedLatentAttention, TokenEncoder
from gradpert.training.v2.optimizer import routes


def test_clipped_swiglu_matches_explicit_formula():
    torch.manual_seed(4)
    layer = GatedFeedForward(8, 0).eval()
    x = torch.randn(2, 5, 8) * 20
    expected = layer.down(
        torch.nn.functional.silu(layer.gate(x).clamp(max=10)) * layer.up(x).clamp(min=-10, max=10)
    )
    torch.testing.assert_close(layer(x), expected)


def test_sparse_mla_is_gene_permutation_equivariant_and_indexer_trains():
    torch.manual_seed(7)
    layer = IndexedLatentAttention(16, 2, 8, 0, 4, 8, 2).eval()
    x = torch.randn(2, 7, 16, requires_grad=True)  # six genes, tail CLS
    perm = torch.tensor([4, 1, 5, 0, 3, 2, 6])
    baseline = layer(x)
    changed = layer(x[:, perm])
    torch.testing.assert_close(changed, baseline[:, perm], atol=1e-5, rtol=1e-5)
    baseline.sum().backward()
    assert layer.index_query.weight.grad is not None
    assert layer.index_key.weight.grad is not None
    assert layer.index_query.weight.grad.abs().sum() > 0
    assert layer.index_key.weight.grad.abs().sum() > 0


def test_sparse_encoder_cls_intervention_and_legacy_architecture():
    torch.manual_seed(9)
    encoder = TokenEncoder(8, 2, 4, 2, 0, "hybrid_sparse", False, "swiglu", 3, 4, 2)
    encoder.eval()
    x = torch.randn(1, 5, 8)
    output = encoder(x)
    blocked = encoder(x, block_cls_to_gene=True)
    assert output.shape == blocked.shape
    assert torch.isfinite(output).all() and torch.isfinite(blocked).all()
    assert V2Architecture().attention == "hybrid"
    assert V2Architecture().ffn_type == "gelu"


def test_sparse_projection_optimizer_routes():
    layer = IndexedLatentAttention(16, 2, 8, 0, 4, 8, 2)
    grouped = {entry["name"]: entry for entry in routes(layer)}
    assert grouped["q_up.weight"]["heads"] == 2
    assert grouped["index_query.weight"]["heads"] == 2
    assert grouped["index_scale"]["optimizer"] == "adamw"


def test_chunk_checkpoint_preserves_sparse_attention_gradients():
    torch.manual_seed(11)
    layer = IndexedLatentAttention(16, 2, 8, 0, 4, 8, 2).train()
    plain = copy.deepcopy(layer)
    plain.checkpoint_chunks = False
    x = torch.randn(2, 7, 16, requires_grad=True)
    x_plain = x.detach().clone().requires_grad_()
    actual = layer(x)
    expected = plain(x_plain)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_plain.grad)
    for (_, parameter), (_, reference) in zip(
        layer.named_parameters(), plain.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, reference.grad)


def test_sparse_chunk_matmul_matches_elementwise_reference_and_chunk_size():
    torch.manual_seed(23)
    q = torch.randn(2, 3, 4, 5, requires_grad=True)
    key = torch.randn(2, 3, 4, 7, 5, requires_grad=True)
    value = torch.randn(2, 3, 4, 7, 5, requires_grad=True)
    logits = torch.matmul(q.unsqueeze(-2), key.transpose(-1, -2)).squeeze(-2)
    weights = logits.softmax(-1)
    actual = torch.matmul(weights.unsqueeze(-2), value).squeeze(-2)
    reference_logits = (q.unsqueeze(-2) * key).sum(-1)
    reference = (reference_logits.softmax(-1).unsqueeze(-1) * value).sum(-2)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward(retain_graph=True)
    gradients = [tensor.grad.clone() for tensor in (q, key, value)]
    for tensor in (q, key, value):
        tensor.grad = None
    reference.square().sum().backward()
    for gradient, tensor in zip(gradients, (q, key, value), strict=True):
        torch.testing.assert_close(gradient, tensor.grad, atol=1e-6, rtol=1e-6)

    layer = IndexedLatentAttention(16, 2, 8, 0, 4, 8, 2).train()
    wider_chunk = copy.deepcopy(layer)
    wider_chunk.query_chunk = 4
    x = torch.randn(2, 7, 16, requires_grad=True)
    x_wider = x.detach().clone().requires_grad_()
    output = layer(x)
    larger = wider_chunk(x_wider)
    torch.testing.assert_close(output, larger, atol=1e-5, rtol=1e-5)
    output.square().sum().backward()
    larger.square().sum().backward()
    torch.testing.assert_close(x.grad, x_wider.grad, atol=1e-5, rtol=1e-5)


def test_nested_encoder_checkpoint_backpropagates():
    torch.manual_seed(12)
    encoder = TokenEncoder(8, 2, 4, 2, 0, "hybrid_sparse", True, "swiglu", 3, 4, 2)
    x = torch.randn(1, 5, 8, requires_grad=True)
    encoder(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    indexer = encoder.layers[6].sublayer
    assert indexer.index_query.weight.grad is not None
    assert indexer.index_query.weight.grad.abs().sum() > 0
