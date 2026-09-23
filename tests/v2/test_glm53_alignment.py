"""Task-adapted sparse MLA and clipped SwiGLU invariants."""

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
