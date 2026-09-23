import numpy as np
import pytest
import torch

from gradpert.features.genept_reduction import pca_genept
from gradpert.modeling.v2 import V2Architecture
from gradpert.modeling.v2.model import GeneGraph, SparseRead


def test_genept_pca_is_reproducible_and_uses_only_seed_matrix():
    values = np.random.default_rng(7).normal(size=(20, 12)).astype(np.float32)
    first, explained = pca_genept(values, 4)
    second, second_explained = pca_genept(values.copy(), 4)
    np.testing.assert_array_equal(first, second)
    assert explained == second_explained
    assert first.shape == (20, 4)
    assert 0 < explained < 1
    np.testing.assert_allclose(first.mean(axis=0), 0, atol=1e-6)


def test_model_width_genept_has_trainable_table_without_adapter():
    options = V2Architecture(width=8, heads=2, projector_hidden=16, prototypes=11)
    graph = GeneGraph(torch.randn(9, 8), options)
    assert isinstance(graph.adapter, torch.nn.Identity)
    assert graph.embedding.weight.requires_grad
    original = GeneGraph(torch.randn(9, 16), options)
    assert isinstance(original.adapter, torch.nn.Linear)


@pytest.mark.parametrize("masked", [False, True])
def test_project_before_gather_matches_original_sparse_read_and_gradients(masked):
    torch.manual_seed(17)
    layer = SparseRead(8, 2, 0).double()
    layer.eval()
    query = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(9, 8, dtype=torch.float64, requires_grad=True)
    if masked:
        memory = memory.clone().index_fill(0, torch.tensor([2, 5]), 0.0)
        memory.retain_grad()
    neighbors = torch.tensor([[0, 1, 1], [2, 3, 4], [5, 6, 7], [4, 4, 8], [8, 0, 2]])
    valid = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]]).bool()
    sources = torch.zeros(5, 3, 4, dtype=torch.float64)
    sources[..., 0] = 1

    def original() -> torch.Tensor:
        n, k = neighbors.shape
        q = layer.query(layer.norm1(query)).reshape(n, layer.heads, layer.head_width)
        selected = memory[neighbors]
        key = layer.key(selected).reshape(n, k, layer.heads, layer.head_width)
        value = layer.value(selected).reshape(n, k, layer.heads, layer.head_width)
        score = torch.einsum("nhd,nkhd->nhk", q.float(), key.float()) / layer.head_width**0.5
        bias = sources.to(layer.source_bias.dtype) @ layer.source_bias
        score = score + bias.permute(0, 2, 1).float()
        weights = score.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
        read = torch.einsum("nhk,nkhd->nhd", weights.to(value.dtype), value).flatten(-2)
        x = query + layer.dropout(layer.output(read))
        return x + layer.dropout(layer.ffn(layer.norm2(x)))

    expected = original()
    actual = layer(query, memory, neighbors, valid, sources)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = (query, memory, *layer.parameters())
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters, retain_graph=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
