import numpy as np
import torch

from gradpert.features.genept_reduction import pca_genept
from gradpert.modeling.v2 import V2Architecture
from gradpert.modeling.v2.model import GeneGraph


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
    ids = torch.arange(9)
    neighbors = torch.stack((ids, ids.roll(1)), dim=-1)
    valid = torch.ones_like(neighbors, dtype=torch.bool)
    sources = torch.zeros(9, 2, 4)
    sources[:, 0, 3] = 1
    sources[:, 1, 0] = 1
    graph(ids, neighbors, valid, sources).square().sum().backward()
    assert graph.embedding.weight.grad is not None
    assert graph.embedding.weight.grad.abs().sum() > 0
