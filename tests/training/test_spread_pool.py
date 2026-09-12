import math

import pytest
import torch

from gradpert.modeling.losses import embedding_spread_loss
from gradpert.training.spread_pool import cell_spread_pool, diagnose_cell_spread


def test_cell_order_and_exact_duplicates_preserve_gradient_accumulation():
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    pool = cell_spread_pool(states, ["a", "b"], ["b", "a", "b", "a"])
    assert torch.equal(pool, states[[1, 0, 1, 0]])
    pool.sum().backward()
    torch.testing.assert_close(states.grad, torch.full_like(states, 2))


def test_duplicates_have_large_loss_but_zero_p_gradient_without_noise():
    states = torch.eye(2, requires_grad=True)
    rng = torch.get_rng_state().clone()
    report = diagnose_cell_spread(states, ["a", "b"], ["a", "b", "a", "b"])
    assert report["same_condition_nearest_fraction"] == 1
    assert report["zero_distance_nearest_fraction"] == 1
    assert report["repeated_p_exactly_equal"]
    assert report["unweighted_loss"] == pytest.approx(-math.log(1e-8))
    assert report["unweighted_unique_p_gradient_norm"] == 0
    assert states.grad is None and torch.equal(rng, torch.get_rng_state())


def test_unique_batch_retains_original_loss():
    states = torch.eye(3)
    pool = cell_spread_pool(states, ["a", "b", "c"], ["a", "b", "c"])
    assert torch.equal(embedding_spread_loss(states)[0], embedding_spread_loss(pool)[0])


def test_unknown_condition_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        cell_spread_pool(torch.eye(2), ["a", "b"], ["ctrl"])
