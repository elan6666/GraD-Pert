from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gradpert.training.v2.reductions import (
    gather_rows,
    nearest_neighbor_terms,
    population_weights,
)


def test_global_condition_weights_skip_invalid_cells():
    ids = torch.tensor([0, 0, 0, 1, 1, 2])
    valid = torch.tensor([True, False, True, True, False, False])
    torch.testing.assert_close(
        population_weights(ids, valid, "condition_mean"),
        torch.tensor([0.25, 0, 0.25, 0.5, 0, 0]),
    )
    torch.testing.assert_close(
        population_weights(ids, valid, "row_mean"),
        torch.tensor([1 / 3, 0, 1 / 3, 1 / 3, 0, 0]),
    )
    assert not population_weights(ids, torch.zeros_like(valid), "condition_mean").any()


def _neighbor_worker(rank, rendezvous, strategy, exclude_condition):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0], [-1.0, 0.2]])
        ids = torch.tensor([0, 0, 1, 1, 1])
        weights = population_weights(ids, torch.ones(5, dtype=torch.bool), strategy)
        reference = values.clone().requires_grad_()
        excluded = ids if exclude_condition else None
        (nearest_neighbor_terms(reference, excluded) * weights).sum().backward()
        start, end = ((0, 2), (2, 5))[rank]
        local = values[start:end].clone().requires_grad_()
        all_rows = gather_rows(local)
        # Local queries may select neighbors on the other rank. Unequal rank
        # populations and condition counts must not change the reference gradient.
        terms = nearest_neighbor_terms(all_rows, excluded)
        (terms[start:end] * weights[start:end]).sum().backward()
        torch.testing.assert_close(local.grad, reference.grad[start:end])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["row_mean", "condition_mean"])
@pytest.mark.parametrize("exclude_condition", [False, True])
def test_cross_rank_nearest_neighbor_gradients_match_full_batch(
    tmp_path, strategy, exclude_condition
):
    mp.spawn(
        _neighbor_worker,
        args=(str(tmp_path / strategy), strategy, exclude_condition),
        nprocs=2,
        join=True,
    )
