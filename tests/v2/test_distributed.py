from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_components import fixture

from gradpert.training.v2.objective import JointObjective


def _center_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model, _ = fixture()
        objective = JointObjective(model)
        objective.ssl2_node_center.fill_(3.0)
        if rank == 0:
            objective._targets("ssl1_node", torch.full((2, model.options.prototypes), 2.0))
            objective._targets("ssl1_cls", torch.full((2, model.options.prototypes), 2.0))
        else:
            # No SSL1 node targets on rank 1, and unequal CLS population sizes.
            objective._targets("ssl1_cls", torch.full((1, model.options.prototypes), 5.0))
        objective.commit_statistics(0.99)
        torch.testing.assert_close(
            objective.ssl1_node_center, torch.full_like(objective.ssl1_node_center, 0.2)
        )
        torch.testing.assert_close(
            objective.ssl1_cls_center, torch.full_like(objective.ssl1_cls_center, 0.3)
        )
        torch.testing.assert_close(
            objective.ssl2_node_center, torch.full_like(objective.ssl2_node_center, 3.0)
        )
        assert not objective.pending
    finally:
        dist.destroy_process_group()


def test_two_rank_centers_handle_empty_local_masks_and_unequal_counts(tmp_path):
    mp.spawn(_center_worker, args=(str(tmp_path / "center-rendezvous"),), nprocs=2, join=True)


def _gradient_worker(rank, rendezvous):
    from gradpert.training.v2.distributed import average_gradients

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = torch.nn.Linear(3, 2)
        extra = torch.nn.Parameter(torch.ones(2))
        model.register_parameter("unused", extra)
        model.weight.grad = torch.full_like(model.weight, float(rank + 1))
        model.bias.grad = torch.full_like(model.bias, 4.0) if rank == 0 else None
        average_gradients(model, bucket_bytes=16)
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 1.5))
        torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 2.0))
        assert extra.grad is None
    finally:
        dist.destroy_process_group()


def test_two_rank_gradient_average_handles_missing_local_and_unused_global(tmp_path):
    mp.spawn(_gradient_worker, args=(str(tmp_path / "gradient-rendezvous"),), nprocs=2, join=True)
