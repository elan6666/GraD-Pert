from datetime import timedelta

import pytest
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


def _three_row_batch():
    from dataclasses import replace

    from gradpert.training.v2.objective import CellView

    model, batch = fixture()
    rows = torch.tensor([0, 1, 0])
    masks = [torch.zeros((3, 4), dtype=torch.bool) for _ in batch.cell_views]
    masks[0][0, :2] = True
    masks[0][2, 3] = True
    masks[1][2, 2] = True
    return model, replace(
        batch,
        control=batch.control[rows],
        truth=batch.truth[rows],
        condition_index=batch.condition_index[rows],
        cell_views=tuple(
            CellView(v.positions, mask) for v, mask in zip(batch.cell_views, masks, strict=True)
        ),
    )


class _GradientCapture:
    def __init__(self, model):
        self.model = model

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self, lr):
        pass


def _step_worker(rank, rendezvous, reference, ssl2):
    from gradpert.training.v2.engine import optimizer_step, slice_cells

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model, batch = _three_row_batch()
        objective = JointObjective(
            model, lambda1=1, lambda2=0.1 if ssl2 else 0, ssl2_weights=(0.8, 0.4, 0)
        )
        local = slice_cells(batch, 0, 1) if rank == 0 else slice_cells(batch, 1, 3)
        metrics = optimizer_step(
            objective,
            _GradientCapture(model),
            local,
            microbatch=1,
            lr=0.001,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index,
        )
        expected = torch.load(reference, weights_only=True)
        for name, parameter in model.named_parameters():
            if expected["gradients"][name] is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected["gradients"][name], atol=3e-6, rtol=1e-4
                )
        assert abs(metrics["prediction"] - expected["prediction"]) < 1e-6
        if ssl2:
            assert abs(metrics["ssl2_ibot"] - expected["ibot"]) < 2e-6
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ssl2", [False, True])
def test_distributed_step_matches_global_additive_gradients_for_unequal_rows(tmp_path, ssl2):
    from gradpert.training.v2.engine import optimizer_step

    model, batch = _three_row_batch()
    objective = JointObjective(
        model, lambda1=1, lambda2=0.1 if ssl2 else 0, ssl2_weights=(0.8, 0.4, 0)
    )
    metrics = optimizer_step(
        objective, _GradientCapture(model), batch, microbatch=3, lr=0.001, momentum=0.99, bf16=False
    )
    reference = str(tmp_path / "reference.pt")
    torch.save(
        {
            "gradients": {name: p.grad for name, p in model.named_parameters()},
            "prediction": metrics["prediction"],
            "ibot": metrics.get("ssl2_ibot"),
        },
        reference,
    )
    mp.spawn(
        _step_worker, args=(str(tmp_path / "step-rendezvous"), reference, ssl2), nprocs=2, join=True
    )


def _checkpoint_worker(rank, rendezvous, checkpoint):
    import random
    from pathlib import Path

    import numpy as np

    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.optimizer import V2Optimizer

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
        optimizer = V2Optimizer(model, 0.001, 0.0)
        generator = np.random.default_rng(100 + rank)
        torch.manual_seed(200 + rank)
        random.seed(300 + rank)
        identity = {"world_size": 2}
        save_checkpoint(
            Path(checkpoint),
            objective,
            optimizer,
            identity=identity,
            progress={"step": 0},
            generator=generator,
        )
        expected = (torch.rand(4), generator.random(4), random.random())
        torch.rand(12)
        generator.random(12)
        random.random()
        progress = load_checkpoint(
            Path(checkpoint), objective, optimizer, identity=identity, generator=generator
        )
        assert progress == {"step": 0}
        torch.testing.assert_close(torch.rand(4), expected[0], rtol=0, atol=0)
        np.testing.assert_array_equal(generator.random(4), expected[1])
        assert random.random() == expected[2]
    finally:
        dist.destroy_process_group()


def test_distributed_checkpoint_restores_each_ranks_distinct_rng(tmp_path):
    mp.spawn(
        _checkpoint_worker,
        args=(str(tmp_path / "checkpoint-rendezvous"), str(tmp_path / "distributed.pt")),
        nprocs=2,
        join=True,
    )
