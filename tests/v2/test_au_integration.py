"""Whole-population AU preserves update, partition and checkpoint semantics."""

import copy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_koleo_accumulation import assert_nested_close
from test_koleo_accumulation import fixture as base_fixture

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.alignment_uniformity import alignment_uniformity
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.distributed import average_gradients
from gradpert.training.v2.engine import optimizer_step, slice_cells
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.reductions import gather_rows
from gradpert.training.v2.runtime import Runtime


def fixture(*, enabled=True, warmup=0, checkpointed=False, well_conditioned=False):
    base, batch = base_fixture(weight=0, checkpointed=checkpointed)
    torch.manual_seed(471)
    student = GraDPertV2(
        base.student.graph.embedding.weight.detach().clone(),
        replace(base.student.options, random_gene_order=False, checkpoint_layers=checkpointed),
    )
    # Avoid a near-zero normalization input in this width-eight gradient oracle.
    # The production initialization is exercised by the calibration/preflight.
    if well_conditioned:
        with torch.no_grad():
            student.control_cls.mul_(5)
            student.response_cls.mul_(5)
    obj = JointObjective(
        student,
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        au_alignment_weight=0.3 if enabled else 0,
        au_uniformity_weight=0.2 if enabled else 0,
        au_warmup_steps=warmup,
    )
    return obj, replace(batch, context_index=torch.tensor([0, 0, 1, 0, 1]))


def step(obj, opt, batch, microbatch):
    return optimizer_step(
        obj, opt, batch, microbatch=microbatch, lr=1e-3, momentum=0.99, bf16=False
    )


class GradientReferenceOptimizer(V2Optimizer):
    """Use a linear update to expose partition errors without Muon amplification.

    Real Muon/Adam state and resume are exercised separately below. Tiny
    cancellation differences caused by differently sized GEMMs can be amplified
    by their normalization, so they are not a reliable gradient-scaling oracle.
    """

    @torch.no_grad()
    def step(self, lr):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-lr)
        self.steps += 1


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_population_matches_microbatch_gradients_update_and_centers(checkpointed):
    whole, batch = fixture(checkpointed=checkpointed, well_conditioned=True)
    chunks = copy.deepcopy(whole)
    first, second = (
        GradientReferenceOptimizer(whole.student, 1e-3, 0),
        GradientReferenceOptimizer(chunks.student, 1e-3, 0),
    )
    torch.manual_seed(800)
    expected = step(whole, first, batch, len(batch.control))
    torch.manual_seed(800)
    actual = step(chunks, second, batch, 2)
    assert_nested_close(actual, expected)
    for (_name, a), (_, b) in zip(
        whole.student.named_parameters(), chunks.student.named_parameters(), strict=True
    ):
        assert_nested_close(a.grad, b.grad)
    assert_nested_close(chunks.state_dict(), whole.state_dict())
    assert not chunks.pending
    assert all(p.grad is None for p in chunks.teacher.parameters())


def test_au_requires_explicit_population_and_partial_diagnostic_omission():
    obj, batch = fixture()
    with pytest.raises(ValueError, match="complete-population"):
        obj(batch)
    loss, metrics = obj(slice_cells(batch, 0, 1), omit_au_for_diagnostic=True)
    assert torch.isfinite(loss)
    assert not any(k.startswith("au_") for k in metrics)
    with pytest.raises(ValueError, match="cannot collect"):
        obj(batch, omit_au_for_diagnostic=True, deferred_au=[])


def test_unique_input_selection_and_context_follow_both_slices_and_views():
    obj, batch = fixture()
    batch = replace(batch, unique_cell_indices=torch.tensor([0, 2, 4]))
    collected = []
    for start, end in [(0, 2), (2, 5)]:
        part = slice_cells(batch, start, end)
        obj(part, deferred_au=collected)
    assert (
        torch.cat([x[2] for x in collected]).tolist() == batch.condition_index[[0, 2, 4]].tolist()
    )
    assert torch.cat([x[3] for x in collected]).tolist() == [0, 1, 1]
    assert all(a.requires_grad and b.requires_grad for a, b, _, _ in collected)


def test_zero_weights_keep_old_forward_and_update_exactly():
    obj, batch = fixture(enabled=False)
    reference = JointObjective(
        copy.deepcopy(obj.student),
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
    )
    first, second = V2Optimizer(obj.student, 1e-3, 0), V2Optimizer(reference.student, 1e-3, 0)
    torch.manual_seed(5)
    actual = step(obj, first, batch, 2)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(5)
    expected = step(reference, second, replace(batch, context_index=None), 2)
    assert actual == expected
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in obj.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key]), key


def test_successful_step_warmup_survives_checkpoint_resume(tmp_path):
    obj, batch = fixture(warmup=30)
    opt = V2Optimizer(obj.student, 1e-3, 0)
    assert step(obj, opt, batch, 2)["au_warmup"] == 0
    assert step(obj, opt, batch, 2)["au_warmup"] == pytest.approx(1 / 30)
    generator = np.random.default_rng(1)
    path = tmp_path / "state.pt"
    save_checkpoint(
        path, obj, opt, identity={"au": True}, progress={"step": 2}, generator=generator
    )
    expected = step(obj, opt, batch, 2)
    restored, _ = fixture(warmup=30)
    opt2 = V2Optimizer(restored.student, 1e-3, 0)
    load_checkpoint(path, restored, opt2, identity={"au": True}, generator=generator)
    actual = step(restored, opt2, batch, 2)
    assert actual["au_warmup"] == pytest.approx(2 / 30)
    assert_nested_close(actual, expected)
    assert_nested_close(obj.state_dict(), restored.state_dict())
    opt2.steps = 30
    assert step(restored, opt2, batch, 2)["au_warmup"] == 1


def test_failed_population_does_not_advance_optimizer_warmup():
    obj, batch = fixture(warmup=30)
    opt = V2Optimizer(obj.student, 1e-3, 0)
    batch = replace(batch, context_index=torch.arange(len(batch.control)))
    before = copy.deepcopy(obj.state_dict())
    with pytest.raises(ValueError):
        step(obj, opt, batch, 2)
    assert opt.steps == 0 and not obj.pending
    for key, value in before.items():
        assert torch.equal(obj.state_dict()[key], value)


def test_context_codes_use_control_identity_and_data_wide_stable_order():
    data = SimpleNamespace(
        context_ids=("T::B", "T::A", "T::B"), _row_index={"b": 0, "a": 1, "b2": 2}
    )
    runtime = Runtime(
        data=data,
        options=SimpleNamespace(au_alignment_weight=1, au_uniformity_weight=0),
        index=None,
        objective=None,
        optimizer=None,
        generator=None,
        device=None,
        batch_size=3,
        identity={},
    )
    raw = SimpleNamespace(control_row_ids=("b2", "a", "b"), control_expression=torch.zeros(3, 1))
    assert runtime.au_context_indices(raw).tolist() == [1, 0, 1]
    runtime.options = SimpleNamespace(au_alignment_weight=0, au_uniformity_weight=0)
    assert runtime.au_context_indices(raw) is None


def _distributed_worker(rank, rendezvous, path, microbatch):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        obj, batch = fixture(well_conditioned=True)
        opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
        local = slice_cells(batch, 0, 2) if rank == 0 else slice_cells(batch, 2, 5)
        result = optimizer_step(
            obj,
            opt,
            local,
            microbatch=microbatch,
            lr=1e-3,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index,
        )
        expected = torch.load(path, weights_only=True)
        assert_nested_close(result, expected["metrics"])
        assert_nested_close(obj.state_dict(), expected["state"])
        for name, parameter in obj.student.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(
                    parameter.grad, expected["gradients"][name], atol=2e-6, rtol=2e-4, msg=name
                )
            else:
                assert expected["gradients"][name] is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("microbatch", [1, 2, 5])
def test_two_rank_uneven_partitions_match_full_population(tmp_path, microbatch):
    obj, batch = fixture(well_conditioned=True)
    opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
    metrics = step(obj, opt, batch, 5)
    path = str(tmp_path / "reference.pt")
    torch.save(
        {
            "metrics": metrics,
            "state": obj.state_dict(),
            "gradients": {name: p.grad for name, p in obj.student.named_parameters()},
        },
        path,
    )
    mp.spawn(
        _distributed_worker, args=(str(tmp_path / "gloo"), path, microbatch), nprocs=2, join=True
    )


def _unclipped_gradient_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(85)
    model = torch.nn.Linear(5, 8, bias=False)
    reference = copy.deepcopy(model)
    x1, x2 = torch.randn(5, 5), torch.randn(5, 5)
    conditions, contexts = torch.tensor([0, 0, 1, 2, 3]), torch.tensor([0, 0, 0, 1, 1])
    result = alignment_uniformity(reference(x1), reference(x2), conditions, contexts)
    expected = 0.3 * result.alignment + 0.2 * result.uniformity
    expected.backward()
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        indices = range(0, 2) if rank == 0 else range(2, 5)
        # Forward one row at a time, gather across all accumulation pieces/ranks,
        # then differentiate the global scalar on each rank with no world factor.
        local = [torch.cat([model(x[i : i + 1]) for i in indices]) for x in (x1, x2)]
        first, second = (gather_rows(value) for value in local)
        actual = alignment_uniformity(first, second, conditions, contexts)
        loss = 0.3 * actual.alignment + 0.2 * actual.uniformity
        loss.backward()
        average_gradients(model)
        torch.testing.assert_close(loss, expected, atol=2e-7, rtol=2e-5)
        # No clipping, optimizer normalization or EMA can hide a scale error.
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, atol=2e-7, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_cross_rank_accumulation_unclipped_parameter_gradient_is_global_loss_gradient(tmp_path):
    mp.spawn(_unclipped_gradient_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def _round2_options():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/representation_round2_20261022/R2-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    return V2Options.parse_parameters(load_experiment_config(path).model.parameters)


def test_default_au_keys_do_not_enter_legacy_architecture_or_checkpoint_state():
    arch, options = _round2_options()
    assert (options.au_alignment_weight, options.au_uniformity_weight) == (0, 0)
    assert not any(key.startswith("au_") for key in arch.payload())
    obj, _ = fixture(enabled=False)
    old = JointObjective(
        copy.deepcopy(obj.student),
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
    )
    assert obj.state_dict().keys() == old.state_dict().keys()
    old.load_state_dict(obj.state_dict(), strict=True)


@pytest.mark.parametrize(
    "change",
    [
        {"au_alignment_weight": -0.1},
        {"au_uniformity_weight": float("nan")},
        {"au_temperature": 0},
        {"au_temperature": True},
        {"au_warmup_steps": 1.5},
        {"au_warmup_steps": -1},
        {"ssl2_koleo": 0.1},
        {"lambda2": 0},
        {"loss_reduction": "condition_mean"},
        {"max_conditions": 1},
    ],
)
def test_invalid_au_config_fails_before_training(change):
    _, options = _round2_options()
    with pytest.raises(ValueError, match="AU"):
        replace(options, **{"au_alignment_weight": 0.3, **change})
