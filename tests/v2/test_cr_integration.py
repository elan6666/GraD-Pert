"""CR reuses complete-population updates, restore and distributed scaling."""

import copy
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_au_integration import GradientReferenceOptimizer, _round2_options, step
from test_au_integration import fixture as au_fixture
from test_koleo_accumulation import assert_nested_close

from gradpert.training.v2.alignment_uniformity import negative_coding_rate
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.diagnostic_objective import diagnostic_objective
from gradpert.training.v2.engine import optimizer_step, slice_cells
from gradpert.training.v2.joint_validation import evaluate_joint_loss
from gradpert.training.v2.optimizer import V2Optimizer


def fixture(*, alignment=True, warmup=0, checkpointed=False):
    torch.set_num_threads(1)
    obj, batch = au_fixture(enabled=False, checkpointed=checkpointed, well_conditioned=True)
    with torch.no_grad():
        obj.student.control_cls.mul_(5)
        obj.student.response_cls.mul_(5)
        obj.teacher.load_state_dict(obj.student.state_dict())
    obj.cr_weight, obj.cr_epsilon_squared, obj.cr_warmup_steps = 0.00004, 0.05, warmup
    obj.cr_enabled = True
    obj.au_alignment_weight = 0.05 if alignment else 0
    obj.au_uniformity_weight = 0
    obj.au_enabled = alignment
    obj.au_warmup_steps = warmup
    # No eligible U pairs; CR and alignment remain well-defined.
    return obj, replace(batch, context_index=torch.arange(len(batch.control)))


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("alignment", [False, True])
def test_cr_full_microbatch_update_and_centers(checkpointed, alignment):
    whole, batch = fixture(checkpointed=checkpointed, alignment=alignment)
    chunks = copy.deepcopy(whole)
    first = GradientReferenceOptimizer(whole.student, 1e-3, 0)
    second = GradientReferenceOptimizer(chunks.student, 1e-3, 0)
    torch.manual_seed(800)
    expected = step(whole, first, batch, 5)
    torch.manual_seed(800)
    actual = step(chunks, second, batch, 2)
    assert_nested_close(actual, expected)
    assert_nested_close(chunks.state_dict(), whole.state_dict())
    for (_, a), (_, b) in zip(
        whole.student.named_parameters(), chunks.student.named_parameters(), strict=True
    ):
        assert_nested_close(a.grad, b.grad)
    assert "au_uniformity" not in actual
    assert actual["population_aux_weighted"] == pytest.approx(
        actual["cr_weighted"] + actual.get("au_weighted", 0)
    )
    assert actual["joint_loss"] == pytest.approx(
        actual["joint_loss_without_population_aux"] + actual["population_aux_weighted"]
    )


def test_cr_failed_step_warmup_and_exact_resume(tmp_path, monkeypatch):
    obj, batch = fixture(warmup=30)
    opt = V2Optimizer(obj.student, 1e-3, 0)
    before = copy.deepcopy(obj.state_dict())
    import gradpert.training.v2.alignment_uniformity as module

    with monkeypatch.context() as patch:

        def fail(*args):
            raise ValueError("planned CR failure")

        patch.setattr(module, "negative_coding_rate", fail)
        with pytest.raises(ValueError, match="planned CR"):
            step(obj, opt, batch, 2)
    assert opt.steps == 0 and not obj.pending
    assert_nested_close(before, obj.state_dict())
    assert step(obj, opt, batch, 2)["cr_warmup"] == 0
    assert step(obj, opt, batch, 2)["cr_warmup"] == pytest.approx(1 / 30)
    generator = np.random.default_rng(1)
    path = tmp_path / "cr.pt"
    save_checkpoint(
        path, obj, opt, identity={"cr": True}, progress={"step": 2}, generator=generator
    )
    expected = step(obj, opt, batch, 2)
    restored, _ = fixture(warmup=30)
    opt2 = V2Optimizer(restored.student, 1e-3, 0)
    load_checkpoint(path, restored, opt2, identity={"cr": True}, generator=generator)
    actual = step(restored, opt2, batch, 2)
    assert actual["cr_warmup"] == pytest.approx(2 / 30)
    assert_nested_close(actual, expected)
    assert_nested_close(restored.state_dict(), obj.state_dict())


def test_cr_only_partial_probe_and_joint_validation_scope():
    obj, batch = fixture(alignment=False)
    before = copy.deepcopy(obj.state_dict())
    loss, terms, scope = diagnostic_objective(obj, slice_cells(batch, 0, 1))
    assert torch.isfinite(loss) and "cr_negative" not in terms
    assert scope["diagnostic_objective_scope"]["unavailable_terms"] == {
        "cr_negative": "requires_complete_global_population"
    }
    obj.pending.clear()
    result = evaluate_joint_loss(obj, [(batch, "frozen")], bf16=False)
    assert "cr_negative" not in result["components"]
    assert (
        result["diagnostic_objective_scope"]["unavailable_terms"]
        == scope["diagnostic_objective_scope"]["unavailable_terms"]
    )
    assert_nested_close(before, obj.state_dict())


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
        obj, batch = fixture()
        opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
        local = slice_cells(batch, 0, 2) if rank == 0 else slice_cells(batch, 2, 5)
        actual = optimizer_step(
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
        assert_nested_close(actual, expected["metrics"])
        assert_nested_close(obj.state_dict(), expected["state"])
        for name, p in obj.student.named_parameters():
            assert_nested_close(p.grad, expected["gradients"][name])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("microbatch", [1, 5])
def test_cr_two_rank_uneven_population(tmp_path, microbatch):
    obj, batch = fixture()
    opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
    metrics = step(obj, opt, batch, 5)
    path = str(tmp_path / "ref.pt")
    torch.save(
        dict(
            metrics=metrics,
            state=obj.state_dict(),
            gradients={name: p.grad for name, p in obj.student.named_parameters()},
        ),
        path,
    )
    mp.spawn(
        _distributed_worker, args=(str(tmp_path / "gloo"), path, microbatch), nprocs=2, join=True
    )


@pytest.mark.parametrize(
    "change",
    [
        {"cr_weight": -1},
        {"cr_weight": float("nan")},
        {"cr_epsilon_squared": 0.0025},
        {"cr_warmup_steps": True},
        {"lambda2": 0},
        {"max_conditions": 1},
    ],
)
def test_invalid_cr_config(change):
    _, options = _round2_options()
    with pytest.raises(ValueError, match="CR"):
        replace(options, **{"cr_weight": 0.00004, **change})


def test_cr_shape_finite_and_cholesky_failure():
    with pytest.raises(ValueError):
        negative_coding_rate(torch.ones(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError):
        negative_coding_rate(torch.full((2, 3), float("nan")), torch.ones(2, 3))


def _unclipped_cr_worker(rank, rendezvous):
    from gradpert.training.v2.distributed import average_gradients
    from gradpert.training.v2.reductions import gather_rows

    torch.set_num_threads(1)
    torch.manual_seed(85)
    model = torch.nn.Linear(5, 8, bias=False)
    reference = copy.deepcopy(model)
    x = [torch.randn(5, 5), torch.randn(5, 5)]
    expected = negative_coding_rate(*(reference(v) for v in x))
    expected.backward()
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        rows = range(0, 2) if rank == 0 else range(2, 5)
        local = [torch.cat([model(v[i : i + 1]) for i in rows]) for v in x]
        actual = negative_coding_rate(*(gather_rows(v) for v in local))
        actual.backward()
        average_gradients(model)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, atol=2e-7, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_unclipped_cr_global_gradient_scale(tmp_path):
    mp.spawn(_unclipped_cr_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_cr_constructor_defaults_preserve_state_and_validate_options():
    from gradpert.training.v2.objective import JointObjective

    obj, _ = au_fixture(enabled=False)
    fresh = JointObjective(
        copy.deepcopy(obj.student),
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
    )
    assert not fresh.cr_enabled and fresh.cr_weight == 0
    assert fresh.state_dict().keys() == obj.state_dict().keys()
    _, options = _round2_options()
    assert options.cr_weight == 0
    for change in (
        {"cr_weight": float("inf")},
        {"cr_epsilon_squared": 0.0025},
        {"cr_warmup_steps": True},
        {"lambda2": 0},
    ):
        with pytest.raises(ValueError, match="CR"):
            JointObjective(
                copy.deepcopy(obj.student),
                loss_reduction="row_mean",
                ssl2_weights=(1, 1, 0),
                **{"cr_weight": 0.00004, **change},
            )


def test_alignment_cr_partial_scope_omits_disabled_uniformity():
    obj, batch = fixture(alignment=True)
    _, _, scope = diagnostic_objective(obj, slice_cells(batch, 0, 1))
    assert scope["diagnostic_objective_scope"]["unavailable_terms"] == {
        "au_alignment": "requires_complete_global_population",
        "cr_negative": "requires_complete_global_population",
    }
    obj.pending.clear()
    result = evaluate_joint_loss(obj, [(batch, "frozen")], bf16=False)
    assert result["diagnostic_objective_scope"] == scope["diagnostic_objective_scope"]
