"""Complete effective-population wiring, RI parity, clock and DDP derivatives."""

import copy
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_koleo_accumulation import fixture as ri_fixture
from test_unified_integration import PARENT

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.diagnostic_objective import diagnostic_objective
from gradpert.training.v2.engine import optimizer_step, slice_cells
from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.process_diagnostics import _read_only
from gradpert.training.v2.reductions import gather_rows
from gradpert.training.v2.spectral_pair import spectral_pair


def fixture(*, enabled=True, checkpointed=False):
    original, batch = ri_fixture(weight=0, checkpointed=checkpointed)
    original.student.options = replace(original.student.options, random_gene_order=False)
    obj = JointObjective(
        original.student,
        lambda1=0,
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        spectral_rank_target=0.35,
        spectral_rank_weight=0.04 if enabled else 0,
        spectral_cosine_weight=0.20 if enabled else 0,
    )
    return obj, batch


def step(obj, batch, *, microbatch, steps=30):
    opt = V2Optimizer(obj.student, 1e-3, 0.01)
    opt.steps = steps
    return optimizer_step(
        obj, opt, batch, microbatch=microbatch, lr=1e-3, momentum=0.99, bf16=False
    ), opt


@pytest.mark.parametrize("checkpointed", [False, True])
def test_fullbatch_and_accumulation_have_same_auxiliary_values_and_gradients(
    monkeypatch, checkpointed
):
    original, batch = fixture(checkpointed=checkpointed)
    actual = copy.deepcopy(original)
    captured = []
    clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    full, _ = step(original, batch, microbatch=len(batch.control))
    accumulated, _ = step(actual, batch, microbatch=2)
    for name in full:
        assert accumulated[name] == pytest.approx(full[name], rel=2e-4, abs=3e-6)
    for g, h in zip(*captured, strict=True):
        if g is None:
            assert h is None
        else:
            torch.testing.assert_close(g, h, rtol=2e-3, atol=1e-5)
    flat = [torch.cat([g.flatten() for g in gradients if g is not None]) for gradients in captured]
    assert (flat[0] - flat[1]).norm() / flat[0].norm() < 2e-4
    assert full["spectral_input_count"] == 5
    assert full["spectral_warmup"] == 1
    assert full["joint_loss"] == pytest.approx(
        full["joint_loss_without_population_aux"]
        + full["spectral_rank_weighted"]
        + full["spectral_cosine_weighted"]
    )
    assert not actual.pending


def test_zero_new_weights_preserve_original_ri_update_rng_and_metric_keys():
    baseline, batch = fixture(enabled=False)
    explicit = copy.deepcopy(baseline)
    explicit.spectral_rank_target = 0.35
    explicit.spectral_softness = 0.25
    torch.manual_seed(179)
    expected, opt1 = step(baseline, batch, microbatch=2, steps=0)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(179)
    actual, opt2 = step(explicit, batch, microbatch=2, steps=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert expected == actual and not any(k.startswith("spectral") for k in actual)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    assert opt1.steps == opt2.steps == 1


@pytest.mark.parametrize("steps,factor", [(0, 0), (1, 1 / 30), (29, 29 / 30), (30, 1), (31, 1)])
def test_warmup_counts_completed_successful_updates(steps, factor):
    obj, batch = fixture()
    result, opt = step(obj, batch, microbatch=2, steps=steps)
    assert result["spectral_warmup"] == pytest.approx(factor)
    assert result["spectral_rank_weighted"] == pytest.approx(
        result["spectral_rank"] * 0.04 * factor
    )
    assert result["spectral_cosine_weighted"] == pytest.approx(
        result["spectral_cosine"] * 0.20 * factor
    )
    assert opt.steps == steps + 1


def test_checkpoint_restore_retains_warmup_clock_and_exact_next_update(tmp_path):
    obj, batch = fixture(checkpointed=True)
    _, optimizer = step(obj, batch, microbatch=2, steps=28)
    generator = np.random.default_rng(71)
    path = tmp_path / "synthetic.pt"
    identity = {"synthetic": True, "loss_protocol": "spectral_pair"}
    save_checkpoint(
        path, obj, optimizer, identity=identity, progress={"step": 29}, generator=generator
    )
    expected = optimizer_step(
        obj, optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
    )
    resumed, _ = fixture(checkpointed=True)
    resumed_optimizer = V2Optimizer(resumed.student, 1e-3, 0.01)
    load_checkpoint(path, resumed, resumed_optimizer, identity=identity, generator=generator)
    assert resumed_optimizer.steps == 29
    actual = optimizer_step(
        resumed, resumed_optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
    )
    assert actual == expected and actual["spectral_warmup"] == 29 / 30
    assert resumed_optimizer.steps == 30
    for key, value in obj.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_partial_diagnostics_label_missing_terms_and_leave_state_rng_unchanged():
    obj, batch = fixture()
    state = copy.deepcopy(obj.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="complete-population"):
        obj(batch)
    with _read_only(obj, batch, 431):
        loss, terms, scope = diagnostic_objective(obj, batch)
        assert torch.isfinite(loss)
        assert "spectral_rank" not in terms
        assert scope["diagnostic_objective_scope"]["unavailable_terms"] == {
            "spectral_rank": "requires_complete_global_population",
            "spectral_cosine": "requires_complete_global_population",
        }
    result = evaluate_loss_diagnostics(obj, batch, bf16=False)
    assert result["unavailable_weighted_gradient_l2"]["spectral_rank"] is None
    assert torch.equal(rng, torch.get_rng_state())
    assert not obj.pending and all(p.grad is None for p in obj.parameters())
    for key, value in state.items():
        torch.testing.assert_close(value, obj.state_dict()[key], rtol=0, atol=0)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(231)
    inputs = torch.randn(8, 5, dtype=torch.float64)
    other = torch.randn(8, 5, dtype=torch.float64)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    reference = spectral_pair(inputs @ weight, other @ weight, target=0.35)
    expected = torch.autograd.grad(0.04 * reference.rank + 0.2 * reference.cosine, weight)[0]
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        local = weight.detach().clone().requires_grad_()
        selection = slice(0, 3) if rank == 0 else slice(3, 8)
        # Multiple local microbatches concatenate before the cross-rank gather.
        first = gather_rows(torch.cat([chunk @ local for chunk in inputs[selection].split(2)]))
        second = gather_rows(torch.cat([chunk @ local for chunk in other[selection].split(2)]))
        result = spectral_pair(first, second, target=0.35)
        (0.04 * result.rank + 0.2 * result.cosine).backward()
        dist.all_reduce(local.grad)
        local.grad /= 2
        torch.testing.assert_close(local.grad, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(result.rho, reference.rho)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_uneven_microbatches_match_fullpopulation_parameter_gradient(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def _distributed_engine_worker(rank, rendezvous):
    torch.set_num_threads(1)
    reference, batch = fixture(checkpointed=True)
    actual = copy.deepcopy(reference)
    captured = []
    clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return clip(parameters, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
        expected, _ = step(reference, batch, microbatch=2)
        dist.init_process_group(
            "gloo",
            init_method="file://" + rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=60),
        )
        try:
            local = slice_cells(batch, 0, 2) if rank == 0 else slice_cells(batch, 2, 5)
            opt = V2Optimizer(actual.student, 1e-3, 0.01)
            opt.steps = 30
            result = optimizer_step(
                actual,
                opt,
                local,
                microbatch=2,
                lr=1e-3,
                momentum=0.99,
                bf16=False,
                global_condition_index=batch.condition_index,
            )
            for name in expected:
                assert result[name] == pytest.approx(expected[name], rel=2e-4, abs=3e-6)
            for first, second in zip(*captured, strict=True):
                if first is None:
                    assert second is None
                else:
                    torch.testing.assert_close(first, second, rtol=2e-3, atol=1e-5)
            assert not actual.pending
        finally:
            dist.destroy_process_group()


def test_complete_engine_two_rank_uneven_shards_match_single_rank_gradient(tmp_path):
    mp.spawn(_distributed_engine_worker, args=(str(tmp_path / "engine-gloo"),), nprocs=2, join=True)


def test_collection_retains_only_explicit_unique_control_condition_rows():
    objective, batch = fixture()
    batch = replace(batch, unique_cell_indices=torch.tensor([0, 1, 3]))
    result, _ = step(objective, batch, microbatch=2)
    assert result["spectral_input_count"] == 3
    assert result["spectral_max_rank"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"au_alignment_weight": 0.1},
        {"au_uniformity_weight": 0.01},
        {"cr_weight": 0.1},
        {"ssl2_weights": (1, 1, 0.1)},
        {"population_aux_space": "ssl2_cls_bottleneck"},
    ],
)
def test_new_pair_rejects_conflicting_regularizers_and_projector_space(kwargs):
    obj, _ = fixture(enabled=False)
    options = dict(loss_reduction="row_mean", ssl2_weights=(1, 1, 0), spectral_rank_weight=0.02)
    options.update(kwargs)
    with pytest.raises(ValueError, match="spectral pair"):
        JointObjective(obj.student, **options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spectral_rank_weight": True},
        {"spectral_rank_weight": float("nan")},
        {"spectral_cosine_weight": -0.1},
        {"spectral_rank_target": 0},
        {"spectral_rank_target": 1.1},
        {"spectral_softness": 0.3},
        {"spectral_warmup_steps": 30.0},
        {"spectral_warmup_steps": -1},
    ],
)
def test_strict_config_and_direct_objective_validate_new_numeric_controls(kwargs):
    _, options = V2Options.parse_parameters(load_experiment_config(PARENT).model.parameters)
    with pytest.raises(ValueError, match="spectral"):
        replace(options, **kwargs)
    obj, _ = fixture(enabled=False)
    with pytest.raises(ValueError, match="spectral"):
        JointObjective(obj.student, **kwargs)
