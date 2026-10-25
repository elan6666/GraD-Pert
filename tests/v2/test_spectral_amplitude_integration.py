"""Optional amplitude on native shared-parameter/global-population execution."""

import copy
import subprocess
import sys
import types
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_spectral_pair_integration import fixture as pair_fixture
from test_spectral_pair_integration import step
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
from gradpert.training.v2.spectral_pair import SpectralAmplitudePairResult, spectral_pair

BASE = "8f4fad3e7c9558b10e3a733be3e5e2fcf7046595"
ROOT = Path(__file__).resolve().parents[2]


def fixture(*, checkpointed=False, only_amplitude=False, weight=0.3):
    obj, batch = pair_fixture(checkpointed=checkpointed)
    return JointObjective(
        obj.student,
        lambda1=0,
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        spectral_rank_target=0.35,
        spectral_rank_weight=0 if only_amplitude else 0.04,
        spectral_cosine_weight=0 if only_amplitude else 0.20,
        spectral_amplitude_target=1.0,
        spectral_amplitude_weight=weight,
    ), batch


def capture_gradients(monkeypatch):
    captured = []
    clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    return captured


@pytest.mark.parametrize("checkpointed", [False, True])
def test_active_amplitude_native_fullbatch_and_micro_replay_shared_gradients(
    monkeypatch, checkpointed
):
    full, batch = fixture(checkpointed=checkpointed)
    micro = copy.deepcopy(full)
    captured = capture_gradients(monkeypatch)
    expected, _ = step(full, batch, microbatch=len(batch.control))
    actual, _ = step(micro, batch, microbatch=2)
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], rel=2e-4, abs=3e-6)
    for a, b in zip(*captured, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=2e-3, atol=1e-5)
    assert expected["spectral_amplitude"] > 0
    assert expected["spectral_amplitude_weighted"] == pytest.approx(
        0.3 * expected["spectral_amplitude"]
    )
    assert expected["joint_loss"] == pytest.approx(
        expected["joint_loss_without_population_aux"] + expected["spectral_weighted"]
    )
    assert not micro.pending


@pytest.mark.parametrize("completed,factor", [(0, 0), (1, 1 / 30), (29, 29 / 30), (30, 1)])
def test_amplitude_only_enables_same_collection_and_successful_update_clock(completed, factor):
    obj, batch = fixture(only_amplitude=True)
    batch = replace(batch, unique_cell_indices=torch.tensor([0, 1, 3]))
    assert obj.spectral_enabled
    result, opt = step(obj, batch, microbatch=2, steps=completed)
    assert result["spectral_input_count"] == 3 and result["spectral_max_rank"] == 2
    assert result["spectral_amplitude_weighted"] == pytest.approx(
        0.3 * factor * result["spectral_amplitude"]
    )
    assert result["spectral_weighted"] == result["spectral_amplitude_weighted"]
    assert opt.steps == completed + 1


def test_partial_probes_mark_amplitude_unavailable_and_preserve_state():
    obj, batch = fixture()
    state = copy.deepcopy(obj.state_dict())
    rng = torch.get_rng_state().clone()
    with _read_only(obj, batch, 37):
        _, terms, scope = diagnostic_objective(obj, batch)
        assert "spectral_amplitude" not in terms
        assert scope["diagnostic_objective_scope"]["unavailable_terms"]["spectral_amplitude"] == (
            "requires_complete_global_population"
        )
    diagnostics = evaluate_loss_diagnostics(obj, batch, bf16=False)
    assert diagnostics["unavailable_weighted_gradient_l2"]["spectral_amplitude"] is None
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in state.items():
        torch.testing.assert_close(value, obj.state_dict()[key], atol=0, rtol=0)


def test_active_amplitude_checkpoint_restore_preserves_exact_next_update(tmp_path):
    obj, batch = fixture(checkpointed=True)
    _, opt = step(obj, batch, microbatch=2, steps=28)
    identity = {"synthetic": True, "amplitude": {"target": 1.0, "weight": 0.3}}
    path = tmp_path / "amplitude.pt"
    generator = np.random.default_rng(11)
    save_checkpoint(path, obj, opt, identity=identity, progress={"step": 29}, generator=generator)
    expected = optimizer_step(obj, opt, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False)
    resumed, _ = fixture(checkpointed=True)
    restored = V2Optimizer(resumed.student, 1e-3, 0.01)
    load_checkpoint(path, resumed, restored, identity=identity, generator=generator)
    assert restored.steps == 29
    actual = optimizer_step(
        resumed, restored, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
    )
    assert actual == expected and actual["spectral_warmup"] == 29 / 30
    for key, value in obj.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)


def test_active_amplitude_adds_no_encoder_forward_or_rng_consumption():
    inactive, batch = fixture(checkpointed=True, weight=0)
    active = copy.deepcopy(inactive)
    active.spectral_amplitude_weight = 0.3
    counts = [0, 0]

    def count(index):
        def hook(*_):
            counts[index] += 1

        return hook

    first = inactive.student.response.register_forward_hook(count(0))
    second = active.student.response.register_forward_hook(count(1))
    try:
        torch.manual_seed(34)
        step(inactive, batch, microbatch=2)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(34)
        step(active, batch, microbatch=2)
        assert counts[0] == counts[1] and counts[0] > 0
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        first.remove()
        second.remove()


@pytest.mark.parametrize(
    "bad",
    [
        {"lambda2": 0},
        {"au_alignment_weight": 0.1},
        {"cr_weight": 0.1},
        {"population_response": True},
        {"population_aux_space": "ssl2_cls_bottleneck"},
    ],
)
def test_amplitude_only_rejects_incompatible_population_or_space(bad):
    obj, _ = pair_fixture(enabled=False)
    with pytest.raises(ValueError, match="spectral pair"):
        JointObjective(
            obj.student,
            loss_reduction="row_mean",
            ssl2_weights=(1, 1, 0),
            spectral_amplitude_weight=0.3,
            **bad,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spectral_amplitude_weight": True},
        {"spectral_amplitude_weight": float("nan")},
        {"spectral_amplitude_weight": -0.1},
        {"spectral_amplitude_target": True},
        {"spectral_amplitude_target": 0},
        {"spectral_amplitude_target": 1.01},
        {"spectral_amplitude_target": float("inf")},
    ],
)
def test_config_and_direct_objective_strict_amplitude_parameters(kwargs):
    _, options = V2Options.parse_parameters(load_experiment_config(PARENT).model.parameters)
    assert options.spectral_amplitude_weight == 0 and options.spectral_amplitude_target == 0.0225
    with pytest.raises(ValueError, match="spectral"):
        replace(options, **kwargs)
    obj, _ = pair_fixture(enabled=False)
    with pytest.raises(ValueError, match="spectral"):
        JointObjective(obj.student, **kwargs)


def published_module(name):
    """Read the published baseline Git object, independent of mutable worktree source."""
    module = types.ModuleType(f"gradpert.training.v2._published_{name}")
    module.__file__ = str(ROOT / f"src/gradpert/training/v2/{name}.py")
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/gradpert/training/v2/{name}.py"], cwd=ROOT, text=True
    )
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def test_disabled_amplitude_matches_published_spectral_fields_values_and_gradients_bitwise():
    old = published_module("spectral_pair")
    torch.manual_seed(49)
    x, y = [torch.randn(6, 4, dtype=torch.float64).requires_grad_() for _ in range(2)]
    expected = old.spectral_pair(x, y)
    actual = spectral_pair(x, y)
    assert vars(actual).keys() == vars(expected).keys()
    for key, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(vars(actual)[key], value, rtol=0, atol=0)
    old_grad = torch.autograd.grad(expected.rank + expected.cosine, (x, y))
    new_grad = torch.autograd.grad(actual.rank + actual.cosine, (x, y))
    for a, b in zip(old_grad, new_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_amplitude_native_update_rng_and_fields_match_published_8f(enabled):
    new, batch = pair_fixture(enabled=enabled, checkpointed=True)
    old_objective = published_module("objective")
    old_engine = published_module("engine")
    old = old_objective.JointObjective(
        copy.deepcopy(new.student),
        lambda1=0,
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        spectral_rank_target=0.35,
        spectral_rank_weight=0.04 if enabled else 0,
        spectral_cosine_weight=0.20 if enabled else 0,
    )
    opt_old, opt_new = [V2Optimizer(obj.student, 1e-3, 0.01) for obj in (old, new)]
    opt_old.steps = opt_new.steps = 29
    torch.manual_seed(73)
    expected = old_engine.optimizer_step(
        old, opt_old, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
    )
    rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    actual = optimizer_step(new, opt_new, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False)
    assert expected == actual and torch.equal(rng, torch.get_rng_state())
    assert not any(k.startswith(("spectral_amplitude", "spectral_q_")) for k in actual)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)


def test_actual_shared_parameters_receive_both_global_amplitude_gradients():
    obj, batch = fixture(checkpointed=True)
    collected = []
    for start in (0, 2, 4):
        obj(
            slice_cells(batch, start, min(start + 2, len(batch.control))),
            include_ssl1=False,
            deferred_au=collected,
            checkpoint_deferred_response=True,
        )
    first, second = [torch.cat([c[v] for c in collected]) for v in (0, 1)]
    result = spectral_pair(first, second, amplitude_target=1.0)
    assert isinstance(result, SpectralAmplitudePairResult)
    parameters = [p for n, p in obj.student.named_parameters() if n.startswith("response.")]
    combined = torch.autograd.grad(
        result.amplitude, parameters, retain_graph=True, allow_unused=True
    )
    branch_gradients = []
    for branch in (0, 1):
        only = spectral_pair(
            first if branch == 0 else first.detach(),
            second if branch == 1 else second.detach(),
            amplitude_target=1.0,
        )
        assert isinstance(only, SpectralAmplitudePairResult)
        branch_gradients.append(
            torch.autograd.grad(only.amplitude, parameters, retain_graph=True, allow_unused=True)
        )
    for branch in branch_gradients:
        assert any(g is not None and g.norm() > 0 for g in branch)
    for full, a, b in zip(combined, *branch_gradients, strict=True):
        if full is not None:
            summed = (torch.zeros_like(full) if a is None else a) + (
                torch.zeros_like(full) if b is None else b
            )
            torch.testing.assert_close(full, summed, rtol=3e-5, atol=2e-6)


def _distributed_amplitude_worker(rank, rendezvous):
    torch.set_num_threads(1)
    reference, batch = fixture(checkpointed=True)
    actual = copy.deepcopy(reference)
    with pytest.MonkeyPatch.context() as monkeypatch:
        captured = capture_gradients(monkeypatch)
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


def test_amplitude_full_engine_two_rank_uneven_population_matches_shared_parameter_gradients(
    tmp_path,
):
    mp.spawn(
        _distributed_amplitude_worker, args=(str(tmp_path / "amplitude-gloo"),), nprocs=2, join=True
    )
