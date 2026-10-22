"""Global KoLeo accumulation frees loss branches without changing update semantics."""

import copy
import gc
import math
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_restored_ffn import restored

from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2 import engine
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def fixture(strategy="row_mean", *, checkpointed=False, dropout=0.0, weight=0.1, residual=True):
    base, batch = restored("BR")
    # Two Global and two Local views; an uneven five-row condition population.
    selection = torch.tensor([0, 2, 0, 1, 3])
    batch = replace(
        batch,
        control=batch.control[selection],
        truth=batch.truth[selection],
        condition_index=batch.condition_index[selection],
        unique_cell_indices=None,
        cell_views=tuple(replace(v, mask=v.mask[selection]) for v in batch.cell_views),
    )
    base.student = GraDPertV2(
        base.student.graph.embedding.weight.detach().clone(),
        replace(
            base.student.options,
            response_injection_residual=residual,
            ffn_variant="gelu" if residual else "mixed",
        ),
    )
    for module in base.student.modules():
        if hasattr(module, "checkpoint_layers"):
            module.checkpoint_layers = checkpointed
        if hasattr(module, "checkpoint_chunks"):
            module.checkpoint_chunks = checkpointed
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout
    obj = JointObjective(
        base.student,
        loss_reduction=strategy,
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, weight),
        koleo_exclude_same_condition=True,
    )
    assert len(batch.cell_views) == 4
    return obj, batch


@contextmanager
def combined_backward_reference(monkeypatch, micro_count):
    """Reproduce the old combined cell + KoLeo backward, with identical forwards."""
    original = torch.Tensor.backward
    losses = []
    calls = 0

    def backward(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= micro_count:
            assert kwargs == {"retain_graph": True}
            losses.append(value)
            return None
        if calls == micro_count + 1:
            value = torch.stack(losses).sum() + value
            losses.clear()
        return original(value, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "backward", backward)
        yield
    assert calls == micro_count + 2  # cell micros, global KoLeo, once-only SSL1
    assert not losses


def assert_nested_close(actual, expected, *, atol=2e-6):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_close(actual[key], expected[key], atol=atol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_nested_close(a, e, atol=atol)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-4, abs=atol)
    else:
        assert actual == expected


@pytest.mark.parametrize("strategy", ["row_mean", "condition_mean"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("checkpointed,dropout", [(False, 0.0), (True, 0.1)])
def test_split_backward_matches_combined_fp32_gradients_updates_and_rng(
    monkeypatch, strategy, checkpointed, dropout, residual
):
    original, batch = fixture(
        strategy, checkpointed=checkpointed, dropout=dropout, residual=residual
    )
    actual = copy.deepcopy(original)
    opt_ref = V2Optimizer(original.student, 1e-3, 0.01)
    opt_actual = V2Optimizer(actual.student, 1e-3, 0.01)
    gradients = []
    clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    # First microbatch contains only condition 0: valid neighbors must come from
    # later microbatches. Check fresh and populated optimizer moments from the
    # exact same starting checkpoint; floating-point trajectories need not be
    # bitwise identical after changing gradient accumulation order.
    for seed in (411, 412):
        actual.load_state_dict(original.state_dict())
        opt_actual.load_state_dict(copy.deepcopy(opt_ref.state_dict()))
        torch.manual_seed(seed)
        with combined_backward_reference(monkeypatch, math.ceil(len(batch.control) / 2)):
            expected = engine.optimizer_step(
                original, opt_ref, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
            )
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        result = engine.optimizer_step(
            actual, opt_actual, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
        )
        assert torch.equal(reference_rng, torch.get_rng_state())
        assert_nested_close(result, expected)
        assert_nested_close(gradients[-1], gradients[-2], atol=2e-6 if residual else 2e-5)
        # FP32 addition order perturbs nearly cancelled gradients. Muon/Adam
        # normalize these tiny values: the deliberately collapsed non-RI toy
        # can amplify <1e-8 gradient changes to <1e-4 parameter changes. The
        # current RI arm has a tighter update bound. Neither is bitwise parity.
        state_atol = 2e-5 if residual else 1e-4
        assert_nested_close(actual.state_dict(), original.state_dict(), atol=state_atol)
        assert_nested_close(opt_actual.state_dict(), opt_ref.state_dict(), atol=state_atol)
        assert not actual.pending and not original.pending


class SavedTensor:
    def __init__(self, value):
        self.value = value.detach()


def live_saved_tensors(obj, batch, monkeypatch, *, combined):
    refs = {"head": [], "ancestor": []}
    scope = ["ancestor"]
    snapshots = []

    def pack(value):
        packed = SavedTensor(value)
        refs[scope[0]].append(weakref.ref(packed))
        return packed

    def before_head(*_):
        scope[0] = "head"

    def after_head(*_):
        scope[0] = "ancestor"

    def before_micro(*_):
        gc.collect()
        snapshots.append(
            {key: sum(r() is not None for r in values) for key, values in refs.items()}
        )

    handles = [obj.register_forward_pre_hook(before_micro)]
    for head in (obj.student.ssl2_cls, obj.student.ssl2_node):
        handles.extend(
            [head.register_forward_pre_hook(before_head), head.register_forward_hook(after_head)]
        )
    optimizer = V2Optimizer(obj.student, 1e-3, 0.01)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.value):
            if combined:
                with combined_backward_reference(monkeypatch, 3):
                    engine.optimizer_step(
                        obj, optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
                    )
            else:
                engine.optimizer_step(
                    obj, optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
                )
    finally:
        for handle in handles:
            handle.remove()
    gc.collect()
    assert all(r() is None for values in refs.values() for r in values)
    return snapshots


@pytest.mark.parametrize("checkpointed", [False, True])
def test_saved_projection_tensors_are_released_between_micros(monkeypatch, checkpointed):
    obj, batch = fixture(checkpointed=checkpointed, dropout=0.1)
    old = live_saved_tensors(copy.deepcopy(obj), batch, monkeypatch, combined=True)
    new = live_saved_tensors(obj, batch, monkeypatch, combined=False)
    assert len(new) == len(old) == 3
    assert all(row["head"] > 0 for row in old[1:])
    assert all(row["head"] == 0 for row in new[1:])
    # Retained CLS ancestors still exist and are actually used in global KoLeo.
    assert all(row["ancestor"] > 0 for row in new[1:])


@pytest.mark.parametrize("weight,microbatch", [(0.0, 2), (0.1, 5)])
def test_disabled_and_single_micro_do_not_retain_graph(monkeypatch, weight, microbatch):
    obj, batch = fixture(weight=weight)
    original = torch.Tensor.backward
    calls = []

    def backward(value, *args, **kwargs):
        calls.append(kwargs)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    engine.optimizer_step(
        obj,
        V2Optimizer(obj.student, 1e-3, 0.01),
        batch,
        microbatch=microbatch,
        lr=1e-3,
        momentum=0.99,
        bf16=False,
    )
    assert all(not call.get("retain_graph", False) for call in calls)


def _distributed_worker(rank, rendezvous, strategy):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        obj, batch = fixture(strategy, checkpointed=True, dropout=0.1)
        batch = replace(batch, control=batch.control + 0.13 * rank)
        reference = copy.deepcopy(obj)
        opt = V2Optimizer(obj.student, 1e-3, 0.01)
        ref_opt = V2Optimizer(reference.student, 1e-3, 0.01)
        kwargs = dict(
            microbatch=2,
            lr=1e-3,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index.repeat(2),
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            torch.manual_seed(901 + rank)
            with combined_backward_reference(monkeypatch, 3):
                expected = engine.optimizer_step(reference, ref_opt, batch, **kwargs)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(901 + rank)
            actual = engine.optimizer_step(obj, opt, batch, **kwargs)
        assert torch.equal(rng, torch.get_rng_state())
        assert_nested_close(actual, expected)
        assert_nested_close(obj.state_dict(), reference.state_dict(), atol=2e-5)
        assert_nested_close(opt.state_dict(), ref_opt.state_dict(), atol=2e-5)
        assert not obj.pending
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["row_mean", "condition_mean"])
def test_two_rank_global_neighbor_backward_preserves_update(tmp_path, strategy):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo"), strategy), nprocs=2, join=True)


def test_three_uninterrupted_checkpointed_updates_are_finite_and_commit_once():
    obj, batch = fixture(checkpointed=True, dropout=0.1)
    optimizer = V2Optimizer(obj.student, 1e-3, 0.01)
    for index in range(3):
        before_teacher = copy.deepcopy(obj.teacher.state_dict())
        result = engine.optimizer_step(
            obj, optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
        )
        assert optimizer.steps == index + 1
        assert not obj.pending
        assert all(math.isfinite(value) for value in result.values())
        for name, value in obj.teacher.state_dict().items():
            torch.testing.assert_close(
                value,
                before_teacher[name] * 0.99 + obj.student.state_dict()[name] * 0.01,
                rtol=1e-6,
                atol=1e-7,
            )
        assert all(torch.isfinite(value).all() for value in obj.state_dict().values())
