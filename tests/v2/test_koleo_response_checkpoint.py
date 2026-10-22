"""Outer Global-response checkpointing preserves RNG, statistics and KoLeo gradients."""

import copy
import gc
import weakref
from dataclasses import replace

import pytest
import torch
from test_koleo_accumulation import SavedTensor, assert_nested_close
from test_koleo_accumulation import fixture as accumulation_fixture

from gradpert.training.v2 import objective as objective_module
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer


def fixture(*, checkpointed=True, weight=0.1):
    obj, batch = accumulation_fixture(checkpointed=checkpointed, dropout=0.1, weight=weight)
    # The accumulation fixture can independently disable individual checkpoint
    # modules; this test also controls the architecture's outer-checkpoint gate.
    obj.student.options = replace(obj.student.options, checkpoint_layers=checkpointed)
    obj.teacher.options = obj.student.options
    return obj, batch


def disable_outer(monkeypatch, obj):
    original = obj.cell_loss

    def cell_loss(*args, **kwargs):
        kwargs["checkpoint_deferred_response"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, "cell_loss", cell_loss)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_nested_two_backwards_match_existing_split_path_and_commit_statistics_once(
    monkeypatch, checkpointed
):
    obj, batch = fixture(checkpointed=checkpointed)
    reference = copy.deepcopy(obj)
    disable_outer(monkeypatch, reference)
    opt = V2Optimizer(obj.student, 1e-3, 0.01)
    ref_opt = V2Optimizer(reference.student, 1e-3, 0.01)
    captures = []
    clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captures.append([None if p.grad is None else p.grad.clone() for p in parameters])
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    counts = {"actual": [], "reference": []}
    for name, objective in (("actual", obj), ("reference", reference)):
        targets = objective._targets

        def record(key, logits, *, target=targets, owner=name):
            counts[owner].append((key, tuple(logits.shape)))
            target(key, logits)

        monkeypatch.setattr(objective, "_targets", record)
    for seed in (351, 352):
        torch.manual_seed(seed)
        expected = optimizer_step(
            reference, ref_opt, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
        )
        rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        actual = optimizer_step(obj, opt, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False)
        assert torch.equal(rng, torch.get_rng_state())
        assert_nested_close(actual, expected)
        assert_nested_close(captures[-1], captures[-2])
        assert_nested_close(obj.state_dict(), reference.state_dict(), atol=2e-5)
        assert_nested_close(opt.state_dict(), ref_opt.state_dict(), atol=2e-5)
        assert counts["actual"] == counts["reference"]
        assert not obj.pending and not reference.pending
        assert opt.steps == ref_opt.steps


@pytest.mark.parametrize(
    "checkpointed,weight,microbatch,expected_calls",
    [(True, 0.1, 2, 6), (False, 0.1, 2, 0), (True, 0.0, 2, 0), (True, 0.1, 5, 0)],
)
def test_only_deferred_global_student_views_use_outer_checkpoint(
    monkeypatch, checkpointed, weight, microbatch, expected_calls
):
    obj, batch = fixture(checkpointed=checkpointed, weight=weight)
    calls = []
    checkpoint = objective_module.checkpoint

    def record(function, *args, **kwargs):
        calls.append((function.__self__, kwargs.copy()))
        return checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(objective_module, "checkpoint", record)
    optimizer_step(
        obj,
        V2Optimizer(obj.student, 1e-3, 0.01),
        batch,
        microbatch=microbatch,
        lr=1e-3,
        momentum=0.99,
        bf16=False,
    )
    assert len(calls) == expected_calls
    assert all(owner is obj.student for owner, _ in calls)
    assert all(k["preserve_rng_state"] and not k["use_reentrant"] for _, k in calls)


def test_default_diagnostic_deferred_list_does_not_enable_outer_hooks(monkeypatch):
    obj, batch = fixture()
    monkeypatch.setattr(
        objective_module, "checkpoint", lambda *a, **k: pytest.fail("outer checkpoint used")
    )
    deferred = []
    loss, _ = obj(batch, include_ssl1=False, deferred_koleo=deferred)
    loss.backward()
    assert len(deferred) == 1
    obj.pending.clear()


def test_deferred_cls_clone_keeps_values_gradients_but_not_token_backing_storage():
    obj, batch = fixture()
    deferred = []
    loss, _ = obj(
        batch, include_ssl1=False, deferred_koleo=deferred, checkpoint_deferred_response=True
    )
    assert len(deferred) == 1
    for value in deferred[0][:2]:
        assert value.requires_grad and value.grad_fn is not None
        assert value.is_contiguous()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        assert value.shape == (len(batch.control), obj.student.options.width)
    loss.backward(retain_graph=True)
    sum(v.square().mean() for v in deferred[0][:2]).backward()
    assert obj.student.response_cls.grad is not None
    assert torch.isfinite(obj.student.response_cls.grad).all()
    obj.pending.clear()


def saved_storage_trace(obj, batch):
    refs = []
    snapshots = []

    def pack(value):
        packed = SavedTensor(value)
        refs.append(weakref.ref(packed))
        return packed

    def before_micro(*_):
        gc.collect()
        live = [r() for r in refs if r() is not None]
        storage = {
            x.value.untyped_storage().data_ptr(): x.value.untyped_storage().nbytes() for x in live
        }
        snapshots.append({"objects": len(live), "unique_storage_bytes": sum(storage.values())})

    handle = obj.register_forward_pre_hook(before_micro)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value.value):
            optimizer_step(
                obj,
                V2Optimizer(obj.student, 1e-3, 0.01),
                batch,
                microbatch=2,
                lr=1e-3,
                momentum=0.99,
                bf16=False,
            )
    finally:
        handle.remove()
    gc.collect()
    assert all(r() is None for r in refs)
    return snapshots


def test_outer_checkpoint_releases_layer_boundary_storage_between_micros(monkeypatch):
    obj, batch = fixture()
    reference = copy.deepcopy(obj)
    disable_outer(monkeypatch, reference)
    previous = saved_storage_trace(reference, batch)
    actual = saved_storage_trace(obj, batch)
    assert len(previous) == len(actual) == 3
    for old, new in zip(previous[1:], actual[1:], strict=True):
        assert new["objects"] < old["objects"]
        assert new["unique_storage_bytes"] < old["unique_storage_bytes"]
