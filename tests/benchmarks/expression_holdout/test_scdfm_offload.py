"""Saved tensor storage is a compute policy, not a changed loss or update budget."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from benchmarks.expression_holdout.scdfm import (
    _pack_saved_tensor_layout,
    _training_activation_context,
    _unpack_saved_tensor_layout,
    resolve_training_budget,
)


def test_cpu_and_default_policy_do_not_offload(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("default/CPU path must not enter CUDA offload")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    for training, device in [({}, "cuda:0"), ({"activation_offload": "cpu"}, "cpu")]:
        with _training_activation_context(training, device):
            pass


def test_explicit_cuda_offload_uses_layout_preserving_hooks(monkeypatch):
    calls = []

    def offload(*args):
        calls.append(args)
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.autograd.graph, "saved_tensors_hooks", offload)
    with _training_activation_context({"activation_offload": "cpu"}, "cuda:0"):
        pass
    assert calls == [(_pack_saved_tensor_layout, _unpack_saved_tensor_layout)]


@pytest.mark.parametrize("kind", ["transpose", "slice", "broadcast", "empty", "scalar"])
def test_saved_activation_preserves_values_and_strides(kind):
    base = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    value = {
        "transpose": base.T,
        "slice": base[1::2, 2::2],
        "broadcast": base[:1].expand(5, 8),
        "empty": base[:0],
        "scalar": base[2, 3],
    }[kind]
    restored = _unpack_saved_tensor_layout(_pack_saved_tensor_layout(value))
    assert restored.shape == value.shape and restored.stride() == value.stride()
    assert restored.storage_offset() == value.storage_offset()
    torch.testing.assert_close(restored, value, rtol=0, atol=0)


def test_layout_hooks_preserve_multistep_optimizer_and_gradients():
    def run(hooked):
        torch.manual_seed(2)
        weights = torch.nn.Parameter(torch.randn(8, 6))
        opt = torch.optim.Adam([weights], lr=0.001)
        inputs = torch.arange(48, dtype=torch.float32).reshape(6, 8)[::2]
        for _ in range(3):
            opt.zero_grad()
            context = (
                torch.autograd.graph.saved_tensors_hooks(
                    _pack_saved_tensor_layout, _unpack_saved_tensor_layout
                )
                if hooked
                else contextlib.nullcontext()
            )
            with context:
                loss = (inputs @ weights).square().mean()
            loss.backward()
            opt.step()
        return weights.detach(), weights.grad, opt.state[weights]

    left, right = run(False), run(True)
    for a, b in zip(left[:2], right[:2], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in left[2]:
        torch.testing.assert_close(left[2][key], right[2][key], rtol=0, atol=0)


def test_offload_keeps_training_budget_and_rejects_unsupported_policies():
    data = SimpleNamespace(train_row_ids=tuple(range(47836)))
    _, baseline = resolve_training_budget(data, {"epochs": 4, "batch_size": 48})
    _, actual = resolve_training_budget(
        data, {"epochs": 4, "batch_size": 48, "activation_offload": "cpu"}
    )
    assert actual == baseline
    assert actual["total_optimizer_updates"] == 3988
    with pytest.raises(ValueError, match="activation_offload"):
        resolve_training_budget(data, {"epochs": 4, "activation_offload": "guess"})
