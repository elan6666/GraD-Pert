"""Saved tensor storage is a compute policy, not a changed loss or update budget."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from benchmarks.expression_holdout.scdfm import (
    _training_activation_context,
    resolve_training_budget,
)


def test_cpu_and_default_policy_do_not_offload(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("default/CPU path must not enter CUDA offload")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    for training, device in [({}, "cuda:0"), ({"activation_offload": "cpu"}, "cpu")]:
        with _training_activation_context(training, device):
            pass


def test_explicit_cuda_offload_uses_pinned_saved_tensors(monkeypatch):
    calls = []

    def offload(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", offload)
    with _training_activation_context({"activation_offload": "cpu"}, "cuda:0"):
        pass
    assert calls == [{"pin_memory": True}]


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
