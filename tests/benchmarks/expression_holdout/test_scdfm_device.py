"""Device alias resolution must not weaken the external runner isolation guard."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.expression_holdout.scdfm import _require_accelerator_device


@pytest.mark.parametrize(
    ("requested", "actual", "current", "resolved"),
    [
        ("cuda:0", "cuda", 0, "cuda:0"),
        ("cuda:1", "cuda", 1, "cuda:1"),
        ("cuda", "cuda:1", 1, "cuda:1"),
        ("cuda", "cuda", 0, "cuda:0"),
    ],
)
def test_cuda_alias_resolves_current_device(monkeypatch, requested, actual, current, resolved):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    accelerator = SimpleNamespace(device=torch.device(actual), num_processes=1)
    assert _require_accelerator_device(accelerator, requested) == torch.device(resolved)


@pytest.mark.parametrize(
    ("requested", "actual", "current"),
    [("cuda:0", "cuda", 1), ("cuda:1", "cuda", 0), ("cuda", "cuda:1", 0)],
)
def test_cuda_alias_does_not_accept_another_gpu(monkeypatch, requested, actual, current):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    accelerator = SimpleNamespace(device=torch.device(actual), num_processes=1)
    with pytest.raises(ValueError, match="differs from requested isolated device"):
        _require_accelerator_device(accelerator, requested)


def test_explicit_devices_do_not_query_current_device(monkeypatch):
    def unexpected():
        raise AssertionError("must not query CUDA for already explicit devices")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    accelerator = SimpleNamespace(device=torch.device("cuda:0"), num_processes=1)
    assert _require_accelerator_device(accelerator, "cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="differs from requested isolated device"):
        _require_accelerator_device(accelerator, "cuda:1")


def test_cpu_and_device_type_mismatch_do_not_initialize_cuda(monkeypatch):
    def unexpected():
        raise AssertionError("must not initialize CUDA for CPU or type mismatch")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    cpu = SimpleNamespace(device=torch.device("cpu"), num_processes=1)
    gpu = SimpleNamespace(device=torch.device("cuda"), num_processes=1)
    assert _require_accelerator_device(cpu, "cpu") == torch.device("cpu")
    for accelerator, requested in [(cpu, "cuda:0"), (gpu, "cpu")]:
        with pytest.raises(ValueError, match="differs from requested isolated device"):
            _require_accelerator_device(accelerator, requested)


def test_multi_process_fails_before_device_resolution(monkeypatch):
    def unexpected():
        raise AssertionError("must reject multiple processes before CUDA resolution")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    accelerator = SimpleNamespace(device=torch.device("cuda"), num_processes=2)
    with pytest.raises(ValueError, match="single-device"):
        _require_accelerator_device(accelerator, "cuda:0")
