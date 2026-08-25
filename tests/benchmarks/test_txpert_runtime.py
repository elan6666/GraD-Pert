from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.txpert.runtime import inspect_cuda_runtime, load_runtime_contract
from scripts.server.build_txpert_environment import filtered_requirements

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_contract_is_bound_to_frozen_official_lock(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lock_path = checkout / "uv.lock"
    lock_path.write_text("official-lock-fixture", encoding="utf-8")
    expected_lock_sha256 = json.loads(
        (ROOT / "benchmarks" / "environments" / "txpert-cu128.json").read_text(encoding="utf-8")
    )["official_checkout"]["lock_sha256"]
    monkeypatch.setattr(
        "benchmarks.txpert.runtime.sha256_file",
        lambda path: expected_lock_sha256 if path == lock_path else "unexpected",
    )
    contract_path, contract = load_runtime_contract(
        repository_root=ROOT,
        checkout_root=checkout,
    )

    assert contract_path == ROOT / "benchmarks" / "environments" / "txpert-cu128.json"
    override = contract["hardware_compatibility_override"]
    assert override["module_versions"]["torch"] == "2.7.0+cu128"
    assert override["required_cuda_arch"] == "sm_120"
    assert override["required_device_capability"] == [12, 0]


def test_runtime_contract_rejects_official_lock_drift(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="lock SHA-256"):
        load_runtime_contract(repository_root=ROOT, checkout_root=checkout)


def test_environment_export_replaces_only_declared_cuda_packages() -> None:
    exported = """
numpy==1.26.4
torch==2.6.0
torch-geometric==2.6.1
torch_scatter==2.1.2+pt26cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-ml-py==12.570.86
"""
    contract = json.loads(
        (ROOT / "benchmarks" / "environments" / "txpert-cu128.json").read_text(encoding="utf-8")
    )
    replaced = set(contract["hardware_compatibility_override"]["replaced_distributions"])

    assert filtered_requirements(exported, replaced=replaced) == (
        "numpy==1.26.4",
        "torch-geometric==2.6.1",
        "nvidia-ml-py==12.570.86",
    )


def test_runtime_gate_executes_core_and_pyg_cuda_kernels(monkeypatch) -> None:
    class FakeDevice:
        type = "cuda"

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_device(_device: FakeDevice) -> None:
            return None

        @staticmethod
        def get_device_capability(_device: FakeDevice) -> tuple[int, int]:
            return (12, 0)

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_90", "sm_120"]

        @staticmethod
        def synchronize(_device: FakeDevice) -> None:
            return None

        @staticmethod
        def get_device_name(_device: FakeDevice) -> str:
            return "NVIDIA GeForce RTX 5090"

    fake_torch = SimpleNamespace(
        __version__="2.7.0+cu128",
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="12.8"),
        float32=np.float32,
        long=np.int64,
        device=lambda _value: FakeDevice(),
        arange=lambda size, **_kwargs: np.arange(size, dtype=np.float32),
        cat=lambda arrays: np.concatenate(arrays),
        ones=lambda shape, **_kwargs: np.ones(shape, dtype=np.float32),
        tensor=lambda values, dtype, **_kwargs: np.asarray(values, dtype=dtype),
    )
    module_versions = {
        "torch": "2.7.0+cu128",
        "torchvision": "0.22.0+cu128",
        "torchaudio": "2.7.0+cu128",
        "torch_geometric": "2.6.1",
        "pyg_lib": "0.4.0+pt27cu128",
        "torch_scatter": "2.1.2+pt27cu128",
        "torch_sparse": "0.6.18+pt27cu128",
        "torch_spline_conv": "1.2.2+pt27cu128",
    }
    modules = {
        name: SimpleNamespace(__version__=version) for name, version in module_versions.items()
    }
    modules["torch"] = fake_torch
    modules["torch_scatter"].scatter_add = lambda source, _index: np.asarray(
        [source.sum()], dtype=np.float32
    )
    monkeypatch.setattr("benchmarks.txpert.runtime.platform.python_version", lambda: "3.12.9")
    monkeypatch.setattr(
        "benchmarks.txpert.runtime.importlib.import_module",
        lambda name: modules[name],
    )
    contract = {
        "hardware_compatibility_override": {
            "python": "3.12",
            "module_versions": module_versions,
            "cuda_runtime": "12.8",
            "required_device_capability": [12, 0],
            "required_cuda_arch": "sm_120",
        }
    }

    receipt = inspect_cuda_runtime(contract=contract, device="cuda:0")

    assert receipt["core_cuda_kernel_passed"] is True
    assert receipt["pyg_cuda_extension_passed"] is True
    assert receipt["wheel_architectures"] == ["sm_90", "sm_120"]
