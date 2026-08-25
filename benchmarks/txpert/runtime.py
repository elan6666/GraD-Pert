"""Validated hardware runtime contract for the isolated TxPert benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path
from typing import Any

from gradpert.hashing import sha256_file

RUNTIME_CONTRACT = Path("benchmarks/environments/txpert-cu128.json")


def load_runtime_contract(
    *, repository_root: Path, checkout_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Load the committed override and bind it to the frozen official lock."""

    contract_path = (repository_root.resolve(strict=True) / RUNTIME_CONTRACT).resolve(strict=True)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "txpert-cuda-runtime-v1":
        raise ValueError("unsupported TxPert CUDA runtime contract")
    official = payload.get("official_checkout")
    override = payload.get("hardware_compatibility_override")
    if not isinstance(official, dict) or not isinstance(override, dict):
        raise ValueError("TxPert CUDA runtime contract lacks required mappings")
    lock_name = official.get("lock_file")
    if not isinstance(lock_name, str):
        raise ValueError("TxPert CUDA runtime contract lacks official lock path")
    lock_path = (checkout_root.resolve(strict=True) / lock_name).resolve(strict=True)
    if not lock_path.is_relative_to(checkout_root.resolve()):
        raise ValueError("TxPert official lock resolves outside checkout")
    if sha256_file(lock_path) != official.get("lock_sha256"):
        raise ValueError("TxPert official lock SHA-256 differs from runtime contract")
    return contract_path, payload


def inspect_cuda_runtime(*, contract: dict[str, Any], device: str) -> dict[str, object]:
    """Run import, architecture, core-kernel, and PyG-extension compatibility gates."""

    override = contract["hardware_compatibility_override"]
    python_version = platform.python_version()
    if ".".join(python_version.split(".")[:2]) != override["python"]:
        raise ValueError("TxPert Python runtime version differs from contract")
    expected_versions = override["module_versions"]
    modules = {name: importlib.import_module(name) for name in expected_versions}
    observed_versions = {
        name: str(getattr(module, "__version__", "")) for name, module in modules.items()
    }
    if observed_versions != expected_versions:
        raise ValueError(
            "TxPert CUDA module versions differ from contract: "
            f"expected={expected_versions}, observed={observed_versions}"
        )

    torch = modules["torch"]
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("TxPert CUDA runtime gate requires an available CUDA device")
    torch.cuda.set_device(cuda_device)
    observed_cuda = str(torch.version.cuda)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(cuda_device))
    architecture = f"sm_{capability[0]}{capability[1]}"
    architectures = tuple(str(value) for value in torch.cuda.get_arch_list())
    expected_capability = tuple(int(value) for value in override["required_device_capability"])
    if observed_cuda != override["cuda_runtime"]:
        raise ValueError("TxPert CUDA runtime version differs from contract")
    if capability != expected_capability or architecture != override["required_cuda_arch"]:
        raise ValueError("TxPert CUDA device capability differs from contract")
    if architecture not in architectures:
        raise ValueError("TxPert PyTorch wheel lacks the active device architecture")

    vector = torch.arange(8, dtype=torch.float32, device=cuda_device)
    concatenated = torch.cat((vector, vector))
    matrix = torch.ones((8, 8), dtype=torch.float32, device=cuda_device)
    matrix_sum = float((matrix @ matrix).sum().item())
    scatter_index = torch.tensor([0, 0], dtype=torch.long, device=cuda_device)
    scatter_source = torch.tensor([1.0, 2.0], dtype=torch.float32, device=cuda_device)
    scatter_result = modules["torch_scatter"].scatter_add(scatter_source, scatter_index)
    torch.cuda.synchronize(cuda_device)
    if concatenated.shape != (16,) or matrix_sum != 512.0:
        raise RuntimeError("TxPert core CUDA kernel health check failed")
    if scatter_result.shape != (1,) or float(scatter_result.item()) != 3.0:
        raise RuntimeError("TxPert PyG CUDA extension health check failed")

    return {
        "schema_version": "txpert-cuda-runtime-receipt-v1",
        "python_version": python_version,
        "module_versions": observed_versions,
        "cuda_runtime": observed_cuda,
        "device_name": str(torch.cuda.get_device_name(cuda_device)),
        "device_capability": list(capability),
        "required_architecture": architecture,
        "wheel_architectures": list(architectures),
        "core_cuda_kernel_passed": True,
        "pyg_cuda_extension_passed": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    contract_path, contract = load_runtime_contract(
        repository_root=args.repository_root,
        checkout_root=args.official_checkout,
    )
    payload = {
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        **inspect_cuda_runtime(contract=contract, device=args.device),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
