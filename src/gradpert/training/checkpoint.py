"""Atomic, identity-bound native checkpoints with exact RNG restoration."""

from __future__ import annotations

import fcntl
import hashlib
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradpert.modeling import CenterState, GraDPertJointModel


@dataclass(frozen=True)
class CheckpointIdentity:
    source_commit: str
    source_tree_sha256: str
    config_sha256: str
    environment_sha256: str
    canonical_data_sha256: str
    split_content_sha256: str

    def __post_init__(self) -> None:
        if len(self.source_commit) != 40:
            raise ValueError("source_commit must be a full Git SHA")
        for field, value in (
            ("source_tree_sha256", self.source_tree_sha256),
            ("config_sha256", self.config_sha256),
            ("environment_sha256", self.environment_sha256),
            ("canonical_data_sha256", self.canonical_data_sha256),
            ("split_content_sha256", self.split_content_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field} must be a lowercase SHA256")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor) or torch_cpu_state.dtype != torch.uint8:
        raise ValueError("checkpoint CPU RNG state must be a torch ByteTensor")
    # ``torch.load(..., map_location=<cuda>)`` moves every tensor in the
    # checkpoint, including RNG states, onto the target device.  PyTorch's RNG
    # restoration APIs require CPU ByteTensors even when restoring CUDA RNGs.
    torch.set_rng_state(torch_cpu_state.detach().cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG topology differs from this host")
        if any(
            not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8
            for cuda_state in cuda_states
        ):
            raise ValueError("checkpoint CUDA RNG states must be torch ByteTensors")
        torch.cuda.set_rng_state_all([cuda_state.detach().cpu() for cuda_state in cuda_states])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: GraDPertJointModel,
    optimizer: torch.optim.Optimizer,
    centers: CenterState,
    progress: Mapping[str, Any],
    identity: CheckpointIdentity,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gradpert-training-checkpoint-v2",
        "identity": identity.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "centers": {
            "condition": centers.condition.detach().clone(),
            "masked_node": centers.masked_node.detach().clone(),
        },
        "progress": dict(progress),
        "rng": _rng_state(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(destination)


def clone_training_checkpoint(source: str | Path, destination: str | Path) -> tuple[str, str]:
    """Atomically materialize an identical checkpoint by reflink or byte copy."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("checkpoint clone source must be a regular non-symlink file")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    method = "reflink"
    try:
        with source_path.open("rb") as source_stream, os.fdopen(descriptor, "wb") as target_stream:
            try:
                fcntl.ioctl(target_stream.fileno(), 0x40049409, source_stream.fileno())
            except OSError:
                method = "copy"
                target_stream.seek(0)
                target_stream.truncate(0)
                source_stream.seek(0)
                shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    source_sha256 = _sha256_file(source_path)
    destination_sha256 = _sha256_file(destination_path)
    if destination_sha256 != source_sha256:
        destination_path.unlink(missing_ok=True)
        raise RuntimeError("checkpoint peer differs from serialized source")
    return method, destination_sha256


def load_training_checkpoint(
    path: str | Path,
    *,
    model: GraDPertJointModel,
    optimizer: torch.optim.Optimizer,
    centers: CenterState,
    expected_identity: CheckpointIdentity,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = torch.load(source, map_location=centers.condition.device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "gradpert-training-checkpoint-v2"
    ):
        raise ValueError("unsupported native checkpoint schema")
    if payload.get("identity") != expected_identity.__dict__:
        raise ValueError("checkpoint identity does not match this run")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    center_payload = payload["centers"]
    if center_payload["condition"].shape != centers.condition.shape:
        raise ValueError("checkpoint condition-center shape differs")
    if center_payload["masked_node"].shape != centers.masked_node.shape:
        raise ValueError("checkpoint masked-node center shape differs")
    centers.condition.copy_(center_payload["condition"])
    centers.masked_node.copy_(center_payload["masked_node"])
    _restore_rng_state(payload["rng"])
    progress = payload["progress"]
    if not isinstance(progress, dict):
        raise ValueError("checkpoint progress must be a mapping")
    return progress
