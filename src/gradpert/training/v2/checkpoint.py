"""Version-tagged, atomic v2 checkpoints with exact RNG/optimizer continuation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .objective import JointObjective
from .optimizer import V2Optimizer


def save_checkpoint(
    path: Path,
    objective: JointObjective,
    optimizer: V2Optimizer,
    *,
    identity: dict[str, Any],
    progress: dict[str, Any],
    generator: np.random.Generator,
) -> None:
    if objective.pending:
        raise ValueError("checkpoint only at completed optimizer-step boundaries")
    distributed = torch.distributed.is_initialized()
    rng = {
        "numpy_generator": generator.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    rank_states: list[Any] | None = None
    rank = 0
    if distributed:
        rank = torch.distributed.get_rank()
        rank_states = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_states, rng)
    payload = {
        "model_version": "v2",
        "schema_version": "gradpert-v2-checkpoint-1",
        "architecture": objective.student.options.payload(),
        "identity": identity,
        "progress": progress,
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        **rng,
        "rank_rng_states": rank_states,
    }
    status: list[Any] = [None]
    if rank == 0:
        try:
            _atomic_checkpoint(path, payload)
        except Exception as error:
            if not distributed:
                raise
            status[0] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"distributed checkpoint write failed: {status[0]}")


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    objective: JointObjective,
    optimizer: V2Optimizer,
    *,
    identity: dict[str, Any],
    generator: np.random.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("model_version") != "v2"
        or payload.get("schema_version") != "gradpert-v2-checkpoint-1"
    ):
        raise ValueError("checkpoint is not a supported GraD-Pert v2 checkpoint")
    if (
        payload["identity"] != identity
        or payload["architecture"] != objective.student.options.payload()
    ):
        raise ValueError("checkpoint data/config/architecture identity mismatch")
    rank_states = payload.get("rank_rng_states")
    if torch.distributed.is_initialized():
        if rank_states is None or len(rank_states) != torch.distributed.get_world_size():
            raise ValueError("checkpoint distributed world size differs")
        rng = rank_states[torch.distributed.get_rank()]
    else:
        if rank_states is not None:
            raise ValueError("distributed training resume requires its original world size")
        rng = payload
    objective.load_state_dict(payload["objective"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.bit_generator.state = rng["numpy_generator"]
    random.setstate(rng["python_rng"])
    torch.set_rng_state(rng["torch_rng"])
    if rng["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training resume requires CUDA RNG restoration")
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
    objective.pending.clear()
    return cast(dict[str, Any], payload["progress"])
