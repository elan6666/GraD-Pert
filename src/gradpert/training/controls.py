"""Deterministic per-epoch pairing of perturbed cells to compatible controls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingControlPairing:
    epoch: int
    perturbed_row_ids: tuple[str, ...]
    control_row_ids: tuple[str, ...]
    context_ids: tuple[str, ...]


class TrainingControlPairer:
    """Choose one same-context control per perturbed cell and epoch."""

    def __init__(self, *, run_seed: int) -> None:
        if run_seed < 0:
            raise ValueError("run_seed must be nonnegative")
        self.run_seed = run_seed

    def _seed(self, *, epoch: int, perturbed_row_id: str) -> int:
        payload = f"{self.run_seed}::{epoch}::{perturbed_row_id}::train_control".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big")

    def pair_epoch(
        self,
        *,
        epoch: int,
        perturbed_row_ids: Sequence[str],
        context_ids: Sequence[str],
        control_pools: Mapping[str, Sequence[str]],
    ) -> TrainingControlPairing:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if not perturbed_row_ids or len(perturbed_row_ids) != len(context_ids):
            raise ValueError("perturbed rows and contexts must be non-empty and aligned")
        selected: list[str] = []
        for row_id, context_id in zip(perturbed_row_ids, context_ids, strict=True):
            if not row_id or not context_id:
                raise ValueError("row and context IDs must be non-empty")
            pool = tuple(control_pools.get(context_id, ()))
            if not pool or any(not item for item in pool):
                raise ValueError(f"no valid compatible training controls for context {context_id}")
            if len(pool) != len(set(pool)):
                raise ValueError(f"training control pool has duplicate row IDs: {context_id}")
            rng = np.random.Generator(
                np.random.PCG64(self._seed(epoch=epoch, perturbed_row_id=row_id))
            )
            selected.append(pool[int(rng.integers(0, len(pool)))])
        return TrainingControlPairing(
            epoch=epoch,
            perturbed_row_ids=tuple(perturbed_row_ids),
            control_row_ids=tuple(selected),
            context_ids=tuple(context_ids),
        )
