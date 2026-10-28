"""Fixed TRAIN response targets and complete-condition auxiliary loss primitives.

This module owns no optimizer, update clock, Teacher, numerical target builder or
checkpoint payload. The caller keeps the bank external to the model and adds
weighted group gradients before the existing reduction, clip and optimizer step.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

_WIDTH = 256
_FIELDS = {
    "allowed_gene_indices",
    "condition_targets",
    "condition_anchor_gene_indices",
    "control_expression",
    "control_slot_offsets",
    "control_slot_indices",
}


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _sha(data: bytes, expected: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
        or hashlib.sha256(data).hexdigest() != expected
    ):
        raise ValueError("SHA256 mismatch or invalid expected hash")


@contextmanager
def isolated_rng(seed: int, cuda_devices: Sequence[int] = ()) -> Iterator[None]:
    """Seed private Python/NumPy/CPU/selected-CUDA streams, restore even on error.

    This is process-global state and must not be entered concurrently by threads.
    The caller supplies every CUDA device that its auxiliary forward will use.
    """
    _integer(seed, "seed")
    if seed >= 2**63:
        raise ValueError("seed must be < 2**63")
    devices = tuple(cuda_devices)
    for device in devices:
        _integer(device, "CUDA device")
    if len(set(devices)) != len(devices):
        raise ValueError("duplicate CUDA device")
    if devices and (not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count()):
        raise ValueError("requested CUDA device unavailable")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = {device: torch.cuda.get_rng_state(device) for device in devices}
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        for device, state in cuda_states.items():
            torch.cuda.set_rng_state(state, device)


def attach_anchor_readout(student: nn.Module, width: int, seed: int) -> nn.Linear:
    """Attach independent default Linear before objective/optimizer construction.

    Disabled callers do not call this function; historical keys and RNG then stay
    untouched. Existing names are rejected rather than silently replaced.
    """
    _integer(width, "width", 1)
    if hasattr(student, "response_anchor_readout"):
        raise ValueError("response_anchor_readout already exists")
    reference = next((p for p in student.parameters() if p.is_floating_point()), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = reference.dtype if reference is not None else torch.get_default_dtype()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("readout requires CPU or explicitly scoped CUDA RNG")
    cuda_devices = (device.index or 0,) if device.type == "cuda" else ()
    with isolated_rng(seed, cuda_devices):
        head = nn.Linear(width, width, bias=True, device=device, dtype=dtype)
    student.add_module("response_anchor_readout", head)
    return head


@dataclass(frozen=True)
class FixedResponseBank:
    """Read-only scientific arrays, deliberately not an nn.Module or buffer."""

    manifest_path: Path
    manifest_sha256: str
    arrays_path: Path
    arrays_sha256: str
    condition_ids: tuple[str, ...]
    identity: dict[str, Any]
    allowed_gene_indices: NDArray[np.int64]
    condition_targets: NDArray[np.float32]
    condition_anchor_gene_indices: NDArray[np.int64]
    control_expression: NDArray[np.float32]
    control_slot_offsets: NDArray[np.int64]
    control_slot_indices: NDArray[np.int64]

    @classmethod
    def from_files(
        cls,
        manifest_path: str | Path,
        expected_sha256: str,
        expected_allowed_gene_indices: Sequence[int] | NDArray[np.int64],
    ) -> FixedResponseBank:
        """Check byte hashes, six-array schema, ordered axes and complete slots.

        Identity semantics and server-only path restrictions belong to the native
        caller. This loader requires a nonempty identity object and never infers
        split/source compatibility from a directory name.
        """
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        _sha(raw, expected_sha256)
        metadata = json.loads(raw)
        if not isinstance(metadata, dict) or metadata.get("schema") != (
            "fixed-response-group-anchor-bank-v1"
        ):
            raise ValueError("invalid bank manifest schema")
        conditions = metadata.get("condition_ids")
        if (
            not isinstance(conditions, list)
            or not conditions
            or any(not isinstance(p, str) or not p for p in conditions)
            or len(set(conditions)) != len(conditions)
        ):
            raise ValueError("condition_ids must be nonempty unique strings")
        identity = metadata.get("identity")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("nonempty bound identity required")
        arrays = metadata.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != {"path", "sha256"}:
            raise ValueError("arrays requires exactly path and sha256")
        if not isinstance(arrays["path"], str) or not arrays["path"]:
            raise ValueError("invalid arrays path")
        array_path = (path.parent / arrays["path"]).resolve()
        payload = array_path.read_bytes()
        _sha(payload, arrays["sha256"])
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if len(archive.files) != len(_FIELDS) or set(archive.files) != _FIELDS:
                raise ValueError("npz must contain exactly the six bank fields")
            values = {key: archive[key].copy() for key in _FIELDS}
        p = len(conditions)
        allowed = values["allowed_gene_indices"]
        expected = np.asarray(expected_allowed_gene_indices)
        if (
            allowed.dtype != np.int64
            or allowed.ndim != 1
            or not len(allowed)
            or (allowed < 0).any()
            or (np.diff(allowed) <= 0).any()
            or expected.dtype.kind not in "iu"
            or expected.ndim != 1
            or not np.array_equal(allowed, expected)
        ):
            raise ValueError("ordered allowed gene axis mismatch")
        target = values["condition_targets"]
        control = values["control_expression"]
        if target.dtype != np.float32 or target.shape != (p, _WIDTH):
            raise ValueError("condition_targets must be float32 [P,256]")
        if (
            control.dtype != np.float32
            or control.ndim != 2
            or control.shape[0] < 1
            or control.shape[1] != len(allowed)
        ):
            raise ValueError("control_expression must be float32 [U,A]")
        if not np.isfinite(target).all() or not np.isfinite(control).all():
            raise ValueError("nonfinite bank values")
        if not np.any(target) or np.all(target == target[0]):
            raise ValueError("zero or identical condition targets")
        anchors = values["condition_anchor_gene_indices"]
        offsets = values["control_slot_offsets"]
        slots = values["control_slot_indices"]
        if anchors.dtype != np.int64 or anchors.shape != (p,) or (anchors < 0).any():
            raise ValueError("invalid single-target gene identity axis")
        if offsets.dtype != np.int64 or offsets.shape != (p + 1,):
            raise ValueError("invalid control slot offsets")
        if slots.dtype != np.int64 or slots.ndim != 1:
            raise ValueError("invalid control slot indices")
        counts = np.diff(offsets)
        if (
            offsets[0] != 0
            or offsets[-1] != len(slots)
            or (counts < 4).any()
            or (counts > 40).any()
            or (slots < 0).any()
            or (slots >= len(control)).any()
        ):
            raise ValueError("control slots require complete groups of 4..40 cached rows")
        for value in values.values():
            value.flags.writeable = False
        return cls(
            path,
            expected_sha256,
            array_path,
            arrays["sha256"],
            tuple(conditions),
            identity,
            **values,
        )

    def control_for_condition(self, condition: int) -> NDArray[np.float32]:
        """Return ordered slots (repeats permitted), never target values as input."""
        _integer(condition, "condition")
        if condition >= len(self.condition_ids):
            raise ValueError("condition outside bank")
        start, stop = self.control_slot_offsets[condition : condition + 2]
        return self.control_expression[self.control_slot_indices[start:stop]]


def group_schedule(
    P: int, steps_per_epoch: int, epoch: int, seed: int, world: int = 1, rank: int = 0
) -> tuple[tuple[int, ...], ...]:
    """Hash permutation, step i%T, rank (i//T)%world; each condition once.

    Empty per-rank steps are valid. This plan is independent of main-batch labels
    and RNG. Resume callers index the same plan with their existing step cursor.
    """
    for value, name, minimum in (
        (P, "P", 1),
        (steps_per_epoch, "steps_per_epoch", 1),
        (epoch, "epoch", 0),
        (seed, "seed", 0),
        (world, "world", 1),
        (rank, "rank", 0),
    ):
        _integer(value, name, minimum)
    if rank >= world:
        raise ValueError("rank must be < world")
    order = sorted(
        range(P),
        key=lambda p: (
            hashlib.sha256(
                json.dumps(
                    ["fixed-response-group-anchor-schedule-v1", seed, epoch, p],
                    separators=(",", ":"),
                ).encode()
            ).digest(),
            p,
        ),
    )
    steps: list[list[int]] = [[] for _ in range(steps_per_epoch)]
    for i, condition in enumerate(order):
        if (i // steps_per_epoch) % world == rank:
            steps[i % steps_per_epoch].append(condition)
    return tuple(tuple(step) for step in steps)


def group_loss_coefficient(
    P: int, steps_per_epoch: int, world: int = 1, weight: float = 1.0
) -> float:
    """Compensate rank-mean reduction and T-step averaging: lambda*world*T/P."""
    for value, name in ((P, "P"), (steps_per_epoch, "steps_per_epoch"), (world, "world")):
        _integer(value, name, 1)
    if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    result = weight * world * steps_per_epoch / P
    if not math.isfinite(result):
        raise ValueError("nonfinite group coefficient")
    return result


def group_mean_loss(z: Tensor, target: Tensor, head: nn.Linear) -> Tensor:
    """MSE after the complete group mean; the fixed target is stop-gradient."""
    if (
        z.ndim != 2
        or not 4 <= z.shape[0] <= 40
        or z.shape[1] < 1
        or not z.is_floating_point()
        or not torch.isfinite(z).all()
        or target.shape != (z.shape[1],)
        or not target.is_floating_point()
        or target.device != z.device
        or not torch.isfinite(target).all()
        or not isinstance(head, nn.Linear)
        or head.in_features != z.shape[1]
        or head.out_features != z.shape[1]
        or head.weight.device != z.device
    ):
        raise ValueError("loss requires finite aligned complete [4..40,D] group and linear head")
    loss = F.mse_loss(head(z.mean(0)), target.detach())
    if not torch.isfinite(loss):
        raise ValueError("nonfinite group mean loss")
    return loss
