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


def group_mean_loss(
    z: Tensor,
    target: Tensor,
    head: nn.Linear,
    loss_mode: str = "mse",
    *,
    diagnostics: dict[str, float] | None = None,
) -> Tensor:
    """Read out the complete group mean once; the fixed target is stop-gradient."""
    if loss_mode not in ("mse", "cosine_direction"):
        raise ValueError("response anchor loss must be mse or cosine_direction")
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
    if loss_mode == "mse":
        loss = F.mse_loss(head(z.mean(0)), target.detach())
    else:
        # Keep the original mean and Linear autocast precision. Only direction
        # normalization and the dot product use FP32, after the single readout.
        output = head(z.mean(0))
        with torch.autocast(z.device.type, enabled=False):
            fixed = target.detach().float()
            target_norm = torch.linalg.vector_norm(fixed)
            if not torch.isfinite(target_norm) or target_norm <= 1e-6:
                raise ValueError("cosine direction target norm must be finite and >1e-6")
            output = output.float()
            output_norm = torch.linalg.vector_norm(output)
            if not torch.isfinite(output_norm):
                raise ValueError("nonfinite cosine direction head output norm")
            loss = 1 - torch.dot(
                output / output_norm.clamp_min(1e-6), fixed / target_norm.clamp_min(1e-6)
            )
        if diagnostics is not None:
            value = float(output_norm.detach())
            diagnostics["head_observation_count"] = diagnostics.get("head_observation_count", 0) + 1
            diagnostics["head_norm_min"] = min(diagnostics.get("head_norm_min", value), value)
            diagnostics["head_norm_max"] = max(diagnostics.get("head_norm_max", value), value)
            diagnostics["head_clamped_count"] = diagnostics.get("head_clamped_count", 0) + int(
                value < 1e-6
            )
    if not torch.isfinite(loss):
        raise ValueError("nonfinite group mean loss")
    return loss


PAIR_MODE = "population_prediction_contrast"
PAIR_SCHEDULE_SCHEMA = "fixed-response-population-pair-cycle-v1"
PAIR_FORMULA = (
    "mean_gene(square((mean_cell_fp32(pred_p)-mean_cell_fp32(pred_q))"
    "-(fixed_train_mean_p-fixed_train_mean_q)))"
)


def pair_schedule(
    P: int, steps_per_epoch: int, epoch: int, seed: int, world: int = 1, rank: int = 0
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Two alternating matchings of one private hash permutation; pairs stay atomic."""
    for value, name, minimum in (
        (P, "P", 4),
        (steps_per_epoch, "steps_per_epoch", 1),
        (epoch, "epoch", 0),
        (seed, "seed", 0),
        (world, "world", 1),
        (rank, "rank", 0),
    ):
        _integer(value, name, minimum)
    if P % 2 or epoch > 1 or rank >= world:
        raise ValueError("pair cycle requires even P, epochs 0/1 and rank < world")
    order = sorted(
        range(P),
        key=lambda p: (
            hashlib.sha256(
                json.dumps([PAIR_SCHEDULE_SCHEMA, seed, p], separators=(",", ":")).encode()
            ).digest(),
            p,
        ),
    )
    shifted = order[epoch:] + order[:epoch]
    steps: list[list[tuple[int, int]]] = [[] for _ in range(steps_per_epoch)]
    for i in range(P // 2):
        if (i // steps_per_epoch) % world == rank:
            steps[i % steps_per_epoch].append((shifted[2 * i], shifted[2 * i + 1]))
    return tuple(tuple(step) for step in steps)


def pair_schedule_identity(P: int, steps_per_epoch: int, seed: int) -> dict[str, Any]:
    """Seal ordered schedules and verify exact coverage and connected cycle."""
    epochs = [
        [pair for step in pair_schedule(P, steps_per_epoch, e, seed) for pair in step]
        for e in range(2)
    ]
    adjacency: dict[int, set[int]] = {p: set() for p in range(P)}
    edges = set()
    for pairs in epochs:
        if sorted(p for pair in pairs for p in pair) != list(range(P)):
            raise ValueError("pair epoch coverage differs")
        for p, q in pairs:
            edge = tuple(sorted((p, q)))
            if p == q or edge in edges:
                raise ValueError("self or duplicate pair edge")
            edges.add(edge)
            adjacency[p].add(q)
            adjacency[q].add(p)
    visited, pending = set(), [0]
    while pending:
        p = pending.pop()
        if p not in visited:
            visited.add(p)
            pending.extend(adjacency[p] - visited)
    if len(visited) != P or any(len(n) != 2 for n in adjacency.values()):
        raise ValueError("two-match union is not one connected cycle")
    return dict(
        schema=PAIR_SCHEDULE_SCHEMA,
        seed=seed,
        conditions=P,
        pairs_per_epoch=P // 2,
        group_forwards_per_epoch=P,
        schedule_sha256=hashlib.sha256(
            json.dumps(epochs, separators=(",", ":")).encode()
        ).hexdigest(),
        coverage_once_per_epoch=True,
        no_self_pairs=True,
        distinct_edges=len(edges),
        connected_components=1,
    )


def prediction_contrast_loss(
    pred_p: Tensor, pred_q: Tensor, target_p: Tensor, target_q: Tensor
) -> Tensor:
    """Signed complete-condition FP32 mean contrast, with no one-half factor."""
    for prediction, target in ((pred_p, target_p), (pred_q, target_q)):
        if (
            prediction.ndim != 2
            or not 4 <= prediction.shape[0] <= 40
            or target.shape != prediction.shape[1:]
            or target.requires_grad
            or target.device != prediction.device
            or not prediction.is_floating_point()
            or not target.is_floating_point()
            or not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()
        ):
            raise ValueError("contrast requires finite complete [4..40,Q] groups and fixed targets")
    if pred_p.shape[1:] != pred_q.shape[1:] or pred_p.device != pred_q.device:
        raise ValueError("contrast query axes differ")
    error = (pred_p.float().mean(0) - pred_q.float().mean(0)) - (
        target_p.detach().float() - target_q.detach().float()
    )
    return error.square().mean()
