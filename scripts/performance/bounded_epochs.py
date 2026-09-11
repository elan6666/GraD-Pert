"""Execution-only epoch boundary for original-horizon diagnostic comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def state_content_sha256(value: Any) -> str:
    """Hash checkpoint content independently of serialization/storage IDs."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"torch")
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(
                item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy")
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
            digest.update(b"end_mapping")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
            digest.update(b"end_sequence")
        elif item is None or isinstance(item, (str, bool, int, float)):
            encoded = json.dumps(item, allow_nan=False).encode()
            digest.update(type(item).__name__.encode())
            digest.update(str(len(encoded)).encode() + b":" + encoded)
        else:
            raise TypeError(f"unsupported checkpoint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def audit_trainer_boundary(trainer: Any, epoch: int) -> dict[str, Any]:
    """Inspect real logs/checkpoints after native validation and serialization."""
    import torch

    from gradpert.hashing import sha256_file

    if (
        trainer.max_epochs != 50
        or trainer.engine.total_schedule_steps != 50 * trainer.steps_per_epoch
    ):
        raise ValueError("diagnostic trainer changed the original schedule horizon")
    if epoch == 0:
        if trainer.progress.completed_epochs != 0 or trainer.progress.global_step != 0:
            raise ValueError("nonzero progress at initial boundary")
        return {"completed_epochs": 0, "global_step": 0}
    records = {}
    for role, path in (("last", trainer.last_checkpoint), ("best", trainer.best_checkpoint)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{role} checkpoint missing or unsafe")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "gradpert-training-checkpoint-v2":
            raise ValueError("unsupported checkpoint schema")
        if payload.get("identity") != trainer.identity.__dict__:
            raise ValueError("checkpoint identity differs from trainer")
        records[role] = {
            "sha256": sha256_file(path),
            "progress": payload["progress"],
            "content_sha256": state_content_sha256(payload),
        }
        del payload
    with trainer.receipts.steps_path.open(newline="") as stream:
        steps = list(csv.DictReader(stream))
    with trainer.receipts.validation_path.open(newline="") as stream:
        validations = list(csv.DictReader(stream))
    validate_boundary_records(
        epoch=epoch,
        steps_per_epoch=trainer.steps_per_epoch,
        schedule_steps=trainer.engine.total_schedule_steps,
        progress=trainer.progress.to_payload(),
        steps=steps,
        validations=validations,
        last_progress=records["last"]["progress"],
        best_progress=records["best"]["progress"],
    )
    return {
        "completed_epochs": epoch,
        "global_step": trainer.progress.global_step,
        "checkpoint_identity": trainer.identity.__dict__,
        "checkpoints": records,
        "steps_sha256": sha256_file(trainer.receipts.steps_path),
        "validation_sha256": sha256_file(trainer.receipts.validation_path),
        "validation_rows": validations,
    }


def validate_boundary_records(
    *,
    epoch: int,
    steps_per_epoch: int,
    schedule_steps: int,
    progress: Mapping[str, object],
    steps: Sequence[Mapping[str, object]],
    validations: Sequence[Mapping[str, object]],
    last_progress: Mapping[str, object],
    best_progress: Mapping[str, object],
) -> None:
    """Require a completed training/validation boundary, never merely a file."""
    if epoch not in {1, 2, 3} or steps_per_epoch <= 0:
        raise ValueError("invalid diagnostic boundary")
    if schedule_steps != 50 * steps_per_epoch:
        raise ValueError("original 50-epoch schedule horizon changed")
    count = epoch * steps_per_epoch
    if (
        progress.get("completed_epochs"),
        progress.get("global_step"),
        progress.get("test_evaluations"),
    ) != (epoch, count, 0):
        raise ValueError("live progress is not the requested no-test boundary")
    if last_progress != progress:
        raise ValueError("last checkpoint progress differs from live progress")
    best_epoch = int(best_progress.get("completed_epochs", -1))
    if (
        not 1 <= best_epoch <= epoch
        or best_progress.get("global_step") != best_epoch * steps_per_epoch
    ):
        raise ValueError("best checkpoint is outside the completed interval")
    if best_progress.get("test_evaluations") != 0:
        raise ValueError("best checkpoint already consumed test")
    if len(steps) != count or any(
        int(row["global_step"]) != i or int(row["epoch"]) != i // steps_per_epoch
        for i, row in enumerate(steps)
    ):
        raise ValueError("step log is missing, duplicate or out of order")
    if len(validations) != epoch or any(
        int(row["epoch"]) != i or int(row["global_step"]) != (i + 1) * steps_per_epoch
        for i, row in enumerate(validations)
    ):
        raise ValueError("validation log does not cover each completed epoch")


class BoundedEpochComplete(RuntimeError):
    """Raised before materializing the first batch beyond the diagnostic budget."""


def bounded_epoch_factory(
    factory: Callable[[int], Iterable[T]],
    *,
    stop_before_epoch: int,
    on_boundary: Callable[[int], None],
) -> Callable[[int], Iterable[T]]:
    """Leave trainer/model schedule horizons intact; intercept only data access.

    The native trainer calls the next epoch factory only after the preceding
    validation, logging and checkpoint save. The callback must inspect those
    facts; it must never label an early exception as a completed epoch.
    """
    if stop_before_epoch not in {1, 3}:
        raise ValueError("diagnostic stop boundary must be epoch 1 or 3")

    def bounded(epoch: int) -> Iterable[T]:
        if epoch < 0 or epoch > stop_before_epoch:
            raise ValueError("epoch outside diagnostic execution boundary")
        on_boundary(epoch)
        if epoch == stop_before_epoch:
            raise BoundedEpochComplete(f"completed diagnostic epoch boundary {epoch}")
        return factory(epoch)

    return bounded
