"""Read-only, fixed-view validation of the complete v2 training objective."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

from gradpert.hashing import sha256_json

from .objective import JointObjective, TrainingBatch


def evaluate_joint_loss(
    objective: JointObjective,
    batches: Iterable[tuple[TrainingBatch, str]],
    *,
    bf16: bool,
    on_batch_complete: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Average complete per-batch objectives by validation cell count.

    Views, condition mixing and control pairing come from a separately seeded,
    fixed validation iterator. Teacher parameters and centers are read-only.
    The nearest-neighbor population is each complete validation batch.
    """
    if objective.pending:
        raise ValueError("joint validation cannot begin with pending teacher statistics")
    was_training = objective.training
    device = next(objective.parameters()).device
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    totals: dict[str, float] = {}
    total_cells = 0
    batch_hashes: list[str] = []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            objective.eval()
            for batch, identity_hash in batches:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                    loss, terms = objective(batch)
                cells = len(batch.control)
                value = float(loss.float())
                if cells < 1 or not math.isfinite(value):
                    raise FloatingPointError("nonfinite or empty joint validation batch")
                totals["joint_loss"] = totals.get("joint_loss", 0.0) + value * cells
                for name, term in terms.items():
                    observed = float(term.float())
                    if not math.isfinite(observed):
                        raise FloatingPointError(f"nonfinite joint validation term: {name}")
                    totals[name] = totals.get(name, 0.0) + observed * cells
                total_cells += cells
                batch_hashes.append(identity_hash)
                objective.pending.clear()
                if on_batch_complete is not None:
                    on_batch_complete(len(batch_hashes), totals["joint_loss"] / total_cells)
    finally:
        objective.pending.clear()
        objective.train(was_training)
    if not batch_hashes:
        raise ValueError("joint validation has no batches")
    return {
        "joint_loss": totals.pop("joint_loss") / total_cells,
        "components": {name: value / total_cells for name, value in totals.items()},
        "cell_count": total_cells,
        "batch_count": len(batch_hashes),
        "batch_identity_sha256": sha256_json(batch_hashes),
        "reduction": "cell_weighted_mean_of_complete_validation_batch_losses",
        "teacher_update": "none",
    }
