"""Completed optimizer-step boundaries for accumulated v2 training."""

from __future__ import annotations

from dataclasses import replace

import torch

from .objective import CellView, JointObjective, TrainingBatch
from .optimizer import V2Optimizer


def slice_cells(batch: TrainingBatch, start: int, end: int) -> TrainingBatch:
    unique = batch.unique_cell_indices
    if unique is not None:
        unique = unique[(unique >= start) & (unique < end)] - start
    return replace(
        batch,
        control=batch.control[start:end],
        truth=batch.truth[start:end],
        condition_index=batch.condition_index[start:end],
        unique_cell_indices=unique,
        cell_views=tuple(CellView(v.positions, v.mask[start:end]) for v in batch.cell_views),
    )


def optimizer_step(
    objective: JointObjective,
    optimizer: V2Optimizer,
    batch: TrainingBatch,
    *,
    microbatch: int,
    lr: float,
    momentum: float,
    bf16: bool,
) -> dict[str, float]:
    """Accumulate cell terms; count graph terms once per effective update.

    KoLeo neighborhoods remain local to each microbatch and each global view.
    The exact neighborhood size must be reported alongside effective batch.
    This function currently supports a single process and deliberately rejects
    a distributed process group until distributed-statistic parity is tested.
    """
    if torch.distributed.is_initialized():
        raise ValueError("distributed v2 steps are not yet verified")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if objective.prediction_reduction == "condition_mean" and microbatch < len(batch.control):
        raise ValueError("condition_mean accumulation requires global condition weights")
    objective.train()
    optimizer.zero_grad()
    objective.pending.clear()
    device = batch.control.device.type
    metrics: dict[str, float] = {}
    total = len(batch.control)
    try:
        for start in range(0, total, microbatch):
            micro = slice_cells(batch, start, min(start + microbatch, total))
            fraction = len(micro.control) / total
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                loss, terms = objective(micro, include_ssl1=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite v2 cell loss")
            (loss * fraction).backward()
            for name, value in terms.items():
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) * fraction
        if objective.lambda1:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                graph_terms = objective.graph_loss(batch.graph_views, batch.condition_index)
                graph_loss = (
                    objective.lambda1
                    * torch.stack(
                        [
                            w * v
                            for w, v in zip(objective.weights[0], graph_terms.values(), strict=True)
                        ]
                    ).sum()
                )
            if not torch.isfinite(graph_loss):
                raise FloatingPointError("nonfinite v2 graph loss")
            graph_loss.backward()  # type: ignore[no-untyped-call]
            metrics.update(
                {f"ssl1_{name}": float(value.detach()) for name, value in graph_terms.items()}
            )
        norm = torch.nn.utils.clip_grad_norm_(
            objective.student.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step(lr)
        objective.commit_statistics(momentum)
        metrics["gradient_norm"] = float(norm)
        metrics["learning_rate"] = lr
        metrics["teacher_momentum"] = momentum
        return metrics
    except Exception:
        objective.pending.clear()
        optimizer.zero_grad()
        raise
