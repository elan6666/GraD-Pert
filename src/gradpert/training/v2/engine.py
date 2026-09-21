"""Completed optimizer-step boundaries for accumulated v2 training."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

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
    global_condition_index: torch.Tensor | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, float]:
    """Accumulate cell terms; count graph terms once per effective update.

    KoLeo neighborhoods remain local to each microbatch and each global view.
    The exact neighborhood size must be reported alongside effective batch.
    Distributed callers provide the unsharded condition index; graph terms
    remain once per global update, cell means are weighted by actual rank rows.
    """
    distributed = torch.distributed.is_initialized()
    population_factor = 1.0
    if distributed:
        if global_condition_index is None or objective.prediction_reduction != "cell_mean":
            raise ValueError("distributed steps require global condition IDs and cell_mean loss")
        count = torch.tensor(len(batch.control), device=batch.control.device)
        torch.distributed.all_reduce(count)
        if count.item() != len(global_condition_index) or not len(batch.control):
            raise ValueError("distributed cell partitions differ from the global batch")
        population_factor = (
            torch.distributed.get_world_size() * len(batch.control) / len(global_condition_index)
        )
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
    ibot_population = None
    if objective.lambda2 and objective.weights[1][1]:
        if len(batch.cell_views) < 2:
            raise ValueError("iBOT requires two global views")
        ibot_population = torch.stack(
            (
                batch.cell_views[0].mask.sum(),
                batch.cell_views[1].mask.sum(),
                batch.control.new_tensor(total, dtype=torch.int64),
            )
        )
        if distributed:
            torch.distributed.all_reduce(ibot_population)
        ibot_population = ibot_population.float()
    finite = True
    try:
        for start in range(0, total, microbatch):
            micro = slice_cells(batch, start, min(start + microbatch, total))
            fraction = len(micro.control) / total
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                loss, terms = objective(micro, include_ssl1=False, ibot_population=ibot_population)
            if not torch.isfinite(loss):
                finite = False
                if not distributed:
                    raise FloatingPointError("nonfinite v2 cell loss")
            (loss * fraction * population_factor).backward()
            for name, value in terms.items():
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) * fraction
        if objective.lambda1:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                graph_terms = objective.graph_loss(
                    batch.graph_views,
                    batch.condition_index
                    if global_condition_index is None
                    else global_condition_index,
                )
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
                finite = False
                if not distributed:
                    raise FloatingPointError("nonfinite v2 graph loss")
            graph_loss.backward()  # type: ignore[no-untyped-call]
            metrics.update(
                {f"ssl1_{name}": float(value.detach()) for name, value in graph_terms.items()}
            )
        if distributed:
            from .distributed import average_gradients

            valid = torch.tensor(int(finite), device=batch.control.device)
            torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
            if not valid.item():
                raise FloatingPointError("nonfinite v2 loss on at least one rank")
            if timings is not None and device == "cuda":
                torch.cuda.synchronize(batch.control.device)
            communication_started = perf_counter()
            average_gradients(objective.student)
            if timings is not None:
                if device == "cuda":
                    torch.cuda.synchronize(batch.control.device)
                timings["gradient_reduction_seconds"] = perf_counter() - communication_started
            names = sorted(metrics)
            values = torch.tensor(
                [
                    metrics[name] * (1.0 if name.startswith("ssl1_") else population_factor)
                    for name in names
                ],
                device=batch.control.device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
            metrics = dict(zip(names, values.tolist(), strict=True))
        norm = torch.nn.utils.clip_grad_norm_(
            objective.student.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step(lr)
        objective.commit_statistics(momentum)
        metrics["joint_loss"] = metrics["prediction"] + sum(
            scale
            * sum(
                weight * metrics.get(f"ssl{stage}_{name}", 0.0)
                for weight, name in zip(weights, names, strict=True)
            )
            for stage, scale, weights, names in (
                (1, objective.lambda1, objective.weights[0], ("condition", "node", "spread")),
                (2, objective.lambda2, objective.weights[1], ("dino", "ibot", "koleo")),
            )
        )
        metrics["gradient_norm"] = float(norm)
        metrics["learning_rate"] = lr
        metrics["teacher_momentum"] = momentum
        return metrics
    except Exception:
        objective.pending.clear()
        optimizer.zero_grad()
        raise
