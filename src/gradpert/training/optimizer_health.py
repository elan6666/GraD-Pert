"""Sparse observation of actual optimizer updates without extra model passes."""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from gradpert.training.optimizers import SplitMatrixAdamW


@contextmanager
def observe_optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    interval: int,
    records: list[dict[str, Any]],
) -> Iterator[None]:
    """Sample step zero and interval boundaries; never modify optimizer state.

    Global step is supplied by the trainer, so resumed sampling uses the original
    timeline. Wall time measures host dispatch, not synchronized GPU kernels.
    """
    if interval <= 0 or global_step % interval:
        yield
        return
    named = dict(model.named_parameters())
    roles = (
        {id(named[r["name"]]): r["optimizer"] for r in optimizer.routes}
        if isinstance(optimizer, SplitMatrixAdamW)
        else {id(p): "adamw" for group in optimizer.param_groups for p in group["params"]}
    )
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    before = [p.detach().clone() for p in parameters]
    started = time.perf_counter()
    yield
    dispatch_ms = (time.perf_counter() - started) * 1000.0
    groups: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for parameter, previous in zip(parameters, before, strict=True):
            group = groups.setdefault(
                roles[id(parameter)], {"weight_squared": 0.0, "update_squared": 0.0}
            )
            group["weight_squared"] += float(previous.double().square().sum().item())
            group["update_squared"] += float(
                (parameter.detach() - previous).double().square().sum().item()
            )
    records.append(
        {
            "global_step": global_step,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "host_dispatch_ms": dispatch_ms,
            "groups": {
                name: {
                    "weight_l2": math.sqrt(values["weight_squared"]),
                    "update_l2": math.sqrt(values["update_squared"]),
                }
                for name, values in groups.items()
            },
        }
    )
