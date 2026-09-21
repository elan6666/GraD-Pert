"""Synchronous gradient collectives for v2's explicitly routed optimizer."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar, cast

import torch
from torch import nn


def average_gradients(model: nn.Module, *, bucket_bytes: int = 25 * 1024**2) -> None:
    """Average all rank gradients, preserving globally unused parameters as None.

    Missing local masked-token gradients contribute zero, but every rank enters
    identical collectives. Parameters must have identical order/dtype on ranks.
    """
    if not torch.distributed.is_initialized():
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or bucket_bytes < 1:
        raise ValueError("gradient reduction needs parameters and a positive bucket budget")
    used = torch.tensor(
        [p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32
    )
    torch.distributed.all_reduce(used)
    active = [p for p, count in zip(parameters, used.tolist(), strict=True) if count]
    world = torch.distributed.get_world_size()
    bucket: list[nn.Parameter] = []
    size = 0

    def flush() -> None:
        if not bucket:
            return
        flat = torch.cat(
            [(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in bucket]
        )
        torch.distributed.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for p in bucket:
            p.grad = flat[offset : offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        bucket.clear()

    for p in active:
        if bucket and (
            size + p.numel() * p.element_size() > bucket_bytes or p.dtype != bucket[0].dtype
        ):
            flush()
            size = 0
        bucket.append(p)
        size += p.numel() * p.element_size()
    flush()


T = TypeVar("T")
_PRIMARY_GROUP: Any = None
_PRIMARY_OWNER: Any = None


def _primary_control_group() -> Any:
    """Keep long rank-zero validation waits outside NCCL gradient watchdogs.

    Every rank enters primary_call in the same order. The auxiliary CPU group
    permits up to one day for full validation/test work; NCCL retains the worker's
    short timeout for actual tensor collectives. Reset if a process group changes.
    """
    global _PRIMARY_GROUP, _PRIMARY_OWNER
    if torch.distributed.get_backend() != "nccl":
        return None
    owner = torch.distributed.group.WORLD
    if _PRIMARY_OWNER is not owner:
        _PRIMARY_GROUP = torch.distributed.new_group(backend="gloo", timeout=timedelta(days=1))
        _PRIMARY_OWNER = owner
    return _PRIMARY_GROUP


def primary_call(operation: Callable[[], T]) -> T:
    """Run a filesystem/evaluation operation once and broadcast its outcome."""
    if not torch.distributed.is_initialized():
        return operation()
    group = _primary_control_group()
    outcome: list[Any] = [None]
    if torch.distributed.get_rank() == 0:
        try:
            outcome[0] = {"value": operation(), "error": None}
        except Exception as error:
            outcome[0] = {"value": None, "error": f"{type(error).__name__}: {error}"}
    torch.distributed.broadcast_object_list(outcome, src=0, group=group)
    result = outcome[0]
    if result["error"] is not None:
        raise RuntimeError("primary rank operation failed: " + result["error"])
    return cast(T, result["value"])
