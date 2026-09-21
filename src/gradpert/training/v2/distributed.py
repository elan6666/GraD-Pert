"""Synchronous gradient collectives for v2's explicitly routed optimizer."""

from __future__ import annotations

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
