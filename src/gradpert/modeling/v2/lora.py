"""Native low-rank adapters for a separately versioned v2 fine-tuning stage."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Frozen linear map plus a zero-initialized low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(rank, base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.down), self.up)


def insert_lora(model: nn.Module, *, rank: int, alpha: float) -> tuple[str, ...]:
    """Adapt hidden graph, control and response projections; freeze all base weights.

    Prediction and distillation heads remain frozen. This keeps the new stage a
    genuine adapter fine-tune and makes its parameter ownership unambiguous.
    """
    if rank < 1 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    model.requires_grad_(False)
    selected: list[str] = []
    for prefix in ("graph", "expression", "cell", "condition_fusion", "response"):
        owner = getattr(model, prefix, None)
        if owner is None:
            continue
        if isinstance(owner, nn.Linear):
            setattr(model, prefix, LowRankLinear(owner, rank, alpha))
            selected.append(prefix)
            continue
        for name, module in list(owner.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            path, _, field = name.rpartition(".")
            parent = owner.get_submodule(path) if path else owner
            setattr(parent, field or name, LowRankLinear(module, rank, alpha))
            selected.append(f"{prefix}.{name}")
    if not selected:
        raise ValueError("LoRA found no hidden linear projections")
    return tuple(selected)


def parent_to_lora_state(parent: dict[str, Tensor], model: nn.Module) -> dict[str, Tensor]:
    """Map only frozen base tensors; preserve zero-initialized adapter outputs."""
    adapted = dict(model.state_dict())
    mapped: set[str] = set()
    for name, value in parent.items():
        path, _, field = name.rpartition(".")
        destination = name if name in adapted else f"{path}.base.{field}"
        if destination not in adapted or adapted[destination].shape != value.shape:
            raise ValueError(f"parent tensor differs from LoRA architecture: {name}")
        adapted[destination] = value
        mapped.add(destination)
    missing_base = [
        name for name in adapted if not name.endswith((".down", ".up")) and name not in mapped
    ]
    if missing_base:
        raise ValueError(f"parent is missing base tensors: {missing_base[:3]}")
    return adapted
