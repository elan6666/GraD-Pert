"""Explicit v2 matrix/auxiliary routing; historical v1 optimizer is untouched."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gradpert.modeling.v2.model import SparseRead
from gradpert.modeling.v2.operators import (
    DeltaAttention,
    FullAttention,
    IndexedLatentAttention,
    LatentAttention,
)


def routes(model: nn.Module) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    result = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        path, _, field = name.rpartition(".")
        module = modules[path]
        owner_path, _, projection = path.rpartition(".")
        owner = modules.get(owner_path)
        hidden = (
            isinstance(module, nn.Linear)
            and field == "weight"
            and "prototypes" not in path
            and path != "prediction.2"
            and projection != "route"
        )
        heads = 1
        if (
            hidden
            and isinstance(owner, (SparseRead, DeltaAttention, LatentAttention))
            and projection in ("query", "key", "value")
        ):
            heads = owner.heads
        if hidden and isinstance(owner, FullAttention) and projection == "qkv":
            heads = 3 * owner.heads
        if (
            hidden
            and isinstance(owner, IndexedLatentAttention)
            and projection
            in (
                "q_up",
                "k_up",
                "v_up",
                "index_query",
                "index_key",
            )
        ):
            heads = owner.heads
        if hidden and (parameter.ndim != 2 or parameter.shape[0] % heads):
            raise ValueError("invalid head-wise optimizer route: " + name)
        result.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "heads": heads,
                "optimizer": "muon" if hidden else "adamw",
            }
        )
    return result


class V2Optimizer:
    def __init__(self, model: nn.Module, lr: float, weight_decay: float) -> None:
        self.routes = routes(model)
        parameters = dict(model.named_parameters())
        self.views: list[tuple[nn.Parameter, list[nn.Parameter]]] = []
        matrix, auxiliary = [], []
        for route in self.routes:
            p = parameters[route["name"]]
            if route["optimizer"] == "adamw":
                auxiliary.append(p)
            else:
                parts = [nn.Parameter(t) for t in p.detach().chunk(route["heads"], 0)]
                self.views.append((p, parts))
                matrix.extend(parts)
        self.muon = torch.optim.Muon(
            matrix,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            eps=1e-7,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adamw = torch.optim.AdamW(
            auxiliary, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
        )
        self.model = model
        self.steps = 0

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

    def step(self, lr: float) -> None:
        for p, parts in self.views:
            gradients = [None] * len(parts) if p.grad is None else p.grad.chunk(len(parts), 0)
            for part, grad in zip(parts, gradients, strict=True):
                part.grad = None if grad is None else grad.detach().clone()
        for opt in (self.muon, self.adamw):
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()  # type: ignore[no-untyped-call]
        self.steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "routes": self.routes,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["routes"] != self.routes:
            raise ValueError("checkpoint optimizer routes differ")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.steps = state["steps"]
