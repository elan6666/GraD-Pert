"""Config-selected hidden-matrix optimizer with explicit native head routing.

Uses PyTorch Muon, not an upstream model runtime. The registered RMS adjustment
is a project transfer choice, not an assertion about GLM's unpublished code.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from gradpert.modeling.encoders import _SparseGraphTransformerLayer


def parameter_routes(model: nn.Module) -> list[dict[str, Any]]:
    """One stable, disjoint route per trainable parameter, including exclusions."""
    modules = dict(model.named_modules())
    routes: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        path, _, field = name.rpartition(".")
        module = modules.get(path)
        root = name.split(".")[0]
        hidden = (
            isinstance(module, nn.Linear)
            and field == "weight"
            and root
            in {"student_encoder", "student_projector", "basal_encoder", "expression_decoder"}
            and name != "expression_decoder.network.4.weight"
            and "prototype_layer" not in name
        )
        heads = 1
        owner_path, _, projection = path.rpartition(".")
        owner = modules.get(owner_path)
        if (
            hidden
            and isinstance(owner, _SparseGraphTransformerLayer)
            and projection in {"query", "key", "value"}
        ):
            heads = owner.head_count
        if hidden and (parameter.ndim != 2 or parameter.shape[0] % heads):
            raise ValueError(f"invalid hidden matrix/head layout: {name}")
        rows = parameter.shape[0] // heads if hidden else 0
        correction = 0.2 * math.sqrt(max(rows, parameter.shape[1])) if hidden else 1.0
        routes.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "optimizer": "muon" if hidden else "adamw",
                "heads": heads,
                "head_axis": 0 if heads > 1 else None,
                "lr_multiplier": correction,
            }
        )
    if not routes or not any(r["heads"] > 1 for r in routes):
        raise ValueError("split optimizer requires native sparse-transformer Q/K/V heads")
    return routes


class SplitMatrixAdamW(torch.optim.Optimizer):
    """One external step/checkpoint interface for head-wise Muon and AdamW.

    Muon head parameters are detached views sharing original weight storage.
    Gradients are copied into views each step; model gradients remain intact.
    Teacher parameters never enter either child optimizer.
    """

    def __init__(self, model: nn.Module, *, lr: float) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("base LR must be finite and positive")
        self.routes = parameter_routes(model)
        named = dict(model.named_parameters())
        parameters = [named[r["name"]] for r in self.routes]
        super().__init__(parameters, {"lr": lr})
        self._views: list[tuple[nn.Parameter, list[nn.Parameter]]] = []
        matrices, auxiliary = [], []
        for route in self.routes:
            parameter = named[route["name"]]
            if route["optimizer"] == "adamw":
                auxiliary.append(parameter)
                continue
            views = [nn.Parameter(x) for x in parameter.detach().chunk(route["heads"], dim=0)]
            self._views.append((parameter, views))
            matrices.extend(views)
        self.muon = torch.optim.Muon(
            matrices,
            lr=lr,
            weight_decay=0,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adamw = torch.optim.AdamW(
            auxiliary, lr=lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-8
        )
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) != 1:
            raise ValueError("mixed optimizer has exactly one externally scheduled group")
        lr = float(self.param_groups[0]["lr"])
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("scheduled LR must be finite and nonnegative")
        for parameter, views in self._views:
            if parameter.grad is None:
                for view in views:
                    view.grad = None
            else:
                if parameter.grad.is_sparse:
                    raise ValueError("sparse matrix gradients are unsupported")
                for view, grad in zip(views, parameter.grad.chunk(len(views), dim=0), strict=True):
                    view.grad = grad.clone()
        for optimizer in (self.muon, self.adamw):
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()  # type: ignore[no-untyped-call]
        self.step_count += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "native-split-matrix-adamw-v1",
            "routes": self.routes,
            "step_count": self.step_count,
            "wrapper": super().state_dict(),
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if (
            state_dict.get("schema") != "native-split-matrix-adamw-v1"
            or state_dict.get("routes") != self.routes
        ):
            raise ValueError("optimizer recipe/parameter routing differs from checkpoint")
        count = state_dict.get("step_count")
        if type(count) is not int or count < 0:
            raise ValueError("invalid optimizer step count")
        super().load_state_dict(state_dict["wrapper"])
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.step_count = count
