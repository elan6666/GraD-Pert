"""Explicit independent inference arithmetic after strict training-state restore."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import cast

import torch

from .model import GraDPertV2

FIELDS = ("ffn_compute_precision", "ssl_head_precision", "prediction_precision")


def evaluation_precision(value: object = "native") -> str:
    if type(value) is not str or value not in ("native", "common_ri_native"):
        raise ValueError("evaluation_precision must be native or common_ri_native")
    return value


def evaluation_precision_context(policy: str, device_type: str) -> AbstractContextManager[None]:
    """Reproduce unautocast RI inference, even inside an outer autocast scope."""
    evaluation_precision(policy)
    if policy == "native":
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


def precision_policy(model: GraDPertV2) -> dict[str, str]:
    return {field: getattr(model.options, field) for field in FIELDS}


def evaluation_student(model: GraDPertV2, policy: str = "native") -> GraDPertV2:
    """Build a separate common-policy Student without touching its restored owner."""
    evaluation_precision(policy)
    if policy == "native":
        return model
    if type(model) is not GraDPertV2 or not any(
        value != "autocast" for value in precision_policy(model).values()
    ):
        raise ValueError("common_ri_native requires an explicit precision-ablation Student")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("common_ri_native requires FP32 master parameters")
    # Constructor needs the original raw prior axis, not a reduced adapter output.
    seeds = model.graph.embedding.weight.detach().clone()
    devices = [seeds.device.index] if seeds.is_cuda else []
    # manual_seed inside native constructors can touch every CUDA generator.
    if cast(Callable[[], bool], torch.cuda.is_initialized)():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        result = GraDPertV2(
            seeds,
            replace(
                model.options,
                ffn_compute_precision="autocast",
                ssl_head_precision="autocast",
                prediction_precision="autocast",
            ),
        ).to(device=seeds.device)
        result.load_state_dict(model.state_dict(), strict=True)
    source_modules = dict(model.named_modules())
    for name, module in result.named_modules():
        module.training = source_modules[name].training
    source_parameters = dict(model.named_parameters())
    for name, parameter in result.named_parameters():
        parameter.requires_grad_(source_parameters[name].requires_grad)
    result.randomize_relay_order = model.randomize_relay_order
    return result
