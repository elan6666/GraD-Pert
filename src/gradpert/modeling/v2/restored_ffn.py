"""Scoped restored-B0 FFNs; no implicit normalization, bias or width changes."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture

from .model import GraDPertV2, RelayGraphLayer, SparseRead
from .operators import GatedFeedForward, ManifoldResidual


class UnclippedGate(nn.Module):
    def __init__(
        self, input_width: int, hidden_width: int, output_width: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(input_width, hidden_width, bias=bias)
        self.up = nn.Linear(input_width, hidden_width, bias=bias)
        self.down = nn.Linear(hidden_width, output_width, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down(F.silu(self.gate(x)) * self.up(x)))


def configure_restored(model: GraDPertV2, raw_width: int, options: V2Architecture) -> None:
    d = options.width
    devices = (
        [model.graph.embedding.weight.device.index] if model.graph.embedding.weight.is_cuda else []
    )
    # The raw prior stays frozen; every gene shares these learned mappings.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed((torch.initial_seed() + 0x71F4) % (2**63))
        h, m = options.genept_hidden_width, options.genept_middle_width
        model.graph.adapter = nn.Sequential(
            nn.Linear(raw_width, h), nn.GELU(), nn.Linear(h, m), nn.GELU(), nn.Linear(m, d)
        )
        model.graph.projection_activation = nn.Identity()
        for module in model.modules():
            if isinstance(module, GatedFeedForward):
                module.clipped = options.ffn_clip
        if options.ffn_variant == "swiglu":
            # Hidden widths and bias conventions follow their historical roles.
            # No new pre-norm, dropout, or scalar task output activation is added.
            model.graph.adapter = nn.Sequential(
                UnclippedGate(raw_width, h, m), nn.SiLU(), nn.Linear(m, d)
            )
            model.expression = nn.Sequential(UnclippedGate(1, d, d), nn.LayerNorm(d))
            model.response.injections = nn.ModuleList(
                UnclippedGate(2 * d, d, d) for _ in model.response.injections
            )
            model.prediction = UnclippedGate(2 * d, d, 1)
            for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
                head.features = UnclippedGate(
                    d, options.projector_hidden, options.projector_bottleneck
                )
        if options.ffn_input_norm == "rmsnorm":
            for module in list(model.modules()):
                if isinstance(module, (SparseRead, RelayGraphLayer)):
                    module.norm2 = nn.RMSNorm(d, eps=1e-6)
                elif isinstance(module, ManifoldResidual) and isinstance(
                    module.sublayer, GatedFeedForward
                ):
                    module.norm = nn.RMSNorm(d, eps=1e-6)
            # Lift expression to a vector before normalization; never RMSNorm a scalar.
            model.expression.insert(1, nn.RMSNorm(d, eps=1e-6))
            model.graph.adapter.insert(0, nn.RMSNorm(raw_width, eps=1e-6))
            model.response.injections = nn.ModuleList(
                nn.Sequential(nn.RMSNorm(2 * d, eps=1e-6), block)
                for block in model.response.injections
            )
            model.prediction = nn.Sequential(nn.RMSNorm(2 * d, eps=1e-6), model.prediction)
            for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
                head.features = nn.Sequential(nn.RMSNorm(d, eps=1e-6), head.features)
