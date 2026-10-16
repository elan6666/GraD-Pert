"""Scoped restored-B0 FFNs; no implicit normalization, bias or width changes."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture

from .conditional_graph import ControlGraphModulation, ControlSummary
from .model import GraDPertV2, RelayGraphLayer, RelayResponseEncoder, SparseRead
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


class Float32LayerNorm(nn.LayerNorm):
    """FP32 mean/variance and affine arithmetic, returning the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(x.dtype)


class MatchedGELUFeedForward(nn.Module):
    """Two matrices with 6d hidden width match the original three 4d matrices."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(width, 6 * width, bias=False)
        self.down = nn.Linear(6 * width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(self.down(self.dropout(F.gelu(self.up(x))))))


def configure_restored(model: GraDPertV2, raw_width: int, options: V2Architecture) -> None:
    d = options.width
    response = cast(RelayResponseEncoder, model.response)
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
            response.injections = nn.ModuleList(
                UnclippedGate(2 * d, d, d) for _ in response.injections
            )
            model.prediction = UnclippedGate(2 * d, d, 1)
            for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
                head.features = UnclippedGate(
                    d, options.projector_hidden, options.projector_bottleneck
                )
        if options.ffn_variant == "gelu":
            for module in list(model.modules()):
                if isinstance(module, (SparseRead, RelayGraphLayer)):
                    module.ffn = MatchedGELUFeedForward(d, options.dropout)
                elif isinstance(module, ManifoldResidual) and isinstance(
                    module.sublayer, GatedFeedForward
                ):
                    module.sublayer = MatchedGELUFeedForward(d, options.dropout)
        if options.ffn_input_norm in ("rmsnorm", "layernorm"):

            def pre_norm(width: int) -> nn.Module:
                return (
                    nn.RMSNorm(width, eps=1e-6)
                    if options.ffn_input_norm == "rmsnorm"
                    else Float32LayerNorm(width, eps=1e-6)
                )

            for module in list(model.modules()):
                if isinstance(module, (SparseRead, RelayGraphLayer)):
                    module.norm2 = pre_norm(d)
                elif isinstance(module, ManifoldResidual) and isinstance(
                    module.sublayer, (GatedFeedForward, MatchedGELUFeedForward)
                ):
                    module.norm = pre_norm(d)
            # Lift expression to a vector before normalization; never RMSNorm a scalar.
            model.expression.insert(1, pre_norm(d))
            cast(nn.Sequential, model.graph.adapter).insert(0, pre_norm(raw_width))
            response.injections = nn.ModuleList(
                nn.Sequential(pre_norm(2 * d), block) for block in response.injections
            )
            model.prediction = nn.Sequential(pre_norm(2 * d), model.prediction)
            for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
                head.features = nn.Sequential(pre_norm(d), head.features)
        if options.ffn_linear_bias == "none":
            # These role roots exclude Q/K/V, KDA state gates, mHC routing and norm beta.
            roles: list[nn.Module] = [model.graph.adapter, model.expression, model.prediction]
            roles.extend(response.injections)
            roles.extend(cast(nn.Module, layer.ffn) for layer in model.graph.layers)
            roles.extend(
                module.sublayer
                for module in model.modules()
                if isinstance(module, ManifoldResidual)
                and isinstance(module.sublayer, (GatedFeedForward, MatchedGELUFeedForward))
            )
            roles.extend(
                head.features
                for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node)
            )
            for root in roles:
                for module in root.modules():
                    if isinstance(module, nn.Linear):
                        module.register_parameter("bias", None)
        if options.prototype_weight_norm:
            from torch.nn.utils.parametrizations import weight_norm

            for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
                weight_norm(head.prototypes, name="weight", dim=0)
                # v retains the original Linear initialization; g begins at one.
                with torch.no_grad():
                    dict(head.prototypes.named_parameters())[
                        "parametrizations.weight.original0"
                    ].fill_(1)
                if options.prototype_fixed_scale:
                    dict(head.prototypes.named_parameters())[
                        "parametrizations.weight.original0"
                    ].requires_grad_(False)
    # Optional mechanisms have their own RNG streams; common B0 initialization
    # and dropout streams are identical when an adapter starts at zero.
    if options.gene_conditioned_readout:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed((torch.initial_seed() + 0x71F1) % (2**63))
            final = nn.Linear(max(1, d // 4), d)
            model.prior_readout = nn.Sequential(
                nn.RMSNorm(raw_width, eps=1e-6),
                nn.Linear(raw_width, max(1, d // 4)),
                nn.GELU(),
                final,
            )
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
    if options.control_conditioned_graph:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed((torch.initial_seed() + 0x71F2) % (2**63))
            model.control_summary = ControlSummary(d, gelu=True)
            model.graph_modulation = ControlGraphModulation(d, options.heads)
