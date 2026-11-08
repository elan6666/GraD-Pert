"""Scoped residual-entrance and encoder-exit normalization experiments."""

from typing import cast

import torch
from torch import nn

from .model import GraDPertV2, RelayResponseEncoder, SparseRead
from .operators import LatentAttention, ManifoldResidual
from .standard_attention import StandardAttention


def configure_norm_layout(model: GraDPertV2) -> None:
    """Preserve operator/target normalizers; change only the declared layout."""
    options = model.options
    d = options.width
    response = cast(RelayResponseEncoder, model.response)
    devices = (
        [model.graph.embedding.weight.device.index] if model.graph.embedding.weight.is_cuda else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed((torch.initial_seed() + 0x71F6) % (2**63))
        model.graph.norm = nn.Identity()
        model.graph.final_norm = nn.RMSNorm(d, eps=1e-6)
        expression = model.expression
        if not isinstance(expression[-1], nn.LayerNorm):
            raise ValueError("normalization layout expects the restored expression mapping")
        expression[-1] = nn.Identity()
        for layer in model.graph.layers:
            for name in ("norm1", "norm2"):
                previous = getattr(layer, name)
                if not isinstance(previous, nn.LayerNorm) or previous.weight is None:
                    raise ValueError("normalization layout expects affine graph entrances")
                replacement = nn.RMSNorm(previous.normalized_shape, eps=previous.eps)
                with torch.no_grad():
                    replacement.weight.copy_(previous.weight)
                setattr(layer, name, replacement)
        response.injections = nn.ModuleList(
            nn.Sequential(nn.RMSNorm(2 * d, eps=1e-6), block) for block in response.injections
        )
        if options.terminal_attention == "standard":
            for raw_wrapper in (
                model.cell.layers[-2],
                response.self_layers[-1],
                response.cross_layers[-1],
            ):
                wrapper = cast(ManifoldResidual, raw_wrapper)
                old = cast(LatentAttention, wrapper.sublayer)
                new = StandardAttention(d, options.heads, options.dropout)
                # Preserve common Q/output initial values; only new full-width KV is sampled.
                new.query.load_state_dict(old.query.state_dict())
                new.output.load_state_dict(old.output.state_dict())
                wrapper.sublayer = new
            last = cast(SparseRead, model.graph.layers[-1])
            last.compress = None
            last.latent_norm = None
            last.key = nn.Linear(d, d, bias=False)
            last.value = nn.Linear(d, d, bias=False)
