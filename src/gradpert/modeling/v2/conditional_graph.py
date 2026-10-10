"""Control summaries and source-only conditioning of the fourth graph layer."""

from typing import cast

import torch
from torch import Tensor, nn

from .mlp import UnifiedMLP


class ControlSummary(nn.Module):
    """Sum per-gene contributions from aligned identity/expression embeddings.

    The caller supplies the expression embeddings of the current view, including
    its expression masks. This module neither looks up genes nor reads scalar
    expression, so it cannot silently recover a hidden expression from another
    view. All genes share the three feature, contribution, and summary MLPs.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.feature = UnifiedMLP(2 * width, 4 * width, width)
        self.weight = UnifiedMLP(width, 4 * width, width)
        self.summary = UnifiedMLP(width, 4 * width, width)

    def forward(self, identity: Tensor, expression: Tensor) -> Tensor:
        if identity.ndim != 2 or expression.ndim != 3:
            raise ValueError("expected identity [genes,width] and expression [batch,genes,width]")
        if identity.shape != expression.shape[1:] or identity.shape[-1] != self.width:
            raise ValueError("identity and expression embeddings must share an aligned gene axis")
        if identity.shape[0] == 0:
            raise ValueError("a control summary needs at least one input gene")
        identities = identity.unsqueeze(0).expand(expression.shape[0], -1, -1)
        features = self.feature(torch.cat((identities, expression), dim=-1))
        contributions = self.weight(features).sigmoid()
        # No division by gene count and no softmax competition between genes.
        return cast(Tensor, self.summary((contributions * features).sum(dim=1)))


class ControlGraphModulation(nn.Module):
    """Zero-start control coefficients and learnable per-source directions."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if heads < 1 or width < 1 or width % heads:
            raise ValueError("width must be positive and divisible by the positive head count")
        self.width, self.heads = width, heads
        self.projection = nn.Linear(width, heads * 4, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.directions = nn.Parameter(torch.empty(4, heads, width // heads))
        # Only the coefficients start at zero: nonzero directions allow their
        # zero-start projection to receive a gradient on the first update.
        nn.init.normal_(self.directions, std=0.02)

    def forward(self, summary: Tensor) -> Tensor:
        if summary.ndim != 2 or summary.shape[-1] != self.width:
            raise ValueError("expected control summary [batch,width]")
        return cast(Tensor, self.projection(summary).reshape(-1, self.heads, 4).tanh())


def conditional_sparse_read(
    layer: nn.Module,
    query: Tensor,
    memory: Tensor,
    neighbors: Tensor,
    valid: Tensor,
    sources: Tensor,
    eta: Tensor,
    directions: Tensor,
) -> Tensor:
    """Apply a SparseRead layer with control-conditioned source key gates.

    For source memberships ``s``, the key gate is ``1 + sum_s s*(E+eta*U)``.
    ``E`` is the layer's original source gate, and values remain unchanged.
    Static Q/K/V and the dynamic dot-product coefficients are shared across
    controls; only attention and the resulting residual/FFN are per-control.
    This avoids a batch-expanded ``[B,N,K,H,D]`` key or gate tensor. The caller
    may partition the query axis to bound the logits/FFN live memory.
    """
    if bool(layer.retention):
        raise ValueError("conditional graph sources require the softmax SparseRead path")
    if neighbors.ndim != 2 or neighbors.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("expected neighbor indices and a boolean mask with shape [queries,edges]")
    if not valid.any(-1).all():
        raise ValueError("every query needs at least one valid memory edge")
    if sources.shape != (*neighbors.shape, 4):
        raise ValueError("expected four-source edge memberships")
    heads, head_width = cast(int, layer.heads), cast(int, layer.head_width)
    n, k = neighbors.shape
    if query.shape != (n, heads * head_width) or memory.ndim != 2:
        raise ValueError("query and memory must be aligned graph feature matrices")
    if memory.shape[-1] != heads * head_width:
        raise ValueError("memory width does not match the graph layer")
    if eta.ndim != 3 or eta.shape[1:] != (heads, 4):
        raise ValueError("expected control modulation [batch,heads,4]")
    if directions.shape != (4, heads, head_width):
        raise ValueError("expected source directions [4,heads,head_width]")

    norm1, norm2 = cast(nn.Module, layer.norm1), cast(nn.Module, layer.norm2)
    query_projection = cast(nn.Module, layer.query)
    key_projection = cast(nn.Module, layer.key)
    value_projection = cast(nn.Module, layer.value)
    output_projection = cast(nn.Module, layer.output)
    dropout, ffn = cast(nn.Module, layer.dropout), cast(nn.Module, layer.ffn)
    source_bias = cast(Tensor, layer.source_bias)
    static_gate = cast(Tensor | None, layer.source_key_gate)
    compress = cast(nn.Module | None, layer.compress)
    latent_norm = cast(nn.Module | None, layer.latent_norm)

    q = query_projection(norm1(query)).reshape(n, heads, head_width)
    # Invalid entries may use any padding sentinel; they never select a gene.
    selected = memory[torch.where(valid, neighbors, torch.zeros_like(neighbors))]
    if compress is not None:
        if latent_norm is None:
            raise ValueError("compressed graph keys require latent normalization")
        selected = latent_norm(compress(selected))
    key = key_projection(selected).reshape(n, k, heads, head_width)
    value = value_projection(selected).reshape(n, k, heads, head_width)
    q_float, key_float = q.float(), key.float()
    memberships = sources.to(dtype=torch.float32)
    scoring_key = key_float
    if static_gate is not None:
        # Keep the same static operations as SparseRead for zero-modulation
        # parity; the dynamic part is additive in the key gate, not a product.
        gate = 1 + torch.einsum("nks,shd->nkhd", sources.to(static_gate.dtype), static_gate)
        scoring_key = key_float * gate.float()
    static_score = torch.einsum("nhd,nkhd->nhk", q_float, scoring_key)
    q_directions = q_float[:, None, :, :] * directions.float()[None, :, :, :]
    coefficients = torch.einsum("nshd,nkhd->nkhs", q_directions, key_float)
    coefficients = coefficients * memberships[:, :, None, :]
    dynamic_score = torch.einsum("nkhs,bhs->bnhk", coefficients, eta.float())
    score = (static_score.unsqueeze(0) + dynamic_score) / head_width**0.5
    bias = sources.to(source_bias.dtype) @ source_bias
    score = score + bias.permute(0, 2, 1).float().unsqueeze(0)
    weights = score.masked_fill(~valid[None, :, None, :], float("-inf")).softmax(-1)
    read = torch.einsum("bnhk,nkhd->bnhd", weights.to(value.dtype), value).flatten(-2)
    x = query.unsqueeze(0) + dropout(output_projection(read))
    return cast(Tensor, x + dropout(ffn(norm2(x))))
