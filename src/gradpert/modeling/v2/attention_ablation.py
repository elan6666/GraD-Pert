"""Native attention cores for complete, config-selected functional ablations.

Retention equation provenance is pinned in the functional-ablation plan. No
upstream code or runtime is imported; graph edge features are a local adaptation.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def retention_normalize(x: Tensor) -> Tensor:
    """Non-affine scaled RMS normalization, including a finite all-zero read."""
    value = x.float()
    denominator = torch.linalg.vector_norm(value / math.sqrt(x.shape[-1]), dim=-1, keepdim=True)
    return cast(Tensor, (value / denominator.clamp_min(1e-12)).to(x.dtype))


class ReplacementAttention(nn.Module):
    """Noncausal self/cross attention and independent legal graph-neighborhood reads."""

    def __init__(self, width: int, heads: int, dropout: float, kind: str) -> None:
        super().__init__()
        if kind not in ("softmax", "retention"):
            raise ValueError("unknown native replacement core")
        self.kind, self.heads, self.head_width = kind, heads, width // heads
        self.dropout = dropout
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        # Only graph calls consume edge embeddings; identity is still the
        # original gene ID, regardless of the neighborhood's storage order.
        self.source = nn.Linear(4, width, bias=False)
        self.output_gate = nn.Linear(width, width, bias=False) if kind == "retention" else None

    def forward(
        self, query: Tensor, memory: Tensor | None = None, *, block_cls_to_gene: bool = False
    ) -> Tensor:
        control = query if memory is None else memory
        shape_q = (*query.shape[:2], self.heads, self.head_width)
        shape_k = (*control.shape[:2], self.heads, self.head_width)
        q = self.query(query).reshape(shape_q).transpose(1, 2)
        k = self.key(control).reshape(shape_k).transpose(1, 2)
        v = self.value(control).reshape(shape_k).transpose(1, 2)
        if self.kind == "softmax":
            mask = None
            if block_cls_to_gene and memory is None:
                mask = torch.ones(
                    query.shape[1], query.shape[1], device=query.device, dtype=torch.bool
                )
                mask[:-1, -1] = False
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            q, k = (
                F.relu(q).float() / math.sqrt(self.head_width),
                F.relu(k).float() / math.sqrt(self.head_width),
            )
            state = k.transpose(-1, -2) @ v.float()
            y = q @ state
            if block_cls_to_gene and memory is None:
                gene_state = k[:, :, :-1].transpose(-1, -2) @ v[:, :, :-1].float()
                y = torch.cat((q[:, :, :-1] @ gene_state, y[:, :, -1:]), dim=2)
            y = retention_normalize(y).to(query.dtype)
            assert self.output_gate is not None
            gate = F.silu(self.output_gate(query)).reshape(shape_q).transpose(1, 2)
            y = y * gate
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))

    def graph(
        self,
        query: Tensor,
        memory: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        *,
        neighborhoods_validated: bool = False,
    ) -> Tensor:
        if sources.shape != (*neighbors.shape, 4) or neighbors.shape != valid.shape:
            raise ValueError("invalid graph neighborhood shape")
        if not neighborhoods_validated and not valid.any(-1).all():
            raise ValueError("every graph target needs a valid neighbor")
        selected = memory[neighbors.clamp_min(0)] + self.source(sources.to(memory.dtype))
        n, length = neighbors.shape
        q = self.query(query).reshape(n, self.heads, self.head_width)
        k = self.key(selected).reshape(n, length, self.heads, self.head_width)
        v = self.value(selected).reshape(n, length, self.heads, self.head_width)
        if self.kind == "softmax":
            score = torch.einsum("nhd,nkhd->nhk", q.float(), k.float()) / math.sqrt(self.head_width)
            weights = score.masked_fill(~valid[:, None], -torch.inf).softmax(-1).to(v.dtype)
            weights = F.dropout(weights, self.dropout, self.training)
        else:
            weights = (
                torch.einsum("nhd,nkhd->nhk", F.relu(q).float(), F.relu(k).float())
                / self.head_width
            )
            weights = weights.masked_fill(~valid[:, None], 0)
        y = torch.einsum("nhk,nkhd->nhd", weights.to(v.dtype), v)
        if self.kind == "retention":
            assert self.output_gate is not None
            y = retention_normalize(y) * F.silu(self.output_gate(query)).reshape_as(y)
        return cast(Tensor, self.output(y.flatten(-2)))
