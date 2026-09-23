"""Version-isolated GraD-Pert v2 graph, control and response paths."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture

from .operators import TokenEncoder


class SparseRead(nn.Module):
    """Memory is static gene identity, never control-expression dependent.

    Neighbors contain one entry per edge union with four-bit source membership:
    GO, STRING, expander, self. Padding has valid=False; source sums do not
    duplicate neighbors. No dense query-by-all-gene score matrix is formed.
    """

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(4, heads))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, memory: Tensor, neighbors: Tensor, valid: Tensor, sources: Tensor
    ) -> Tensor:
        if neighbors.shape != valid.shape or not valid.any(-1).all():
            raise ValueError("every query needs at least one valid memory edge")
        if sources.shape != (*neighbors.shape, 4):
            raise ValueError("expected four-source edge memberships")
        n, k = neighbors.shape
        q = self.query(self.norm1(query)).reshape(n, self.heads, self.head_width)
        selected = memory[neighbors.clamp_min(0)]
        key = self.key(selected).reshape(n, k, self.heads, self.head_width)
        value = self.value(selected).reshape(n, k, self.heads, self.head_width)
        score = torch.einsum("nhd,nkhd->nhk", q.float(), key.float()) / self.head_width**0.5
        bias = sources.to(self.source_bias.dtype) @ self.source_bias
        score = score + bias.permute(0, 2, 1).float()
        weights = score.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
        read = torch.einsum("nhk,nkhd->nhd", weights.to(value.dtype), value).flatten(-2)
        x = query + self.dropout(self.output(read))
        return cast(Tensor, x + self.dropout(self.ffn(self.norm2(x))))


class GeneGraph(nn.Module):
    def __init__(self, seeds: Tensor, options: V2Architecture) -> None:
        super().__init__()
        if seeds.ndim != 2 or not torch.isfinite(seeds).all():
            raise ValueError("finite aligned gene seed table required")
        self.embedding = nn.Embedding(seeds.shape[0], seeds.shape[1])
        with torch.no_grad():
            self.embedding.weight.copy_(seeds)
        # An already reduced GenePT table is the model-width representation;
        # the historical 2048-wide route retains its learned adapter unchanged.
        self.adapter = (
            nn.Identity()
            if seeds.shape[1] == options.width
            else nn.Linear(seeds.shape[1], options.width)
        )
        self.norm = nn.LayerNorm(options.width)
        self.mask_token = nn.Parameter(torch.zeros(options.width))
        self.layers = nn.ModuleList(
            [
                SparseRead(options.width, options.heads, options.dropout)
                for _ in range(options.graph_layers)
            ]
        )

    def forward(
        self,
        ids: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        masked_ids: Tensor | None = None,
    ) -> Tensor:
        memory = self.norm(self.adapter(self.embedding.weight))
        if masked_ids is not None:
            memory = memory.index_copy(0, masked_ids, self.mask_token.expand(len(masked_ids), -1))
        x = memory[ids]
        for layer in self.layers:
            x = layer(x, memory, neighbors, valid, sources)
        return cast(Tensor, x)


class Projector(nn.Module):
    def __init__(self, options: V2Architecture) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(options.width, options.projector_hidden),
            nn.GELU(),
            nn.Linear(options.projector_hidden, options.projector_bottleneck),
        )
        self.prototypes = nn.Linear(options.projector_bottleneck, options.prototypes, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.prototypes(F.normalize(self.features(x).float(), dim=-1)))


class GraDPertV2(nn.Module):
    model_version = "v2"

    def __init__(self, seeds: Tensor, options: V2Architecture) -> None:
        super().__init__()
        self.options = options
        d = options.width
        self.graph = GeneGraph(seeds, options)
        self.expression = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.expression_mask = nn.Parameter(torch.zeros(d))
        self.control_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.response_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        args = (
            d,
            options.heads,
            options.latent_rank,
            options.streams,
            options.dropout,
            options.attention,
            options.checkpoint_layers,
            options.ffn_type,
            options.sparse_topk,
            options.sparse_index_dim,
            options.sparse_query_chunk,
        )
        self.cell = TokenEncoder(*args)
        self.condition_fusion = nn.Linear(2 * d, d)
        self.response = TokenEncoder(*args)
        self.prediction = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.ssl1_cls = Projector(options)
        self.ssl1_node = Projector(options)
        self.ssl2_cls = Projector(options)
        self.ssl2_node = Projector(options)

    @staticmethod
    def aggregate_targets(graph: Tensor, positions: Tensor, valid: Tensor) -> Tensor:
        if positions.shape != valid.shape or not valid.any(-1).all():
            raise ValueError("each condition must have at least one target")
        values = graph[positions.clamp_min(0)] * valid.unsqueeze(-1)
        return values.sum(-2) / valid.sum(-1, keepdim=True)

    def encode_response(
        self,
        gene: Tensor,
        control: Tensor,
        condition: Tensor,
        expression_mask: Tensor | None = None,
        *,
        block_response_cls_to_gene: bool = False,
    ) -> dict[str, Tensor]:
        if control.ndim != 2 or control.shape[1] != gene.shape[0]:
            raise ValueError("control expression must align with query gene IDs")
        expression = self.expression(control.unsqueeze(-1))
        if expression_mask is not None:
            if expression_mask.shape != control.shape or expression_mask.dtype != torch.bool:
                raise ValueError("expression mask must be aligned boolean tensor")
            expression = torch.where(
                expression_mask.unsqueeze(-1), self.expression_mask, expression
            )
        x = expression + gene.unsqueeze(0)
        encoded = self.cell(torch.cat((x, self.control_cls.expand(len(control), -1, -1)), dim=1))
        basal, control_cls = encoded[:, :-1], encoded[:, -1]
        joint = torch.cat((basal, condition[:, None, :].expand_as(basal)), dim=-1)
        response = self.response(
            torch.cat(
                (self.condition_fusion(joint), self.response_cls.expand(len(control), -1, -1)),
                dim=1,
            ),
            block_cls_to_gene=block_response_cls_to_gene,
        )
        delta = self.prediction(response[:, :-1]).squeeze(-1)
        return {
            "prediction": control + delta,
            "delta": delta,
            "control_cls": control_cls,
            "response_cls": response[:, -1],
            "response_tokens": response[:, :-1],
        }
