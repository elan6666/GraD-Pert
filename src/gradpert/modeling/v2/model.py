"""Version-isolated GraD-Pert v2 graph, control and response paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from gradpert.config.v2 import V2Architecture

from .operators import (
    CrossManifoldResidual,
    GatedFeedForward,
    LatentAttention,
    LatentCrossAttention,
    ManifoldResidual,
    RelayDeltaAttention,
    TokenEncoder,
    relay_order,
)


@dataclass
class GraphContext:
    """Aligned graph context for exact selected-node propagation or relay reads."""

    ids: Tensor
    neighbors: Tensor
    valid: Tensor
    sources: Tensor
    query_positions: Tensor
    query_neighbors: Tensor


class SparseRead(nn.Module):
    """Read gene identities or updated graph states, never control expression.

    Neighbors contain one entry per edge union with four-bit source membership:
    GO, STRING, expander, self. Padding has valid=False; source sums do not
    duplicate neighbors. No dense query-by-all-gene score matrix is formed.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float,
        rank: int | None = None,
        ffn_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.compress = nn.Linear(width, rank, bias=False) if rank is not None else None
        self.latent_norm = nn.RMSNorm(rank) if rank is not None else None
        self.key = nn.Linear(rank or width, width, bias=False)
        self.value = nn.Linear(rank or width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(4, heads))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = (
            GatedFeedForward(width, dropout)
            if ffn_type == "swiglu"
            else nn.Sequential(
                nn.Linear(width, 4 * width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * width, width),
            )
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
        if self.compress is not None:
            assert self.latent_norm is not None
            selected = self.latent_norm(self.compress(selected))
        key = self.key(selected).reshape(n, k, self.heads, self.head_width)
        value = self.value(selected).reshape(n, k, self.heads, self.head_width)
        score = torch.einsum("nhd,nkhd->nhk", q.float(), key.float()) / self.head_width**0.5
        bias = sources.to(self.source_bias.dtype) @ self.source_bias
        score = score + bias.permute(0, 2, 1).float()
        weights = score.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
        read = torch.einsum("nhk,nkhd->nhd", weights.to(value.dtype), value).flatten(-2)
        x = query + self.dropout(self.output(read))
        return cast(Tensor, x + self.dropout(self.ffn(self.norm2(x))))


class RelayGraphLayer(nn.Module):
    """One target receives one output from only its own randomized neighborhood."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float,
        chunk_rows: int = 64,
        checkpoint_chunks: bool = False,
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.read = RelayDeltaAttention(width, heads)
        self.ffn = GatedFeedForward(width, dropout)
        self.dropout = nn.Dropout(dropout)
        self.chunk_rows = chunk_rows
        self.checkpoint_chunks = checkpoint_chunks
        self.validate_once = False

    def _chunk(
        self,
        query: Tensor,
        normalized_query: Tensor,
        normalized: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
    ) -> Tensor:
        read = self.read.graph(
            query=normalized_query,
            memory=normalized,
            neighbors=neighbors,
            valid=valid,
            sources=sources,
            neighborhoods_validated=self.validate_once,
        )
        x = query + self.dropout(read)
        return cast(Tensor, x + self.dropout(self.ffn(self.norm2(x))))

    def forward(self, memory: Tensor, neighbors: Tensor, valid: Tensor, sources: Tensor) -> Tensor:
        if self.validate_once and not valid.any(-1).all():
            raise ValueError("every graph target needs a valid neighbor")
        normalized = self.norm1(memory)
        outputs = []
        for start in range(0, len(memory), self.chunk_rows):
            end = min(start + self.chunk_rows, len(memory))
            args = (
                memory[start:end],
                normalized[start:end],
                normalized,
                neighbors[start:end],
                valid[start:end],
                sources[start:end],
            )
            outputs.append(
                checkpoint(self._chunk, *args, use_reentrant=False)
                if self.checkpoint_chunks and self.training and torch.is_grad_enabled()
                else self._chunk(*args)
            )
        return torch.cat(outputs)


class RelayResponseEncoder(nn.Module):
    """Per-layer perturbation injection, self read, control cross read, FFN."""

    def __init__(self, options: V2Architecture) -> None:
        super().__init__()
        d, streams = options.width, options.streams
        self.streams = streams
        self.checkpoint_layers = options.checkpoint_layers
        self.injections = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(options.kda_layers + 1)
        )
        self.self_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for index in range(options.kda_layers + 1):
            self.self_layers.append(
                ManifoldResidual(
                    d,
                    streams,
                    RelayDeltaAttention(d, options.heads)
                    if index < options.kda_layers
                    else LatentAttention(d, options.heads, options.latent_rank, options.dropout),
                )
            )
            self.cross_layers.append(
                CrossManifoldResidual(
                    d,
                    streams,
                    RelayDeltaAttention(d, options.heads)
                    if index < options.kda_layers
                    else LatentCrossAttention(
                        d, options.heads, options.latent_rank, options.dropout
                    ),
                )
            )
            self.ffn_layers.append(
                ManifoldResidual(d, streams, GatedFeedForward(d, options.dropout))
            )
        self.norm = nn.RMSNorm(d)

    def _layer(
        self,
        x: Tensor,
        control: Tensor,
        condition: Tensor,
        order: Tensor,
        index: int,
        block_cls_to_gene: bool,
    ) -> Tensor:
        p = condition[:, None, None, :].expand_as(x)
        x = self.injections[index](torch.cat((x, p), dim=-1))
        x = self.self_layers[index](x, order=order, block_cls_to_gene=block_cls_to_gene)
        x = self.cross_layers[index](x, memory=control, order=order)
        return cast(Tensor, self.ffn_layers[index](x))

    def forward(
        self,
        x: Tensor,
        control: Tensor,
        condition: Tensor,
        orders: tuple[Tensor, ...],
        *,
        block_cls_to_gene: bool = False,
    ) -> Tensor:
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.streams, x.shape[-1])
        for index in range(len(self.injections)):
            order = orders[index] if index < len(orders) else orders[-1]
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda state, memory, perturbation, scan_order, layer_index=index: self._layer(
                        state,
                        memory,
                        perturbation,
                        scan_order,
                        layer_index,
                        block_cls_to_gene,
                    ),
                    x,
                    control,
                    condition,
                    order,
                    use_reentrant=False,
                )
            else:
                x = self._layer(x, control, condition, order, index, block_cls_to_gene)
        return cast(Tensor, self.norm(x.mean(-2)))


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
        if options.graph_read_mode == "relay":
            # Zero queries stay zero across bias-free relay blocks; repeated
            # normalize(0) derivatives otherwise amplify masked-node gradients.
            nn.init.normal_(self.mask_token, std=0.02)
        self.layers = nn.ModuleList(
            (
                [
                    RelayGraphLayer(
                        options.width,
                        options.heads,
                        options.dropout,
                        checkpoint_chunks=options.checkpoint_layers,
                    )
                    for _ in range(3)
                ]
                + [
                    SparseRead(
                        options.width,
                        options.heads,
                        options.dropout,
                        options.latent_rank,
                        options.ffn_type,
                    )
                ]
            )
            if options.graph_read_mode == "relay"
            else [
                SparseRead(options.width, options.heads, options.dropout)
                for _ in range(options.graph_layers)
            ]
        )
        self.read_mode = options.graph_read_mode

    def forward(
        self,
        ids: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        masked_ids: Tensor | None = None,
        context: GraphContext | None = None,
    ) -> Tensor:
        memory = self.norm(self.adapter(self.embedding.weight))
        if masked_ids is not None:
            memory = memory.index_copy(0, masked_ids, self.mask_token.expand(len(masked_ids), -1))
        if self.read_mode == "static":
            if context is not None:
                raise ValueError("static graph read cannot use propagation context")
            x = memory[ids]
            for layer in self.layers:
                x = layer(x, memory, neighbors, valid, sources)
            return cast(Tensor, x)
        if self.read_mode == "relay":
            if context is None or len(self.layers) != 4:
                raise ValueError("relay graph read requires four layers and a context")
            x = memory[context.ids]
            for layer in self.layers[:3]:
                x = layer(x, context.neighbors, context.valid, context.sources)
            x = self.layers[3](x, x, context.neighbors, context.valid, context.sources)
            return cast(Tensor, x[context.query_positions])
        if context is None or len(self.layers) != 2:
            raise ValueError("propagated graph read requires two layers and a context")
        first = self.layers[0](
            memory[context.ids], memory, context.neighbors, context.valid, context.sources
        )
        return cast(
            Tensor,
            self.layers[1](
                first[context.query_positions],
                first,
                context.query_neighbors,
                valid,
                sources,
            ),
        )


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
    randomize_relay_order: bool

    def __init__(self, seeds: Tensor, options: V2Architecture) -> None:
        super().__init__()
        self.options = options
        self.randomize_relay_order = False
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
            options.kda_layers,
        )
        self.cell = TokenEncoder(*args)
        self.condition_fusion = (
            nn.Identity() if options.attention == "relay_full" else nn.Linear(2 * d, d)
        )
        self.response = (
            RelayResponseEncoder(options)
            if options.attention == "relay_full"
            else TokenEncoder(*args)
        )
        prediction_width = 2 * d if options.attention == "relay_full" else d
        self.prediction = nn.Sequential(nn.Linear(prediction_width, d), nn.GELU(), nn.Linear(d, 1))
        self.ssl1_cls = Projector(options)
        self.ssl1_node = Projector(options)
        self.ssl2_cls = Projector(options)
        self.ssl2_node = Projector(options)
        if options.attention == "relay_full":
            self.set_relay_order_randomization(self.training)
        for module in self.modules():
            if isinstance(module, ManifoldResidual):
                module.sinkhorn_backend = options.sinkhorn_backend
            if isinstance(module, RelayGraphLayer):
                module.validate_once = options.relay_validate_once
            if isinstance(module, RelayDeltaAttention):
                module.write_passes = options.relay_passes
                module.compiled_chunks = options.relay_kernel == "inductor"
                module.replay_sequences = options.relay_kernel == "cudagraphs"
                module.eval_seed = (
                    options.relay_eval_seed if options.relay_eval_seed is not None else 1
                )

    def set_relay_order_randomization(self, enabled: bool) -> None:
        """Teacher can remain in eval mode while receiving randomized training views."""
        self.randomize_relay_order = enabled
        for module in self.modules():
            if isinstance(module, RelayDeltaAttention):
                module.randomize_order = enabled

    def train(self, mode: bool = True) -> GraDPertV2:
        super().train(mode)
        if self.options.attention == "relay_full":
            self.set_relay_order_randomization(mode)
        return self

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
        relay = self.options.attention == "relay_full"
        order = (
            tuple(
                relay_order(
                    len(control),
                    control.shape[1],
                    control.device,
                    getattr(self, "randomize_relay_order", self.training),
                    layer,
                    self.options.relay_eval_seed if self.options.relay_eval_seed is not None else 1,
                )
                for layer in range(self.options.kda_layers)
            )
            if relay
            else None
        )
        encoded = self.cell(
            torch.cat((x, self.control_cls.expand(len(control), -1, -1)), dim=1), order=order
        )
        basal, control_cls = encoded[:, :-1], encoded[:, -1]
        if relay:
            if not isinstance(order, tuple):
                raise AssertionError("relay order was not initialized")
            if block_response_cls_to_gene and self.training:
                raise ValueError("CLS edge intervention is an evaluation-only diagnostic")
            response = self.response(
                torch.cat((basal, self.response_cls.expand(len(control), -1, -1)), dim=1),
                basal,
                condition,
                order,
                block_cls_to_gene=block_response_cls_to_gene,
            )
            joint = torch.cat((response[:, :-1], condition[:, None, :].expand_as(basal)), dim=-1)
            delta = self.prediction(joint).squeeze(-1)
            return {
                "prediction": control + delta,
                "delta": delta,
                "control_cls": control_cls,
                "response_cls": response[:, -1],
                "response_tokens": response[:, :-1],
            }
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
