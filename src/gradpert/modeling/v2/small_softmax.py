"""Small graph and expression encoders with single-pass manifold connections.

Behavioral provenance and task adaptations are recorded in the versioned design.
No upstream runtime is required.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from gradpert.config.v2 import V2Architecture

    from .model import GraphContext


class FeatureRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
        return (normalized * self.weight).to(x.dtype)


class DenseSwiGLU(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, 2 * width, bias=False)
        self.up = nn.Linear(width, 2 * width, bias=False)
        self.output = nn.Linear(2 * width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate(x), self.up(x)
        # Frozen Flash config uses a limit of 10. Gate has no lower clamp.
        product = F.silu(gate.float().clamp(max=10)) * value.float().clamp(-10, 10)
        return cast(Tensor, self.output(product.to(gate.dtype)))


class StreamConnection(nn.Module):
    """Consume the preceding read coefficients and produce the next ones."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.route = nn.Linear(4 * width, 24, bias=False)
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        base = torch.zeros(24)
        base[:4] = math.log(1 / 3)
        base[4:8] = math.log(1 / 7)
        base[8:] = (4 * torch.eye(4)).flatten()
        self.base = nn.Parameter(base)
        self.norm = FeatureRMSNorm(width)
        nn.init.normal_(self.route.weight, std=(4 * width) ** -0.5)

    def prepare(self, state: Tensor, previous: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.autocast(device_type=state.device.type, enabled=False):
            flat = state.flatten(-2).float()
            scores = self.route(flat) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
            read = (scores[..., :4] * self.scale[0] + self.base[:4]).sigmoid() + 1e-6
            write = 2 * (scores[..., 4:8] * self.scale[1] + self.base[4:8]).sigmoid()
            matrix = (scores[..., 8:] * self.scale[2] + self.base[8:]).unflatten(-1, (4, 4))
            matrix = matrix.softmax(-1) + 1e-6
            matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
            for _ in range(19):
                matrix = matrix / (matrix.sum(-1, keepdim=True) + 1e-6)
                matrix = matrix / (matrix.sum(-2, keepdim=True) + 1e-6)
            collapsed = (state.float() * previous.unsqueeze(-1)).sum(-2).to(state.dtype)
        return self.norm(collapsed), read, write, matrix

    @staticmethod
    def finish(state: Tensor, update: Tensor, write: Tensor, matrix: Tensor) -> Tensor:
        mixed = (matrix.unsqueeze(-1) * state.float().unsqueeze(-2)).sum(-3)
        return (mixed + write.unsqueeze(-1) * update.float().unsqueeze(-2)).to(update.dtype)


def start_streams(x: Tensor) -> tuple[Tensor, Tensor]:
    state = x.unsqueeze(-2).expand(*x.shape[:-1], 4, x.shape[-1])
    read = x.new_zeros((*x.shape[:-1], 4), dtype=torch.float32)
    read[..., 0] = 1
    return state, read


def collapse_streams(state: Tensor, read: Tensor) -> Tensor:
    return (state.float() * read.unsqueeze(-1)).sum(-2).to(state.dtype)


class SharedKVAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.heads = 4
        self.head_width = width // 4
        self.q_down = nn.Linear(width, width // 2, bias=False)
        self.q_norm = FeatureRMSNorm(width // 2)
        self.q_up = nn.Linear(width // 2, width, bias=False)
        self.kv = nn.Linear(width, self.head_width, bias=False)
        self.kv_norm = FeatureRMSNorm(self.head_width)
        self.sink = nn.Parameter(torch.zeros(4))
        self.group_output = nn.ModuleList(
            nn.Linear(width // 2, self.head_width, bias=False) for _ in range(2)
        )
        self.output = nn.Linear(width // 2, width, bias=False)

    def forward(
        self, query: Tensor, memory: Tensor | None = None, *, block_cls: bool = False
    ) -> Tensor:
        memory = query if memory is None else memory
        q = self.q_up(self.q_norm(self.q_down(query))).unflatten(-1, (4, self.head_width))
        kv = self.kv_norm(self.kv(memory))
        # Chunk queries to bound the exact dense softmax workspace. The sink is
        # a separate zero-value entry and is included in the stable denominator.
        outputs = []
        for start in range(0, query.shape[1], 64):
            with torch.autocast(device_type=query.device.type, enabled=False):
                score = torch.einsum("bqhd,bkd->bhqk", q[:, start : start + 64].float(), kv.float())
            score = score * self.head_width**-0.5
            if block_cls:
                rows = torch.arange(start, start + score.shape[2], device=query.device)
                score[..., -1] = score[..., -1].masked_fill(
                    rows[None, None, :] < query.shape[1] - 1, -torch.inf
                )
            sink = self.sink[None, :, None, None].expand(*score.shape[:-1], 1)
            weights = torch.cat((score, sink), -1).softmax(-1)[..., :-1]
            outputs.append(torch.einsum("bhqk,bkd->bqhd", weights.to(kv.dtype), kv))
        value = torch.cat(outputs, 1).reshape(*query.shape[:-1], 2, query.shape[-1] // 2)
        projected = torch.cat(
            [layer(value[..., i, :]) for i, layer in enumerate(self.group_output)], -1
        )
        return cast(Tensor, self.output(projected))


class NeighborhoodAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.heads = 4
        self.value_width = width // 4
        self.relation_width = width // 16
        self.content_width = self.value_width - self.relation_width
        self.q_down = nn.Linear(width, width // 2, bias=False)
        self.q_norm = FeatureRMSNorm(width // 2)
        self.kv_down = nn.Linear(width, width // 2, bias=False)
        self.kv_norm = FeatureRMSNorm(width // 2)
        self.q_up = nn.Linear(width // 2, 4 * self.content_width, bias=False)
        self.k_up = nn.Linear(width // 2, 4 * self.content_width, bias=False)
        self.v_up = nn.Linear(width // 2, width, bias=False)
        self.relation_query = nn.Linear(width // 2, 4 * self.relation_width, bias=False)
        self.relation_key = nn.Linear(width, self.relation_width, bias=False)
        self.source_transform = nn.Parameter(
            torch.zeros(4, self.relation_width, self.relation_width)
        )
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, neighbors: Tensor, valid: Tensor, sources: Tensor) -> Tensor:
        if neighbors.shape != valid.shape or sources.shape != (*neighbors.shape, 4):
            raise ValueError("unaligned graph neighborhood")
        compressed_q = self.q_norm(self.q_down(x))
        compressed_kv = self.kv_norm(self.kv_down(x))
        q = self.q_up(compressed_q).unflatten(-1, (4, self.content_width))
        k = self.k_up(compressed_kv).unflatten(-1, (4, self.content_width))
        v = self.v_up(compressed_kv).unflatten(-1, (4, self.value_width))
        rq = self.relation_query(compressed_q).unflatten(-1, (4, self.relation_width))
        rk = self.relation_key(x)
        outputs = []
        for start in range(0, len(x), 64):
            edge = neighbors[start : start + 64].clamp_min(0)
            mask = valid[start : start + 64]
            with torch.autocast(device_type=x.device.type, enabled=False):
                transform = torch.eye(self.relation_width, device=x.device) + torch.einsum(
                    "nks,sij->nkij",
                    sources[start : start + 64].float(),
                    self.source_transform.float(),
                )
                query_relation = torch.einsum(
                    "nkij,nhj->nkhi", transform, rq[start : start + 64].float()
                )
                key_relation = torch.einsum("nkij,nkj->nki", transform, rk[edge].float())
                scores = torch.einsum(
                    "nhd,nkhd->nhk", q[start : start + 64].float(), k[edge].float()
                )
                scores = scores + torch.einsum("nkhi,nki->nhk", query_relation, key_relation)
            scores = scores * self.value_width**-0.5
            scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
            safe = torch.where(mask.any(-1)[:, None, None], scores, torch.zeros_like(scores))
            weights = safe.softmax(-1).masked_fill(~mask[:, None, :], 0)
            outputs.append(torch.einsum("nhk,nkhd->nhd", weights.to(v.dtype), v[edge]).flatten(-2))
        return cast(Tensor, self.output(torch.cat(outputs)))


class SmallGraphLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attention = NeighborhoodAttention(width)
        self.ffn = DenseSwiGLU(width)
        self.attention_connection = StreamConnection(width)
        self.ffn_connection = StreamConnection(width)

    def forward(
        self, state: Tensor, read: Tensor, neighbors: Tensor, valid: Tensor, sources: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, following, write, matrix = self.attention_connection.prepare(state, read)
        state = self.attention_connection.finish(
            state, self.attention(x, neighbors, valid, sources), write, matrix
        )
        x, read, write, matrix = self.ffn_connection.prepare(state, following)
        return self.ffn_connection.finish(state, self.ffn(x), write, matrix), read


class SmallGeneGraph(nn.Module):
    read_mode = "relay"

    def __init__(
        self, seeds: Tensor | int, options: V2Architecture, *, gene_seed: int | None
    ) -> None:
        super().__init__()
        d = options.width
        self.adapter: nn.Module
        if options.small_gene_table:
            if not isinstance(seeds, int) or seeds <= 0 or type(gene_seed) is not int:
                raise ValueError("trainable gene table requires a graph node count and run seed")
            generator = torch.Generator(device="cpu").manual_seed(gene_seed)
            weight = torch.randn(seeds, d, generator=generator, device="cpu", dtype=torch.float32)
            self.embedding = nn.Embedding(seeds, d, _weight=weight)
            self.adapter = nn.Identity()
        else:
            if (
                not isinstance(seeds, Tensor)
                or seeds.ndim != 2
                or seeds.shape[1] != 2048
                or not torch.isfinite(seeds).all()
                or gene_seed is not None
            ):
                raise ValueError("small softmax requires finite raw 2048-dimensional priors")
            self.embedding = nn.Embedding(
                seeds.shape[0], seeds.shape[1], _weight=seeds.detach().clone(), _freeze=True
            )
            self.adapter = nn.Sequential(
                FeatureRMSNorm(2048),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.layers = nn.ModuleList(SmallGraphLayer(d) for _ in range(options.graph_layers))
        self.final_norm = FeatureRMSNorm(d)
        self.checkpoint_layers = options.checkpoint_layers
        self.checkpoint_genept_mapping = options.checkpoint_genept_mapping

    def identities(self, ids: Tensor | None = None) -> Tensor:
        seeds = self.embedding.weight if ids is None else self.embedding(ids)
        if self.checkpoint_genept_mapping and torch.is_grad_enabled():
            return cast(Tensor, checkpoint(self.adapter, seeds, use_reentrant=False))
        return cast(Tensor, self.adapter(seeds))

    def forward(
        self,
        ids: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        masked_ids: Tensor | None = None,
        context: GraphContext | None = None,
        modulation: Any = None,
        prefix: Any = None,
    ) -> Tensor:
        if context is None or modulation is not None or prefix is not None:
            raise ValueError("small softmax requires its native graph context")
        x = self.identities(context.ids)
        if masked_ids is not None:
            mask = torch.isin(context.ids, masked_ids)
            x = torch.where(mask[:, None], self.mask_token.to(x), x)
        state, read = start_streams(x)
        for layer in self.layers:
            args = (state, read, context.neighbors, context.valid, context.sources)
            state, read = (
                checkpoint(layer, *args, use_reentrant=False)
                if self.checkpoint_layers and torch.is_grad_enabled()
                else layer(*args)
            )
        return cast(Tensor, self.final_norm(collapse_streams(state, read)))[context.query_positions]


class SmallTokenLayer(nn.Module):
    def __init__(self, width: int, response: bool) -> None:
        super().__init__()
        self.response = response
        self.injection = (
            nn.Sequential(
                FeatureRMSNorm(2 * width),
                nn.Linear(2 * width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            if response
            else None
        )
        self.self_attention = SharedKVAttention(width)
        self.self_connection = StreamConnection(width)
        self.cross_attention = SharedKVAttention(width) if response else None
        self.cross_connection = StreamConnection(width) if response else None
        self.ffn = DenseSwiGLU(width)
        self.ffn_connection = StreamConnection(width)

    def forward(
        self,
        state: Tensor,
        read: Tensor,
        memory: Tensor | None,
        condition: Tensor | None,
        block_cls: bool,
    ) -> tuple[Tensor, Tensor]:
        if self.injection is not None:
            assert condition is not None
            prior = condition[:, None, None, :].expand_as(state)
            state = state + self.injection(torch.cat((state, prior), -1))
        x, following, write, matrix = self.self_connection.prepare(state, read)
        state = self.self_connection.finish(
            state, self.self_attention(x, block_cls=block_cls), write, matrix
        )
        if self.cross_attention is not None:
            assert self.cross_connection is not None and memory is not None
            x, read, write, matrix = self.cross_connection.prepare(state, following)
            state = self.cross_connection.finish(
                state, self.cross_attention(x, memory), write, matrix
            )
            following = read
        x, read, write, matrix = self.ffn_connection.prepare(state, following)
        return self.ffn_connection.finish(state, self.ffn(x), write, matrix), read


class SmallTokenEncoder(nn.Module):
    def __init__(self, options: V2Architecture, response: bool = False) -> None:
        super().__init__()
        depth = 3 if response else options.small_cell_layers
        self.layers = nn.ModuleList(SmallTokenLayer(options.width, response) for _ in range(depth))
        self.norm = FeatureRMSNorm(options.width)
        self.checkpoint_layers = options.checkpoint_layers

    def forward(
        self,
        x: Tensor,
        memory: Tensor | None = None,
        condition: Tensor | None = None,
        order: Any = None,
        *,
        block_cls_to_gene: bool = False,
    ) -> Tensor:
        state, read = start_streams(x)
        for layer in self.layers:
            args = (state, read, memory, condition, block_cls_to_gene)
            state, read = (
                checkpoint(layer, *args, use_reentrant=False)
                if self.checkpoint_layers and torch.is_grad_enabled()
                else layer(*args)
            )
        return cast(Tensor, self.norm(collapse_streams(state, read)))


def initialize_small_model(
    model: Any, seeds: Tensor | int, options: V2Architecture, *, gene_seed: int | None
) -> None:
    d = options.width
    model.graph = SmallGeneGraph(seeds, options, gene_seed=gene_seed)
    model.expression = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
    model.expression_mask = nn.Parameter(torch.zeros(d))
    model.control_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
    model.response_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
    model.gene_expression_fusion = nn.Linear(2 * d, d)
    model.cell = SmallTokenEncoder(options)
    model.response = SmallTokenEncoder(options, response=True)
    layers: list[nn.Module] = [nn.Linear(2 * d, d), nn.GELU()]
    for _ in range(options.small_prediction_hidden_layers - 1):
        layers.extend((nn.Linear(d, d), nn.GELU()))
    model.prediction = nn.Sequential(*layers, nn.Linear(d, 1))
    model.prior_readout = None
    model.control_summary = None
    model.graph_modulation = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith(".route"):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
