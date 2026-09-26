"""Native v2 attention and residual operators.

Equation provenance and deliberate architectural differences are recorded in
``docs/provenance/GRADPERT_V2_OPERATORS.md``. No upstream runtime is imported.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def delta_scan(q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor) -> Tensor:
    """Channel-decayed delta recurrence; [batch,time,head,channel].

    State is float32. Reads use the decayed state before the rank-one write;
    queries read the updated state. This reference is intentionally unfused.
    """
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32)
    output = []
    for t in range(q.shape[1]):
        key = k[:, t].float()
        state = state * log_decay[:, t].float().exp().unsqueeze(-1)
        error = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", key, state)
        state = state + (beta[:, t].float().unsqueeze(-1) * key).unsqueeze(-1) * error.unsqueeze(-2)
        output.append(torch.einsum("bhk,bhkv->bhv", q[:, t].float(), state))
    return torch.stack(output, dim=1).to(v.dtype)


def chunk_delta_scan(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, chunk_size: int = 32
) -> Tensor:
    """Native differentiable block solve of the same delta recurrence.

    The strict lower triangle encodes past writes erased by subsequent keys.
    Solving for all writes in a block avoids a Python loop over every token.
    Log-decay differences outside the causal triangle are set to zero BEFORE
    exponentiation, avoiding overflow from unused positive exponents.
    """
    shape = q.shape
    state = q.new_zeros(shape[0], shape[2], shape[3], v.shape[-1], dtype=torch.float32)
    chunks = []
    for start in range(0, shape[1], chunk_size):
        end = min(start + chunk_size, shape[1])
        query, key, value = [t[:, start:end].float().transpose(1, 2) for t in (q, k, v)]
        gates = log_decay[:, start:end].float().transpose(1, 2).cumsum(-2)
        write_rate = beta[:, start:end].float().transpose(1, 2).unsqueeze(-1)
        length = end - start
        causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
        decay = (
            (gates.unsqueeze(-2) - gates.unsqueeze(-3))
            .masked_fill(~causal[None, None, :, :, None], 0)
            .exp()
        )
        past_keys = (key.unsqueeze(-2) * key.unsqueeze(-3) * decay).sum(-1)
        triangular = (past_keys * write_rate).tril(-1)
        triangular = triangular + torch.eye(length, device=q.device)
        old_read = (key * gates.exp()) @ state
        writes = torch.linalg.solve_triangular(
            triangular, write_rate * (value - old_read), upper=False
        )
        query_keys = (query.unsqueeze(-2) * key.unsqueeze(-3) * decay).sum(-1)
        outputs = (query * gates.exp()) @ state + query_keys.masked_fill(~causal, 0) @ writes
        final_keys = key * (gates[..., -1:, :] - gates).exp()
        state = (
            gates[..., -1, :].exp().unsqueeze(-1) * state + final_keys.transpose(-1, -2) @ writes
        )
        chunks.append(outputs.transpose(1, 2))
    return torch.cat(chunks, dim=1).to(v.dtype)


def delta_final_state(
    k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, state: Tensor | None = None
) -> Tensor:
    """Reference write-only delta scan, with an optional carried-in state."""
    if state is None:
        state = k.new_zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[-1], dtype=torch.float32)
    for t in range(k.shape[1]):
        key = k[:, t].float()
        state = state * log_decay[:, t].float().exp().unsqueeze(-1)
        error = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", key, state)
        state = state + (beta[:, t].float().unsqueeze(-1) * key).unsqueeze(-1) * error.unsqueeze(-2)
    return state


def chunk_delta_final_state(
    k: Tensor,
    v: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    state: Tensor | None = None,
    chunk_size: int = 32,
    *,
    compiled: bool = False,
) -> Tensor:
    """Exact block solve for the final state only; avoids per-token query reads."""
    if state is None:
        state = k.new_zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[-1], dtype=torch.float32)
    block = compiled_delta_block() if compiled else delta_final_block
    for start in range(0, k.shape[1], chunk_size):
        end = min(start + chunk_size, k.shape[1])
        state = block(
            k[:, start:end], v[:, start:end], log_decay[:, start:end], beta[:, start:end], state
        )
    return state


def delta_final_block(
    k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, state: Tensor
) -> Tensor:
    """One unchanged delta block; regional compilation can fuse elementwise work.

    Keep the causal mask before exp to avoid overflowing unused upper entries.
    This region has no random draws, parameter mutation or device synchronization.
    """
    key, value = [t.float().transpose(1, 2) for t in (k, v)]
    gates = log_decay.float().transpose(1, 2).cumsum(-2)
    write_rate = beta.float().transpose(1, 2).unsqueeze(-1)
    length = k.shape[1]
    causal = torch.ones(length, length, device=k.device, dtype=torch.bool).tril()
    decay = (
        (gates.unsqueeze(-2) - gates.unsqueeze(-3))
        .masked_fill(~causal[None, None, :, :, None], 0)
        .exp()
    )
    past_keys = (key.unsqueeze(-2) * key.unsqueeze(-3) * decay).sum(-1)
    triangular = (past_keys * write_rate).tril(-1)
    triangular = triangular + torch.eye(length, device=k.device)
    old_read = (key * gates.exp()) @ state
    writes = torch.linalg.solve_triangular(triangular, write_rate * (value - old_read), upper=False)
    final_keys = key * (gates[..., -1:, :] - gates).exp()
    return cast(
        Tensor,
        gates[..., -1, :].exp().unsqueeze(-1) * state + final_keys.transpose(-1, -2) @ writes,
    )


@lru_cache(maxsize=1)
def compiled_delta_block() -> Callable[..., Tensor]:
    """Opt-in Inductor region shared across graph/cell/response and EMA copies.

    Dynamic shapes cover cropped views and graph tail chunks. Fail on compiler
    errors instead of silently recording eager execution as a compiled result.
    Compilation caches and warmup costs belong in the performance receipt.
    """
    import torch._functorch.config as compiler_config

    compiled = torch.compile(delta_final_block, fullgraph=True, dynamic=True)

    def invoke(*args: Tensor) -> Tensor:
        # The training engine leaves autocast before backward. AOTAutograd's
        # default same_as_forward assumption would silently change that policy.
        with compiler_config.patch(backward_pass_autocast="off"):
            return compiled(*args)

    return invoke


def relay_order(
    batch: int, genes: int, device: torch.device, training: bool, layer: int = 0, seed: int = 1
) -> Tensor:
    """Random training order; stable seeded order at evaluation."""
    if training:
        return torch.rand(batch, genes, device=device).argsort(-1)
    generator = torch.Generator(device=device).manual_seed(seed + layer)
    # One evaluation permutation per layer/view, independent of cell chunking.
    return torch.rand(1, genes, device=device, generator=generator).argsort(-1).expand(batch, -1)


class DeltaAttention(nn.Module):
    """KDA equation with normalized Q/K, channel decay and gated output.

    No short convolution or rotary position transformation is added. Head
    width is d/heads. Explicit sequence order is a data contract.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.decay_down = nn.Linear(width, self.head_width, bias=False)
        self.decay_up = nn.Linear(self.head_width, width, bias=False)
        self.log_rate = nn.Parameter(torch.zeros(heads))
        self.decay_bias = nn.Parameter(torch.zeros(width))
        self.write = nn.Linear(width, heads)
        self.output_gate = nn.Linear(width, width)
        self.head_norm = nn.RMSNorm(self.head_width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        shape = (*x.shape[:2], self.heads, self.head_width)
        q = F.normalize(self.query(x).reshape(shape).float(), dim=-1)
        k = F.normalize(self.key(x).reshape(shape).float(), dim=-1)
        v = self.value(x).reshape(shape)
        gate = self.decay_up(self.decay_down(x)).float() + self.decay_bias
        decay = -self.log_rate.float().exp()[None, None, :, None] * F.softplus(gate.reshape(shape))
        beta = self.write(x).float().sigmoid()
        y = chunk_delta_scan(q / math.sqrt(self.head_width), k, v, decay, beta)
        y = self.head_norm(y).flatten(-2) * F.silu(self.output_gate(x))
        return cast(Tensor, self.output(y))


class RelayDeltaAttention(DeltaAttention):
    """Two-pass carried-state KDA whose tokens read only the final state."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__(width, heads)
        self.source = nn.Linear(4, width, bias=False)
        self.randomize_order: bool | None = None
        self.eval_seed: int = 1
        self.compiled_chunks = False

    def random_order_enabled(self) -> bool:
        return self.training if self.randomize_order is None else self.randomize_order

    def _project_writes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = (*x.shape[:2], self.heads, self.head_width)
        k = F.normalize(self.key(x).reshape(shape).float(), dim=-1)
        v = self.value(x).reshape(shape)
        gate = self.decay_up(self.decay_down(x)).float() + self.decay_bias
        decay = -self.log_rate.float().exp()[None, None, :, None] * F.softplus(gate.reshape(shape))
        beta = self.write(x).float().sigmoid()
        return k, v, decay, beta

    def _read(self, query: Tensor, state: Tensor) -> Tensor:
        shape = (*query.shape[:2], self.heads, self.head_width)
        q = F.normalize(self.query(query).reshape(shape).float(), dim=-1)
        y = torch.einsum("bthk,bhkv->bthv", q / math.sqrt(self.head_width), state)
        y = self.head_norm(y.to(query.dtype)).flatten(-2) * F.silu(self.output_gate(query))
        return cast(Tensor, self.output(y))

    @staticmethod
    def _ordered(tensor: Tensor, order: Tensor) -> Tensor:
        return torch.gather(
            tensor,
            1,
            order.reshape(*order.shape, *([1] * (tensor.ndim - 2))).expand(
                -1, -1, *tensor.shape[2:]
            ),
        )

    def _two_pass(self, writes: tuple[Tensor, Tensor, Tensor, Tensor], order: Tensor) -> Tensor:
        key, value, decay, beta = (self._ordered(t, order) for t in writes)
        # A single scan preserves the exact forward final state as the reverse
        # initial state; reverse writes use the very same projected tokens.
        return chunk_delta_final_state(
            torch.cat((key, key.flip(1)), dim=1),
            torch.cat((value, value.flip(1)), dim=1),
            torch.cat((decay, decay.flip(1)), dim=1),
            torch.cat((beta, beta.flip(1)), dim=1),
            compiled=self.compiled_chunks,
        )

    def forward(
        self,
        x: Tensor,
        *,
        order: Tensor | None = None,
        has_cls: bool = True,
        block_cls_to_gene: bool = False,
    ) -> Tensor:
        genes = x.shape[1] - int(has_cls)
        if order is None:
            order = relay_order(
                len(x), genes, x.device, self.random_order_enabled(), seed=self.eval_seed
            )
        if order.shape != (len(x), genes):
            raise ValueError("relay permutation must cover every gene exactly once")
        state = self._two_pass(self._project_writes(x[:, :genes]), order)
        gene_state = state
        if has_cls:
            # Forward CLS is read-only; after both gene scans it writes exactly once.
            cls_key, cls_value, cls_decay, cls_beta = self._project_writes(x[:, genes:])
            state = chunk_delta_final_state(
                cls_key, cls_value, cls_decay, cls_beta, state=state, compiled=self.compiled_chunks
            )
        if block_cls_to_gene and has_cls:
            return torch.cat(
                (self._read(x[:, :genes], gene_state), self._read(x[:, genes:], state)), dim=1
            )
        return self._read(x, state)

    def cross(self, query: Tensor, control: Tensor, *, order: Tensor) -> Tensor:
        if len(query) != len(control) or order.shape != control.shape[:2]:
            raise ValueError("cross KDA needs aligned control order")
        state = self._two_pass(self._project_writes(control), order)
        return self._read(query, state)

    def graph(
        self,
        query: Tensor,
        memory: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        *,
        order: Tensor | None = None,
    ) -> Tensor:
        if neighbors.shape != valid.shape or sources.shape != (*neighbors.shape, 4):
            raise ValueError("invalid graph neighborhood shape")
        if not valid.any(-1).all():
            raise ValueError("every graph target needs a valid neighbor")
        n, length = neighbors.shape
        selected = memory[neighbors.clamp_min(0)] + self.source(sources.to(memory.dtype))
        key, value, decay, beta = self._project_writes(selected)
        decay = decay.masked_fill(~valid[:, :, None, None], 0)
        beta = beta.masked_fill(~valid[:, :, None], 0)
        if order is None:
            scores = (
                torch.rand(n, length, device=query.device)
                if self.random_order_enabled()
                else (
                    ((neighbors.long() * 1103515245 + 12345 * self.eval_seed) & 0x7FFFFFFF).float()
                    / 0x7FFFFFFF
                )
            )
            order = scores.masked_fill(~valid, float("inf")).argsort(-1)
        state = self._two_pass((key, value, decay, beta), order)
        return self._read(query.unsqueeze(1), state).squeeze(1)


class LatentAttention(nn.Module):
    """Noncausal attention with a shared compressed KV latent, without RoPE."""

    def __init__(self, width: int, heads: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.query = nn.Linear(width, width, bias=False)
        self.compress = nn.Linear(width, rank, bias=False)
        self.latent_norm = nn.RMSNorm(rank)
        self.key = nn.Linear(rank, width, bias=False)
        self.value = nn.Linear(rank, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, *, block_cls_to_gene: bool = False) -> Tensor:
        latent = self.latent_norm(self.compress(x))
        shape = (*x.shape[:2], self.heads, self.head_width)
        q, k, v = [
            a.reshape(shape).transpose(1, 2)
            for a in (self.query(x), self.key(latent), self.value(latent))
        ]
        if block_cls_to_gene:
            genes = x.shape[1] - 1
            y = torch.cat(
                (
                    F.scaled_dot_product_attention(
                        q[:, :, :genes],
                        k[:, :, :genes],
                        v[:, :, :genes],
                        dropout_p=self.dropout if self.training else 0.0,
                    ),
                    F.scaled_dot_product_attention(
                        q[:, :, genes:],
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                    ),
                ),
                dim=2,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))


class LatentCrossAttention(LatentAttention):
    """Response queries read the control gene states through compressed KV."""

    def forward(
        self, query: Tensor, memory: Tensor | None = None, *, block_cls_to_gene: bool = False
    ) -> Tensor:
        if memory is None:
            return super().forward(query, block_cls_to_gene=block_cls_to_gene)
        latent = self.latent_norm(self.compress(memory))
        q = self.query(query).reshape(len(query), query.shape[1], self.heads, self.head_width)
        k = self.key(latent).reshape(len(memory), memory.shape[1], self.heads, self.head_width)
        v = self.value(latent).reshape(len(memory), memory.shape[1], self.heads, self.head_width)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))


class IndexedLatentAttention(nn.Module):
    """Content-indexed, noncausal latent attention for unordered genes.

    Gene queries reserve slots for self and CLS; the CLS query reads all genes.
    A selected-score bias keeps the independent indexer trainable despite hard
    top-k. Query chunks bound gathered KV memory. No positional pooling is used.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        rank: int,
        dropout: float,
        topk: int,
        index_dim: int,
        query_chunk: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.topk = topk
        self.query_chunk = query_chunk
        self.checkpoint_chunks = True
        self.q_down = nn.Linear(width, rank, bias=False)
        self.q_norm = nn.RMSNorm(rank)
        self.q_up = nn.Linear(rank, width, bias=False)
        self.kv_down = nn.Linear(width, rank, bias=False)
        self.kv_norm = nn.RMSNorm(rank)
        self.k_up = nn.Linear(rank, width, bias=False)
        self.v_up = nn.Linear(rank, width, bias=False)
        self.index_query = nn.Linear(width, heads * index_dim, bias=False)
        self.index_key = nn.Linear(width, heads * index_dim, bias=False)
        self.index_dim = index_dim
        self.index_scale = nn.Parameter(torch.tensor(0.1))
        self.output = nn.Linear(width, width, bias=False)

    def _chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        iq: Tensor,
        ik: Tensor,
        start: int,
        end: int,
        genes: int,
        has_cls: bool,
    ) -> Tensor:
        batch, _, _, _ = q.shape
        size = end - start
        query_ids = torch.arange(start, end, device=q.device)
        count = self.topk - 1 - int(has_cls)
        with torch.no_grad():
            scores = torch.matmul(
                iq[:, :, start:end].float(), ik[:, :, :genes].float().transpose(-1, -2)
            ) / math.sqrt(self.index_dim)
            scores = scores.scatter(
                -1,
                query_ids.view(1, 1, size, 1).expand(batch, self.heads, -1, -1),
                -float("inf"),
            )
            selected = scores.topk(count, dim=-1).indices
        self_ids = query_ids.view(1, 1, size, 1).expand(batch, self.heads, -1, -1)
        selected = torch.cat((self_ids, selected), dim=-1)
        if has_cls:
            selected = torch.cat((selected, torch.full_like(self_ids, genes)), dim=-1)
        key = torch.gather(
            k.unsqueeze(2).expand(-1, -1, size, -1, -1),
            3,
            selected.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_width),
        )
        value = torch.gather(
            v.unsqueeze(2).expand(-1, -1, size, -1, -1),
            3,
            selected.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_width),
        )
        logits = torch.matmul(
            q[:, :, start:end].float().unsqueeze(-2), key.float().transpose(-1, -2)
        ).squeeze(-2) / math.sqrt(self.head_width)
        index_key = torch.gather(
            ik.unsqueeze(2).expand(-1, -1, size, -1, -1),
            3,
            selected.unsqueeze(-1).expand(-1, -1, -1, -1, self.index_dim),
        )
        index_bias = torch.matmul(
            iq[:, :, start:end].float().unsqueeze(-2), index_key.float().transpose(-1, -2)
        ).squeeze(-2) / math.sqrt(self.index_dim)
        logits = logits + self.index_scale.tanh() * index_bias
        weights = logits.softmax(-1).to(value.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        return torch.matmul(weights.unsqueeze(-2), value).squeeze(-2)

    def forward(self, x: Tensor, *, has_cls: bool = True) -> Tensor:
        batch, length, _ = x.shape
        genes = length - int(has_cls)
        q = (
            self.q_up(self.q_norm(self.q_down(x)))
            .reshape(batch, length, self.heads, self.head_width)
            .transpose(1, 2)
        )
        latent = self.kv_norm(self.kv_down(x))
        k = self.k_up(latent).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = self.v_up(latent).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        if genes <= self.topk - int(has_cls):
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
            return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))

        iq = self.index_query(x).reshape(batch, length, self.heads, self.index_dim)
        ik = self.index_key(x).reshape(batch, length, self.heads, self.index_dim)
        iq, ik = iq.transpose(1, 2), ik.transpose(1, 2)
        outputs = []
        for start in range(0, genes, self.query_chunk):
            end = min(start + self.query_chunk, genes)
            args = (q, k, v, iq, ik, start, end, genes, has_cls)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                # Exact recomputation trades extra FLOPs for bounded saved
                # activations, even when the outer encoder is checkpointed.
                outputs.append(checkpoint(self._chunk, *args, use_reentrant=False))
            else:
                outputs.append(self._chunk(*args))
        if has_cls:
            cls_output = F.scaled_dot_product_attention(
                q[:, :, genes:], k, v, dropout_p=self.dropout if self.training else 0.0
            )
            outputs.append(cls_output)
        y = torch.cat(outputs, dim=2)
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))


class GatedFeedForward(nn.Module):
    """GLM-5.3-style clipped SwiGLU with task-specific model width."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(width, 4 * width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate(x).clamp(max=10)
        up = self.up(x).clamp(min=-10, max=10)
        return cast(Tensor, self.dropout(self.down(self.dropout(F.silu(gate) * up))))


class FullAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.heads, self.dropout, self.causal = heads, dropout, causal

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = [
            t.reshape(*x.shape[:2], self.heads, -1).transpose(1, 2)
            for t in self.qkv(x).chunk(3, -1)
        ]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
        )
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))


def sinkhorn(logits: Tensor, iterations: int = 20) -> Tensor:
    z = logits.float()
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return z.exp()


class ManifoldResidual(nn.Module):
    """Per-token pre/post/residual maps, wrapping one attention or FFN."""

    def __init__(self, width: int, streams: int, sublayer: nn.Module) -> None:
        super().__init__()
        self.streams, self.sublayer = streams, sublayer
        self.norm = nn.RMSNorm(width)
        if streams > 1:
            self.route = nn.Linear(streams * width, 2 * streams + streams * streams, bias=False)
            self.scales = nn.Parameter(torch.full((3,), 0.01))
            bias = torch.zeros(2 * streams + streams * streams)
            bias[:streams] = math.log(1 / (streams - 1))
            bias[2 * streams :] = (torch.eye(streams) * 4).flatten()
            self.bias = nn.Parameter(bias)

    def maps(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = self.streams
        flat = x.flatten(-2)
        normed = F.rms_norm(flat.float(), (flat.shape[-1],)).to(flat.dtype)
        a, b, c = self.route(normed).float().split((n, n, n * n), dim=-1)
        a = (a * self.scales[0] + self.bias[:n]).sigmoid()
        b = 2 * (b * self.scales[1] + self.bias[n : 2 * n]).sigmoid()
        c = sinkhorn((c * self.scales[2] + self.bias[2 * n :]).unflatten(-1, (n, n)))
        return a.to(x.dtype), b.to(x.dtype), c.to(x.dtype)

    def forward(
        self,
        x: Tensor,
        *,
        has_cls: bool = True,
        order: Tensor | None = None,
        block_cls_to_gene: bool = False,
    ) -> Tensor:
        if self.streams == 1:
            h = self.norm(x.squeeze(-2))
            y = (
                self.sublayer(h, has_cls=has_cls)
                if isinstance(self.sublayer, IndexedLatentAttention)
                else self.sublayer(
                    h,
                    order=order,
                    has_cls=has_cls,
                    block_cls_to_gene=block_cls_to_gene,
                )
                if isinstance(self.sublayer, RelayDeltaAttention)
                else self.sublayer(h, block_cls_to_gene=block_cls_to_gene)
                if isinstance(self.sublayer, LatentAttention)
                else self.sublayer(h)
            )
            return x + cast(Tensor, y).unsqueeze(-2)
        pre, post, residual = self.maps(x)
        h = (pre.unsqueeze(-1) * x).sum(-2)
        h = self.norm(h)
        y = cast(
            Tensor,
            self.sublayer(h, has_cls=has_cls)
            if isinstance(self.sublayer, IndexedLatentAttention)
            else self.sublayer(
                h,
                order=order,
                has_cls=has_cls,
                block_cls_to_gene=block_cls_to_gene,
            )
            if isinstance(self.sublayer, RelayDeltaAttention)
            else self.sublayer(h, block_cls_to_gene=block_cls_to_gene)
            if isinstance(self.sublayer, LatentAttention)
            else self.sublayer(h),
        )
        return torch.einsum("...ij,...jd->...id", residual, x) + post.unsqueeze(-1) * y.unsqueeze(
            -2
        )


class CrossManifoldResidual(ManifoldResidual):
    """mHC response residual whose read memory comes from control genes."""

    def forward(
        self,
        x: Tensor,
        *,
        has_cls: bool = True,
        order: Tensor | None = None,
        block_cls_to_gene: bool = False,
        memory: Tensor | None = None,
    ) -> Tensor:
        if memory is None or order is None:
            raise ValueError("cross residual needs aligned control memory and order")
        if self.streams == 1:
            h = self.norm(x.squeeze(-2))
            y = (
                self.sublayer.cross(h, memory, order=order)
                if isinstance(self.sublayer, RelayDeltaAttention)
                else self.sublayer(h, memory)
            )
            return x + cast(Tensor, y).unsqueeze(-2)
        pre, post, residual = self.maps(x)
        h = self.norm((pre.unsqueeze(-1) * x).sum(-2))
        y = (
            self.sublayer.cross(h, memory, order=order)
            if isinstance(self.sublayer, RelayDeltaAttention)
            else self.sublayer(h, memory)
        )
        return torch.einsum("...ij,...jd->...id", residual, x) + post.unsqueeze(-1) * y.unsqueeze(
            -2
        )


class TokenEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        rank: int,
        streams: int,
        dropout: float,
        attention: str,
        checkpoint_layers: bool,
        ffn_type: str = "gelu",
        sparse_topk: int = 500,
        sparse_index_dim: int = 64,
        sparse_query_chunk: int = 8,
        kda_layers: int = 3,
    ) -> None:
        super().__init__()
        if kda_layers < 1:
            raise ValueError("encoder needs at least one pre-read layer")
        self.kda_layers = kda_layers
        self.streams, self.checkpoint_layers = streams, checkpoint_layers
        self.per_gene = attention == "per_gene"
        layers = []
        attention_layer: nn.Module
        for index in range(kda_layers + 1):
            if self.per_gene:
                attention_layer = nn.Sequential(
                    nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
                )
            elif index < kda_layers and attention == "relay_full":
                attention_layer = RelayDeltaAttention(width, heads)
            elif index < kda_layers and attention in ("hybrid", "hybrid_sparse", "delta_full"):
                attention_layer = DeltaAttention(width, heads)
            elif index == kda_layers and attention == "hybrid_sparse":
                attention_layer = IndexedLatentAttention(
                    width, heads, rank, dropout, sparse_topk, sparse_index_dim, sparse_query_chunk
                )
            elif index == kda_layers and attention in ("hybrid", "full_latent", "relay_full"):
                attention_layer = LatentAttention(width, heads, rank, dropout)
            else:
                attention_layer = FullAttention(width, heads, dropout, causal=index < kda_layers)
            layers.append(ManifoldResidual(width, streams, attention_layer))
            ffn: nn.Module = (
                GatedFeedForward(width, dropout)
                if ffn_type == "swiglu"
                else nn.Sequential(
                    nn.Linear(width, 4 * width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * width, width),
                    nn.Dropout(dropout),
                )
            )
            layers.append(
                ManifoldResidual(
                    width,
                    streams,
                    ffn,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(width)

    def forward(
        self,
        x: Tensor,
        *,
        block_cls_to_gene: bool = False,
        order: Tensor | tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        if block_cls_to_gene and self.training:
            raise ValueError("CLS edge intervention is an evaluation-only diagnostic")
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.streams, x.shape[-1])
        for index, layer in enumerate(self.layers):
            layer_order = (
                order[index // 2]
                if isinstance(order, tuple) and index // 2 < len(order)
                else order
                if isinstance(order, Tensor)
                else None
            )
            if block_cls_to_gene and index == 2 * self.kda_layers:
                # Only the final attention layer is noncausal. Earlier causal
                # layers cannot transmit the tail CLS to preceding gene slots.
                # Retain the normal CLS readout, while genes attend to genes only.
                full = layer(x, order=layer_order)
                genes = layer(x[:, :-1], has_cls=False, order=layer_order)
                x = torch.cat((genes, full[:, -1:]), dim=1)
            elif self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda z, block=layer, block_order=layer_order: block(z, order=block_order),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, order=layer_order)
        result = self.norm(x.mean(-2))
        if self.per_gene:
            # The final slot is a readout only: its pooled summary never feeds
            # gene outputs. Distillation can still compare cell/response states.
            result = torch.cat((result[:, :-1], result[:, :-1].mean(1, keepdim=True)), dim=1)
        return cast(Tensor, result)
