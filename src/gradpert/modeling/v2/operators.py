"""Native v2 attention and residual operators.

Equation provenance and deliberate architectural differences are recorded in
``docs/provenance/GRADPERT_V2_OPERATORS.md``. No upstream runtime is imported.
"""

from __future__ import annotations

import math
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

    def forward(self, x: Tensor) -> Tensor:
        latent = self.latent_norm(self.compress(x))
        shape = (*x.shape[:2], self.heads, self.head_width)
        q, k, v = [
            a.reshape(shape).transpose(1, 2)
            for a in (self.query(x), self.key(latent), self.value(latent))
        ]
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))


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

    def forward(self, x: Tensor) -> Tensor:
        if self.streams == 1:
            return x + cast(Tensor, self.sublayer(self.norm(x.squeeze(-2)))).unsqueeze(-2)
        pre, post, residual = self.maps(x)
        h = (pre.unsqueeze(-1) * x).sum(-2)
        y = cast(Tensor, self.sublayer(self.norm(h)))
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
    ) -> None:
        super().__init__()
        self.streams, self.checkpoint_layers = streams, checkpoint_layers
        self.per_gene = attention == "per_gene"
        layers = []
        attention_layer: nn.Module
        for index in range(4):
            if self.per_gene:
                attention_layer = nn.Sequential(
                    nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
                )
            elif index < 3 and attention in ("hybrid", "delta_full"):
                attention_layer = DeltaAttention(width, heads)
            elif index == 3 and attention in ("hybrid", "full_latent"):
                attention_layer = LatentAttention(width, heads, rank, dropout)
            else:
                attention_layer = FullAttention(width, heads, dropout, causal=index < 3)
            layers.append(ManifoldResidual(width, streams, attention_layer))
            layers.append(
                ManifoldResidual(
                    width,
                    streams,
                    nn.Sequential(
                        nn.Linear(width, 4 * width),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(4 * width, width),
                        nn.Dropout(dropout),
                    ),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(width)

    def forward(self, x: Tensor, *, block_cls_to_gene: bool = False) -> Tensor:
        if block_cls_to_gene and self.training:
            raise ValueError("CLS edge intervention is an evaluation-only diagnostic")
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.streams, x.shape[-1])
        for index, layer in enumerate(self.layers):
            if block_cls_to_gene and index == 6:
                # Only the fourth attention layer is noncausal. Earlier causal
                # layers cannot transmit the tail CLS to preceding gene slots.
                # Retain the normal CLS readout, while genes attend to genes only.
                full = layer(x)
                genes = layer(x[:, :-1])
                x = torch.cat((genes, full[:, -1:]), dim=1)
            elif self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        result = self.norm(x.mean(-2))
        if self.per_gene:
            # The final slot is a readout only: its pooled summary never feeds
            # gene outputs. Distillation can still compare cell/response states.
            result = torch.cat((result[:, :-1], result[:, :-1].mean(1, keepdim=True)), dim=1)
        return cast(Tensor, result)
