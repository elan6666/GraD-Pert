"""Native graph mHC routes around the existing legal-neighborhood operators."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .model import RelayGraphLayer, SparseRead
from .operators import ManifoldResidual


class GraphManifoldLayer(nn.Module):
    """Carry graph streams through separate attention and FFN residual routes.

    Routing combines streams of each gene independently. The original core owns
    all attention, source-membership, normalization, FFN and dropout parameters.
    Only its legal neighbors enter an attention read; no additional node is used.
    """

    def __init__(
        self,
        core: RelayGraphLayer | SparseRead,
        width: int,
        streams: int,
        sinkhorn_backend: str = "native",
    ) -> None:
        super().__init__()
        if not isinstance(core, (RelayGraphLayer, SparseRead)):
            raise TypeError("graph mHC requires RelayGraphLayer or SparseRead")
        if streams < 1:
            raise ValueError("graph mHC requires at least one stream")
        self.core, self.width, self.streams = core, width, streams
        self.attention_route = ManifoldResidual(width, streams, nn.Identity())
        self.ffn_route = ManifoldResidual(width, streams, nn.Identity())
        for route in (self.attention_route, self.ffn_route):
            route.add_module("norm", nn.Identity())
            route.sinkhorn_backend = sinkhorn_backend

    def _prepare(
        self, route: ManifoldResidual, x: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if self.streams == 1:
            return x.squeeze(-2), None, None
        pre, post, residual = route.maps(x)
        return (pre.unsqueeze(-1) * x).sum(-2), post, residual

    @staticmethod
    def _inject(x: Tensor, read: Tensor, post: Tensor | None, residual: Tensor | None) -> Tensor:
        if post is None:
            return x + read.unsqueeze(-2)
        assert residual is not None
        return torch.einsum("...ij,...jd->...id", residual, x) + post.unsqueeze(
            -1
        ) * read.unsqueeze(-2)

    def _ffn(self, x: Tensor) -> Tensor:
        collapsed, post, residual = self._prepare(self.ffn_route, x)
        read = self.core.dropout(self.core.ffn(self.core.norm2(collapsed)))
        return self._inject(x, read, post, residual)

    def _relay_chunk(
        self,
        x: Tensor,
        normalized_query: Tensor,
        normalized_memory: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        sources: Tensor,
        post: Tensor | None,
        residual: Tensor | None,
    ) -> Tensor:
        core = self.core
        assert isinstance(core, RelayGraphLayer)
        read = core.read.graph(
            query=normalized_query,
            memory=normalized_memory,
            neighbors=neighbors,
            valid=valid,
            sources=sources,
            neighborhoods_validated=core.validate_once,
        )
        x = self._inject(x, core.dropout(read), post, residual)
        return self._ffn(x)

    def forward(self, memory: Tensor, neighbors: Tensor, valid: Tensor, sources: Tensor) -> Tensor:
        if memory.ndim != 3 or memory.shape[1:] != (self.streams, self.width):
            raise ValueError("graph mHC expects node-by-stream-by-width memory")
        if self.streams == 1:
            # Reuse the exact original autograd graph as well as its residual
            # arithmetic, avoiding extra view-dependent gradient accumulation.
            collapsed = memory.squeeze(-2)
            output = (
                self.core(collapsed, collapsed, neighbors, valid, sources)
                if isinstance(self.core, SparseRead)
                else self.core(collapsed, neighbors, valid, sources)
            )
            return cast(Tensor, output.unsqueeze(-2))
        collapsed, post, residual = self._prepare(self.attention_route, memory)
        core = self.core
        if isinstance(core, SparseRead):
            read = core.attention(collapsed, collapsed, neighbors, valid, sources)
            return self._ffn(self._inject(memory, core.dropout(read), post, residual))

        if core.validate_once and not valid.any(-1).all():
            raise ValueError("every graph target needs a valid neighbor")
        normalized = core.norm1(collapsed)
        outputs = []
        for start in range(0, len(memory), core.chunk_rows):
            end = min(start + core.chunk_rows, len(memory))
            args = (
                memory[start:end],
                normalized[start:end],
                normalized,
                neighbors[start:end],
                valid[start:end],
                sources[start:end],
                None if post is None else post[start:end],
                None if residual is None else residual[start:end],
            )
            outputs.append(
                checkpoint(self._relay_chunk, *args, use_reentrant=False)
                if core.checkpoint_chunks and torch.is_grad_enabled()
                else self._relay_chunk(*args)
            )
        return torch.cat(outputs)
