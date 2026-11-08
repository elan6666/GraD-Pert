"""Full-width terminal self/cross reads with native CLS visibility contracts."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StandardAttention(nn.Module):
    """Noncausal softmax attention; each Q/K/V directly spans model width."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("attention width must divide into heads")
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, memory: Tensor | None = None, *, block_cls_to_gene: bool = False
    ) -> Tensor:
        cross = memory is not None
        memory = x if memory is None else memory
        q = self.query(x).unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)
        k = self.key(memory).unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)
        v = self.value(memory).unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)
        dropout = self.dropout if self.training else 0.0
        if block_cls_to_gene and not cross:
            # Genes exclude the tail CLS key; the tail CLS still reads all tokens.
            y = torch.cat(
                (
                    F.scaled_dot_product_attention(
                        q[:, :, :-1], k[:, :, :-1], v[:, :, :-1], dropout_p=dropout
                    ),
                    F.scaled_dot_product_attention(q[:, :, -1:], k, v, dropout_p=dropout),
                ),
                dim=2,
            )
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        return cast(Tensor, self.output(y.transpose(1, 2).flatten(-2)))
