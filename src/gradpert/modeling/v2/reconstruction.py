"""Training-only scalar control reconstruction from masked basal states."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ControlReconstruction(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.mask_token = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.gene = nn.Linear(width, 1)
        self.cls_projection = nn.Linear(width, width, bias=False)
        self.gene_projection = nn.Linear(width, width, bias=False)
        self.cls_bias = nn.Parameter(torch.zeros(()))

    def forward(self, basal: Tensor, cls: Tensor, gene: Tensor) -> tuple[Tensor, Tensor]:
        token = self.gene(basal).squeeze(-1)
        cell = torch.einsum(
            "bd,gd->bg", self.cls_projection(cls), self.gene_projection(gene)
        ) / math.sqrt(self.width)
        return token, cell + self.cls_bias
