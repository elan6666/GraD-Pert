"""Explicit expression-loss reductions; no direction term or gene filtering."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def expression_loss(
    prediction: Tensor, target: Tensor, condition_ids: Sequence[str], *, reduction: str
) -> Tensor:
    if reduction == "cell_mean":
        return F.mse_loss(prediction, target)
    if reduction != "condition_mean":
        raise ValueError("unknown expression reduction")
    if (
        prediction.ndim != 2
        or prediction.shape != target.shape
        or prediction.shape[0] != len(condition_ids)
        or not condition_ids
        or prediction.shape[1] == 0
        or any(not isinstance(c, str) or not c for c in condition_ids)
    ):
        raise ValueError("condition reduction requires aligned nonempty cells/genes/IDs")
    # Square BEFORE averaging; complete condition strings are the group keys.
    cell_error = (prediction - target).square().mean(dim=1)
    groups: dict[str, list[int]] = {}
    for index, condition in enumerate(condition_ids):
        groups.setdefault(condition, []).append(index)
    return torch.stack([cell_error[groups[c]].mean() for c in sorted(groups)]).mean()
