"""Small shared interface; fitting receives sanitized train/validation data only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedData:
    adata: Any
    gene_ids: tuple[str, ...]
    graph_gene_ids: tuple[str, ...]
    train_conditions: tuple[str, ...]
    val_conditions: tuple[str, ...]
    allowed_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...]
    train_row_ids: tuple[str, ...]
    val_row_ids: tuple[str, ...]
    control_row_ids: tuple[str, ...]
    row_context_ids: tuple[str, ...]
    train_control_indices: dict[str, tuple[int, ...]]
    val_control_indices: dict[str, tuple[int, ...]]
    receipt: dict[str, Any]


@dataclass
class FittedAdapter:
    model: Any
    receipt: dict[str, Any]
