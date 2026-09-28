"""Freeze test metric groups by actual training-expression exposure at a checkpoint."""

from __future__ import annotations

from typing import Any

from gradpert.hashing import sha256_json


def checkpoint_expression_groups(
    history: list[dict[str, Any]],
    *,
    checkpoint_epoch: int,
    expression_gene_ids: tuple[str, ...],
    allowed_gene_indices: tuple[int, ...] | None,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    if checkpoint_epoch < 1 or checkpoint_epoch > len(history):
        raise ValueError("exposure requires a committed checkpoint epoch")
    record = history[checkpoint_epoch - 1]
    if record["epoch"] != checkpoint_epoch or "seen_expression_gene_indices" not in record:
        raise ValueError("checkpoint lacks actual training-expression exposure evidence")
    seen = tuple(record["seen_expression_gene_indices"])
    axis = tuple(range(len(expression_gene_ids)))
    allowed = set(axis if allowed_gene_indices is None else allowed_gene_indices)
    if (
        not seen
        or any(type(index) is not int for index in seen)
        or tuple(sorted(set(seen))) != seen
        or not set(seen).issubset(allowed)
        or not allowed.issubset(axis)
    ):
        raise ValueError("recorded expression exposure differs from the frozen gene axis")
    seen_set = set(seen)
    unseen = tuple(index for index in axis if index not in seen_set)
    groups = {"seen_expression": seen, "unseen_expression": unseen}
    return groups, {
        "definition": "numeric_control_and_truth_expression_in_completed_training_updates",
        "checkpoint_epoch": checkpoint_epoch,
        "groups": {
            name: {
                "gene_count": len(indices),
                "gene_ids_sha256": sha256_json([expression_gene_ids[i] for i in indices]),
            }
            for name, indices in groups.items()
        },
    }
