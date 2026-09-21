"""Training expression visibility derived only from frozen condition identities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from gradpert.hashing import sha256_json


def expression_policy(
    gene_ids: tuple[str, ...],
    test_conditions: tuple[str, ...],
    control_condition_id: str,
    *,
    enabled: bool,
) -> tuple[np.ndarray[Any, Any] | None, dict[str, Any]]:
    if not gene_ids or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("expression axis must be nonempty and unique")
    targets = sorted(
        {g for c in test_conditions for g in c.split("+") if g != control_condition_id}
    )
    target_set = set(targets)
    excluded = [g for g in gene_ids if enabled and g in target_set]
    excluded_set = set(excluded)
    allowed = np.array([i for i, g in enumerate(gene_ids) if g not in excluded_set], dtype=np.int64)
    if not len(allowed):
        raise ValueError("test-target exclusion leaves no training expression genes")
    receipt = {
        "schema": "training-expression-policy-v1",
        "exclude_test_target_expression": enabled,
        "test_conditions_sha256": sha256_json(list(test_conditions)),
        "test_target_gene_ids": targets,
        "excluded_expression_gene_ids": excluded,
        "excluded_expression_gene_ids_sha256": sha256_json(excluded),
        "expression_gene_order_sha256": sha256_json(list(gene_ids)),
        "allowed_expression_gene_ids_sha256": sha256_json([gene_ids[i] for i in allowed]),
        "graph_identity_visibility": "unchanged",
        "evaluation_control_expression": "available",
    }
    return (allowed if enabled else None), receipt


def restrict_control(control: Tensor, allowed: tuple[int, ...] | None) -> Tensor:
    """Keep v1's fixed input width without exposing excluded expression values."""
    if allowed is None:
        return control
    result = torch.zeros_like(control)
    result[:, list(allowed)] = control[:, list(allowed)]
    return result


def select_expression(values: Tensor, allowed: tuple[int, ...] | None) -> Tensor:
    return values if allowed is None else values[:, list(allowed)]
