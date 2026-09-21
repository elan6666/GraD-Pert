"""Response-independent expression-column partitions for the separate G1 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gradpert.hashing import sha256_file, sha256_json


def make_partition(
    gene_ids: tuple[str, ...],
    *,
    heldout_count: int,
    seed: int,
    excluded_gene_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    if len(set(gene_ids)) != len(gene_ids) or not 0 < heldout_count < len(gene_ids):
        raise ValueError("unique expression axis and a nonempty proper holdout required")
    excluded = set(excluded_gene_ids)
    if not excluded.issubset(gene_ids):
        raise ValueError("excluded genes must belong to expression axis")
    candidates = [i for i, g in enumerate(gene_ids) if g not in excluded]
    if not 0 < heldout_count < len(candidates):
        raise ValueError("G1 holdout must leave eligible training genes")
    heldout = set(np.random.default_rng(seed).choice(candidates, heldout_count, replace=False))
    result = {
        "schema_version": "gradpert-v2-expression-holdout-1",
        "selection": "preregistered_uniform_without_expression_values",
        "seed": seed,
        "expression_gene_order_sha256": sha256_json(list(gene_ids)),
        "training_gene_ids": [g for i, g in enumerate(gene_ids) if i not in heldout],
        "heldout_gene_ids": [g for i, g in enumerate(gene_ids) if i in heldout],
    }
    if excluded:
        result["schema_version"] = "gradpert-v2-expression-holdout-2"
        result["excluded_from_sampling_gene_ids"] = [g for g in gene_ids if g in excluded]
    return result


def load_partition(path: Path, digest: str, gene_ids: tuple[str, ...]) -> np.ndarray[Any, Any]:
    if sha256_file(path) != digest:
        raise ValueError("expression holdout manifest hash mismatch")
    data = json.loads(path.read_text())
    if (
        data["schema_version"]
        not in {"gradpert-v2-expression-holdout-1", "gradpert-v2-expression-holdout-2"}
        or data["selection"] != "preregistered_uniform_without_expression_values"
        or data["expression_gene_order_sha256"] != sha256_json(list(gene_ids))
    ):
        raise ValueError("expression holdout schema, source or gene axis mismatch")
    expected = make_partition(
        gene_ids,
        heldout_count=len(data["heldout_gene_ids"]),
        seed=data["seed"],
        excluded_gene_ids=tuple(data.get("excluded_from_sampling_gene_ids", ())),
    )
    if data != expected:
        raise ValueError("expression partition differs from its deterministic preregistered recipe")
    allowed = set(data["training_gene_ids"])
    return np.array([i for i, g in enumerate(gene_ids) if g in allowed], dtype=np.int64)
