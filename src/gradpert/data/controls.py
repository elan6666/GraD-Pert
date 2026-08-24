"""Shared Scouter-style 300-control manifest construction."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from gradpert.config.schema import DatasetId
from gradpert.contracts import ControlDraw, EvaluationControlManifest, SplitManifest
from gradpert.hashing import sha256_json


def stable_draw_seed(
    *, dataset_id: str, split_name: str, condition_id: str, evaluation_seed: int = 20260824
) -> int:
    """Derive a platform-stable 128-bit PCG64 seed for one condition draw."""

    key = f"{evaluation_seed}::{dataset_id}::{split_name}::{condition_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:16], byteorder="big")


def build_evaluation_control_manifest(
    *,
    dataset_id: DatasetId,
    protocol_id: str,
    split_name: Literal["val", "test"],
    split_manifest: SplitManifest,
    control_pools: Mapping[str, Mapping[str, Sequence[str]]],
    truth_context_ids: Mapping[str, Sequence[str]],
    evaluation_seed: Literal[20260824] = 20260824,
) -> EvaluationControlManifest:
    """Sample exact ordered row IDs once for all models, with replacement."""

    if dataset_id != split_manifest.dataset_id or protocol_id != split_manifest.protocol_id:
        raise ValueError("control manifest and split manifest identities differ")
    if split_name not in {"val", "test"}:
        raise ValueError("evaluation controls are defined only for val or test")
    condition_ids = (
        split_manifest.val_conditions if split_name == "val" else split_manifest.test_conditions
    )
    draws: list[ControlDraw] = []
    for condition_id in condition_ids:
        if condition_id not in control_pools or condition_id not in truth_context_ids:
            raise ValueError(f"missing eligible control pools/truth contexts for {condition_id}")
        pools = control_pools[condition_id]
        truth_contexts = tuple(str(value) for value in truth_context_ids[condition_id])
        if not truth_contexts or any(not value for value in truth_contexts):
            raise ValueError(f"truth contexts are empty/invalid for {condition_id}")
        normalized_pools: dict[str, list[str]] = {}
        for context_id in sorted(set(truth_contexts)):
            pool = [str(row_id) for row_id in pools.get(context_id, ())]
            if not pool or any(not row_id for row_id in pool):
                raise ValueError(
                    f"eligible control pool is empty/invalid for {condition_id}:{context_id}"
                )
            if len(pool) != len(set(pool)):
                raise ValueError(
                    f"source control row IDs are not unique for {condition_id}:{context_id}"
                )
            normalized_pools[context_id] = pool
        rng = np.random.Generator(
            np.random.PCG64(
                stable_draw_seed(
                    dataset_id=dataset_id,
                    split_name=split_name,
                    condition_id=condition_id,
                    evaluation_seed=evaluation_seed,
                )
            )
        )
        selected_contexts = [
            str(context_id) for context_id in rng.choice(truth_contexts, size=300, replace=True)
        ]
        selected = [
            str(rng.choice(normalized_pools[context_id])) for context_id in selected_contexts
        ]
        draws.append(
            ControlDraw(
                condition_id=condition_id,
                context_policy="truth_cell_context_resampling",
                source_pool_sha256=sha256_json(
                    {context_id: sorted(pool) for context_id, pool in normalized_pools.items()}
                ),
                ordered_context_ids=selected_contexts,
                ordered_context_ids_sha256=sha256_json(selected_contexts),
                ordered_row_ids=selected,
                ordered_row_ids_sha256=sha256_json(selected),
            )
        )
    return EvaluationControlManifest(
        schema_version="evaluation-controls-v1",
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        split_name=split_name,
        split_content_sha256=split_manifest.split_content_sha256,
        evaluation_seed=evaluation_seed,
        rng="numpy_pcg64",
        sample_with_replacement=True,
        context_policy="truth_cell_context_resampling",
        n_controls_per_condition=300,
        draws=draws,
    )
