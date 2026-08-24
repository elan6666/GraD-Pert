"""Model-independent condition split construction."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from gradpert.config.schema import DatasetId
from gradpert.contracts import SplitManifest
from gradpert.hashing import sha256_json

_SPLIT_NAMES = ("train", "val", "test")
_SPLIT_FRACTIONS = (0.5625, 0.1875, 0.25)


def _apportion_counts(total: int) -> tuple[int, int, int]:
    """Allocate exact total by largest remainder with stable split tie order."""

    if total < 3:
        raise ValueError("at least three perturbation conditions are required")
    quotas = [total * fraction for fraction in _SPLIT_FRACTIONS]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total - sum(counts)
    ranking = sorted(range(3), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in ranking[:remainder]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("condition count is too small for non-empty train/val/test")
    return counts[0], counts[1], counts[2]


def build_grouped_split_manifest(
    *,
    dataset_id: DatasetId,
    protocol_id: str,
    conditions: Iterable[str],
    control_condition_id: str = "ctrl",
    split_seed: int = 42,
) -> SplitManifest:
    """Build the one canonical unseen-condition split with NumPy PCG64."""

    if split_seed != 42:
        raise ValueError("v1 split seed is frozen at 42")
    values = [value for value in conditions if value != control_condition_id]
    if any(not value for value in values):
        raise ValueError("condition IDs must be non-empty")
    unique = sorted(set(values))
    if len(values) != len(unique):
        raise ValueError("condition input contains duplicate perturbation IDs")
    counts = _apportion_counts(len(unique))
    permutation = np.random.Generator(np.random.PCG64(split_seed)).permutation(len(unique))
    shuffled = [unique[int(index)] for index in permutation]
    train_end = counts[0]
    val_end = train_end + counts[1]
    partitions = {
        "train_conditions": shuffled[:train_end],
        "val_conditions": shuffled[train_end:val_end],
        "test_conditions": shuffled[val_end:],
    }
    content: dict[str, object] = {
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "policy_id": "grouped_0.5625_0.1875_0.25",
        "split_seed": split_seed,
        "control_condition_id": control_condition_id,
        **partitions,
    }
    return SplitManifest.model_validate(
        {
            "schema_version": "split-manifest-v1",
            **content,
            "split_content_sha256": sha256_json(content),
        }
    )


def build_norman_combo_seen2_split_manifest(
    *,
    conditions: Iterable[str],
    split_seed: int = 42,
) -> SplitManifest:
    """Freeze the GEARS combo-seen-2 policy with all singles retained in train."""

    if split_seed != 42:
        raise ValueError("v1 split seed is frozen at 42")
    values = list(conditions)
    ordered = list(dict.fromkeys(values))
    if len(ordered) != len(values):
        raise ValueError("condition input contains duplicate perturbation IDs")
    perturbations = [condition for condition in ordered if condition != "ctrl"]
    singles = [
        condition
        for condition in perturbations
        if len([part for part in condition.split("+") if part != "ctrl"]) == 1
    ]
    combinations = [
        condition
        for condition in perturbations
        if len([part for part in condition.split("+") if part != "ctrl"]) == 2
        and "ctrl" not in condition.split("+")
    ]
    if len(singles) + len(combinations) != len(perturbations):
        raise ValueError("Norman combo-seen-2 accepts only canonical singles and doubles")
    if len(combinations) < 20:
        raise ValueError("Norman combo-seen-2 requires enough doubles for val and test")

    rng = np.random.RandomState(split_seed)
    test_count = int(len(combinations) * 0.1)
    test_set = set(rng.choice(combinations, test_count, replace=False).tolist())
    remaining = [condition for condition in perturbations if condition not in test_set]
    remaining_combinations = [condition for condition in combinations if condition not in test_set]
    val_count = int(len(remaining_combinations) * 0.1)
    val_set = set(rng.choice(remaining_combinations, val_count, replace=False).tolist())
    train = [condition for condition in remaining if condition not in val_set]
    val = [condition for condition in combinations if condition in val_set]
    test = [condition for condition in combinations if condition in test_set]

    trained_single_genes = {
        part
        for condition in train
        if "ctrl" in condition.split("+")
        for part in condition.split("+")
        if part != "ctrl"
    }
    for condition in [*val, *test]:
        if not set(condition.split("+")).issubset(trained_single_genes):
            raise ValueError(f"combo_seen2 condition lacks both training singles: {condition}")

    content: dict[str, object] = {
        "dataset_id": "norman",
        "protocol_id": "norman_combo_seen2",
        "policy_id": "gears_predefined_combo_seen2",
        "split_seed": split_seed,
        "control_condition_id": "ctrl",
        "train_conditions": train,
        "val_conditions": val,
        "test_conditions": test,
    }
    return SplitManifest.model_validate(
        {
            "schema_version": "split-manifest-v1",
            **content,
            "split_content_sha256": sha256_json(content),
        }
    )


def apply_benchmark_condition_policy(
    manifest: SplitManifest,
    *,
    policy_id: str,
    excluded_conditions: Iterable[str],
) -> SplitManifest:
    """Remove frozen unsupported conditions without reshuffling retained conditions."""

    excluded = tuple(excluded_conditions)
    if excluded != tuple(sorted(set(excluded))):
        raise ValueError("benchmark condition exclusions must be unique and sorted")
    partition = {
        *manifest.train_conditions,
        *manifest.val_conditions,
        *manifest.test_conditions,
    }
    missing = sorted(set(excluded) - partition)
    if missing:
        raise ValueError(f"benchmark condition exclusions are absent from source split: {missing}")
    excluded_set = set(excluded)
    content: dict[str, object] = {
        **manifest.content_payload(),
        "policy_id": f"{manifest.policy_id}__{policy_id}",
        "train_conditions": [
            value for value in manifest.train_conditions if value not in excluded_set
        ],
        "val_conditions": [value for value in manifest.val_conditions if value not in excluded_set],
        "test_conditions": [
            value for value in manifest.test_conditions if value not in excluded_set
        ],
    }
    return SplitManifest.model_validate(
        {
            "schema_version": manifest.schema_version,
            **content,
            "split_content_sha256": sha256_json(content),
        }
    )
