from __future__ import annotations

from pathlib import Path

from gradpert.contracts import SplitManifest
from gradpert.data.controls import build_evaluation_control_manifest, stable_draw_seed
from gradpert.data.registry import load_dataset_registry
from gradpert.data.split import (
    apply_benchmark_condition_policy,
    build_grouped_split_manifest,
    build_norman_combo_seen2_split_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_V2_SPLITS = {
    "replogle_k562_essential": (
        (612, 202, 273),
        "4d8dffc4e8e217f6b5b5d794666af346169789c3c6761d61b493a94522da5d69",
    ),
    "replogle_rpe1_essential": (
        (1335, 448, 590),
        "19a1cc511fb98cc6063ef1b8bcfdc73db9ea0c674b86c3f4112f4388acd80204",
    ),
    "nadig_jurkat": (
        (1335, 445, 592),
        "ecb2099cea9c231e9bf16c7db9bf94f2ac715df6bffece79caf0b661e863cdd0",
    ),
    "nadig_hepg2": (
        (1334, 448, 591),
        "540b214bd879aa3cb445c762447046bb319b58d6230a6e660c0a798380331bc6",
    ),
    "norman": (
        (206, 11, 13),
        "ebcdde39b87d9e98ef02b84db6677a5ae7404d5953bcbe6fd4ccbb61e8a13ed7",
    ),
}


def _split():  # type: ignore[no-untyped-def]
    return build_grouped_split_manifest(
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        conditions=["ctrl", *(f"gene-{index:02d}" for index in range(16))],
    )


def test_grouped_split_has_frozen_ratio_and_is_deterministic() -> None:
    first = _split()
    second = _split()
    assert (len(first.train_conditions), len(first.val_conditions), len(first.test_conditions)) == (
        9,
        3,
        4,
    )
    assert first == second
    assert not (
        set(first.train_conditions) & set(first.val_conditions)
        or set(first.train_conditions) & set(first.test_conditions)
        or set(first.val_conditions) & set(first.test_conditions)
    )


def test_benchmark_condition_policy_removes_without_reshuffling() -> None:
    source = _split()
    excluded = sorted([source.train_conditions[1], source.test_conditions[0]])
    filtered = apply_benchmark_condition_policy(
        source,
        policy_id="gears_default_graph_intersection_v1",
        excluded_conditions=excluded,
    )
    assert filtered.policy_id == ("grouped_0.5625_0.1875_0.25__gears_default_graph_intersection_v1")
    assert filtered.train_conditions == [
        value for value in source.train_conditions if value not in excluded
    ]
    assert filtered.val_conditions == source.val_conditions
    assert filtered.test_conditions == [
        value for value in source.test_conditions if value not in excluded
    ]


def test_five_registry_exclusion_sets_reproduce_the_verified_v2_split_hashes() -> None:
    archived_root = ROOT / "registry" / "prepared" / "superseded" / "datasets-v1"
    for registry_path in sorted((ROOT / "registry" / "datasets").glob("*.yaml")):
        entry = load_dataset_registry(registry_path)
        archived_split_path = next(
            (archived_root / entry.dataset_id).glob("*/manifests/split.json")
        )
        source = SplitManifest.model_validate_json(archived_split_path.read_text(encoding="utf-8"))
        policy = entry.benchmark_condition_policy
        filtered = apply_benchmark_condition_policy(
            source,
            policy_id=policy.policy_id,
            excluded_conditions=policy.excluded_conditions,
        )
        expected_counts, expected_hash = EXPECTED_V2_SPLITS[entry.dataset_id]
        assert (
            len(filtered.train_conditions),
            len(filtered.val_conditions),
            len(filtered.test_conditions),
        ) == expected_counts
        assert filtered.split_content_sha256 == expected_hash
        assert set(source.train_conditions) | set(source.val_conditions) | set(
            source.test_conditions
        ) == (
            set(filtered.train_conditions)
            | set(filtered.val_conditions)
            | set(filtered.test_conditions)
            | set(policy.excluded_conditions)
        )


def test_control_manifest_reuses_exact_draws_and_samples_with_replacement() -> None:
    split = _split()
    pools = {
        condition: {
            "K562::batch-1": ["ctrl-a", "ctrl-b"],
            "K562::batch-2": ["ctrl-c", "ctrl-d"],
        }
        for condition in split.test_conditions
    }
    truth_contexts = {
        condition: ["K562::batch-1", "K562::batch-1", "K562::batch-2"]
        for condition in split.test_conditions
    }
    first = build_evaluation_control_manifest(
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        split_name="test",
        split_manifest=split,
        control_pools=pools,
        truth_context_ids=truth_contexts,
    )
    second = build_evaluation_control_manifest(
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        split_name="test",
        split_manifest=split,
        control_pools=pools,
        truth_context_ids=truth_contexts,
    )
    assert first == second
    assert all(len(draw.ordered_row_ids) == 300 for draw in first.draws)
    assert all(len(draw.ordered_context_ids) == 300 for draw in first.draws)
    assert all(
        set(draw.ordered_context_ids) == set(truth_contexts[draw.condition_id])
        for draw in first.draws
    )
    assert all(
        set(draw.ordered_row_ids) == {"ctrl-a", "ctrl-b", "ctrl-c", "ctrl-d"}
        for draw in first.draws
    )


def test_draw_seed_namespaces_conditions_and_splits() -> None:
    base = stable_draw_seed(dataset_id="norman", split_name="test", condition_id="A+B")
    assert base != stable_draw_seed(dataset_id="norman", split_name="val", condition_id="A+B")
    assert base != stable_draw_seed(dataset_id="norman", split_name="test", condition_id="A+C")


def test_norman_combo_seen2_keeps_all_component_singles_in_training() -> None:
    genes = [f"G{index:02d}" for index in range(12)]
    singles = [f"{gene}+ctrl" for gene in genes]
    doubles = [
        f"{genes[left]}+{genes[right]}"
        for left in range(len(genes))
        for right in range(left + 1, len(genes))
    ]
    manifest = build_norman_combo_seen2_split_manifest(conditions=["ctrl", *singles, *doubles])
    assert manifest.policy_id == "gears_predefined_combo_seen2"
    assert set(singles).issubset(manifest.train_conditions)
    assert set(manifest.val_conditions).issubset(doubles)
    assert set(manifest.test_conditions).issubset(doubles)
    assert manifest == build_norman_combo_seen2_split_manifest(
        conditions=["ctrl", *singles, *doubles]
    )
