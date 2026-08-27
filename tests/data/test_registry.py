from __future__ import annotations

from pathlib import Path

import pytest

from gradpert.data.registry import load_dataset_registry, verify_dataset_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = ROOT / "registry" / "datasets"


def test_registry_is_exact_and_has_no_cross_cell_substitution() -> None:
    report = verify_dataset_registry(REGISTRY_ROOT)
    assert report["count"] == 5
    assert len({entry["sha256"] for entry in report["entries"]}) == 5
    for path in sorted(REGISTRY_ROOT.glob("*.yaml")):
        entry = load_dataset_registry(path)
        assert entry.schema_version == "dataset-registry-v2"
        assert entry.benchmark_condition_policy.policy_id == "gears_default_graph_intersection_v1"
        assert entry.benchmark_condition_policy.excluded_conditions
        if entry.dataset_id != "replogle_k562_essential":
            assert "K562_cross_cell_lines" not in entry.source.url


def test_nadig_registry_preserves_observed_raw_to_canonical_mapping() -> None:
    for dataset_id in ("nadig_hepg2", "nadig_jurkat"):
        entry = load_dataset_registry(REGISTRY_ROOT / f"{dataset_id}.yaml")
        assert entry.source_metadata.condition_column == "gene"
        assert entry.source_metadata.batch_column == "gem_group"
        assert entry.source_metadata.control_identifier == "non-targeting"
        assert entry.source_metadata.condition_transform == (
            "append_ctrl_suffix_then_collapse_control"
        )
        assert entry.canonical_metadata.condition_column == "condition"
        assert entry.canonical_metadata.batch_column == "batch"


def test_rpe1_registry_uses_the_independently_audited_scperturb_source() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "replogle_rpe1_essential.yaml")
    assert entry.source.availability == "ready_for_download"
    assert entry.source.blocked_reason is None
    assert entry.source.checksum.value == "cc7f1ec50aeb3a3e1b4a6cfa713d80fa"
    assert entry.source_metadata.audit_state == "verified_from_frozen_reference"
    assert entry.source_metadata.condition_column == "gene"
    assert entry.source_metadata.batch_column == "batch"
    assert entry.source_metadata.gene_symbol_location == "var_index"
    assert entry.source_metadata.gene_symbol_column is None


def test_registry_separates_raw_transforms_from_preserved_processed_sources() -> None:
    for dataset_id in ("replogle_rpe1_essential", "nadig_jurkat", "nadig_hepg2"):
        entry = load_dataset_registry(REGISTRY_ROOT / f"{dataset_id}.yaml")
        assert entry.source.semantics == "raw_single_cell"
        assert entry.preprocessing.input_expression_state == "raw_integer_counts"
        assert entry.preprocessing.expression_scale_action == ("normalize_total_4000_then_log1p")
        assert entry.preprocessing.normalize_total == 4000
        assert entry.preprocessing.log1p

    for dataset_id in ("replogle_k562_essential", "norman"):
        entry = load_dataset_registry(REGISTRY_ROOT / f"{dataset_id}.yaml")
        assert entry.source.semantics == "upstream_processed_archive"
        assert entry.preprocessing.input_expression_state == "verified_upstream_log1p"
        assert entry.preprocessing.expression_scale_action == "preserve_verified_upstream"
        assert entry.preprocessing.normalize_total is None
        assert not entry.preprocessing.log1p


def test_registry_rejects_identity_mismatch(tmp_path: Path) -> None:
    source = REGISTRY_ROOT / "nadig_hepg2.yaml"
    target = tmp_path / "nadig_jurkat.yaml"
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="identity/path mismatch"):
        load_dataset_registry(target)
