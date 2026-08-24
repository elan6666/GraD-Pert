from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradpert.contracts import (
    CanonicalDataManifest,
    DatasetGraphManifest,
    EvaluationStateManifest,
    SplitManifest,
)
from gradpert.hashing import sha256_file, sha256_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARED_ROOT = PROJECT_ROOT / "registry" / "prepared"
EXPECTED_V2_SPLIT_HASHES = {
    "replogle_k562_essential": ("4d8dffc4e8e217f6b5b5d794666af346169789c3c6761d61b493a94522da5d69"),
    "replogle_rpe1_essential": ("19a1cc511fb98cc6063ef1b8bcfdc73db9ea0c674b86c3f4112f4388acd80204"),
    "nadig_jurkat": "ecb2099cea9c231e9bf16c7db9bf94f2ac715df6bffece79caf0b661e863cdd0",
    "nadig_hepg2": "540b214bd879aa3cb445c762447046bb319b58d6230a6e660c0a798380331bc6",
    "norman": "ebcdde39b87d9e98ef02b84db6677a5ae7404d5953bcbe6fd4ccbb61e8a13ed7",
}


def test_prepared_receipt_sync_state_names_the_current_protocol() -> None:
    state = json.loads((PREPARED_ROOT / "CURRENT_STATE.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == "prepared-receipt-sync-v1"
    assert state["registry_version"] == "datasets-v2"
    assert state["server_verification"] == "passed"
    assert state["split_content_sha256"] == EXPECTED_V2_SPLIT_HASHES


def _require_current_receipts() -> None:
    state = json.loads((PREPARED_ROOT / "CURRENT_STATE.json").read_text(encoding="utf-8"))
    assert state["registry_version"] == "datasets-v2"
    if not state["local_receipt_chain_available"]:
        pytest.skip("datasets-v2 small server receipts are pending allowlisted synchronization")
    assert state["status"] == "synchronized_verified"


def test_all_five_small_prepared_receipt_chains_are_self_consistent() -> None:
    _require_current_receipts()
    canonical_paths = sorted(PREPARED_ROOT.glob("*/*/manifests/canonical.json"))
    assert len(canonical_paths) == 5
    assert {path.parents[2].name for path in canonical_paths} == {
        "replogle_k562_essential",
        "replogle_rpe1_essential",
        "nadig_jurkat",
        "nadig_hepg2",
        "norman",
    }

    for canonical_path in canonical_paths:
        manifest_root = canonical_path.parent
        canonical_root = manifest_root.parent / "canonical"
        canonical = CanonicalDataManifest.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        split = SplitManifest.model_validate_json(
            (manifest_root / "split.json").read_text(encoding="utf-8")
        )
        assert split.dataset_id == canonical.dataset_id
        assert split.protocol_id == canonical.protocol_id
        assert split.split_content_sha256 == canonical.split_content_sha256
        assert sha256_file(manifest_root / "source.json") == canonical.source_manifest_sha256
        assert (
            sha256_file(manifest_root / "preprocessing.json")
            == canonical.preprocessing_manifest_sha256
        )
        assert sha256_file(manifest_root / "qc.json") == canonical.qc_manifest_sha256
        assert sha256_file(manifest_root / "split.json") == canonical.split_manifest_sha256

        expression_genes = (
            (canonical_root / "expression_gene_ids.txt").read_text(encoding="utf-8").splitlines()
        )
        graph_genes = (
            (canonical_root / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
        )
        assert len(expression_genes) == canonical.n_expression_genes
        assert len(graph_genes) == canonical.n_graph_genes
        assert graph_genes[: len(expression_genes)] == expression_genes
        assert sha256_json(expression_genes) == canonical.expression_gene_order_sha256
        assert sha256_json(graph_genes) == canonical.graph_gene_order_sha256

        checksums = (manifest_root / "checksums.sha256").read_text(encoding="utf-8")
        assert f"{canonical.canonical_adata_sha256}  canonical/adata.h5ad" in checksums
        checksum_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
        assert checksum_payload["state"] == "canonical_ready"


def test_all_five_small_graph_receipt_chains_are_self_consistent() -> None:
    _require_current_receipts()
    graph_paths = sorted(PREPARED_ROOT.glob("*/*/graphs/manifest.json"))
    assert len(graph_paths) == 5
    for graph_path in graph_paths:
        dataset_root = graph_path.parents[1]
        canonical = CanonicalDataManifest.model_validate_json(
            (dataset_root / "manifests" / "canonical.json").read_text(encoding="utf-8")
        )
        graph = DatasetGraphManifest.model_validate_json(graph_path.read_text(encoding="utf-8"))
        assert (graph.dataset_id, graph.protocol_id) == (
            canonical.dataset_id,
            canonical.protocol_id,
        )
        assert graph.canonical_data_sha256 == canonical.canonical_adata_sha256
        assert graph.graph_gene_order_sha256 == canonical.graph_gene_order_sha256
        assert graph.graph_gene_count == canonical.n_graph_genes

        coverage_root = graph_path.parent / "graph_coverage"
        coverage_path = coverage_root / "graph_coverage.json"
        assert sha256_file(coverage_path) == graph.coverage_report_sha256
        assert sha256_file(coverage_root / "missing_genes.csv") == graph.missing_genes_sha256
        assert sha256_file(coverage_root / "isolated_genes.csv") == graph.isolated_genes_sha256
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        assert coverage["candidate_target_count"] == graph.candidate_target_count
        assert (
            coverage["both_sources_missing_target_count"] == graph.both_sources_missing_target_count
        )
        graph_genes = set(
            (dataset_root / "canonical" / "graph_gene_ids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert set(coverage["candidate_targets"]) <= graph_genes


def test_all_five_evaluation_state_receipts_bind_the_same_frozen_splits() -> None:
    _require_current_receipts()
    state_paths = sorted(PREPARED_ROOT.glob("*/*/evaluation/state_manifest.json"))
    assert len(state_paths) == 5
    expected_unavailable_counts = {
        "replogle_k562_essential": 0,
        "replogle_rpe1_essential": 1,
        "nadig_jurkat": 4,
        "nadig_hepg2": 0,
        "norman": 0,
    }
    for state_path in state_paths:
        dataset_root = state_path.parents[1]
        canonical = CanonicalDataManifest.model_validate_json(
            (dataset_root / "manifests" / "canonical.json").read_text(encoding="utf-8")
        )
        split = SplitManifest.model_validate_json(
            (dataset_root / "manifests" / "split.json").read_text(encoding="utf-8")
        )
        state = EvaluationStateManifest.model_validate_json(state_path.read_text(encoding="utf-8"))
        assert (state.dataset_id, state.protocol_id) == (
            canonical.dataset_id,
            canonical.protocol_id,
        )
        assert state.canonical_data_sha256 == canonical.canonical_adata_sha256
        assert state.split_content_sha256 == split.split_content_sha256
        assert state.expression_gene_order_sha256 == canonical.expression_gene_order_sha256
        assert state.condition_ids == [*split.val_conditions, *split.test_conditions]
        assert state.systema_reference_condition_ids == [
            *split.train_conditions,
            *split.val_conditions,
        ]
        assert len(state.de_unavailable_reasons) == expected_unavailable_counts[state.dataset_id]
        assert not (state_path.parent / "state_arrays.npz").exists()
