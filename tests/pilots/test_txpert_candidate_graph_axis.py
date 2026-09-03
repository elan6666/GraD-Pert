from __future__ import annotations

import csv

import pytest

from gradpert.hashing import sha256_json
from gradpert.pilots import txpert_candidate_graph_axis as candidate


def test_candidate_csv_preserves_the_exact_public_id_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gears_gene_set.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "0"])
        writer.writeheader()
        for index in range(candidate.TXPERT_CANDIDATE_GENE_COUNT):
            writer.writerow({"": index, "0": f"G{index:04d}"})
    monkeypatch.setattr(
        candidate,
        "sha256_file",
        lambda _path: candidate.TXPERT_CANDIDATE_GENE_SET_SHA256,
    )
    genes = candidate._read_txpert_candidate_genes(path)
    assert len(genes) == 9853
    assert genes[:2] == ("G0000", "G0001")
    assert genes[-1] == "G9852"


def test_candidate_manifest_rejects_a_graph_axis_reordering() -> None:
    genes = [f"G{index:04d}" for index in range(candidate.TXPERT_CANDIDATE_GENE_COUNT)]
    artifacts = {"go": "b" * 64, "string": "c" * 64}
    graph_hash = sha256_json(genes)
    payload = {
        "schema_version": "txpert-candidate-gene-graph-v1",
        "dataset_id": "nadig_jurkat",
        "protocol_id": "within_cell_unseen_single",
        "canonical_data_sha256": "a" * 64,
        "split_content_sha256": "d" * 64,
        "source_h5ad_sha256": "e" * 64,
        "source_registry_sha256": "f" * 64,
        "graph_axis_policy": "txpert_candidate_gene_universe",
        "selection_method": "frozen_txpert_gears_gene_set_order",
        "txpert_public_commit": candidate.TXPERT_PUBLIC_COMMIT,
        "candidate_gene_set_path": (
            "/data/yilangliu/GraD-Pert/upstreams/txpert/data/gears_gene_set.csv"
        ),
        "candidate_gene_set_sha256": candidate.TXPERT_CANDIDATE_GENE_SET_SHA256,
        "requested_gene_count": 9853,
        "expression_gene_count": 5000,
        "candidate_gene_ids": genes,
        "candidate_gene_order_sha256": graph_hash,
        "candidate_target_ids": ["G0001"],
        "candidate_target_order_sha256": sha256_json(["G0001"]),
        "graph_gene_ids": genes,
        "graph_gene_order_sha256": graph_hash,
        "graph_gene_count": 9853,
        "source_artifact_sha256": artifacts,
        "source_pruned_nonself_edge_count": {"go": 1, "string": 1},
        "topology_content_sha256": sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifacts}
        ),
        "top_k_incoming_per_source": 20,
        "control_graph_node_included": False,
        "gene_feature_policy": "learned_id",
        "materialization_wall_ms": 1.0,
    }
    manifest = candidate.TxPertCandidateGraphManifest.model_validate(payload)
    assert manifest.graph_gene_count == 9853
    payload["graph_gene_ids"] = [genes[1], genes[0], *genes[2:]]
    with pytest.raises(ValueError, match="must equal the frozen TxPert candidate order"):
        candidate.TxPertCandidateGraphManifest.model_validate(payload)
