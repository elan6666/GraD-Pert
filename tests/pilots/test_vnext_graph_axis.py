from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gradpert.features import MissingGenePTTargetsError
from gradpert.hashing import sha256_json
from gradpert.pilots import vnext_graph_axis
from gradpert.pilots.vnext_graph_axis import (
    GenePTAvailabilityReceipt,
    VNextGraphManifest,
    _direct_hvg512,
    materialize_genept_vnext_graph,
)

HASH = "a" * 64


def _manifest(*, frozen_hash: str | None = None) -> VNextGraphManifest:
    hvgs = [f"G{index:03d}" for index in range(512)]
    ranked_hvgs = list(reversed(hvgs))
    targets = ["G001", "PERT"]
    axis = [*hvgs, "PERT"]
    graph_hash = sha256_json(axis)
    artifacts = {"go": "b" * 64, "string": "c" * 64}
    return VNextGraphManifest(
        schema_version="recomputed-hvg512-graph-v2",
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_data_sha256=HASH,
        split_content_sha256=HASH,
        source_h5ad_sha256=HASH,
        source_registry_sha256=HASH,
        hvg_method="scanpy.pp.highly_variable_genes",
        hvg_flavor="seurat",
        normalize_total=4000,
        log1p=True,
        hvg_subset=True,
        requested_hvg_count=512,
        expression_gene_count=5000,
        hvg_fit_scope="full_filtered_cell_line_pre_split",
        hvg_fit_cell_count=1000,
        hvg_fit_condition_ids=["A", "B", "C", "ctrl"],
        hvg_fit_condition_ids_sha256=sha256_json(["A", "B", "C", "ctrl"]),
        direct_hvg_gene_ids=hvgs,
        direct_hvg_gene_order_sha256=sha256_json(hvgs),
        normalized_dispersion_ranked_hvg_gene_ids=ranked_hvgs,
        frozen_rank_hvg_gene_order_sha256=frozen_hash or sha256_json(ranked_hvgs),
        normalized_dispersion_ranking_sha256="e" * 64,
        candidate_target_ids=targets,
        candidate_target_order_sha256=sha256_json(targets),
        graph_gene_ids=axis,
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=len(axis),
        source_artifact_sha256=artifacts,
        source_pruned_nonself_edge_count={"go": 5, "string": 7},
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifacts}
        ),
        materialization_wall_ms=1.0,
    )


def test_vnext_graph_manifest_seals_hvg512_targets_and_topology() -> None:
    manifest = _manifest()
    assert manifest.requested_hvg_count == 512
    assert manifest.hvg_fit_scope == "full_filtered_cell_line_pre_split"
    assert manifest.hvg_subset is True
    assert manifest.graph_gene_ids[-1] == "PERT"
    assert manifest.graph_gene_count == 513


def test_vnext_graph_manifest_rejects_ranking_drift() -> None:
    with pytest.raises(ValidationError, match="normalized-dispersion HVG rank hash"):
        _manifest(frozen_hash="d" * 64)


def test_vnext_graph_manifest_rejects_unrelated_extra_graph_gene() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["graph_gene_ids"].append("UNRELATED")
    payload["graph_gene_count"] = len(payload["graph_gene_ids"])
    payload["graph_gene_order_sha256"] = sha256_json(payload["graph_gene_ids"])
    payload["topology_content_sha256"] = sha256_json(
        {
            "graph_gene_order_sha256": payload["graph_gene_order_sha256"],
            "sources": payload["source_artifact_sha256"],
        }
    )
    with pytest.raises(ValidationError, match="must equal exactly HVG512 union targets"):
        VNextGraphManifest.model_validate(payload)


def test_vnext_graph_manifest_rejects_duplicate_hvg_and_shuffled_axis() -> None:
    duplicate = _manifest().model_dump(mode="json")
    duplicate["direct_hvg_gene_ids"][-1] = duplicate["direct_hvg_gene_ids"][0]
    duplicate["direct_hvg_gene_order_sha256"] = sha256_json(duplicate["direct_hvg_gene_ids"])
    with pytest.raises(ValidationError, match="direct HVG receipt contains duplicate"):
        VNextGraphManifest.model_validate(duplicate)

    shuffled = _manifest().model_dump(mode="json")
    shuffled["graph_gene_ids"] = list(reversed(shuffled["graph_gene_ids"]))
    shuffled["graph_gene_order_sha256"] = sha256_json(shuffled["graph_gene_ids"])
    shuffled["topology_content_sha256"] = sha256_json(
        {
            "graph_gene_order_sha256": shuffled["graph_gene_order_sha256"],
            "sources": shuffled["source_artifact_sha256"],
        }
    )
    with pytest.raises(ValidationError, match="preserve the direct HVG512 prefix"):
        VNextGraphManifest.model_validate(shuffled)


def test_vnext_graph_manifest_rejects_partial_genept_identity() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["gene_feature_policy"] = "genept_emb_b_exact"
    payload["genept_source_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="requires parent, source, and removal"):
        VNextGraphManifest.model_validate(payload)


def test_direct_hvg512_mirrors_txpert_full_pre_split_pipeline(monkeypatch) -> None:
    genes = [f"G{index:03d}" for index in range(520)]
    canonical = SimpleNamespace(
        obs=pd.DataFrame({"condition": ["train", "validation", "test", "ctrl"]}),
        var=pd.DataFrame({"gene_name": genes}),
        n_obs=4,
    )
    calls: list[tuple[str, object]] = []

    def normalize_total(adata, *, target_sum: int) -> None:
        assert adata is canonical
        calls.append(("normalize_total", target_sum))

    def log1p(adata) -> None:
        assert adata is canonical
        calls.append(("log1p", None))

    def highly_variable_genes(adata, *, flavor: str, n_top_genes: int, subset: bool) -> None:
        assert adata is canonical
        calls.append(("highly_variable_genes", (flavor, n_top_genes, subset)))
        adata.var = adata.var.iloc[:512].copy()
        adata.var["highly_variable"] = True
        adata.var["dispersions_norm"] = np.arange(512, dtype=np.float64)

    fake_scanpy = SimpleNamespace(
        pp=SimpleNamespace(
            normalize_total=normalize_total,
            log1p=log1p,
            highly_variable_genes=highly_variable_genes,
        )
    )
    monkeypatch.setattr(
        vnext_graph_axis,
        "filter_cells_by_perturbation_effect",
        lambda source, entry: (source, object()),
    )
    monkeypatch.setattr(
        vnext_graph_axis,
        "canonicalize_metadata",
        lambda filtered, entry: (canonical, object()),
    )
    monkeypatch.setattr(
        vnext_graph_axis.importlib,
        "import_module",
        lambda name: fake_scanpy if name == "scanpy" else None,
    )

    subset_ids, ranked_ids, _, condition_ids, cell_count = _direct_hvg512(
        object(),
        SimpleNamespace(canonical_metadata=SimpleNamespace(condition_column="condition")),
    )

    assert calls == [
        ("normalize_total", 4000),
        ("log1p", None),
        ("highly_variable_genes", ("seurat", 512, True)),
    ]
    assert condition_ids == ("ctrl", "test", "train", "validation")
    assert cell_count == 4
    assert subset_ids == tuple(genes[:512])
    assert ranked_ids == tuple(reversed(genes[:512]))


def test_genept_unavailable_receipt_seals_exact_missing_targets() -> None:
    receipt = GenePTAvailabilityReceipt(
        schema_version="genept-vnext-availability-v1",
        status="unavailable_missing_perturbation_targets",
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        missing_non_target_policy="remove_preserving_canonical_order",
        missing_perturbation_target_policy="skip_variant_before_model_construction",
        parent_topology_content_sha256="1" * 64,
        candidate_target_order_sha256="2" * 64,
        genept_source_sha256="3" * 64,
        missing_perturbation_target_gene_ids=["MISSING_TARGET"],
        missing_perturbation_target_gene_ids_sha256=sha256_json(["MISSING_TARGET"]),
    )
    assert receipt.status == "unavailable_missing_perturbation_targets"
    assert receipt.result_topology_content_sha256 is None


def test_genept_missing_target_aborts_before_graph_destination(monkeypatch, tmp_path) -> None:
    parent = _manifest()
    source_sha = "9" * 64
    destination = tmp_path / "genept-graph"
    availability = tmp_path / "availability.json"

    monkeypatch.setattr(
        vnext_graph_axis,
        "load_vnext_graph_topology",
        lambda root: (object(), parent),
    )
    monkeypatch.setattr(
        vnext_graph_axis,
        "verify_genept_emb_b",
        lambda path: SimpleNamespace(source_sha256=source_sha),
    )

    def reject_missing_target(*args, **kwargs):
        raise MissingGenePTTargetsError(("MISSING_TARGET",))

    monkeypatch.setattr(vnext_graph_axis, "build_genept_coverage_plan", reject_missing_target)

    with pytest.raises(MissingGenePTTargetsError, match="MISSING_TARGET"):
        materialize_genept_vnext_graph(
            parent_root=tmp_path / "parent",
            destination=destination,
            genept_artifact_path=tmp_path / "emb_b.pkl",
            availability_receipt_path=availability,
            source_registry=SimpleNamespace(),
            official_checkout=tmp_path / "official",
        )

    assert not destination.exists()
    receipt = GenePTAvailabilityReceipt.model_validate_json(
        availability.read_text(encoding="utf-8")
    )
    assert receipt.status == "unavailable_missing_perturbation_targets"
    assert receipt.missing_perturbation_target_gene_ids == ["MISSING_TARGET"]
