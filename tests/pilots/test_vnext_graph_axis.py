from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gradpert.features import MissingGenePTTargetsError
from gradpert.features.text_prior import GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256
from gradpert.hashing import sha256_json
from gradpert.pilots import vnext_graph_axis
from gradpert.pilots.vnext_graph_axis import (
    GenePTAvailabilityReceipt,
    GenePTSeedAvailabilityReceipt,
    VNextGraphManifest,
    VNextGraphScaleAuditReceipt,
    _direct_hvg512,
    _direct_hvgs,
    _ordered_hvg_target_union,
    _ranking_receipt_name,
    _require_existing_graph_lineage,
    audit_vnext_hvg_graph_axes,
    materialize_genept_vnext_graph,
    preflight_genept_seed_vnext,
)

HASH = "a" * 64
SUPPORTED_HVG_COUNTS = (512, 1024, 2048, 5000)


def _manifest(
    *,
    hvg_count: int = 512,
    frozen_hash: str | None = None,
) -> VNextGraphManifest:
    hvgs = [f"G{index:04d}" for index in range(hvg_count)]
    ranked_hvgs = list(reversed(hvgs))
    targets = [hvgs[1], "PERT"]
    axis = [*hvgs, "PERT"]
    graph_hash = sha256_json(axis)
    artifacts = {"go": "b" * 64, "string": "c" * 64}
    return VNextGraphManifest(
        schema_version=(
            "recomputed-hvg512-graph-v2" if hvg_count == 512 else "recomputed-hvg-graph-v3"
        ),
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
        requested_hvg_count=hvg_count,
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


def _scale_manifest(hvg_count: int, *, gene_prefix: str = "G") -> VNextGraphManifest:
    master_ranking = [f"{gene_prefix}{index:04d}" for index in reversed(range(5000))]
    ranked_hvgs = master_ranking[:hvg_count]
    direct_hvgs = sorted(ranked_hvgs)
    axis = [*direct_hvgs, "PERT"]
    graph_hash = sha256_json(axis)
    artifacts = {
        "go": f"{hvg_count // 512:x}"[-1] * 64,
        "string": f"{(hvg_count // 512) + 8:x}"[-1] * 64,
    }
    return VNextGraphManifest(
        schema_version=(
            "recomputed-hvg512-graph-v2" if hvg_count == 512 else "recomputed-hvg-graph-v3"
        ),
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
        requested_hvg_count=hvg_count,
        expression_gene_count=5000,
        hvg_fit_scope="full_filtered_cell_line_pre_split",
        hvg_fit_cell_count=1000,
        hvg_fit_condition_ids=["A", "B", "C", "ctrl"],
        hvg_fit_condition_ids_sha256=sha256_json(["A", "B", "C", "ctrl"]),
        direct_hvg_gene_ids=direct_hvgs,
        direct_hvg_gene_order_sha256=sha256_json(direct_hvgs),
        normalized_dispersion_ranked_hvg_gene_ids=ranked_hvgs,
        frozen_rank_hvg_gene_order_sha256=sha256_json(ranked_hvgs),
        normalized_dispersion_ranking_sha256=sha256_json(
            [{"gene_id": gene, "dispersions_norm_hex": "0x1.0p+0"} for gene in ranked_hvgs]
        ),
        candidate_target_ids=["PERT"],
        candidate_target_order_sha256=sha256_json(["PERT"]),
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


@pytest.mark.parametrize("hvg_count", SUPPORTED_HVG_COUNTS)
def test_vnext_graph_manifest_seals_requested_hvgs_targets_and_topology(
    hvg_count: int,
) -> None:
    manifest = _manifest(hvg_count=hvg_count)
    assert manifest.requested_hvg_count == hvg_count
    assert manifest.hvg_fit_scope == "full_filtered_cell_line_pre_split"
    assert manifest.hvg_subset is True
    assert manifest.graph_gene_ids[-1] == "PERT"
    assert manifest.graph_gene_count == hvg_count + 1
    assert manifest.graph_gene_ids[:hvg_count] == manifest.direct_hvg_gene_ids


def test_vnext_graph_manifest_preserves_legacy_hvg512_schema() -> None:
    manifest = _manifest()
    assert manifest.schema_version == "recomputed-hvg512-graph-v2"
    assert _ranking_receipt_name(manifest.requested_hvg_count) == "hvg512_dispersion_ranking.json"

    payload = _manifest(hvg_count=1024).model_dump(mode="json")
    payload["schema_version"] = "recomputed-hvg512-graph-v2"
    with pytest.raises(ValidationError, match="legacy HVG512 graph schema"):
        VNextGraphManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("hvg_count", "receipt_name"),
    (
        (512, "hvg512_dispersion_ranking.json"),
        (1024, "hvg1024_dispersion_ranking.json"),
        (2048, "hvg2048_dispersion_ranking.json"),
        (5000, "hvg5000_dispersion_ranking.json"),
    ),
)
def test_vnext_graph_ranking_receipt_name_is_count_specific(
    hvg_count: int,
    receipt_name: str,
) -> None:
    assert _ranking_receipt_name(hvg_count) == receipt_name


def test_vnext_graph_rejects_unregistered_hvg_count() -> None:
    with pytest.raises(ValueError, match="must be one of 512, 1024, 2048, 5000"):
        _ranking_receipt_name(4096)


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
    with pytest.raises(ValidationError, match="must equal exactly HVGs union targets"):
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
    with pytest.raises(ValidationError, match="preserve the direct HVG prefix"):
        VNextGraphManifest.model_validate(shuffled)


def test_vnext_graph_manifest_rejects_partial_genept_identity() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["gene_feature_policy"] = "genept_emb_b_exact"
    payload["genept_source_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="requires parent, source, and removal"):
        VNextGraphManifest.model_validate(payload)


def test_existing_vnext_graph_requires_current_input_lineage() -> None:
    manifest = _manifest()
    entry = SimpleNamespace(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
    )
    canonical = SimpleNamespace(canonical_adata_sha256=HASH)
    split = SimpleNamespace(split_content_sha256=HASH)

    _require_existing_graph_lineage(
        manifest,
        entry=entry,
        canonical=canonical,
        split=split,
        requested_hvg_count=512,
        source_h5ad_sha256=HASH,
        source_registry_sha256=HASH,
    )
    with pytest.raises(ValueError, match="source_registry_sha256"):
        _require_existing_graph_lineage(
            manifest,
            entry=entry,
            canonical=canonical,
            split=split,
            requested_hvg_count=512,
            source_h5ad_sha256=HASH,
            source_registry_sha256="d" * 64,
        )


def test_real_artifact_scale_audit_seals_nested_sets_order_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifests = {hvg_count: _scale_manifest(hvg_count) for hvg_count in SUPPORTED_HVG_COUNTS}
    roots = {hvg_count: tmp_path / f"hvg{hvg_count}" for hvg_count in SUPPORTED_HVG_COUNTS}
    for _hvg_count, root in roots.items():
        root.mkdir()
        (root / "manifest.json").write_text(
            manifests[_hvg_count].model_dump_json(),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        vnext_graph_axis,
        "load_vnext_graph_topology",
        lambda root: (object(), manifests[int(str(root).rsplit("hvg", 1)[1])]),
    )
    destination = tmp_path / "hvg-scale-audit.json"

    receipt = audit_vnext_hvg_graph_axes(roots, destination=destination)

    assert receipt.status == "passed"
    assert receipt.nested_pairs == ["512<1024", "1024<2048", "2048<5000"]
    assert set(receipt.manifest_file_sha256_by_hvg_count) == {
        "512",
        "1024",
        "2048",
        "5000",
    }
    assert (
        VNextGraphScaleAuditReceipt.model_validate_json(destination.read_text(encoding="utf-8"))
        == receipt
    )


def test_real_artifact_scale_audit_rejects_non_nested_hvg_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifests = {hvg_count: _scale_manifest(hvg_count) for hvg_count in SUPPORTED_HVG_COUNTS}
    manifests[1024] = _scale_manifest(1024, gene_prefix="X")
    roots = {hvg_count: tmp_path / f"hvg{hvg_count}" for hvg_count in SUPPORTED_HVG_COUNTS}
    for _hvg_count, root in roots.items():
        root.mkdir()
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        vnext_graph_axis,
        "load_vnext_graph_topology",
        lambda root: (object(), manifests[int(str(root).rsplit("hvg", 1)[1])]),
    )

    with pytest.raises(ValueError, match="direct HVGs are not nested"):
        audit_vnext_hvg_graph_axes(roots, destination=tmp_path / "audit.json")


def _install_fake_hvg_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    genes: list[str],
) -> tuple[SimpleNamespace, list[tuple[str, object]]]:
    calls: list[tuple[str, object]] = []

    def canonicalize_metadata(filtered, entry):
        return (
            SimpleNamespace(
                obs=pd.DataFrame({"condition": ["train", "validation", "test", "ctrl"]}),
                var=pd.DataFrame({"gene_name": genes}),
                n_obs=4,
            ),
            object(),
        )

    def normalize_total(adata, *, target_sum: int) -> None:
        calls.append(("normalize_total", target_sum))

    def log1p(adata) -> None:
        calls.append(("log1p", None))

    def highly_variable_genes(adata, *, flavor: str, n_top_genes: int, subset: bool) -> None:
        calls.append(("highly_variable_genes", (flavor, n_top_genes, subset)))
        adata.var = adata.var.iloc[:n_top_genes].copy()
        adata.var["highly_variable"] = True
        adata.var["dispersions_norm"] = np.arange(n_top_genes, dtype=np.float64)

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
        canonicalize_metadata,
    )
    monkeypatch.setattr(
        vnext_graph_axis.importlib,
        "import_module",
        lambda name: fake_scanpy if name == "scanpy" else None,
    )
    return SimpleNamespace(canonical_metadata=SimpleNamespace(condition_column="condition")), calls


@pytest.mark.parametrize("hvg_count", SUPPORTED_HVG_COUNTS)
def test_direct_hvgs_mirror_same_full_pre_split_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    hvg_count: int,
) -> None:
    genes = [f"G{index:04d}" for index in range(5100)]
    entry, calls = _install_fake_hvg_pipeline(monkeypatch, genes)

    subset_ids, ranked_ids, _, condition_ids, cell_count = _direct_hvgs(
        object(),
        entry,
        requested_hvg_count=hvg_count,
    )

    assert calls == [
        ("normalize_total", 4000),
        ("log1p", None),
        ("highly_variable_genes", ("seurat", hvg_count, True)),
    ]
    assert condition_ids == ("ctrl", "test", "train", "validation")
    assert cell_count == 4
    assert subset_ids == tuple(genes[:hvg_count])
    assert ranked_ids == tuple(reversed(genes[:hvg_count]))


def test_direct_hvg_axes_are_nested_under_one_frozen_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genes = [f"G{index:04d}" for index in range(5100)]
    entry, _ = _install_fake_hvg_pipeline(monkeypatch, genes)
    selected = {
        hvg_count: _direct_hvgs(
            object(),
            entry,
            requested_hvg_count=hvg_count,
        )[0]
        for hvg_count in SUPPORTED_HVG_COUNTS
    }

    for smaller_count, larger_count in pairwise(SUPPORTED_HVG_COUNTS):
        assert selected[larger_count][:smaller_count] == selected[smaller_count]
        assert set(selected[smaller_count]) < set(selected[larger_count])


@pytest.mark.parametrize("hvg_count", SUPPORTED_HVG_COUNTS)
def test_hvg_target_union_preserves_hvg_prefix_and_canonical_target_order(
    hvg_count: int,
) -> None:
    hvgs = tuple(f"G{index:04d}" for index in range(hvg_count))
    canonical_graph = (*hvgs, "TARGET_B", "IGNORED", "TARGET_A")
    targets = (hvgs[1], "TARGET_A", "TARGET_B")

    assert _ordered_hvg_target_union(hvgs, canonical_graph, targets) == (
        *hvgs,
        "TARGET_B",
        "TARGET_A",
    )


def test_hvg_target_union_rejects_target_outside_canonical_graph() -> None:
    with pytest.raises(ValueError, match="candidate targets are absent"):
        _ordered_hvg_target_union(("HVG",), ("HVG",), ("MISSING",))


def test_direct_hvg512_compatibility_helper_uses_512(monkeypatch) -> None:
    observed: list[int] = []
    result = (("HVG",), ("HVG",), [], ("ctrl",), 1)

    def direct_hvgs(source, entry, *, requested_hvg_count: int):
        observed.append(requested_hvg_count)
        return result

    monkeypatch.setattr(vnext_graph_axis, "_direct_hvgs", direct_hvgs)
    assert _direct_hvg512(object(), SimpleNamespace()) == result
    assert observed == [512]


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


def test_genept_seed_preflight_preserves_parent_graph_and_seals_superset(
    monkeypatch, tmp_path
) -> None:
    parent = _manifest()
    selected_count = len(parent.graph_gene_ids)
    prior = SimpleNamespace(
        source_path=(tmp_path / "seed-go-protein-pathway.npz").resolve(),
        source_sha256=GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256,
        source_size_bytes=123,
        model="doubao-embedding-vision",
        embedding_width=2048,
        source_gene_count=selected_count + 2,
        source_gene_order_sha256="4" * 64,
        requested_runtime_gene_ids=tuple(parent.graph_gene_ids),
        requested_runtime_gene_order_sha256=parent.graph_gene_order_sha256,
        gene_ids=tuple(parent.graph_gene_ids),
        gene_order_sha256=parent.graph_gene_order_sha256,
        selected_matrix_sha256="5" * 64,
        extra_source_gene_count=2,
        extra_source_gene_ids_sha256="6" * 64,
        ignored_missing_non_perturbation_gene_ids=(),
        ignored_missing_non_perturbation_gene_ids_sha256=sha256_json([]),
        perturbation_target_gene_ids=tuple(parent.candidate_target_ids),
        perturbation_target_gene_ids_sha256=parent.candidate_target_order_sha256,
        zero_vector_gene_ids=(),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        vnext_graph_axis,
        "load_vnext_graph_topology",
        lambda root: (object(), parent),
    )

    def verify(path, **kwargs):  # type: ignore[no-untyped-def]
        observed.update(kwargs)
        return prior

    monkeypatch.setattr(vnext_graph_axis, "verify_text_prior_npz", verify)
    availability = tmp_path / "availability.json"
    runtime_graph_root = "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
    parent_root = tmp_path / runtime_graph_root
    parent_root.mkdir(parents=True)
    (parent_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    receipt = preflight_genept_seed_vnext(
        parent_root=parent_root,
        genept_artifact_path=prior.source_path,
        expected_genept_sha256=GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256,
        runtime_graph_root=runtime_graph_root,
        availability_receipt_path=availability,
    )

    assert observed["expected_gene_ids"] == tuple(parent.graph_gene_ids)
    assert observed["perturbation_target_gene_ids"] == tuple(parent.candidate_target_ids)
    assert receipt.status == "available"
    assert receipt.result_topology_content_sha256 == parent.topology_content_sha256
    assert receipt.source_gene_count == selected_count + 2
    assert receipt.extra_source_gene_count == 2
    assert receipt.requested_runtime_gene_count == selected_count
    assert receipt.ignored_missing_non_perturbation_gene_count == 0
    assert receipt.missing_non_perturbation_gene_policy == "omit_preserving_canonical_order"
    assert (
        GenePTSeedAvailabilityReceipt.model_validate_json(availability.read_text(encoding="utf-8"))
        == receipt
    )


def test_genept_seed_preflight_receipts_ordered_non_target_omission(monkeypatch, tmp_path) -> None:
    parent = _manifest()
    missing = parent.graph_gene_ids[0]
    retained = tuple(gene_id for gene_id in parent.graph_gene_ids if gene_id != missing)
    prior = SimpleNamespace(
        source_path=(tmp_path / "seed-go-protein-pathway.npz").resolve(),
        source_sha256=GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256,
        source_size_bytes=123,
        model="doubao-embedding-vision",
        embedding_width=2048,
        source_gene_count=len(retained) + 2,
        source_gene_order_sha256="4" * 64,
        requested_runtime_gene_ids=tuple(parent.graph_gene_ids),
        requested_runtime_gene_order_sha256=parent.graph_gene_order_sha256,
        gene_ids=retained,
        gene_order_sha256=sha256_json(list(retained)),
        selected_matrix_sha256="5" * 64,
        extra_source_gene_count=2,
        extra_source_gene_ids_sha256="6" * 64,
        ignored_missing_non_perturbation_gene_ids=(missing,),
        ignored_missing_non_perturbation_gene_ids_sha256=sha256_json([missing]),
        perturbation_target_gene_ids=tuple(parent.candidate_target_ids),
        perturbation_target_gene_ids_sha256=parent.candidate_target_order_sha256,
        zero_vector_gene_ids=(),
    )
    monkeypatch.setattr(
        vnext_graph_axis,
        "load_vnext_graph_topology",
        lambda root: (object(), parent),
    )
    monkeypatch.setattr(vnext_graph_axis, "verify_text_prior_npz", lambda *args, **kwargs: prior)
    runtime_graph_root = "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
    parent_root = tmp_path / runtime_graph_root
    parent_root.mkdir(parents=True)
    (parent_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    receipt = preflight_genept_seed_vnext(
        parent_root=parent_root,
        genept_artifact_path=prior.source_path,
        expected_genept_sha256=GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256,
        runtime_graph_root=runtime_graph_root,
        availability_receipt_path=tmp_path / "availability.json",
    )

    assert receipt.status == "available"
    assert receipt.requested_runtime_gene_count == len(parent.graph_gene_ids)
    assert receipt.selected_gene_count == len(retained)
    assert receipt.ignored_missing_non_perturbation_gene_count == 1
    assert receipt.ignored_missing_non_perturbation_gene_ids_sha256 == sha256_json([missing])
    assert receipt.result_topology_content_sha256 is None


def test_genept_seed_preflight_rejects_nonsealed_artifact_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"sealed Protein\+Reactome\+SIGNOR"):
        preflight_genept_seed_vnext(
            parent_root=tmp_path / "parent",
            genept_artifact_path=tmp_path / "other.npz",
            expected_genept_sha256="0" * 64,
            runtime_graph_root="vnext/graph_axes/nadig_jurkat/hvg512_plus_targets",
            availability_receipt_path=tmp_path / "availability.json",
        )


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
