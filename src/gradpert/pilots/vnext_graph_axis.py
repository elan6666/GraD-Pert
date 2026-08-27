"""Receipt-backed HVG512-plus-target runtime graph for B2-vNext."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from gradpert.contracts.base import NonEmpty, Sha256, StrictManifest
from gradpert.contracts.manifests import CanonicalDataManifest, SplitManifest
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, atomic_text, read_json
from gradpert.data.preprocessing import (
    canonicalize_metadata,
    filter_cells_by_perturbation_effect,
)
from gradpert.data.schema import DatasetRegistryEntry
from gradpert.features import (
    MissingGenePTTargetsError,
    build_genept_coverage_plan,
    verify_genept_emb_b,
)
from gradpert.graphs.materialization import (
    _atomic_graph,
    _load_pruned_graph,
    _read_filtered_edges,
    verify_graph_source_checkout,
)
from gradpert.graphs.pruning import prune_incoming_edges
from gradpert.graphs.registry import GraphSourceRegistry
from gradpert.graphs.views import GraphTopology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.pilots.graph_axis import (
    _candidate_targets,
    _rank_selected_hvgs,
)


class VNextGraphManifest(StrictManifest):
    schema_version: Literal["recomputed-hvg512-graph-v2"]
    dataset_id: Literal["nadig_jurkat"]
    protocol_id: NonEmpty
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    source_h5ad_sha256: Sha256
    source_registry_sha256: Sha256
    hvg_method: Literal["scanpy.pp.highly_variable_genes"]
    hvg_flavor: Literal["seurat"]
    normalize_total: Literal[4000]
    log1p: Literal[True]
    hvg_subset: Literal[True]
    requested_hvg_count: Literal[512]
    expression_gene_count: Literal[5000]
    hvg_fit_scope: Literal["full_filtered_cell_line_pre_split"]
    hvg_fit_cell_count: int = Field(ge=1)
    hvg_fit_condition_ids: list[NonEmpty]
    hvg_fit_condition_ids_sha256: Sha256
    direct_hvg_gene_ids: list[NonEmpty]
    direct_hvg_gene_order_sha256: Sha256
    normalized_dispersion_ranked_hvg_gene_ids: list[NonEmpty]
    frozen_rank_hvg_gene_order_sha256: Sha256
    normalized_dispersion_ranking_sha256: Sha256
    candidate_target_ids: list[NonEmpty]
    candidate_target_order_sha256: Sha256
    graph_gene_ids: list[NonEmpty]
    graph_gene_order_sha256: Sha256
    graph_gene_count: int = Field(ge=1)
    source_artifact_sha256: dict[Literal["go", "string"], Sha256]
    source_pruned_nonself_edge_count: dict[Literal["go", "string"], int]
    topology_content_sha256: Sha256
    gene_feature_policy: Literal["learned_id", "genept_emb_b_exact"] = "learned_id"
    parent_topology_content_sha256: Sha256 | None = None
    genept_source_sha256: Sha256 | None = None
    genept_removed_non_target_gene_ids: list[NonEmpty] = Field(default_factory=list)
    genept_removed_non_target_gene_ids_sha256: Sha256 | None = None
    materialization_wall_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def enforce_content(self) -> VNextGraphManifest:
        if len(self.direct_hvg_gene_ids) != self.requested_hvg_count:
            raise ValueError("direct HVG receipt count differs from requested_hvg_count")
        if len(self.direct_hvg_gene_ids) != len(set(self.direct_hvg_gene_ids)):
            raise ValueError("direct HVG receipt contains duplicate genes")
        if len(self.graph_gene_ids) != self.graph_gene_count:
            raise ValueError("vNext graph gene count differs from its axis")
        if len(self.graph_gene_ids) != len(set(self.graph_gene_ids)):
            raise ValueError("vNext graph axis contains duplicate genes")
        if sha256_json(self.direct_hvg_gene_ids) != self.direct_hvg_gene_order_sha256:
            raise ValueError("direct HVG gene hash differs")
        if sha256_json(self.hvg_fit_condition_ids) != self.hvg_fit_condition_ids_sha256:
            raise ValueError("HVG fit condition hash differs")
        if len(self.normalized_dispersion_ranked_hvg_gene_ids) != self.requested_hvg_count:
            raise ValueError("normalized-dispersion HVG receipt count differs")
        if len(self.normalized_dispersion_ranked_hvg_gene_ids) != len(
            set(self.normalized_dispersion_ranked_hvg_gene_ids)
        ):
            raise ValueError("normalized-dispersion HVG ranking contains duplicate genes")
        if sha256_json(self.normalized_dispersion_ranked_hvg_gene_ids) != (
            self.frozen_rank_hvg_gene_order_sha256
        ):
            raise ValueError("normalized-dispersion HVG rank hash differs")
        if set(self.direct_hvg_gene_ids) != set(self.normalized_dispersion_ranked_hvg_gene_ids):
            raise ValueError("subset HVG512 differs from normalized-dispersion selection")
        if sha256_json(self.candidate_target_ids) != self.candidate_target_order_sha256:
            raise ValueError("candidate target hash differs")
        if len(self.candidate_target_ids) != len(set(self.candidate_target_ids)):
            raise ValueError("candidate target receipt contains duplicate genes")
        if sha256_json(self.graph_gene_ids) != self.graph_gene_order_sha256:
            raise ValueError("vNext graph gene hash differs")
        missing = sorted(set(self.candidate_target_ids) - set(self.graph_gene_ids))
        if missing:
            raise ValueError(f"vNext graph omits perturbation targets: {missing}")
        requested_axis = set(self.direct_hvg_gene_ids) | set(self.candidate_target_ids)
        observed_axis = set(self.graph_gene_ids)
        if self.gene_feature_policy == "learned_id":
            if observed_axis != requested_axis:
                raise ValueError(
                    "learned-ID vNext graph axis must equal exactly HVG512 union targets"
                )
            if self.graph_gene_ids[: self.requested_hvg_count] != self.direct_hvg_gene_ids:
                raise ValueError("learned-ID vNext graph must preserve the direct HVG512 prefix")
        else:
            removed = set(self.genept_removed_non_target_gene_ids)
            if not removed <= set(self.direct_hvg_gene_ids) or removed & set(
                self.candidate_target_ids
            ):
                raise ValueError("GenePT may remove only non-target HVG genes")
            if observed_axis != requested_axis - removed:
                raise ValueError(
                    "GenePT vNext graph axis must equal HVG512 union targets "
                    "minus receipted removals"
                )
            retained_hvgs = [gene for gene in self.direct_hvg_gene_ids if gene not in removed]
            if self.graph_gene_ids[: len(retained_hvgs)] != retained_hvgs:
                raise ValueError("GenePT vNext graph must preserve retained direct-HVG order")
        expected_topology = sha256_json(
            {
                "graph_gene_order_sha256": self.graph_gene_order_sha256,
                "sources": self.source_artifact_sha256,
            }
        )
        if expected_topology != self.topology_content_sha256:
            raise ValueError("vNext topology content hash differs")
        if self.gene_feature_policy == "learned_id":
            if (
                any(
                    value is not None
                    for value in (
                        self.parent_topology_content_sha256,
                        self.genept_source_sha256,
                        self.genept_removed_non_target_gene_ids_sha256,
                    )
                )
                or self.genept_removed_non_target_gene_ids
            ):
                raise ValueError("learned_id graph must not carry GenePT filtering state")
        else:
            if (
                self.parent_topology_content_sha256 is None
                or self.genept_source_sha256 is None
                or self.genept_removed_non_target_gene_ids_sha256 is None
            ):
                raise ValueError("GenePT graph requires parent, source, and removal hashes")
            if sha256_json(self.genept_removed_non_target_gene_ids) != (
                self.genept_removed_non_target_gene_ids_sha256
            ):
                raise ValueError("GenePT removed-gene hash differs")
        return self


class GenePTAvailabilityReceipt(StrictManifest):
    """Fail-closed server preflight outcome for the frozen GenePT artifact."""

    schema_version: Literal["genept-vnext-availability-v1"]
    status: Literal["available", "unavailable_missing_perturbation_targets"]
    dataset_id: Literal["nadig_jurkat"]
    identifier_matching: Literal["exact_case_sensitive"]
    missing_non_target_policy: Literal["remove_preserving_canonical_order"]
    missing_perturbation_target_policy: Literal["skip_variant_before_model_construction"]
    parent_topology_content_sha256: Sha256
    candidate_target_order_sha256: Sha256
    genept_source_sha256: Sha256
    missing_perturbation_target_gene_ids: list[NonEmpty]
    missing_perturbation_target_gene_ids_sha256: Sha256
    result_topology_content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def enforce_availability(self) -> GenePTAvailabilityReceipt:
        if sha256_json(self.missing_perturbation_target_gene_ids) != (
            self.missing_perturbation_target_gene_ids_sha256
        ):
            raise ValueError("GenePT unavailable target hash differs")
        if self.status == "available":
            if self.missing_perturbation_target_gene_ids:
                raise ValueError("available GenePT receipt cannot contain missing targets")
            if self.result_topology_content_sha256 is None:
                raise ValueError("available GenePT receipt requires result topology")
        elif (
            not self.missing_perturbation_target_gene_ids
            or self.result_topology_content_sha256 is not None
        ):
            raise ValueError("unavailable GenePT receipt requires only missing target IDs")
        return self


def _genept_availability_receipt(
    *,
    parent: VNextGraphManifest,
    genept_source_sha256: str,
    missing_targets: tuple[str, ...],
    result_topology_content_sha256: str | None,
) -> GenePTAvailabilityReceipt:
    return GenePTAvailabilityReceipt(
        schema_version="genept-vnext-availability-v1",
        status=("unavailable_missing_perturbation_targets" if missing_targets else "available"),
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        missing_non_target_policy="remove_preserving_canonical_order",
        missing_perturbation_target_policy="skip_variant_before_model_construction",
        parent_topology_content_sha256=parent.topology_content_sha256,
        candidate_target_order_sha256=parent.candidate_target_order_sha256,
        genept_source_sha256=genept_source_sha256,
        missing_perturbation_target_gene_ids=list(missing_targets),
        missing_perturbation_target_gene_ids_sha256=sha256_json(list(missing_targets)),
        result_topology_content_sha256=result_topology_content_sha256,
    )


def _direct_hvg512(
    source: object,
    entry: DatasetRegistryEntry,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, str]], tuple[str, ...], int]:
    """Mirror TxPert within-cell HVG selection before condition splitting."""

    filtered, _ = filter_cells_by_perturbation_effect(source, entry)
    canonical, _ = canonicalize_metadata(filtered, entry)
    condition_column = entry.canonical_metadata.condition_column
    observed = np.asarray(canonical.obs[condition_column], dtype=str)
    if observed.size == 0:
        raise ValueError("filtered full-cell-line HVG scope contains no cells")
    fit_condition_ids = tuple(sorted(set(observed.tolist())))
    fit_cell_count = int(canonical.n_obs)
    scanpy = importlib.import_module("scanpy")
    scanpy.pp.normalize_total(canonical, target_sum=4000)
    scanpy.pp.log1p(canonical)
    scanpy.pp.highly_variable_genes(
        canonical,
        flavor="seurat",
        n_top_genes=512,
        subset=True,
    )
    subset_gene_ids = tuple(str(value) for value in canonical.var["gene_name"])
    if len(subset_gene_ids) != 512:
        raise ValueError("TxPert-style subset=True did not retain exactly 512 genes")
    ranked = _rank_selected_hvgs(canonical, expected_count=512)
    gene_names = tuple(str(value) for value in canonical.var["gene_name"])
    if len(gene_names) != len(set(gene_names)):
        raise ValueError("full-cell-line HVG ranking requires unique canonical gene names")
    dispersion_by_gene = {
        gene: float(value)
        for gene, value in zip(
            gene_names,
            np.asarray(canonical.var["dispersions_norm"], dtype=np.float64),
            strict=True,
        )
    }
    ranking_receipt = [
        {
            "gene_id": gene,
            "dispersions_norm_hex": dispersion_by_gene[gene].hex(),
        }
        for gene in ranked
    ]
    if set(subset_gene_ids) != set(ranked):
        raise ValueError("TxPert-style subset genes differ from normalized-dispersion ranking")
    return subset_gene_ids, ranked, ranking_receipt, fit_condition_ids, fit_cell_count


def materialize_vnext_hvg512_graph(
    *,
    entry: DatasetRegistryEntry,
    data_root: str | Path,
    destination: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> VNextGraphManifest:
    """Build the immutable Nadig HVG512-plus-target graph without changing H5AD."""

    if entry.dataset_id != "nadig_jurkat" or entry.source.semantics != "raw_single_cell":
        raise ValueError("the first B2-vNext graph is frozen to raw Nadig Jurkat")
    started = time.perf_counter()
    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        return load_vnext_graph_topology(target)[1]
    target.mkdir(parents=True, exist_ok=True)
    canonical = CanonicalDataManifest.model_validate_json(
        layout.canonical_manifest.read_text(encoding="utf-8")
    )
    split = SplitManifest.model_validate_json(
        (layout.manifests / "split.json").read_text(encoding="utf-8")
    )
    source_path = layout.source / entry.source.filename
    if entry.source.checksum.algorithm != "sha256":
        raise ValueError("the raw Nadig Jurkat source requires a frozen SHA-256")
    observed_source_sha = sha256_file(source_path, chunk_size=8 * 1024 * 1024)
    if observed_source_sha != entry.source.checksum.value:
        raise ValueError("raw source H5AD differs from the frozen registry")

    anndata = importlib.import_module("anndata")
    source = anndata.read_h5ad(source_path)
    (
        direct_hvgs,
        dispersion_ranked_hvgs,
        dispersion_ranking,
        fit_condition_ids,
        fit_cell_count,
    ) = _direct_hvg512(
        source,
        entry,
    )
    if getattr(source, "isbacked", False):
        source.file.close()
    dispersion_ranking_hash = sha256_json(dispersion_ranking)

    canonical_graph_genes = tuple(
        (layout.canonical / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    candidate_targets = _candidate_targets(split)
    missing_targets = sorted(set(candidate_targets) - set(canonical_graph_genes))
    if missing_targets:
        raise ValueError(f"candidate targets are absent from canonical graph: {missing_targets}")
    hvg_set = set(direct_hvgs)
    target_set = set(candidate_targets)
    graph_gene_ids = direct_hvgs + tuple(
        gene for gene in canonical_graph_genes if gene in target_set and gene not in hvg_set
    )

    source_paths = verify_graph_source_checkout(official_checkout, source_registry)
    artifact_hashes: dict[Literal["go", "string"], str] = {}
    edge_counts: dict[Literal["go", "string"], int] = {}
    for source_name in ("go", "string"):
        edges, _, _ = _read_filtered_edges(
            source_paths[source_name], source_registry.sources[source_name], graph_gene_ids
        )
        graph = prune_incoming_edges(
            source_name=source_name,
            gene_ids=graph_gene_ids,
            weighted_edges=edges,
            top_k=20,
        )
        artifact_path = target / f"{source_name}.npz"
        _atomic_graph(artifact_path, graph)
        artifact_hashes[source_name] = sha256_file(artifact_path)
        edge_counts[source_name] = len(graph.edges)
    atomic_text(target / "graph_gene_ids.txt", "\n".join(graph_gene_ids) + "\n")
    atomic_json(
        target / "hvg512_dispersion_ranking.json",
        {
            "schema_version": "txpert-full-cell-line-hvg-dispersions-norm-v2",
            "fit_scope": "full_filtered_cell_line_pre_split",
            "fit_cell_count": fit_cell_count,
            "fit_condition_ids": list(fit_condition_ids),
            "fit_condition_ids_sha256": sha256_json(list(fit_condition_ids)),
            "ordered_entries": dispersion_ranking,
            "ordered_entries_sha256": dispersion_ranking_hash,
        },
    )
    graph_hash = sha256_json(list(graph_gene_ids))
    manifest = VNextGraphManifest(
        schema_version="recomputed-hvg512-graph-v2",
        dataset_id="nadig_jurkat",
        protocol_id=entry.protocol_id,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        split_content_sha256=split.split_content_sha256,
        source_h5ad_sha256=observed_source_sha,
        source_registry_sha256=sha256_file(source_registry_path),
        hvg_method="scanpy.pp.highly_variable_genes",
        hvg_flavor="seurat",
        normalize_total=4000,
        log1p=True,
        hvg_subset=True,
        requested_hvg_count=512,
        expression_gene_count=5000,
        hvg_fit_scope="full_filtered_cell_line_pre_split",
        hvg_fit_cell_count=fit_cell_count,
        hvg_fit_condition_ids=list(fit_condition_ids),
        hvg_fit_condition_ids_sha256=sha256_json(list(fit_condition_ids)),
        direct_hvg_gene_ids=list(direct_hvgs),
        direct_hvg_gene_order_sha256=sha256_json(list(direct_hvgs)),
        normalized_dispersion_ranked_hvg_gene_ids=list(dispersion_ranked_hvgs),
        frozen_rank_hvg_gene_order_sha256=sha256_json(list(dispersion_ranked_hvgs)),
        normalized_dispersion_ranking_sha256=dispersion_ranking_hash,
        candidate_target_ids=list(candidate_targets),
        candidate_target_order_sha256=sha256_json(list(candidate_targets)),
        graph_gene_ids=list(graph_gene_ids),
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=len(graph_gene_ids),
        source_artifact_sha256=artifact_hashes,
        source_pruned_nonself_edge_count=edge_counts,
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifact_hashes}
        ),
        materialization_wall_ms=(time.perf_counter() - started) * 1000.0,
    )
    atomic_json(target / "manifest.json", manifest.model_dump(mode="json"))
    load_vnext_graph_topology(target)
    return manifest


def load_vnext_graph_topology(
    root: str | Path,
) -> tuple[GraphTopology, VNextGraphManifest]:
    source = Path(root)
    manifest = VNextGraphManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    gene_ids = tuple((source / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines())
    if sha256_json(list(gene_ids)) != manifest.graph_gene_order_sha256:
        raise ValueError("vNext runtime graph axis differs from its manifest")
    ranking = read_json(source / "hvg512_dispersion_ranking.json")
    if (
        not isinstance(ranking, dict)
        or ranking.get("schema_version") != "txpert-full-cell-line-hvg-dispersions-norm-v2"
        or ranking.get("fit_scope") != manifest.hvg_fit_scope
        or ranking.get("fit_cell_count") != manifest.hvg_fit_cell_count
        or ranking.get("fit_condition_ids") != manifest.hvg_fit_condition_ids
        or ranking.get("fit_condition_ids_sha256") != manifest.hvg_fit_condition_ids_sha256
    ):
        raise ValueError("vNext full-cell-line HVG ranking receipt identity differs")
    ordered_entries = ranking.get("ordered_entries")
    if not isinstance(ordered_entries, list) or sha256_json(ordered_entries) != (
        manifest.normalized_dispersion_ranking_sha256
    ):
        raise ValueError("vNext normalized-dispersion ranking hash differs")
    ranked_gene_ids = [
        entry.get("gene_id") if isinstance(entry, dict) else None for entry in ordered_entries
    ]
    if ranked_gene_ids != manifest.normalized_dispersion_ranked_hvg_gene_ids:
        raise ValueError("vNext normalized-dispersion gene order differs")
    graphs = {}
    for source_name in ("go", "string"):
        artifact = source / f"{source_name}.npz"
        if sha256_file(artifact) != manifest.source_artifact_sha256[source_name]:
            raise ValueError(f"vNext graph artifact hash mismatch: {source_name}")
        graphs[source_name] = _load_pruned_graph(
            artifact, source_name=source_name, gene_ids=gene_ids
        )
    return GraphTopology(gene_ids=gene_ids, sources=graphs), manifest


def materialize_genept_vnext_graph(
    *,
    parent_root: str | Path,
    destination: str | Path,
    genept_artifact_path: str | Path,
    availability_receipt_path: str | Path | None = None,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> VNextGraphManifest:
    """Filter only missing non-targets, then re-prune both graph sources."""

    started = time.perf_counter()
    _, parent = load_vnext_graph_topology(parent_root)
    if parent.gene_feature_policy != "learned_id":
        raise ValueError("GenePT filtering requires the unfiltered learned-ID parent graph")
    artifact = verify_genept_emb_b(genept_artifact_path)
    try:
        coverage = build_genept_coverage_plan(
            artifact,
            ordered_graph_gene_ids=parent.graph_gene_ids,
            perturbation_target_gene_ids=parent.candidate_target_ids,
        )
    except MissingGenePTTargetsError as error:
        if availability_receipt_path is not None:
            receipt = _genept_availability_receipt(
                parent=parent,
                genept_source_sha256=artifact.source_sha256,
                missing_targets=error.missing_target_gene_ids,
                result_topology_content_sha256=None,
            )
            atomic_json(availability_receipt_path, receipt.model_dump(mode="json"))
        raise
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        observed = load_vnext_graph_topology(target)[1]
        if (
            observed.gene_feature_policy != "genept_emb_b_exact"
            or observed.parent_topology_content_sha256 != parent.topology_content_sha256
            or observed.genept_source_sha256 != artifact.source_sha256
        ):
            raise ValueError("existing GenePT graph identity differs from requested inputs")
        return observed
    target.mkdir(parents=True, exist_ok=True)
    graph_gene_ids = coverage.retained_graph_gene_ids
    source_paths = verify_graph_source_checkout(official_checkout, source_registry)
    artifact_hashes: dict[Literal["go", "string"], str] = {}
    edge_counts: dict[Literal["go", "string"], int] = {}
    for source_name in ("go", "string"):
        edges, _, _ = _read_filtered_edges(
            source_paths[source_name], source_registry.sources[source_name], graph_gene_ids
        )
        graph = prune_incoming_edges(
            source_name=source_name,
            gene_ids=graph_gene_ids,
            weighted_edges=edges,
            top_k=20,
        )
        artifact_path = target / f"{source_name}.npz"
        _atomic_graph(artifact_path, graph)
        artifact_hashes[source_name] = sha256_file(artifact_path)
        edge_counts[source_name] = len(graph.edges)
    atomic_text(target / "graph_gene_ids.txt", "\n".join(graph_gene_ids) + "\n")
    parent_ranking = read_json(Path(parent_root) / "hvg512_dispersion_ranking.json")
    if sha256_json(parent_ranking["ordered_entries"]) != (
        parent.normalized_dispersion_ranking_sha256
    ):
        raise ValueError("parent normalized-dispersion receipt differs")
    atomic_json(target / "hvg512_dispersion_ranking.json", parent_ranking)
    graph_hash = sha256_json(list(graph_gene_ids))
    manifest = VNextGraphManifest(
        schema_version=parent.schema_version,
        dataset_id=parent.dataset_id,
        protocol_id=parent.protocol_id,
        canonical_data_sha256=parent.canonical_data_sha256,
        split_content_sha256=parent.split_content_sha256,
        source_h5ad_sha256=parent.source_h5ad_sha256,
        source_registry_sha256=parent.source_registry_sha256,
        hvg_method=parent.hvg_method,
        hvg_flavor=parent.hvg_flavor,
        normalize_total=parent.normalize_total,
        log1p=parent.log1p,
        hvg_subset=parent.hvg_subset,
        requested_hvg_count=parent.requested_hvg_count,
        expression_gene_count=parent.expression_gene_count,
        hvg_fit_scope=parent.hvg_fit_scope,
        hvg_fit_cell_count=parent.hvg_fit_cell_count,
        hvg_fit_condition_ids=parent.hvg_fit_condition_ids,
        hvg_fit_condition_ids_sha256=parent.hvg_fit_condition_ids_sha256,
        direct_hvg_gene_ids=parent.direct_hvg_gene_ids,
        direct_hvg_gene_order_sha256=parent.direct_hvg_gene_order_sha256,
        normalized_dispersion_ranked_hvg_gene_ids=(
            parent.normalized_dispersion_ranked_hvg_gene_ids
        ),
        frozen_rank_hvg_gene_order_sha256=parent.frozen_rank_hvg_gene_order_sha256,
        normalized_dispersion_ranking_sha256=parent.normalized_dispersion_ranking_sha256,
        candidate_target_ids=parent.candidate_target_ids,
        candidate_target_order_sha256=parent.candidate_target_order_sha256,
        graph_gene_ids=list(graph_gene_ids),
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=len(graph_gene_ids),
        source_artifact_sha256=artifact_hashes,
        source_pruned_nonself_edge_count=edge_counts,
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifact_hashes}
        ),
        gene_feature_policy="genept_emb_b_exact",
        parent_topology_content_sha256=parent.topology_content_sha256,
        genept_source_sha256=artifact.source_sha256,
        genept_removed_non_target_gene_ids=list(coverage.removed_non_target_gene_ids),
        genept_removed_non_target_gene_ids_sha256=(coverage.removed_non_target_gene_ids_sha256),
        materialization_wall_ms=(time.perf_counter() - started) * 1000.0,
    )
    atomic_json(target / "manifest.json", manifest.model_dump(mode="json"))
    atomic_json(target / "genept_coverage.json", coverage.to_receipt())
    load_vnext_graph_topology(target)
    if availability_receipt_path is not None:
        receipt = _genept_availability_receipt(
            parent=parent,
            genept_source_sha256=artifact.source_sha256,
            missing_targets=(),
            result_topology_content_sha256=manifest.topology_content_sha256,
        )
        atomic_json(availability_receipt_path, receipt.model_dump(mode="json"))
    return manifest
