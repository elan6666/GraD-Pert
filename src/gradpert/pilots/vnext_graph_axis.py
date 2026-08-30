"""Receipt-backed HVG-plus-target runtime graphs for B2-vNext."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Literal, cast

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
    verify_text_prior_npz,
)
from gradpert.features.text_prior import GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256
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

SupportedHVGCount = Literal[512, 1024, 2048, 5000]
VNextGraphSchemaVersion = Literal[
    "recomputed-hvg512-graph-v2",
    "recomputed-hvg-graph-v3",
]
_SUPPORTED_HVG_COUNTS = (512, 1024, 2048, 5000)


def _validated_hvg_count(requested_hvg_count: int) -> SupportedHVGCount:
    if requested_hvg_count not in _SUPPORTED_HVG_COUNTS:
        raise ValueError(
            "vNext graph HVG count must be one of "
            f"{', '.join(str(value) for value in _SUPPORTED_HVG_COUNTS)}"
        )
    return cast(SupportedHVGCount, requested_hvg_count)


def _ranking_receipt_name(requested_hvg_count: int) -> str:
    return f"hvg{_validated_hvg_count(requested_hvg_count)}_dispersion_ranking.json"


class VNextGraphManifest(StrictManifest):
    schema_version: VNextGraphSchemaVersion
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
    requested_hvg_count: SupportedHVGCount
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
        if self.schema_version == "recomputed-hvg512-graph-v2" and (
            self.requested_hvg_count != 512
        ):
            raise ValueError("legacy HVG512 graph schema requires requested_hvg_count=512")
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
            raise ValueError("subset HVGs differ from normalized-dispersion selection")
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
                    "learned-ID vNext graph axis must equal exactly HVGs union targets"
                )
            if self.graph_gene_ids[: self.requested_hvg_count] != self.direct_hvg_gene_ids:
                raise ValueError("learned-ID vNext graph must preserve the direct HVG prefix")
        else:
            removed = set(self.genept_removed_non_target_gene_ids)
            if not removed <= set(self.direct_hvg_gene_ids) or removed & set(
                self.candidate_target_ids
            ):
                raise ValueError("GenePT may remove only non-target HVG genes")
            if observed_axis != requested_axis - removed:
                raise ValueError(
                    "GenePT vNext graph axis must equal HVGs union targets minus receipted removals"
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


class VNextGraphScaleAuditReceipt(StrictManifest):
    """Cross-axis proof that every formal H graph shares one frozen lineage."""

    schema_version: Literal["vnext-hvg-scale-audit-v1"]
    status: Literal["passed"]
    dataset_id: Literal["nadig_jurkat"]
    protocol_id: NonEmpty
    requested_hvg_counts: list[SupportedHVGCount]
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    source_h5ad_sha256: Sha256
    source_registry_sha256: Sha256
    hvg_fit_condition_ids_sha256: Sha256
    candidate_target_order_sha256: Sha256
    manifest_file_sha256_by_hvg_count: dict[str, Sha256]
    direct_hvg_order_sha256_by_hvg_count: dict[str, Sha256]
    ranked_hvg_order_sha256_by_hvg_count: dict[str, Sha256]
    graph_gene_order_sha256_by_hvg_count: dict[str, Sha256]
    topology_content_sha256_by_hvg_count: dict[str, Sha256]
    nested_pairs: list[NonEmpty]

    @model_validator(mode="after")
    def enforce_scale_audit(self) -> VNextGraphScaleAuditReceipt:
        expected_counts = list(_SUPPORTED_HVG_COUNTS)
        if self.requested_hvg_counts != expected_counts:
            raise ValueError("vNext graph-scale audit requires all supported HVG counts in order")
        expected_keys = {str(value) for value in expected_counts}
        for values in (
            self.manifest_file_sha256_by_hvg_count,
            self.direct_hvg_order_sha256_by_hvg_count,
            self.ranked_hvg_order_sha256_by_hvg_count,
            self.graph_gene_order_sha256_by_hvg_count,
            self.topology_content_sha256_by_hvg_count,
        ):
            if set(values) != expected_keys:
                raise ValueError("vNext graph-scale audit hash keys differ from supported counts")
        expected_pairs = [
            f"{smaller}<{larger}" for smaller, larger in pairwise(_SUPPORTED_HVG_COUNTS)
        ]
        if self.nested_pairs != expected_pairs:
            raise ValueError("vNext graph-scale audit nested-pair coverage differs")
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


class GenePTSeedAvailabilityReceipt(StrictManifest):
    """Sealed coverage preflight for the Seed-GO-ProteinPathway superset."""

    schema_version: Literal["genept-seed-go-protein-pathway-availability-v2"]
    status: Literal["available"]
    dataset_id: Literal["nadig_jurkat"]
    identifier_matching: Literal["exact_case_sensitive"]
    extra_source_gene_policy: Literal["ignore_preserving_runtime_axis"]
    missing_non_perturbation_gene_policy: Literal["omit_preserving_canonical_order"]
    missing_perturbation_target_policy: Literal["fail_before_model_construction"]
    parent_topology_content_sha256: Sha256
    parent_graph_gene_order_sha256: Sha256
    candidate_target_order_sha256: Sha256
    prior_contract_id: Literal["seed_go_protein_pathway_master_v1"]
    runtime_graph_root: NonEmpty
    parent_graph_manifest_sha256: Sha256
    genept_source_path: NonEmpty
    genept_source_size_bytes: int = Field(gt=0)
    genept_source_sha256: Sha256
    genept_model: NonEmpty
    embedding_width: int = Field(gt=0)
    source_gene_count: int = Field(gt=0)
    source_gene_order_sha256: Sha256
    requested_runtime_gene_count: int = Field(gt=0)
    requested_runtime_gene_order_sha256: Sha256
    selected_gene_count: int = Field(gt=0)
    selected_gene_order_sha256: Sha256
    selected_matrix_sha256: Sha256
    extra_source_gene_count: int = Field(ge=0)
    extra_source_gene_ids_sha256: Sha256
    ignored_missing_non_perturbation_gene_count: int = Field(ge=0)
    ignored_missing_non_perturbation_gene_ids_sha256: Sha256
    perturbation_target_gene_count: int = Field(gt=0)
    perturbation_target_gene_ids_sha256: Sha256
    zero_vector_gene_count: Literal[0]
    result_topology_content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def enforce_seed_availability(self) -> GenePTSeedAvailabilityReceipt:
        if self.requested_runtime_gene_order_sha256 != self.parent_graph_gene_order_sha256:
            raise ValueError("GenePT Seed requested axis differs from the parent graph axis")
        if self.perturbation_target_gene_ids_sha256 != self.candidate_target_order_sha256:
            raise ValueError("GenePT Seed target identity differs from the parent target union")
        if self.source_gene_count != self.selected_gene_count + self.extra_source_gene_count:
            raise ValueError("GenePT Seed source/selected/extra gene counts do not reconcile")
        if self.requested_runtime_gene_count != (
            self.selected_gene_count + self.ignored_missing_non_perturbation_gene_count
        ):
            raise ValueError("GenePT Seed requested/selected/ignored gene counts do not reconcile")
        if self.ignored_missing_non_perturbation_gene_count == 0:
            if self.ignored_missing_non_perturbation_gene_ids_sha256 != sha256_json([]):
                raise ValueError("GenePT Seed zero ignored-gene count requires the empty hash")
            if self.selected_gene_order_sha256 != self.parent_graph_gene_order_sha256:
                raise ValueError("full-coverage GenePT Seed selected axis differs from parent")
            if self.result_topology_content_sha256 != self.parent_topology_content_sha256:
                raise ValueError("full-coverage GenePT Seed must preserve parent topology")
        elif self.result_topology_content_sha256 is not None:
            raise ValueError(
                "GenePT Seed omissions require a separately materialized result topology"
            )
        return self


def preflight_genept_seed_vnext(
    *,
    parent_root: str | Path,
    genept_artifact_path: str | Path,
    expected_genept_sha256: str,
    runtime_graph_root: str,
    availability_receipt_path: str | Path,
) -> GenePTSeedAvailabilityReceipt:
    """Verify the sealed prior against the unchanged H512+targets runtime graph."""

    if expected_genept_sha256 != GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256:
        raise ValueError("GenePT Seed preflight requires the sealed ProteinPathway artifact")
    relative_runtime_root = Path(runtime_graph_root)
    if (
        relative_runtime_root.is_absolute()
        or ".." in relative_runtime_root.parts
        or not relative_runtime_root.parts
    ):
        raise ValueError("GenePT Seed runtime graph root must be a safe relative path")
    parent_path = Path(parent_root).resolve(strict=True)
    if not str(parent_path).endswith(str(relative_runtime_root)):
        raise ValueError("GenePT Seed parent path differs from the declared runtime graph root")
    parent_manifest_path = parent_path / "manifest.json"
    _, parent = load_vnext_graph_topology(parent_path)
    if parent.gene_feature_policy != "learned_id":
        raise ValueError("GenePT Seed preflight requires the unfiltered learned-ID parent graph")
    prior = verify_text_prior_npz(
        genept_artifact_path,
        expected_sha256=expected_genept_sha256,
        expected_gene_ids=tuple(parent.graph_gene_ids),
        perturbation_target_gene_ids=tuple(parent.candidate_target_ids),
    )
    if prior.zero_vector_gene_ids:
        raise AssertionError("sealed Seed-GO-ProteinPathway preflight returned zero vectors")
    receipt = GenePTSeedAvailabilityReceipt(
        schema_version="genept-seed-go-protein-pathway-availability-v2",
        status="available",
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        extra_source_gene_policy="ignore_preserving_runtime_axis",
        missing_non_perturbation_gene_policy="omit_preserving_canonical_order",
        missing_perturbation_target_policy="fail_before_model_construction",
        parent_topology_content_sha256=parent.topology_content_sha256,
        parent_graph_gene_order_sha256=parent.graph_gene_order_sha256,
        candidate_target_order_sha256=parent.candidate_target_order_sha256,
        prior_contract_id="seed_go_protein_pathway_master_v1",
        runtime_graph_root=runtime_graph_root,
        parent_graph_manifest_sha256=sha256_file(parent_manifest_path),
        genept_source_path=str(prior.source_path),
        genept_source_size_bytes=prior.source_size_bytes,
        genept_source_sha256=prior.source_sha256,
        genept_model=prior.model,
        embedding_width=prior.embedding_width,
        source_gene_count=prior.source_gene_count,
        source_gene_order_sha256=prior.source_gene_order_sha256,
        requested_runtime_gene_count=len(prior.requested_runtime_gene_ids),
        requested_runtime_gene_order_sha256=prior.requested_runtime_gene_order_sha256,
        selected_gene_count=len(prior.gene_ids),
        selected_gene_order_sha256=prior.gene_order_sha256,
        selected_matrix_sha256=prior.selected_matrix_sha256,
        extra_source_gene_count=prior.extra_source_gene_count,
        extra_source_gene_ids_sha256=prior.extra_source_gene_ids_sha256,
        ignored_missing_non_perturbation_gene_count=len(
            prior.ignored_missing_non_perturbation_gene_ids
        ),
        ignored_missing_non_perturbation_gene_ids_sha256=(
            prior.ignored_missing_non_perturbation_gene_ids_sha256
        ),
        perturbation_target_gene_count=len(prior.perturbation_target_gene_ids),
        perturbation_target_gene_ids_sha256=prior.perturbation_target_gene_ids_sha256,
        zero_vector_gene_count=0,
        result_topology_content_sha256=(
            parent.topology_content_sha256
            if not prior.ignored_missing_non_perturbation_gene_ids
            else None
        ),
    )
    destination = Path(availability_receipt_path)
    atomic_json(destination, receipt.model_dump(mode="json"))
    sealed = GenePTSeedAvailabilityReceipt.model_validate_json(
        destination.read_text(encoding="utf-8")
    )
    if sealed != receipt:
        raise RuntimeError("GenePT Seed availability receipt round-trip differs")
    return receipt


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


def _direct_hvgs(
    source: object,
    entry: DatasetRegistryEntry,
    *,
    requested_hvg_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, str]], tuple[str, ...], int]:
    """Mirror TxPert within-cell HVG selection before condition splitting."""

    hvg_count = _validated_hvg_count(requested_hvg_count)
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
        n_top_genes=hvg_count,
        subset=True,
    )
    subset_gene_ids = tuple(str(value) for value in canonical.var["gene_name"])
    if len(subset_gene_ids) != hvg_count:
        raise ValueError(f"TxPert-style subset=True did not retain the requested {hvg_count} genes")
    ranked = _rank_selected_hvgs(canonical, expected_count=hvg_count)
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


def _direct_hvg512(
    source: object,
    entry: DatasetRegistryEntry,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, str]], tuple[str, ...], int]:
    """Retain the frozen HVG512 helper contract for existing callers."""

    return _direct_hvgs(source, entry, requested_hvg_count=512)


def _ordered_hvg_target_union(
    direct_hvgs: tuple[str, ...],
    canonical_graph_genes: tuple[str, ...],
    candidate_targets: tuple[str, ...],
) -> tuple[str, ...]:
    """Keep the HVG prefix and append missing targets in canonical graph order."""

    target_set = set(candidate_targets)
    missing_targets = sorted(target_set - set(canonical_graph_genes))
    if missing_targets:
        raise ValueError(f"candidate targets are absent from canonical graph: {missing_targets}")
    hvg_set = set(direct_hvgs)
    return direct_hvgs + tuple(
        gene for gene in canonical_graph_genes if gene in target_set and gene not in hvg_set
    )


def _require_existing_graph_lineage(
    manifest: VNextGraphManifest,
    *,
    entry: DatasetRegistryEntry,
    canonical: CanonicalDataManifest,
    split: SplitManifest,
    requested_hvg_count: SupportedHVGCount,
    source_h5ad_sha256: str,
    source_registry_sha256: str,
) -> None:
    expected = {
        "dataset_id": entry.dataset_id,
        "protocol_id": entry.protocol_id,
        "canonical_data_sha256": canonical.canonical_adata_sha256,
        "split_content_sha256": split.split_content_sha256,
        "requested_hvg_count": requested_hvg_count,
        "source_h5ad_sha256": source_h5ad_sha256,
        "source_registry_sha256": source_registry_sha256,
    }
    observed = {name: getattr(manifest, name) for name in expected}
    if observed != expected:
        differing = sorted(name for name in expected if observed[name] != expected[name])
        raise ValueError(
            "existing vNext graph lineage differs from the request: " + ", ".join(differing)
        )


def materialize_vnext_hvg512_graph(
    *,
    entry: DatasetRegistryEntry,
    data_root: str | Path,
    destination: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
    requested_hvg_count: int = 512,
) -> VNextGraphManifest:
    """Build a supported Nadig HVG-plus-target graph without changing H5AD."""

    if entry.dataset_id != "nadig_jurkat" or entry.source.semantics != "raw_single_cell":
        raise ValueError("the first B2-vNext graph is frozen to raw Nadig Jurkat")
    hvg_count = _validated_hvg_count(requested_hvg_count)
    started = time.perf_counter()
    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    target = Path(destination)
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
    observed_registry_sha = sha256_file(source_registry_path)
    if target.exists() and any(target.iterdir()):
        existing = load_vnext_graph_topology(target)[1]
        _require_existing_graph_lineage(
            existing,
            entry=entry,
            canonical=canonical,
            split=split,
            requested_hvg_count=hvg_count,
            source_h5ad_sha256=observed_source_sha,
            source_registry_sha256=observed_registry_sha,
        )
        return existing
    target.mkdir(parents=True, exist_ok=True)

    anndata = importlib.import_module("anndata")
    source = anndata.read_h5ad(source_path)
    (
        direct_hvgs,
        dispersion_ranked_hvgs,
        dispersion_ranking,
        fit_condition_ids,
        fit_cell_count,
    ) = _direct_hvgs(
        source,
        entry,
        requested_hvg_count=hvg_count,
    )
    if getattr(source, "isbacked", False):
        source.file.close()
    dispersion_ranking_hash = sha256_json(dispersion_ranking)

    canonical_graph_genes = tuple(
        (layout.canonical / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    candidate_targets = _candidate_targets(split)
    graph_gene_ids = _ordered_hvg_target_union(
        direct_hvgs,
        canonical_graph_genes,
        candidate_targets,
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
    schema_version: VNextGraphSchemaVersion = (
        "recomputed-hvg512-graph-v2" if hvg_count == 512 else "recomputed-hvg-graph-v3"
    )
    ranking_receipt: dict[str, object] = {
        "schema_version": (
            "txpert-full-cell-line-hvg-dispersions-norm-v2"
            if hvg_count == 512
            else "txpert-full-cell-line-hvg-dispersions-norm-v3"
        ),
        "fit_scope": "full_filtered_cell_line_pre_split",
        "fit_cell_count": fit_cell_count,
        "fit_condition_ids": list(fit_condition_ids),
        "fit_condition_ids_sha256": sha256_json(list(fit_condition_ids)),
        "ordered_entries": dispersion_ranking,
        "ordered_entries_sha256": dispersion_ranking_hash,
    }
    if hvg_count != 512:
        ranking_receipt["requested_hvg_count"] = hvg_count
    atomic_json(target / _ranking_receipt_name(hvg_count), ranking_receipt)
    graph_hash = sha256_json(list(graph_gene_ids))
    manifest = VNextGraphManifest(
        schema_version=schema_version,
        dataset_id="nadig_jurkat",
        protocol_id=entry.protocol_id,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        split_content_sha256=split.split_content_sha256,
        source_h5ad_sha256=observed_source_sha,
        source_registry_sha256=observed_registry_sha,
        hvg_method="scanpy.pp.highly_variable_genes",
        hvg_flavor="seurat",
        normalize_total=4000,
        log1p=True,
        hvg_subset=True,
        requested_hvg_count=hvg_count,
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
    ranking = read_json(source / _ranking_receipt_name(manifest.requested_hvg_count))
    expected_ranking_schema = (
        "txpert-full-cell-line-hvg-dispersions-norm-v2"
        if manifest.schema_version == "recomputed-hvg512-graph-v2"
        else "txpert-full-cell-line-hvg-dispersions-norm-v3"
    )
    if (
        not isinstance(ranking, dict)
        or ranking.get("schema_version") != expected_ranking_schema
        or (
            manifest.schema_version == "recomputed-hvg-graph-v3"
            and ranking.get("requested_hvg_count") != manifest.requested_hvg_count
        )
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


def _is_ordered_subsequence(smaller: list[str], larger: list[str]) -> bool:
    larger_iterator = iter(larger)
    return all(any(candidate == gene for candidate in larger_iterator) for gene in smaller)


def audit_vnext_hvg_graph_axes(
    roots_by_hvg_count: Mapping[int, str | Path],
    *,
    destination: str | Path,
) -> VNextGraphScaleAuditReceipt:
    """Seal the real four-axis H lineage before any formal matrix launch."""

    if set(roots_by_hvg_count) != set(_SUPPORTED_HVG_COUNTS):
        raise ValueError("vNext graph-scale audit requires exactly H512/H1024/H2048/H5000")
    roots = {hvg_count: Path(roots_by_hvg_count[hvg_count]) for hvg_count in _SUPPORTED_HVG_COUNTS}
    manifests = {
        hvg_count: load_vnext_graph_topology(roots[hvg_count])[1]
        for hvg_count in _SUPPORTED_HVG_COUNTS
    }
    shared_fields = (
        "dataset_id",
        "protocol_id",
        "canonical_data_sha256",
        "split_content_sha256",
        "source_h5ad_sha256",
        "source_registry_sha256",
        "hvg_method",
        "hvg_flavor",
        "normalize_total",
        "log1p",
        "hvg_subset",
        "expression_gene_count",
        "hvg_fit_scope",
        "hvg_fit_cell_count",
        "hvg_fit_condition_ids",
        "hvg_fit_condition_ids_sha256",
        "candidate_target_ids",
        "candidate_target_order_sha256",
        "gene_feature_policy",
    )
    reference = manifests[512]
    if any(manifest.gene_feature_policy != "learned_id" for manifest in manifests.values()):
        raise ValueError("formal H graph-scale audit accepts only learned-ID parent axes")
    for hvg_count, manifest in manifests.items():
        if manifest.requested_hvg_count != hvg_count:
            raise ValueError(f"H{hvg_count} graph manifest count differs from its audit slot")
        differing = [
            field
            for field in shared_fields
            if getattr(manifest, field) != getattr(reference, field)
        ]
        if differing:
            raise ValueError(
                f"H{hvg_count} graph lineage differs from H512: " + ", ".join(differing)
            )

    nested_pairs: list[str] = []
    for smaller_count, larger_count in pairwise(_SUPPORTED_HVG_COUNTS):
        smaller = manifests[smaller_count]
        larger = manifests[larger_count]
        smaller_hvgs = set(smaller.direct_hvg_gene_ids)
        larger_hvgs = set(larger.direct_hvg_gene_ids)
        if not smaller_hvgs < larger_hvgs:
            raise ValueError(f"H{smaller_count} direct HVGs are not nested in H{larger_count}")
        if not _is_ordered_subsequence(
            smaller.direct_hvg_gene_ids,
            larger.direct_hvg_gene_ids,
        ):
            raise ValueError(
                f"H{smaller_count} direct-HVG order is not preserved in H{larger_count}"
            )
        if larger.normalized_dispersion_ranked_hvg_gene_ids[:smaller_count] != (
            smaller.normalized_dispersion_ranked_hvg_gene_ids
        ):
            raise ValueError(
                f"H{smaller_count} normalized-dispersion ranking is not the H{larger_count} prefix"
            )
        if not set(smaller.graph_gene_ids) < set(larger.graph_gene_ids):
            raise ValueError(f"H{smaller_count} graph axis is not nested in H{larger_count}")
        nested_pairs.append(f"{smaller_count}<{larger_count}")

    receipt = VNextGraphScaleAuditReceipt(
        schema_version="vnext-hvg-scale-audit-v1",
        status="passed",
        dataset_id="nadig_jurkat",
        protocol_id=reference.protocol_id,
        requested_hvg_counts=[
            _validated_hvg_count(hvg_count) for hvg_count in _SUPPORTED_HVG_COUNTS
        ],
        canonical_data_sha256=reference.canonical_data_sha256,
        split_content_sha256=reference.split_content_sha256,
        source_h5ad_sha256=reference.source_h5ad_sha256,
        source_registry_sha256=reference.source_registry_sha256,
        hvg_fit_condition_ids_sha256=reference.hvg_fit_condition_ids_sha256,
        candidate_target_order_sha256=reference.candidate_target_order_sha256,
        manifest_file_sha256_by_hvg_count={
            str(hvg_count): sha256_file(roots[hvg_count] / "manifest.json")
            for hvg_count in _SUPPORTED_HVG_COUNTS
        },
        direct_hvg_order_sha256_by_hvg_count={
            str(hvg_count): manifests[hvg_count].direct_hvg_gene_order_sha256
            for hvg_count in _SUPPORTED_HVG_COUNTS
        },
        ranked_hvg_order_sha256_by_hvg_count={
            str(hvg_count): manifests[hvg_count].frozen_rank_hvg_gene_order_sha256
            for hvg_count in _SUPPORTED_HVG_COUNTS
        },
        graph_gene_order_sha256_by_hvg_count={
            str(hvg_count): manifests[hvg_count].graph_gene_order_sha256
            for hvg_count in _SUPPORTED_HVG_COUNTS
        },
        topology_content_sha256_by_hvg_count={
            str(hvg_count): manifests[hvg_count].topology_content_sha256
            for hvg_count in _SUPPORTED_HVG_COUNTS
        },
        nested_pairs=nested_pairs,
    )
    destination_path = Path(destination)
    atomic_json(destination_path, receipt.model_dump(mode="json"))
    sealed = VNextGraphScaleAuditReceipt.model_validate_json(
        destination_path.read_text(encoding="utf-8")
    )
    if sealed != receipt:
        raise RuntimeError("vNext graph-scale audit receipt round-trip differs")
    return receipt


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
    ranking_name = _ranking_receipt_name(parent.requested_hvg_count)
    parent_ranking = read_json(Path(parent_root) / ranking_name)
    if sha256_json(parent_ranking["ordered_entries"]) != (
        parent.normalized_dispersion_ranking_sha256
    ):
        raise ValueError("parent normalized-dispersion receipt differs")
    atomic_json(target / ranking_name, parent_ranking)
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
