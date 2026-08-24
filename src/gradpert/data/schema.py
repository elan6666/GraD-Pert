"""Strict schema for one standalone dataset source registry file."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gradpert.config.schema import DatasetId


class StrictRegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class UpstreamChecksum(StrictRegistryModel):
    algorithm: Literal["md5", "sha256"]
    value: str = Field(pattern=r"^[0-9a-f]+$")

    @model_validator(mode="after")
    def enforce_length(self) -> UpstreamChecksum:
        expected = 32 if self.algorithm == "md5" else 64
        if len(self.value) != expected:
            raise ValueError(f"{self.algorithm} checksum must contain {expected} hex characters")
        return self


class DatasetSource(StrictRegistryModel):
    url: str = Field(pattern=r"^https://")
    filename: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    checksum: UpstreamChecksum
    license_id: str = Field(min_length=1)
    semantics: Literal["raw_single_cell", "upstream_processed_archive"]
    archive_h5ad_member: str | None
    evidence_url: str = Field(pattern=r"^https://")
    immutable_source_id: str = Field(min_length=1)
    availability: Literal["ready_for_download", "blocked_upstream"]
    blocked_reason: str | None

    @model_validator(mode="after")
    def enforce_availability(self) -> DatasetSource:
        if "/" in self.filename or "\\" in self.filename or self.filename in {".", ".."}:
            raise ValueError("source filename must be a plain basename")
        if self.availability == "blocked_upstream" and not self.blocked_reason:
            raise ValueError("blocked source requires blocked_reason")
        if self.availability == "ready_for_download" and self.blocked_reason is not None:
            raise ValueError("ready source cannot declare blocked_reason")
        if self.semantics == "upstream_processed_archive":
            member = self.archive_h5ad_member
            if member is None or not member.endswith(".h5ad") or member.startswith("/"):
                raise ValueError("processed archive requires an explicit relative H5AD member")
            if any(part in {"", ".", ".."} for part in member.split("/")):
                raise ValueError("archive H5AD member must be a safe relative path")
        elif self.archive_h5ad_member is not None:
            raise ValueError("raw H5AD source forbids archive_h5ad_member")
        return self


class SourceMetadataMapping(StrictRegistryModel):
    """Observed upstream layout and its explicit canonicalization rule."""

    audit_state: Literal["verified_from_frozen_reference", "requires_source_audit"]
    condition_column: str | None
    batch_column: str | None
    constant_batch_value: str | None
    cell_type_column: str | None
    observed_cell_type_values: list[str] | None
    control_identifier: str | None
    gene_symbol_location: Literal["var_column", "var_index"] | None
    gene_symbol_column: str | None
    gene_symbol_duplicate_policy: Literal["reject", "suffix_later_with_var_index"] | None
    canonical_cell_type_value: str = Field(min_length=1)
    condition_transform: (
        Literal[
            "identity",
            "append_ctrl_suffix_then_collapse_control",
            "normalize_perturbation_components",
        ]
        | None
    )

    @model_validator(mode="after")
    def require_verified_layout(self) -> SourceMetadataMapping:
        required = (
            self.condition_column,
            self.control_identifier,
            self.condition_transform,
        )
        if self.audit_state == "verified_from_frozen_reference" and any(
            value is None for value in required
        ):
            raise ValueError("verified source mapping requires all non-cell-type fields")
        if self.audit_state == "verified_from_frozen_reference":
            if (self.batch_column is None) == (self.constant_batch_value is None):
                raise ValueError(
                    "verified source mapping requires exactly one batch column or constant"
                )
            if self.cell_type_column is None:
                if self.observed_cell_type_values is not None:
                    raise ValueError("absent cell-type column forbids observed values")
            elif not self.observed_cell_type_values:
                raise ValueError("cell-type column requires frozen observed values")
        if self.audit_state == "verified_from_frozen_reference":
            if self.gene_symbol_location == "var_column" and self.gene_symbol_column is None:
                raise ValueError("var_column gene symbols require gene_symbol_column")
            if self.gene_symbol_location == "var_index" and self.gene_symbol_column is not None:
                raise ValueError("var_index gene symbols forbid gene_symbol_column")
            if self.gene_symbol_location is None:
                raise ValueError("verified source mapping requires gene_symbol_location")
            if self.gene_symbol_duplicate_policy is None:
                raise ValueError("verified source mapping requires gene duplicate policy")
        if self.audit_state == "requires_source_audit" and any(
            value is not None
            for value in (
                self.condition_column,
                self.batch_column,
                self.constant_batch_value,
                self.cell_type_column,
                self.observed_cell_type_values,
                self.control_identifier,
                self.gene_symbol_location,
                self.gene_symbol_column,
                self.gene_symbol_duplicate_policy,
                self.condition_transform,
            )
        ):
            raise ValueError("unaudited source mapping must not guess upstream fields")
        return self


class CanonicalMetadataSchema(StrictRegistryModel):
    """Column names every model adapter receives after canonicalization."""

    condition_column: Literal["condition"]
    batch_column: Literal["batch"]
    cell_type_column: Literal["cell_type"]
    control_column: Literal["control"]
    condition_name_column: Literal["condition_name"]
    gene_symbol_column: Literal["gene_name"]


class PreprocessingSpec(StrictRegistryModel):
    profile_id: Literal["txpert_within_cell_v1", "gears_norman_audited_v1"]
    signal_filter: str
    normalize_total: int | None
    log1p: bool
    hvg_count: int | None
    hvg_fit_scope: Literal["independent_cell_context", "upstream_frozen_and_audited"]
    forced_non_hvg_policy: Literal["known_candidate_targets_only"]


class BenchmarkConditionPolicy(StrictRegistryModel):
    """Shared condition universe required by every benchmarked model."""

    policy_id: Literal["gears_default_graph_intersection_v1"]
    official_repository: Literal["https://github.com/snap-stanford/GEARS.git"]
    official_commit: Literal["f374e43e197b295016d80395d7a54ddb81cc6769"]
    gene2go_resource_sha256: Literal[
        "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771"
    ]
    essential_gene_resource_sha256: Literal[
        "46c3dfe354d8ad5c0da22c69f3d0ca451987b1a61ed9d984279b22b9565ff8d7"
    ]
    excluded_conditions: list[str]

    @model_validator(mode="after")
    def enforce_exclusions(self) -> BenchmarkConditionPolicy:
        if self.excluded_conditions != sorted(set(self.excluded_conditions)):
            raise ValueError("benchmark condition exclusions must be unique and sorted")
        if not self.excluded_conditions:
            raise ValueError("benchmark condition policy must record observed exclusions")
        if any(
            not condition
            or condition == "ctrl"
            or not [part for part in condition.split("+") if part != "ctrl"]
            for condition in self.excluded_conditions
        ):
            raise ValueError("benchmark condition exclusions contain an invalid perturbation")
        return self


class DatasetRegistryEntry(StrictRegistryModel):
    schema_version: Literal["dataset-registry-v2"]
    dataset_id: DatasetId
    protocol_id: Literal["within_cell_unseen_single", "norman_combo_seen2"]
    cell_context: Literal["K562", "RPE1", "Jurkat", "HepG2"]
    perturbation_modality: Literal["CRISPRi", "CRISPRa"]
    source: DatasetSource
    source_metadata: SourceMetadataMapping
    canonical_metadata: CanonicalMetadataSchema
    preprocessing: PreprocessingSpec
    benchmark_condition_policy: BenchmarkConditionPolicy
    control_condition_id: Literal["ctrl"]
    split_policy: Literal["grouped_0.5625_0.1875_0.25", "gears_predefined_combo_seen2"]
    split_seed: Literal[42]

    @model_validator(mode="after")
    def enforce_protocol(self) -> DatasetRegistryEntry:
        is_norman = self.dataset_id == "norman"
        if is_norman != (self.protocol_id == "norman_combo_seen2"):
            raise ValueError("Norman protocol identity mismatch")
        if is_norman != (self.split_policy == "gears_predefined_combo_seen2"):
            raise ValueError("Norman split policy identity mismatch")
        if (
            self.dataset_id != "replogle_k562_essential"
            and "K562_cross_cell_lines" in self.source.url
        ):
            raise ValueError("cross-cell cache cannot source an independent within-cell dataset")
        if self.source_metadata.canonical_cell_type_value.casefold() != (
            self.cell_context.casefold()
        ):
            raise ValueError("canonical cell type must match registry context ignoring case")
        return self
