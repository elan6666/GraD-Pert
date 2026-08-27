"""Auditable within-cell preprocessing primitives for canonical AnnData views."""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from gradpert.data.schema import DatasetRegistryEntry


@dataclass(frozen=True)
class MetadataCanonicalizationReport:
    n_observations: int
    n_genes: int
    n_conditions: int
    n_controls: int
    source_condition_column: str
    source_batch_column: str
    source_gene_symbol_column: str
    canonical_cell_type_value: str
    condition_transform: str


@dataclass(frozen=True)
class PerturbationFilterConditionReport:
    source_condition_id: str
    target_gene_present: bool
    cells_before: int
    cells_after: int


@dataclass(frozen=True)
class PerturbationFilterReport:
    percentile: int
    control_identifier: str
    n_cells_before: int
    n_cells_after: int
    conditions: tuple[PerturbationFilterConditionReport, ...]


@dataclass(frozen=True)
class WithinCellPreprocessingReport:
    dataset_id: str
    input_expression_state: str
    expression_scale_action: str
    n_cells_input: int
    n_cells_filtered: int
    n_genes_input: int
    n_hvg: int
    n_forced_candidate_targets: int
    n_expression_genes: int
    n_graph_genes: int
    missing_candidate_targets: tuple[str, ...]
    metadata: MetadataCanonicalizationReport
    perturbation_filter: PerturbationFilterReport


@dataclass(frozen=True)
class UpstreamProcessedPreprocessingReport:
    dataset_id: str
    input_expression_state: str
    n_cells_input: int
    n_cells_output: int
    n_expression_genes: int
    n_graph_genes: int
    n_source_conditions: int
    n_canonical_conditions: int
    n_condition_aliases_collapsed: int
    expression_scale_action: str
    forced_candidate_targets: tuple[str, ...]
    missing_candidate_targets: tuple[str, ...]
    metadata: MetadataCanonicalizationReport


def _require_verified_mapping(entry: DatasetRegistryEntry) -> None:
    if entry.source_metadata.audit_state != "verified_from_frozen_reference":
        raise RuntimeError(f"{entry.dataset_id} source metadata has not passed audit")


def _required_source_fields(entry: DatasetRegistryEntry) -> tuple[str, str, str]:
    _require_verified_mapping(entry)
    mapping = entry.source_metadata
    fields = (
        mapping.condition_column,
        mapping.control_identifier,
        mapping.condition_transform,
    )
    if any(value is None for value in fields):
        raise RuntimeError("verified source mapping is incomplete")
    return cast(tuple[str, str, str], fields)


def _source_batches(adata: Any, entry: DatasetRegistryEntry) -> tuple[np.ndarray[Any, Any], str]:
    mapping = entry.source_metadata
    if mapping.batch_column is not None:
        _require_columns(adata.obs, (mapping.batch_column,), "adata.obs")
        return _string_values(adata.obs[mapping.batch_column], mapping.batch_column), (
            mapping.batch_column
        )
    if mapping.constant_batch_value is not None:
        return (
            np.repeat(mapping.constant_batch_value, int(adata.n_obs)).astype(str),
            f"<constant:{mapping.constant_batch_value}>",
        )
    raise RuntimeError("verified source batch mapping is incomplete")


def _normalize_perturbation_condition(value: str, control_identifier: str) -> str:
    if value == control_identifier:
        return "ctrl"
    components = value.split("+")
    if any(not component for component in components):
        raise ValueError(f"condition contains an empty perturbation component: {value}")
    active = [component for component in components if component != control_identifier]
    if not active or len(active) > 2 or len(active) != len(set(active)):
        raise ValueError(f"condition has invalid active perturbation components: {value}")
    ordered = sorted(active)
    if len(ordered) == 1:
        return f"{ordered[0]}+ctrl"
    return "+".join(ordered)


def _require_columns(frame: Any, columns: tuple[str, ...], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing frozen source columns: {missing}")


def _string_values(series: Any, field: str) -> np.ndarray[Any, np.dtype[np.str_]]:
    if bool(series.isna().any()):
        raise ValueError(f"{field} contains null values")
    return np.asarray(series.astype(str).to_numpy(), dtype=str)


def _source_gene_names(adata: Any, entry: DatasetRegistryEntry) -> tuple[np.ndarray[Any, Any], str]:
    mapping = entry.source_metadata
    if mapping.gene_symbol_location == "var_index":
        values = np.asarray(adata.var.index.astype(str).to_numpy(), dtype=str)
        source = "var.index"
    elif mapping.gene_symbol_location == "var_column" and mapping.gene_symbol_column is not None:
        _require_columns(adata.var, (mapping.gene_symbol_column,), "adata.var")
        values = _string_values(adata.var[mapping.gene_symbol_column], mapping.gene_symbol_column)
        source = mapping.gene_symbol_column
    else:
        raise RuntimeError("verified source gene-symbol mapping is incomplete")

    if mapping.gene_symbol_duplicate_policy == "suffix_later_with_var_index":
        var_ids = np.asarray(adata.var.index.astype(str).to_numpy(), dtype=str)
        seen: set[str] = set()
        disambiguated: list[str] = []
        for symbol, var_id in zip(values.tolist(), var_ids.tolist(), strict=True):
            if symbol in seen:
                disambiguated.append(f"{symbol}__{var_id}")
            else:
                disambiguated.append(symbol)
                seen.add(symbol)
        values = np.asarray(disambiguated, dtype=str)
        source = f"{source}[suffix_later_with_var_index]"
    if len(values) != len(set(values.tolist())):
        raise ValueError("source gene symbols remain non-unique after frozen duplicate policy")
    return values, source


def canonicalize_metadata(
    adata: Any,
    entry: DatasetRegistryEntry,
) -> tuple[Any, MetadataCanonicalizationReport]:
    """Map observed upstream fields into the one model-independent schema."""

    condition_column, control_identifier, transform = _required_source_fields(entry)
    result = adata.copy()
    _require_columns(result.obs, (condition_column,), "adata.obs")
    if not bool(result.obs.index.is_unique):
        raise ValueError("observation row IDs must be unique before canonicalization")

    source_conditions = _string_values(result.obs[condition_column], condition_column)
    source_batches, source_batch_description = _source_batches(result, entry)
    control_mask = source_conditions == control_identifier
    if not bool(control_mask.any()):
        raise ValueError(f"source control identifier is absent: {control_identifier}")

    if transform == "append_ctrl_suffix_then_collapse_control":
        canonical_conditions = np.char.add(source_conditions, "+ctrl")
        canonical_conditions[control_mask] = entry.control_condition_id
    elif transform == "identity":
        canonical_conditions = source_conditions.copy()
        canonical_conditions[control_mask] = entry.control_condition_id
    elif transform == "normalize_perturbation_components":
        canonical_conditions = np.asarray(
            [
                _normalize_perturbation_condition(value, control_identifier)
                for value in source_conditions
            ],
            dtype=str,
        )
    else:  # pragma: no cover - closed by the registry schema
        raise ValueError(f"unsupported condition transform: {transform}")

    source_cell_type_column = entry.source_metadata.cell_type_column
    expected_cell_type = entry.source_metadata.canonical_cell_type_value
    if source_cell_type_column is not None:
        _require_columns(result.obs, (source_cell_type_column,), "adata.obs")
        observed_cell_types = set(
            _string_values(result.obs[source_cell_type_column], source_cell_type_column)
        )
        expected_observed = set(entry.source_metadata.observed_cell_type_values or ())
        if observed_cell_types != expected_observed:
            raise ValueError(
                "source cell type does not match frozen observed values: "
                f"observed={sorted(observed_cell_types)}, expected={sorted(expected_observed)}"
            )

    source_gene_names, gene_source = _source_gene_names(result, entry)
    if len(source_gene_names) != len(set(source_gene_names.tolist())):
        raise ValueError("source gene symbols must be unique")

    canonical = entry.canonical_metadata
    result.obs[canonical.condition_column] = canonical_conditions
    result.obs[canonical.batch_column] = source_batches
    result.obs[canonical.cell_type_column] = expected_cell_type
    result.obs[canonical.control_column] = control_mask.astype(np.int8)
    result.obs[canonical.condition_name_column] = [
        f"{expected_cell_type}_{condition}_1+1" for condition in canonical_conditions
    ]
    result.var[canonical.gene_symbol_column] = source_gene_names

    report = MetadataCanonicalizationReport(
        n_observations=int(result.n_obs),
        n_genes=int(result.n_vars),
        n_conditions=len(set(canonical_conditions.tolist())),
        n_controls=int(control_mask.sum()),
        source_condition_column=condition_column,
        source_batch_column=source_batch_description,
        source_gene_symbol_column=gene_source,
        canonical_cell_type_value=expected_cell_type,
        condition_transform=transform,
    )
    return result, report


def _to_dense(matrix: Any) -> np.ndarray[Any, Any]:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _matrix_values(matrix: Any) -> np.ndarray[Any, Any]:
    values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix).reshape(-1)
    return np.asarray(values)


def _require_finite_nonnegative(matrix: Any, *, label: str) -> None:
    values = _matrix_values(matrix)
    if values.size and (not bool(np.isfinite(values).all()) or float(values.min()) < 0):
        raise ValueError(f"{label} must contain only finite nonnegative values")


def _require_raw_integer_counts(matrix: Any, *, label: str) -> None:
    """Fail closed before applying the raw-count-only TxPert preprocessing path."""

    _require_finite_nonnegative(matrix, label=label)
    values = _matrix_values(matrix).reshape(-1)
    chunk_size = 1_000_000
    for start in range(0, int(values.size), chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        if not bool(np.all(np.abs(chunk - np.rint(chunk)) <= 1e-6)):
            raise ValueError(
                f"{label} must be raw integer counts; refusing to normalize/log1p "
                "a previously transformed matrix"
            )


def _require_scanpy() -> Any:
    try:
        sc = importlib.import_module("scanpy")
    except ImportError as error:  # pragma: no cover - exercised in the server data environment
        raise RuntimeError("scanpy is required for formal dataset preprocessing") from error
    return sc


def _candidate_targets(canonical_conditions: np.ndarray[Any, Any]) -> set[str]:
    targets: set[str] = set()
    for condition in canonical_conditions.tolist():
        if condition == "ctrl":
            continue
        targets.update(component for component in condition.split("+") if component != "ctrl")
    return targets


def _append_graph_only_targets(
    adata: Any,
    entry: DatasetRegistryEntry,
    targets: set[str],
) -> tuple[Any, tuple[str, ...]]:
    gene_column = entry.canonical_metadata.gene_symbol_column
    genes = _string_values(adata.var[gene_column], gene_column)
    if len(genes) != len(set(genes.tolist())):
        raise ValueError("canonical gene symbols must be unique")
    missing_targets = tuple(sorted(targets - set(genes.tolist())))
    if not missing_targets:
        return adata, ()

    try:
        ad = importlib.import_module("anndata")
        pd = importlib.import_module("pandas")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as error:  # pragma: no cover - server data environment
        raise RuntimeError(
            "anndata, pandas and scipy are required to append graph-only targets"
        ) from error
    dtype = adata.X.dtype
    if sparse.issparse(adata.X):
        extra_x = sparse.csr_matrix((int(adata.n_obs), len(missing_targets)), dtype=dtype)
    else:
        extra_x = np.zeros((int(adata.n_obs), len(missing_targets)), dtype=dtype)
    extra_var = pd.DataFrame(
        {
            gene_column: list(missing_targets),
            "highly_variable": [False] * len(missing_targets),
            "forced_candidate_target": [True] * len(missing_targets),
            "expression_output_gene": [False] * len(missing_targets),
        },
        index=[f"graph_only::{target}" for target in missing_targets],
    )
    graph_only = ad.AnnData(
        X=extra_x,
        obs=adata.obs.copy(),
        var=extra_var,
    )
    combined = ad.concat(
        [adata, graph_only],
        axis=1,
        join="outer",
        merge="same",
    )
    combined.uns = deepcopy(adata.uns)
    combined_genes = _string_values(combined.var[gene_column], gene_column)
    if combined_genes[: len(genes)].tolist() != genes.tolist():
        raise ValueError("appending graph-only targets changed the expression gene order")
    if set(combined_genes.tolist()) != set(genes.tolist()) | targets:
        raise ValueError("graph-only target append did not produce the frozen union")
    return combined, missing_targets


def _mark_full_upstream_axes(
    adata: Any,
    entry: DatasetRegistryEntry,
) -> tuple[Any, tuple[str, ...]]:
    targets = _candidate_targets(
        _string_values(adata.obs[entry.canonical_metadata.condition_column], "condition")
    )
    adata.var["highly_variable"] = np.ones(int(adata.n_vars), dtype=bool)
    adata.var["forced_candidate_target"] = np.zeros(int(adata.n_vars), dtype=bool)
    adata.var["expression_output_gene"] = np.ones(int(adata.n_vars), dtype=bool)
    return _append_graph_only_targets(adata, entry, targets)


def preprocess_upstream_within_cell(
    adata: Any,
    entry: DatasetRegistryEntry,
) -> tuple[Any, UpstreamProcessedPreprocessingReport]:
    """Audit and canonicalize the frozen upstream 5,000-gene within-cell artifact."""

    if entry.dataset_id != "replogle_k562_essential":
        raise ValueError("upstream within-cell preprocessing is frozen to Replogle K562")
    if entry.source.semantics != "upstream_processed_archive":
        raise ValueError("upstream within-cell input must come from the frozen archive")
    if (
        entry.preprocessing.input_expression_state != "verified_upstream_log1p"
        or entry.preprocessing.expression_scale_action != "preserve_verified_upstream"
        or entry.preprocessing.normalize_total is not None
        or entry.preprocessing.log1p
    ):
        raise ValueError("upstream K562 must preserve its frozen processed expression scale")
    if int(adata.n_vars) != 5000:
        raise ValueError("frozen upstream within-cell artifact must contain exactly 5,000 genes")
    _require_finite_nonnegative(adata.X, label="upstream within-cell expression")
    source_condition_count = int(adata.obs[entry.source_metadata.condition_column].nunique())
    canonical, metadata = canonicalize_metadata(adata, entry)
    canonical, forced_targets = _mark_full_upstream_axes(canonical, entry)
    canonical.uns["gradpert_preprocessing"] = {
        "profile_id": entry.preprocessing.profile_id,
        "input_expression_state": entry.preprocessing.input_expression_state,
        "expression_scale_action": "preserved_verified_upstream_log1p_hvg5000",
        "expression_axis": "all_frozen_upstream_hvg5000",
        "graph_axis": "expression_hvg5000_then_forced_candidate_targets",
        "forced_candidate_targets": list(forced_targets),
        "metadata": asdict(metadata),
    }
    report = UpstreamProcessedPreprocessingReport(
        dataset_id=entry.dataset_id,
        input_expression_state=entry.preprocessing.input_expression_state,
        n_cells_input=int(adata.n_obs),
        n_cells_output=int(canonical.n_obs),
        n_expression_genes=5000,
        n_graph_genes=int(canonical.n_vars),
        n_source_conditions=source_condition_count,
        n_canonical_conditions=metadata.n_conditions,
        n_condition_aliases_collapsed=source_condition_count - metadata.n_conditions,
        expression_scale_action="preserved_verified_upstream_log1p_hvg5000",
        forced_candidate_targets=forced_targets,
        missing_candidate_targets=(),
        metadata=metadata,
    )
    return canonical, report


def preprocess_norman(
    adata: Any,
    entry: DatasetRegistryEntry,
) -> tuple[Any, UpstreamProcessedPreprocessingReport]:
    """Canonicalize metadata while preserving the frozen GEARS-processed matrix."""

    if entry.dataset_id != "norman" or entry.preprocessing.profile_id != (
        "gears_norman_audited_v1"
    ):
        raise ValueError("Norman preprocessing requires the frozen Norman registry entry")
    if entry.source.semantics != "upstream_processed_archive":
        raise ValueError("Norman input must come from the frozen processed archive")
    if (
        entry.preprocessing.input_expression_state != "verified_upstream_log1p"
        or entry.preprocessing.expression_scale_action != "preserve_verified_upstream"
        or entry.preprocessing.normalize_total is not None
        or entry.preprocessing.log1p
    ):
        raise ValueError("Norman must preserve its verified upstream expression scale")
    if int(adata.n_vars) != 5045:
        raise ValueError("frozen upstream Norman artifact must contain exactly 5,045 genes")
    _require_finite_nonnegative(adata.X, label="upstream Norman log expression")
    source_condition_count = int(adata.obs[entry.source_metadata.condition_column].nunique())
    canonical, metadata = canonicalize_metadata(adata, entry)
    _require_finite_nonnegative(canonical.X, label="canonical Norman expression")
    expression_gene_count = int(canonical.n_vars)
    canonical, forced_targets = _mark_full_upstream_axes(canonical, entry)
    for stale_key in (
        "rank_genes_groups_cov_all",
        "top_non_dropout_de_20",
        "top_non_zero_de_20",
        "non_dropout_gene_idx",
        "non_zeros_gene_idx",
    ):
        canonical.uns.pop(stale_key, None)
    canonical.uns["gradpert_preprocessing"] = {
        "profile_id": entry.preprocessing.profile_id,
        "input_expression_state": entry.preprocessing.input_expression_state,
        "expression_scale_action": "preserved_verified_gears_processed_log_expression",
        "condition_alias_policy": "sort_active_components_and_place_ctrl_last",
        "upstream_cell_type_audit": entry.source_metadata.observed_cell_type_values,
        "canonical_cell_type": entry.source_metadata.canonical_cell_type_value,
        "forced_candidate_targets": list(forced_targets),
        "metadata": asdict(metadata),
    }
    report = UpstreamProcessedPreprocessingReport(
        dataset_id=entry.dataset_id,
        input_expression_state=entry.preprocessing.input_expression_state,
        n_cells_input=int(adata.n_obs),
        n_cells_output=int(canonical.n_obs),
        n_expression_genes=expression_gene_count,
        n_graph_genes=int(canonical.n_vars),
        n_source_conditions=source_condition_count,
        n_canonical_conditions=metadata.n_conditions,
        n_condition_aliases_collapsed=source_condition_count - metadata.n_conditions,
        expression_scale_action="preserved_verified_gears_processed_log_expression",
        forced_candidate_targets=forced_targets,
        missing_candidate_targets=(),
        metadata=metadata,
    )
    return canonical, report


def filter_cells_by_perturbation_effect(
    adata: Any,
    entry: DatasetRegistryEntry,
    *,
    percentile: int = 10,
) -> tuple[Any, PerturbationFilterReport]:
    """Apply the frozen target-expression lower-tail filter on raw target IDs."""

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    condition_column, control_identifier, _ = _required_source_fields(entry)
    _require_columns(adata.obs, (condition_column,), "adata.obs")
    conditions = _string_values(adata.obs[condition_column], condition_column)
    genes, _ = _source_gene_names(adata, entry)
    if len(genes) != len(set(genes.tolist())):
        raise ValueError("source gene symbols must be unique")
    gene_locations = {gene: index for index, gene in enumerate(genes.tolist())}

    control_mask = conditions == control_identifier
    if not bool(control_mask.any()):
        raise ValueError(f"source control identifier is absent: {control_identifier}")
    control_expression = _to_dense(adata.X[control_mask, :])
    thresholds = np.percentile(control_expression, percentile, axis=0)

    kept_positions: list[int] = []
    reports: list[PerturbationFilterConditionReport] = []
    for condition in dict.fromkeys(conditions.tolist()):
        positions = np.flatnonzero(conditions == condition)
        gene_location = gene_locations.get(condition)
        if condition == control_identifier or gene_location is None:
            selected = positions
        else:
            expression = _to_dense(adata.X[positions, gene_location]).reshape(-1)
            selected = positions[expression <= thresholds[gene_location]]
        kept_positions.extend(int(position) for position in selected)
        reports.append(
            PerturbationFilterConditionReport(
                source_condition_id=condition,
                target_gene_present=gene_location is not None,
                cells_before=len(positions),
                cells_after=len(selected),
            )
        )

    filtered = adata[kept_positions, :].copy()
    report = PerturbationFilterReport(
        percentile=percentile,
        control_identifier=control_identifier,
        n_cells_before=int(adata.n_obs),
        n_cells_after=int(filtered.n_obs),
        conditions=tuple(reports),
    )
    return filtered, report


def preprocess_raw_within_cell(
    adata: Any,
    entry: DatasetRegistryEntry,
    *,
    known_candidate_targets: set[str],
) -> tuple[Any, WithinCellPreprocessingReport]:
    """Filter, normalize/log, independently select HVGs, and retain candidates."""

    if entry.source.semantics != "raw_single_cell":
        raise ValueError("raw within-cell preprocessing requires raw_single_cell source semantics")
    if entry.preprocessing.profile_id != "txpert_within_cell_v1":
        raise ValueError("raw within-cell preprocessing requires txpert_within_cell_v1")
    if entry.preprocessing.normalize_total != 4000 or not entry.preprocessing.log1p:
        raise ValueError("frozen within-cell normalization must be normalize_total(4000)+log1p")
    if entry.preprocessing.hvg_count != 5000:
        raise ValueError("frozen within-cell HVG count must be 5000")
    if (
        entry.preprocessing.input_expression_state != "raw_integer_counts"
        or entry.preprocessing.expression_scale_action != "normalize_total_4000_then_log1p"
    ):
        raise ValueError("raw within-cell registry must declare the raw-count transform")

    _require_raw_integer_counts(adata.X, label="raw within-cell expression")

    filtered, filter_report = filter_cells_by_perturbation_effect(adata, entry)
    canonical, metadata_report = canonicalize_metadata(filtered, entry)
    sc = _require_scanpy()

    sc.pp.normalize_total(canonical, target_sum=4000)
    sc.pp.log1p(canonical)
    sc.pp.highly_variable_genes(canonical, n_top_genes=5000, subset=False)

    gene_column = entry.canonical_metadata.gene_symbol_column
    gene_names = _string_values(canonical.var[gene_column], gene_column)
    candidate_mask = np.isin(gene_names, sorted(known_candidate_targets))
    hvg_mask = canonical.var["highly_variable"].to_numpy(dtype=bool)
    canonical.var["forced_candidate_target"] = candidate_mask & ~hvg_mask
    canonical.var["expression_output_gene"] = hvg_mask
    expression_positions = np.flatnonzero(hvg_mask)
    forced_positions = np.flatnonzero(candidate_mask & ~hvg_mask)
    graph_positions = np.concatenate((expression_positions, forced_positions))
    prepared = canonical[:, graph_positions].copy()
    prepared, graph_only_targets = _append_graph_only_targets(
        prepared,
        entry,
        known_candidate_targets,
    )
    prepared.uns["gradpert_preprocessing"] = {
        "profile_id": entry.preprocessing.profile_id,
        "input_expression_state": entry.preprocessing.input_expression_state,
        "input_expression_audit": "finite_nonnegative_integer_counts_full_matrix",
        "expression_scale_action": entry.preprocessing.expression_scale_action,
        "filter_before_condition_encoding": True,
        "normalize_total": 4000,
        "log1p": True,
        "hvg_count": 5000,
        "forced_non_hvg_policy": "known_candidate_targets_only",
        "expression_axis": "highly_variable_true_only",
        "graph_axis": "highly_variable_or_known_candidate_target",
        "graph_only_candidate_targets": list(graph_only_targets),
        "metadata": asdict(metadata_report),
    }

    report = WithinCellPreprocessingReport(
        dataset_id=entry.dataset_id,
        input_expression_state=entry.preprocessing.input_expression_state,
        expression_scale_action=entry.preprocessing.expression_scale_action,
        n_cells_input=int(adata.n_obs),
        n_cells_filtered=int(prepared.n_obs),
        n_genes_input=int(adata.n_vars),
        n_hvg=int(hvg_mask.sum()),
        n_forced_candidate_targets=int((candidate_mask & ~hvg_mask).sum())
        + len(graph_only_targets),
        n_expression_genes=int(hvg_mask.sum()),
        n_graph_genes=int(prepared.n_vars),
        missing_candidate_targets=(),
        metadata=metadata_report,
        perturbation_filter=filter_report,
    )
    return prepared, report
