"""Five-dataset registry and deterministic fairness services."""

from gradpert.data.acquisition import (
    SourceFileStatus,
    checksum_file,
    download_source,
    inspect_source_file,
    require_downloadable,
    safe_extract_zip,
)
from gradpert.data.controls import build_evaluation_control_manifest, stable_draw_seed
from gradpert.data.preparation import (
    DatasetLayout,
    DatasetPreparationResult,
    dataset_status,
    prepare_dataset,
    refresh_dataset_protocol,
    verify_prepared_dataset,
)
from gradpert.data.preprocessing import (
    MetadataCanonicalizationReport,
    PerturbationFilterReport,
    UpstreamProcessedPreprocessingReport,
    WithinCellPreprocessingReport,
    canonicalize_metadata,
    filter_cells_by_perturbation_effect,
    preprocess_norman,
    preprocess_raw_within_cell,
    preprocess_upstream_within_cell,
)
from gradpert.data.registry import load_dataset_registry, verify_dataset_registry
from gradpert.data.split import (
    apply_benchmark_condition_policy,
    build_grouped_split_manifest,
    build_norman_combo_seen2_split_manifest,
)

__all__ = [
    "DatasetLayout",
    "DatasetPreparationResult",
    "MetadataCanonicalizationReport",
    "PerturbationFilterReport",
    "SourceFileStatus",
    "UpstreamProcessedPreprocessingReport",
    "WithinCellPreprocessingReport",
    "apply_benchmark_condition_policy",
    "build_evaluation_control_manifest",
    "build_grouped_split_manifest",
    "build_norman_combo_seen2_split_manifest",
    "canonicalize_metadata",
    "checksum_file",
    "dataset_status",
    "download_source",
    "filter_cells_by_perturbation_effect",
    "inspect_source_file",
    "load_dataset_registry",
    "prepare_dataset",
    "preprocess_norman",
    "preprocess_raw_within_cell",
    "preprocess_upstream_within_cell",
    "refresh_dataset_protocol",
    "require_downloadable",
    "safe_extract_zip",
    "stable_draw_seed",
    "verify_dataset_registry",
    "verify_prepared_dataset",
]
