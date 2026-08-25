from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_json
from gradpert.pilots.graph_axis import ReducedGraphManifest, _rank_selected_hvgs

HASH = "a" * 64


def test_hvg_ranking_uses_descending_normalized_dispersion_with_stable_ties() -> None:
    genes = [f"G{index:03d}" for index in range(501)]
    dispersions = np.arange(501, dtype=np.float64)
    dispersions[10] = dispersions[11]
    adata = SimpleNamespace(
        var=pd.DataFrame(
            {
                "gene_name": genes,
                "highly_variable": [True] * 500 + [False],
                "dispersions_norm": dispersions,
            }
        )
    )
    ranked = _rank_selected_hvgs(adata, expected_count=500)
    assert ranked[:3] == ("G499", "G498", "G497")
    assert ranked.index("G010") < ranked.index("G011")


def _manifest(*, frozen_hash: str | None = None) -> ReducedGraphManifest:
    top500 = [f"G{index:03d}" for index in range(500)]
    graph_hash = sha256_json(top500)
    artifacts = {"go": "b" * 64, "string": "c" * 64}
    return ReducedGraphManifest(
        schema_version="recomputed-top500-graph-v1",
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_data_sha256=HASH,
        split_content_sha256=HASH,
        source_h5ad_sha256=HASH,
        source_registry_sha256=HASH,
        hvg_method="scanpy.pp.highly_variable_genes",
        normalize_total=4000,
        log1p=True,
        requested_hvg_count=500,
        expression_gene_count=5000,
        direct_top500_gene_ids=top500,
        direct_top500_gene_order_sha256=sha256_json(top500),
        frozen_rank_top500_gene_order_sha256=frozen_hash or sha256_json(top500),
        candidate_target_count=3,
        graph_gene_ids=top500,
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=500,
        source_artifact_sha256=artifacts,
        source_pruned_nonself_edge_count={"go": 10, "string": 12},
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifacts}
        ),
        materialization_wall_ms=1.0,
    )


def test_reduced_graph_manifest_seals_direct_and_frozen_top500_identity() -> None:
    manifest = _manifest()
    assert manifest.direct_top500_gene_order_sha256 == (
        manifest.frozen_rank_top500_gene_order_sha256
    )
    with pytest.raises(ValidationError, match="frozen dispersion ranking"):
        _manifest(frozen_hash="d" * 64)


def test_b1_config_is_explicit_graph_only_and_metrics_only() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/pilots/perf_b1_graph_only/gradpert_b2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(path)
    assert config.model.parameters["performance_pilot_variant"].value == ("perf_b1_graph_only")
    assert config.model.parameters["graph_axis_policy"].value == (
        "recomputed_top500_union_candidate_targets"
    )
    assert config.model.parameters["systems_optimizations"].value == "disabled"
    assert config.artifacts.result_mode == "metrics_only"


def test_b2_config_enables_exact_all_seven_systems_contract() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/pilots/perf_b2_systems_only/gradpert_b2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(path)
    parameters = config.model.parameters
    assert parameters["performance_pilot_variant"].value == "perf_b2_systems_only"
    assert parameters["graph_axis_policy"].value == "canonical_full"
    assert parameters["systems_optimizations"].value == ("all_seven_semantics_preserving_v1")
    seven = (
        "systems_merged_hdf5_reads",
        "systems_control_expression_cache",
        "systems_background_prefetch",
        "systems_resident_graph_tensors",
        "systems_validation_expression_cache",
        "systems_buffered_training_logs",
        "systems_single_checkpoint_serialization",
    )
    assert all(parameters[name].value is True for name in seven)
    assert parameters["systems_pin_memory"].value is True
    assert parameters["systems_nonblocking_transfer"].value is True
    assert parameters["systems_prefetch_depth"].value == 2
    assert parameters["systems_log_buffer_steps"].value == 64
    assert config.artifacts.result_mode == "metrics_only"
