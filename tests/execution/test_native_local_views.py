from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.execution.native import (
    _ordered_perturbation_target_gene_ids,
    _read_local_view_realization_receipt,
    _text_prior_receipt,
)
from gradpert.features import TextPriorArtifact
from gradpert.graphs import resolve_local_view_contract
from gradpert.hashing import sha256_json


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "global_step": 0,
            "unique_condition_count": 2,
            "local_view_realization_count": 8,
            "local_node_count_sum": 28,
            "local_node_count_min": 3,
            "local_node_count_max": 4,
            "local_budget_hit_count": 4,
            "local_node_counts_sha256": "a" * 64,
            "masked_local_assignment_count": 4,
            "masked_local_index_counts_json": "[1,1,1,1]",
            "masked_local_assignments_sha256": "b" * 64,
        },
        {
            "global_step": 1,
            "unique_condition_count": 1,
            "local_view_realization_count": 4,
            "local_node_count_sum": 12,
            "local_node_count_min": 2,
            "local_node_count_max": 4,
            "local_budget_hit_count": 2,
            "local_node_counts_sha256": "c" * 64,
            "masked_local_assignment_count": 2,
            "masked_local_index_counts_json": "[0,1,0,1]",
            "masked_local_assignments_sha256": "d" * 64,
        },
    ]


def _contract():  # type: ignore[no-untyped-def]
    return resolve_local_view_contract(
        graph_node_count=8,
        local_view_count=4,
        node_budget_ratio=(1, 2),
        mask_view_ratio=(1, 2),
    )


def test_local_view_realization_receipt_reduces_compact_step_evidence(tmp_path: Path) -> None:
    steps = tmp_path / "train_steps.csv"
    _write_rows(steps, _rows())

    receipt = _read_local_view_realization_receipt(steps, contract=_contract())

    assert receipt["training_step_count"] == 2
    assert receipt["realized_local_view_count"] == 12
    assert receipt["node_count"] == {"min": 2, "mean": 40 / 12, "max": 4, "sum": 40}
    assert receipt["graph_coverage"] == {
        "min": 0.25,
        "mean": (40 / 12) / 8,
        "max": 0.5,
    }
    assert receipt["budget_hit_count"] == 6
    assert receipt["masked_local_assignment_count"] == 6
    assert receipt["masked_local_assignment_counts_by_index"] == [1, 2, 1, 2]
    assert len(str(receipt["ordered_step_evidence_sha256"])) == 64


def test_local_view_realization_receipt_rejects_budget_overrun(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["local_node_count_max"] = 5
    steps = tmp_path / "train_steps.csv"
    _write_rows(steps, rows)

    with pytest.raises(ValueError, match="realized local-view count or budget"):
        _read_local_view_realization_receipt(steps, contract=_contract())


def test_genept_seed_receipt_binds_superset_selection_without_large_extra_ids(
    tmp_path: Path,
) -> None:
    selected = np.zeros((2, 3), dtype=np.float32)
    artifact = TextPriorArtifact(
        source_path=tmp_path / "seed-go-protein-pathway.npz",
        source_sha256="a" * 64,
        source_size_bytes=123,
        gene_ids=("A", "B"),
        values=selected,
        model="doubao-embedding-vision",
        embedding_width=3,
        gene_order_sha256="b" * 64,
        zero_vector_gene_ids=(),
        source_gene_count=4,
        source_gene_order_sha256="c" * 64,
        extra_source_gene_count=2,
        extra_source_gene_ids=("EXTRA_A", "EXTRA_B"),
        extra_source_gene_ids_sha256="d" * 64,
        perturbation_target_gene_ids=("A",),
        perturbation_target_gene_ids_sha256="e" * 64,
        selected_matrix_sha256="f" * 64,
        requested_runtime_gene_ids=("A", "B", "MISSING"),
        requested_runtime_gene_order_sha256=sha256_json(["A", "B", "MISSING"]),
        ignored_missing_non_perturbation_gene_ids=("MISSING",),
        ignored_missing_non_perturbation_gene_ids_sha256=sha256_json(["MISSING"]),
    )

    receipt = _text_prior_receipt(artifact, feature_mode="genept_frozen")

    assert receipt["schema_version"] == "sealed-superset-text-prior-v2"
    assert receipt["source_gene_count"] == 4
    assert receipt["requested_runtime_gene_count"] == 3
    assert receipt["selected_gene_count"] == 2
    assert receipt["extra_source_gene_count"] == 2
    assert receipt["extra_source_gene_ids_sha256"] == "d" * 64
    assert "extra_source_gene_ids" not in receipt
    assert receipt["ignored_missing_non_perturbation_gene_count"] == 1
    assert receipt["ignored_missing_non_perturbation_gene_ids_sha256"] == sha256_json(["MISSING"])
    assert receipt["missing_non_perturbation_gene_policy"] == ("omit_preserving_canonical_order")
    assert "missing_runtime_gene_policy" not in receipt
    assert receipt["zero_fill_policy"] == "forbidden"


def test_genept_target_union_uses_all_sealed_split_partitions() -> None:
    training_data = SimpleNamespace(
        split=SimpleNamespace(
            train_conditions=("A+ctrl", "B"),
            val_conditions=("C+A",),
            test_conditions=("D+ctrl",),
            control_condition_id="ctrl",
        )
    )

    assert _ordered_perturbation_target_gene_ids(training_data) == ("A", "B", "C", "D")
