from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.execution.native import _read_local_view_realization_receipt
from gradpert.graphs import resolve_local_view_contract


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
