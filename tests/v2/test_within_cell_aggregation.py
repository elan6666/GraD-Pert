"""Condition means preserve native per-cell rank ceilings and missing data."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
from aggregate_within_cell import aggregate


def test_mean_of_normalized_cell_ranks_not_ratio_of_pooled_means(tmp_path):
    path = tmp_path / "cells.csv"
    rows = [
        dict(
            condition_id="p",
            context_id="c",
            feature="response",
            effective_rank=2,
            rank_ceiling=4,
            relative_variance="NA",
        ),
        dict(
            condition_id="p",
            context_id="c",
            feature="response",
            effective_rank=6,
            rank_ceiling=8,
            relative_variance="NA",
        ),
        dict(
            condition_id="q",
            context_id="c",
            feature="response",
            effective_rank=1,
            rank_ceiling=8,
            relative_variance=0.1,
        ),
    ]
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seal, result = aggregate(path, "medium")
    assert result[0]["cells"] == 2 and result[0]["effective_rank_mean"] == 4
    assert result[0]["normalized_rank_mean"] == 0.625
    assert result[0]["relative_variance_mean"] is None
    assert result[1]["cells"] == 1 and result[1]["normalized_rank_mean"] == 0.125
    assert seal["bytes"] == path.stat().st_size


def test_panel_keeps_only_gene_axis_and_context_separate(tmp_path):
    path = tmp_path / "panel.csv"
    columns = ["statistical_axis", "condition_id", "context_id", "effective_rank", "rank_ceiling"]
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(
            [
                ["genes_within_cell", "p", "c1", 2, 4],
                ["cells_within_gene", "p", "c1", 99, 100],
                ["genes_within_cell", "p", "c2", 3, 4],
            ]
        )
    _, result = aggregate(path, "panel")
    assert len(result) == 2 and all(r["cells"] == 1 for r in result)
    assert [r["effective_rank_mean"] for r in result] == [2, 3]
    with pytest.raises(ValueError, match="unknown"):
        aggregate(path, "mixed")
