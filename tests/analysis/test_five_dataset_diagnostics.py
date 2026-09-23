"""Invariants for the descriptive five-dataset protocol."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/analysis/five_dataset_diagnostics.py")
)


def test_control_single_double_classification() -> None:
    classify = MODULE["perturbation_kind"]
    assert classify("ctrl", True) == 0
    assert classify("STAT1", False) == 1
    assert classify("STAT1+IRF1", False) == 2
    assert classify("STAT1+ctrl", False) == 1
    with pytest.raises(ValueError):
        classify("STAT1", True)


def test_batch_stratified_halves_are_disjoint_balanced_and_repeatable() -> None:
    assign = MODULE["split_half_assignments"]
    conditions = np.asarray(["ctrl", "A", "A", "A", "A", "A", "B", "B", "B"])
    batches = np.asarray(["x", "x", "x", "y", "y", "y", "x", "y", "y"])
    codes, halves, names = assign(conditions, batches, ["A", "B"], repeats=5, seed=19)
    assert names == ["A", "B"]
    assert np.all(halves[:, codes < 0] == -1)
    for repeat in range(5):
        for code in (0, 1):
            rows = np.flatnonzero(codes == code)
            assert set(halves[repeat, rows]) == {0, 1}
            assert (
                abs(int((halves[repeat, rows] == 0).sum()) - int((halves[repeat, rows] == 1).sum()))
                <= 1
            )
    _, again, _ = assign(conditions, batches, ["A", "B"], repeats=5, seed=19)
    np.testing.assert_array_equal(halves, again)


def test_visualization_caps_each_condition() -> None:
    select = MODULE["select_visualization_rows"]
    kinds = np.asarray([0] * 10 + [1] * 20 + [1] * 20 + [2] * 10)
    conditions = np.asarray(["ctrl"] * 10 + ["A"] * 20 + ["B"] * 20 + ["C"] * 10)
    rows = select(kinds, conditions, per_kind=12, per_condition_cap=4, seed=5)
    assert int((kinds[rows] == 0).sum()) == 10
    assert int((kinds[rows] == 1).sum()) == 8
    assert int((kinds[rows] == 2).sum()) == 4
