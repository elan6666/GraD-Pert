"""Scientific invariants for the bounded observed distribution audit."""

import numpy as np
from scipy.spatial.distance import cdist

from scripts.analysis.dataset_distribution_audit import (
    MAX_CONDITIONS_PER_LINE,
    _bh_fdr,
    _energy_distance,
    _select_stratified,
)


def test_energy_distance_detects_shift_and_is_symmetric() -> None:
    values = np.array([[0.0], [0.1], [3.0], [3.1]])
    distance = cdist(values, values)
    left, right = np.array([0, 1]), np.array([2, 3])
    score = _energy_distance(distance, left, right)
    assert score > 5
    assert np.isclose(score, _energy_distance(distance, right, left))


def test_bh_fdr_is_monotone_in_p_value() -> None:
    q = _bh_fdr([0.04, 0.01, 0.9, 0.02])
    assert np.allclose(q, [0.0533333333, 0.04, 0.9, 0.04])


def test_stratified_selection_is_seeded_and_bounded() -> None:
    eligible = [(f"c{i:03d}", "batch", i + 20) for i in range(400)]
    chosen = _select_stratified(eligible, seed=42)
    assert len(chosen) == MAX_CONDITIONS_PER_LINE
    assert chosen == _select_stratified(eligible, seed=42)
    assert len({condition for condition, _, _ in chosen}) == MAX_CONDITIONS_PER_LINE
    counts = np.array([count for _, _, count in chosen])
    assert np.sum(counts < 120) == 30
    assert np.sum((counts >= 120) & (counts < 220)) == 30
    assert np.sum((counts >= 220) & (counts < 320)) == 30
    assert np.sum(counts >= 320) == 30
