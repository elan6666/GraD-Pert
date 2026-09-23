"""Invariants for the observed-only perturbation atlas."""

import numpy as np

from scripts.analysis.dataset_effect_atlas import (
    _crosscell_summary,
    _effective_rank,
    _json_safe,
    _norman_interactions,
    _top_gene_agreement,
)


def test_top_gene_agreement_requires_shared_signed_markers() -> None:
    left = np.array([5.0, -4.0, 0.1, 0.2])
    right = np.array([6.0, -3.0, -0.1, 0.1])
    jaccard, sign = _top_gene_agreement(left, right, k=2)
    assert jaccard == 1.0
    assert sign == 1.0


def test_effective_rank_distinguishes_collinear_and_orthogonal_effects() -> None:
    line = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    square = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    assert np.isclose(_effective_rank(line, seed=1), 1.0)
    assert np.isclose(_effective_rank(square, seed=1), 2.0)


def test_norman_fitted_interaction_recovers_linear_combination() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    observed = 2 * a + 3 * b
    result = _norman_interactions({"A+ctrl": a, "B+ctrl": b, "A+B": observed}, {})
    assert result["fitted_double_conditions"] == 1
    assert np.isclose(result["coefficient_a_median"], 2.0)
    assert np.isclose(result["coefficient_b_median"], 3.0)
    assert np.isclose(result["fitted_relative_residual_median"], 0.0)


def test_crosscell_pair_counts_and_nan_are_json_safe() -> None:
    deltas = {
        line: {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
        for line in ("K562", "RPE1", "jurkat", "hepg2")
    }
    result = _crosscell_summary(deltas)
    assert result["K562<-RPE1"]["shared_conditions"] == 2
    assert np.isclose(result["K562<-RPE1"]["median_pearson_delta"], 1.0)
    assert _json_safe({"missing": float("nan")}) == {"missing": None}
