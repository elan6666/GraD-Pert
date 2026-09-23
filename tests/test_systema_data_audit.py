"""Small invariants for the data-only Systema audit."""

import numpy as np

from scripts.analysis.systema_data_audit import _centroid_accuracy, _cosine


def test_systematic_shift_alignment_uses_vector_cosine() -> None:
    common = np.array([1.0, 0.0])
    assert _cosine(np.array([2.0, 0.0]), common) == 1.0
    assert _cosine(np.array([-2.0, 0.0]), common) == -1.0
    assert np.isnan(_cosine(np.zeros(2), common))


def test_constant_prediction_centroid_accuracy_is_half_on_average() -> None:
    truth = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    result = _centroid_accuracy(np.zeros(2), truth, max_conditions=3, seed=42)
    assert result["conditions"] == 3
    assert result["mean"] == 0.5


def test_centroid_accuracy_ties_are_not_counted_as_success() -> None:
    truth = np.array([[1.0, 0.0], [-1.0, 0.0]])
    result = _centroid_accuracy(np.zeros(2), truth, max_conditions=2, seed=42)
    assert result["mean"] == 0.0
