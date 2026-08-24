from __future__ import annotations

import numpy as np
import pytest

from gradpert.baselines import (
    GeneralTrainDeltaBaseline,
    GlobalTrainDeltaBaseline,
    MatchedControlBaseline,
    fit_training_deltas,
    parse_condition_components,
)


def _fit():  # type: ignore[no-untyped-def]
    return fit_training_deltas(
        train_perturbed_expression=np.asarray(
            [
                [3.0, 2.0],
                [5.0, 4.0],
                [11.0, 13.0],
                [8.0, 9.0],
            ]
        ),
        train_condition_ids=["A+ctrl", "A+ctrl", "B+ctrl", "A+B"],
        train_context_ids=["K562"] * 4,
        train_batch_ids=["b1", "b1", "b2", "b1"],
        train_control_expression=np.asarray([[1.0, 1.0], [1.0, 1.0], [10.0, 10.0]]),
        control_context_ids=["K562", "K562", "K562"],
        control_batch_ids=["b1", "b1", "b2"],
    )


def test_condition_parser_is_order_stable_and_control_aware() -> None:
    assert parse_condition_components("B+A+ctrl") == ("A", "B")
    assert parse_condition_components("A+ctrl") == ("A",)
    with pytest.raises(ValueError, match="control"):
        parse_condition_components("ctrl")


def test_matched_control_returns_the_exact_300_rows_without_aliasing() -> None:
    controls = np.arange(600, dtype=float).reshape(300, 2)
    prediction = MatchedControlBaseline.predict(controls)
    np.testing.assert_array_equal(prediction, controls)
    assert prediction is not controls


def test_fitted_deltas_use_training_batch_controls_only() -> None:
    fitted = _fit()
    np.testing.assert_allclose(fitted.exact(("A",)), [3.0, 2.0])
    np.testing.assert_allclose(fitted.exact(("B",)), [1.0, 3.0])
    np.testing.assert_allclose(fitted.exact(("A", "B")), [7.0, 8.0])
    np.testing.assert_allclose(fitted.global_single(), [7.0 / 3, 7.0 / 3])


def test_global_baseline_adds_one_global_single_delta_per_component() -> None:
    baseline = GlobalTrainDeltaBaseline(_fit())
    controls = np.zeros((300, 2))
    single = baseline.predict(condition_id="UNSEEN+ctrl", input_controls=controls)
    double = baseline.predict(condition_id="UNSEEN+OTHER", input_controls=controls)
    np.testing.assert_allclose(single[0], [7.0 / 3, 7.0 / 3])
    np.testing.assert_allclose(double[0], [14.0 / 3, 14.0 / 3])
    assert single.shape == double.shape == (300, 2)


def test_general_baseline_uses_exact_seen_then_additive_component_fallback() -> None:
    baseline = GeneralTrainDeltaBaseline(_fit())
    controls = np.zeros((300, 2))
    seen_double = baseline.predict(condition_id="B+A", input_controls=controls)
    unseen_double = baseline.predict(condition_id="A+C", input_controls=controls)
    np.testing.assert_allclose(seen_double[0], [7.0, 8.0])
    np.testing.assert_allclose(unseen_double[0], [16.0 / 3, 13.0 / 3])


def test_fit_fails_when_a_perturbed_batch_has_no_training_control() -> None:
    with pytest.raises(ValueError, match="no training control mean"):
        fit_training_deltas(
            train_perturbed_expression=np.ones((1, 2)),
            train_condition_ids=["A+ctrl"],
            train_context_ids=["K562"],
            train_batch_ids=["missing"],
            train_control_expression=np.ones((1, 2)),
            control_context_ids=["K562"],
            control_batch_ids=["present"],
        )
