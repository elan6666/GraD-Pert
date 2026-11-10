import numpy as np
import pytest

from gradpert.evaluation.response_diagnostics import (
    normalized_bank,
    recover_top_effects,
    response_rank,
    summarize_condition,
)


def test_no_response_rank_is_exact_chance_and_zero_truth_is_excluded():
    bank, valid = normalized_bank(np.array([[0, 0], [1, 0], [0, 1], [-1, 0]]))
    assert response_rank(np.array([0, 0]), bank, valid, 1) == pytest.approx(
        {"rank": 2, "reciprocal_rank": (1 + 1 / 2 + 1 / 3) / 3, "top1": 1 / 3, "top5": 1}
    )
    assert response_rank(np.array([1, 0]), bank, valid, 1)["rank"] == 1
    assert response_rank(np.array([1, 0]), bank, valid, 3)["rank"] == 3
    assert all(v is None for v in response_rank(np.array([1, 0]), bank, valid, 0).values())


def test_top_effect_recovery_ties_and_zero_predictions_are_explicit():
    truth = np.array([0, 3, -3, 1])
    eligible = np.array([1, 2, 3])
    recovered = recover_top_effects(np.array([0, -5, -5, 0]), truth, eligible, 1)
    assert recovered == {
        "truth_top_count": 1,
        "predicted_top_count": 1,
        "recall": 1,
        "precision": 1,
        "direction_accuracy": 0,
    }
    no_effect = recover_top_effects(np.zeros(4), truth, eligible, 2)
    assert no_effect["recall"] == no_effect["direction_accuracy"] == 0
    assert no_effect["precision"] is None


def test_amplitude_and_direction_are_separate_and_downstream_excludes_target():
    truth = np.array([100.0, 4.0, 2.0])
    pred = np.array([100.0, -4.0, 2.0])
    control = np.zeros(3)
    bank, valid = normalized_bank([truth, -truth])
    result = summarize_condition(
        prediction=pred,
        truth=truth,
        paired_control=control,
        pool_control=control,
        train_mean=control,
        target_indices=(0,),
        position=0,
        bank=bank,
        valid_positions=valid,
    )
    assert result["effect_rms_ratio"] == 1
    assert result["target_direction_accuracy"] == 1
    assert result["top_effects"]["20"]["model"]["direction_accuracy"] == 0.5
    assert result["top_effects"]["20"]["model"]["truth_top_count"] == 2
    assert result["mse_gain_over_copy"] > 0
