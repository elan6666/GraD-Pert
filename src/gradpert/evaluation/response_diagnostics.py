"""Descriptive diagnostics for frozen unseen-perturbation validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

SCHEMA = "frozen-response-diagnostics-v1"
TOP_K = (20, 50)
MODELS = ("model", "control_copy", "train_mean")


def centroid(cells: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return cells.mean(axis=0, dtype=np.float64)


def rms(vector: np.ndarray[Any, Any]) -> float:
    return float(np.sqrt(np.mean(np.square(vector))))


def normalized_bank(
    responses: Sequence[np.ndarray[Any, Any]],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    bank = np.asarray(responses, dtype=np.float64)
    norms = np.linalg.norm(bank, axis=1)
    valid = norms > 0
    return bank[valid] / norms[valid, None], np.flatnonzero(valid)


def response_rank(
    query: np.ndarray[Any, Any],
    bank: np.ndarray[Any, Any],
    valid_positions: np.ndarray[Any, Any],
    target: int,
) -> dict[str, float | None]:
    """Expected retrieval scores under a uniform ordering of cosine ties."""
    identity = np.flatnonzero(valid_positions == target)
    if not len(identity):
        return dict.fromkeys(("rank", "reciprocal_rank", "top1", "top5"))
    norm = np.linalg.norm(query)
    scores = bank @ (query / norm) if norm else np.zeros(len(bank))
    correct = scores[identity[0]]
    ties = np.isclose(scores, correct, atol=1e-12, rtol=0)
    above = int(np.count_nonzero((scores > correct) & ~ties))
    tied = int(ties.sum())
    first, last = above + 1, above + tied
    return {
        "rank": (first + last) / 2,
        "reciprocal_rank": float(np.mean(1 / np.arange(first, last + 1))),
        "top1": max(0, min(1, last) - above) / tied,
        "top5": max(0, min(5, last) - above) / tied,
    }


def recover_top_effects(
    predicted: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    eligible: np.ndarray[Any, Any],
    k: int,
) -> dict[str, float | int | None]:
    """Proxy recovery of large effects, with stable ties and no target gene."""
    true_candidates = eligible[truth[eligible] != 0]
    pred_candidates = eligible[predicted[eligible] != 0]
    true_top = true_candidates[np.argsort(-np.abs(truth[true_candidates]), kind="stable")[:k]]
    pred_top = pred_candidates[np.argsort(-np.abs(predicted[pred_candidates]), kind="stable")[:k]]
    overlap = np.intersect1d(true_top, pred_top)
    return {
        "truth_top_count": len(true_top),
        "predicted_top_count": len(pred_top),
        "recall": len(overlap) / len(true_top) if len(true_top) else None,
        "precision": len(overlap) / len(pred_top) if len(pred_top) else None,
        "direction_accuracy": (
            float(np.mean(np.sign(predicted[true_top]) == np.sign(truth[true_top])))
            if len(true_top)
            else None
        ),
    }


def effect_geometry(predicted: np.ndarray[Any, Any], truth: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Exact radial/angular decomposition of squared response error in expression units."""
    true_rms, pred_rms = rms(truth), rms(predicted)
    product = true_rms * pred_rms
    cosine = float(np.clip(np.mean(predicted * truth) / product, -1, 1)) if product else None
    return {
        "truth_effect_rms": true_rms,
        "pred_effect_rms": pred_rms,
        "effect_rms_ratio": pred_rms / true_rms if true_rms else None,
        "effect_cosine": cosine,
        "amplitude_mse": (pred_rms - true_rms) ** 2,
        "direction_mse": 2 * product * (1 - cosine) if cosine is not None else 0.0,
    }


def summarize_condition(
    *,
    prediction: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    paired_control: np.ndarray[Any, Any],
    pool_control: np.ndarray[Any, Any],
    train_mean: np.ndarray[Any, Any],
    target_indices: tuple[int, ...],
    position: int,
    bank: np.ndarray[Any, Any],
    valid_positions: np.ndarray[Any, Any],
    groups: dict[str, tuple[int, ...]] | None = None,
    downstream_bank: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    | None = None,
) -> dict[str, Any]:
    effects = truth - paired_control
    eligible = np.delete(
        np.arange(len(effects)), [i for i in target_indices if 0 <= i < len(effects)]
    )
    predicted_effect = prediction - paired_control
    means = dict(zip(MODELS, (prediction, paired_control, train_mean), strict=True))
    mse = {name: rms(mean - truth) ** 2 for name, mean in means.items()}
    targets = sorted(set(target_indices).intersection(range(len(effects))))
    return {
        **effect_geometry(predicted_effect, effects),
        "effect_groups": {
            name: effect_geometry(predicted_effect[np.asarray(ix)], effects[np.asarray(ix)])
            for name, ix in (groups or {}).items()
            if ix
        },
        **(
            {
                "retrieval_without_val_targets": {
                    name: response_rank(
                        (mean - pool_control)[downstream_bank[0]],
                        downstream_bank[1],
                        downstream_bank[2],
                        position,
                    )
                    for name, mean in means.items()
                }
            }
            if downstream_bank is not None
            else {}
        ),
        "target_direction_accuracy": (
            float(
                np.mean(
                    np.sign(effects[targets]) == np.sign((prediction - paired_control)[targets])
                )
            )
            if targets
            else None
        ),
        "mse": mse,
        "mse_gain_over_copy": mse["control_copy"] - mse["model"],
        "mse_gain_over_train_mean": mse["train_mean"] - mse["model"],
        "retrieval": {
            name: response_rank(mean - pool_control, bank, valid_positions, position)
            for name, mean in means.items()
        },
        "top_effects": {
            str(k): {
                name: recover_top_effects(mean - paired_control, effects, eligible, k)
                for name, mean in means.items()
            }
            for k in TOP_K
        },
    }
