"""Condition-mean error and TxPert-style retrieval on frozen canonical rows."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def centered_unit_rows(values):
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(norm).all():
        raise ValueError("retrieval bank contains a constant or nonfinite vector")
    return centered / norm


class FrozenConditionResponse:
    """Build the test-only bank from means of cell-level batch-centered truth."""

    def __init__(self, data, *, columns=slice(None)):
        conditions = tuple(data.split.test_conditions)
        self.conditions = tuple(sorted(conditions))
        self.rank_keys = {
            condition: "+".join(p for p in condition.split("+") if p != "ctrl")
            for condition in self.conditions
        }
        if len(set(self.rank_keys.values())) != len(conditions):
            raise ValueError("TxPert retrieval keys are ambiguous")
        self.positions = {condition: idx for idx, condition in enumerate(self.conditions)}
        genes = np.asarray(data.expression_gene_ids)[columns]
        self.axis = {gene: index for index, gene in enumerate(genes)}
        control_ids = defaultdict(list)
        self.context_by_row = dict(zip(data.row_ids, data.context_ids, strict=True))
        for row_id, condition, context in zip(
            data.row_ids, data.condition_ids, data.context_ids, strict=True
        ):
            if condition == data.split.control_condition_id:
                control_ids[context].append(row_id)
        baselines = {
            context: data.load_control_rows(tuple(ids))
            .expression[:, columns]
            .mean(0, dtype=np.float64)
            for context, ids in control_ids.items()
        }
        bank = []
        for condition in self.conditions:
            truth = data.load_truth_rows(condition)
            contexts = [self.context_by_row[row_id] for row_id in truth.ordered_row_ids]
            weights = {context: contexts.count(context) for context in sorted(set(contexts))}
            mean_baseline = sum(baselines[k] * n for k, n in weights.items()) / len(contexts)
            bank.append(truth.expression[:, columns].mean(0, dtype=np.float64) - mean_baseline)
        self.bank = centered_unit_rows(np.asarray(bank))

    def __call__(self, condition, prediction, controls, truth):
        pred_mean = prediction.mean(0, dtype=np.float64)
        truth_mean = truth.expression.mean(0, dtype=np.float64)
        error = pred_mean - truth_mean
        delta = pred_mean - controls.mean(0, dtype=np.float64)
        target_ids = [self.axis[g] for g in condition.split("+") if g in self.axis and g != "ctrl"]
        centered = delta - delta.mean()
        norm = np.linalg.norm(centered)
        rank = None
        if norm > 0 and np.isfinite(norm):
            similarities = self.bank @ (centered / norm)
            score = similarities[self.positions[condition]]
            # Official RetrievalMetric sorts ascending (similarity, perturbation).
            key = self.rank_keys[condition]
            rank = sum(
                sim < score or (sim == score and self.rank_keys[c] < key)
                for sim, c in zip(similarities, self.conditions, strict=True)
            ) / len(self.conditions)
        return {
            "condition_mean_mse": float(np.mean(error**2)),
            "direct_target_mean_mse": float(np.mean(error[target_ids] ** 2))
            if target_ids
            else None,
            "txpert_normalized_retrieval": rank,
        }


def summarize_response(rows):
    result = {}
    for key in ("condition_mean_mse", "direct_target_mean_mse", "txpert_normalized_retrieval"):
        values = [
            row["response_metrics"][key] for row in rows if row["response_metrics"][key] is not None
        ]
        result[key] = {
            "macro_mean": float(np.mean(values)) if values else None,
            "valid_conditions": len(values),
            "total_conditions": len(rows),
        }
    return result
