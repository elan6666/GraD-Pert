"""Training-only global and condition-aware additive delta baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


def parse_condition_components(condition_id: str) -> tuple[str, ...]:
    if not condition_id:
        raise ValueError("condition ID must be non-empty")
    components = tuple(sorted(part for part in condition_id.split("+") if part != "ctrl"))
    if not components:
        raise ValueError("control is not a perturbation condition")
    if len(components) > 2:
        raise ValueError("v1 nonlearned baselines support at most two perturbation components")
    if len(components) != len(set(components)):
        raise ValueError("condition contains a duplicate perturbation component")
    return components


@dataclass(frozen=True)
class WeightedDelta:
    mean: np.ndarray[Any, Any]
    sample_count: int


@dataclass(frozen=True)
class FittedDeltaRegistry:
    gene_count: int
    exact_by_context: dict[str, dict[tuple[str, ...], WeightedDelta]]
    global_single_by_context: dict[str, WeightedDelta]
    training_condition_ids: tuple[str, ...]

    def _weighted_across_contexts(
        self,
        candidates: Sequence[WeightedDelta],
    ) -> np.ndarray[Any, Any]:
        if not candidates:
            raise ValueError("no training delta is available for the requested condition")
        total = sum(candidate.sample_count for candidate in candidates)
        if total <= 0:
            raise ValueError("training delta has invalid sample count")
        numerator = sum(
            (candidate.mean * candidate.sample_count for candidate in candidates),
            start=np.zeros(self.gene_count, dtype=np.float64),
        )
        return cast(np.ndarray[Any, Any], numerator / total)

    def global_single(self) -> np.ndarray[Any, Any]:
        return self._weighted_across_contexts(list(self.global_single_by_context.values()))

    def exact(self, components: tuple[str, ...]) -> np.ndarray[Any, Any] | None:
        candidates = [
            by_condition[components]
            for by_condition in self.exact_by_context.values()
            if components in by_condition
        ]
        return self._weighted_across_contexts(candidates) if candidates else None


def _validate_parallel_rows(
    expression: np.ndarray[Any, Any],
    *metadata: Sequence[str],
) -> np.ndarray[Any, Any]:
    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("expression must be a non-empty rank-two matrix")
    if any(len(items) != values.shape[0] for items in metadata):
        raise ValueError("metadata row count differs from expression")
    if not np.isfinite(values).all():
        raise ValueError("expression contains non-finite values")
    return values


def fit_training_deltas(
    *,
    train_perturbed_expression: np.ndarray[Any, Any],
    train_condition_ids: Sequence[str],
    train_context_ids: Sequence[str],
    train_batch_ids: Sequence[str],
    train_control_expression: np.ndarray[Any, Any],
    control_context_ids: Sequence[str],
    control_batch_ids: Sequence[str],
) -> FittedDeltaRegistry:
    """Fit batch-centered deltas without accepting validation/test truth."""

    perturbed = _validate_parallel_rows(
        train_perturbed_expression,
        train_condition_ids,
        train_context_ids,
        train_batch_ids,
    )
    controls = _validate_parallel_rows(
        train_control_expression,
        control_context_ids,
        control_batch_ids,
    )
    if perturbed.shape[1] != controls.shape[1]:
        raise ValueError("perturbed/control gene dimensions differ")

    control_groups: dict[tuple[str, str], list[np.ndarray[Any, Any]]] = defaultdict(list)
    for row, context_id, batch_id in zip(
        controls,
        control_context_ids,
        control_batch_ids,
        strict=True,
    ):
        control_groups[(context_id, batch_id)].append(row)
    control_means = {
        key: np.stack(rows, axis=0).mean(axis=0) for key, rows in control_groups.items()
    }

    delta_rows: dict[tuple[str, tuple[str, ...]], list[np.ndarray[Any, Any]]] = defaultdict(list)
    for row, condition_id, context_id, batch_id in zip(
        perturbed,
        train_condition_ids,
        train_context_ids,
        train_batch_ids,
        strict=True,
    ):
        key = (context_id, batch_id)
        if key not in control_means:
            raise ValueError(f"no training control mean for context/batch {key}")
        components = parse_condition_components(condition_id)
        delta_rows[(context_id, components)].append(row - control_means[key])

    exact_by_context: dict[str, dict[tuple[str, ...], WeightedDelta]] = defaultdict(dict)
    global_single_rows: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    for (context_id, components), rows in delta_rows.items():
        stacked = np.stack(rows, axis=0)
        exact_by_context[context_id][components] = WeightedDelta(
            mean=stacked.mean(axis=0),
            sample_count=stacked.shape[0],
        )
        if len(components) == 1:
            global_single_rows[context_id].extend(rows)
    if not global_single_rows:
        raise ValueError("at least one training single perturbation is required")
    global_single_by_context = {
        context_id: WeightedDelta(
            mean=np.stack(rows, axis=0).mean(axis=0),
            sample_count=len(rows),
        )
        for context_id, rows in global_single_rows.items()
    }
    return FittedDeltaRegistry(
        gene_count=perturbed.shape[1],
        exact_by_context=dict(exact_by_context),
        global_single_by_context=global_single_by_context,
        training_condition_ids=tuple(sorted(set(train_condition_ids))),
    )


def _validate_prediction_controls(
    input_controls: np.ndarray[Any, Any],
    gene_count: int,
) -> np.ndarray[Any, Any]:
    controls = np.asarray(input_controls, dtype=np.float64)
    if controls.shape != (300, gene_count):
        raise ValueError(f"prediction controls must have shape (300, {gene_count})")
    if not np.isfinite(controls).all():
        raise ValueError("prediction controls contain non-finite values")
    return controls


class GlobalTrainDeltaBaseline:
    def __init__(self, fitted: FittedDeltaRegistry) -> None:
        self.fitted = fitted

    def predict(
        self,
        *,
        condition_id: str,
        input_controls: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        components = parse_condition_components(condition_id)
        controls = _validate_prediction_controls(input_controls, self.fitted.gene_count)
        delta = self.fitted.global_single() * len(components)
        return cast(np.ndarray[Any, Any], controls + delta)


class GeneralTrainDeltaBaseline:
    def __init__(self, fitted: FittedDeltaRegistry) -> None:
        self.fitted = fitted

    def predict(
        self,
        *,
        condition_id: str,
        input_controls: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        components = parse_condition_components(condition_id)
        controls = _validate_prediction_controls(input_controls, self.fitted.gene_count)
        exact = self.fitted.exact(components)
        if exact is not None:
            return cast(np.ndarray[Any, Any], controls + exact)
        component_deltas = [self.fitted.exact((component,)) for component in components]
        global_delta = self.fitted.global_single()
        resolved = [delta if delta is not None else global_delta for delta in component_deltas]
        return cast(
            np.ndarray[Any, Any],
            controls + sum(resolved, start=np.zeros(self.fitted.gene_count)),
        )
