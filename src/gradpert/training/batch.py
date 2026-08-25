"""Typed cell-level batch consumed by the native B2 step engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class GraDPertTrainingBatch:
    control_expression: Tensor
    target_expression: Tensor
    condition_ids: tuple[str, ...]
    anchors_by_condition: Mapping[str, tuple[int, ...]]
    perturbed_row_ids: tuple[str, ...]
    control_row_ids: tuple[str, ...]
    data_read_ms: float = 0.0
    host_to_device_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.control_expression.ndim != 2:
            raise ValueError("control expression must be rank two")
        if self.control_expression.shape != self.target_expression.shape:
            raise ValueError("control and target expression shapes differ")
        rows = self.control_expression.shape[0]
        if not rows:
            raise ValueError("training batch must contain cells")
        for values, field in (
            (self.condition_ids, "condition_ids"),
            (self.perturbed_row_ids, "perturbed_row_ids"),
            (self.control_row_ids, "control_row_ids"),
        ):
            if len(values) != rows or any(not value for value in values):
                raise ValueError(f"{field} must contain one non-empty value per row")
        unique_conditions = set(self.condition_ids)
        if set(self.anchors_by_condition) != unique_conditions:
            raise ValueError("active-anchor mapping must equal the batch's unique conditions")
        for condition_id, anchors in self.anchors_by_condition.items():
            if not condition_id or not anchors or len(anchors) != len(set(anchors)):
                raise ValueError("each condition requires unique active anchors")
            if any(anchor < 0 for anchor in anchors):
                raise ValueError("active anchor IDs must be nonnegative")
        if self.data_read_ms < 0 or self.host_to_device_ms < 0:
            raise ValueError("batch stage timings must be nonnegative")
