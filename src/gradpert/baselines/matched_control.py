"""Matched-control population floor."""

from __future__ import annotations

from typing import Any

import numpy as np


class MatchedControlBaseline:
    """Return the exact evaluator-selected control population unchanged."""

    @staticmethod
    def predict(input_controls: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        controls = np.asarray(input_controls)
        if controls.ndim != 2 or controls.shape[0] != 300:
            raise ValueError("matched-control prediction requires [300, genes] input")
        if not np.isfinite(controls).all():
            raise ValueError("matched-control input contains non-finite values")
        return controls.copy()
