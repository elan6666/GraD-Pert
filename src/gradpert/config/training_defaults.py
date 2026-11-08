"""Explicit defaults for newly materialized training configs, never legacy loading."""

from __future__ import annotations

import copy
from typing import Any


def with_training_throughput_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Copy a native relay training recipe and record the authorized execution preset."""
    result = copy.deepcopy(config)
    if result.get("model_id") != "gradpert_v2":
        raise ValueError("throughput defaults apply only to native v2 training")
    parameters = result["model"]["parameters"]
    if parameters.get("attention", {}).get("value") != "relay_full":
        raise ValueError("throughput defaults require relay attention")
    for name in ("relay_validate_once", "cache_kda_constants"):
        parameters[name] = {
            "value": True,
            "source": "user_locked",
            "reference": "docs/experiments/TRAINING_EXECUTION_DEFAULTS.md",
        }
    return result
