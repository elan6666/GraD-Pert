"""Preregistered paired ABBA timing statistics; not an identity validator."""

from __future__ import annotations

import math
from statistics import median

from scripts.performance.profile_native_a0 import _percentiles


def summarize_abba(arms: list[dict]) -> dict:
    """Consume A1/B1/B2/A2 after receipt and resource validation elsewhere."""
    if len(arms) != 4:
        raise ValueError("requires exactly A1/B1/B2/A2")
    for arm in arms:
        values = arm["wall_ms"]
        if len(values) != 20 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("requires twenty finite positive measured timings")
        for key in ("peak_allocated", "peak_reserved"):
            if not math.isfinite(arm[key]) or arm[key] <= 0:
                raise ValueError("missing or invalid peak memory")
    pairs = []
    for a, b in ((arms[0], arms[1]), (arms[3], arms[2])):
        old = median(a["wall_ms"])
        new = median(b["wall_ms"])
        old_percentiles = _percentiles(a["wall_ms"])
        new_percentiles = _percentiles(b["wall_ms"])
        assert old_percentiles is not None and new_percentiles is not None
        pairs.append(
            {
                "ratio": new / old,
                "reduction_ms": old - new,
                "p90_ratio": new_percentiles["p90"] / old_percentiles["p90"],
                "allocated_ratio": b["peak_allocated"] / a["peak_allocated"],
                "reserved_ratio": b["peak_reserved"] / a["peak_reserved"],
            }
        )
    predicates = {
        "median_reduction_at_least_10_percent": median(p["ratio"] for p in pairs) <= 0.9,
        "median_reduction_at_least_100_ms": median(p["reduction_ms"] for p in pairs) >= 100,
        "both_p90_within_5_percent": all(p["p90_ratio"] <= 1.05 for p in pairs),
        "both_allocated_within_5_percent": all(p["allocated_ratio"] <= 1.05 for p in pairs),
        "both_reserved_within_5_percent": all(p["reserved_ratio"] <= 1.05 for p in pairs),
    }
    return {
        "raw_arms": arms,
        "arm_percentiles": [_percentiles(a["wall_ms"]) for a in arms],
        "pairs": pairs,
        "median_paired_ratio": median(p["ratio"] for p in pairs),
        "predicates": predicates,
        "timing_thresholds_passed": all(predicates.values()),
        "scientific_completion": False,
    }
