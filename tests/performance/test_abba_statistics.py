from copy import deepcopy

import pytest

from scripts.performance.abba_statistics import summarize_abba


def arms():
    return [
        {"wall_ms": [v] * 20, "peak_allocated": 1000, "peak_reserved": 2000}
        for v in (1000, 800, 880, 1100)
    ]


def test_paired_ratio_and_raw_values():
    values = arms()
    result = summarize_abba(values)
    assert result["median_paired_ratio"] == pytest.approx(0.8)
    assert result["timing_thresholds_passed"]
    assert result["raw_arms"] == values
    assert set(result["arm_percentiles"][0]) == {"p50", "p90", "p95", "p99"}


@pytest.mark.parametrize("key", ["peak_allocated", "peak_reserved"])
def test_either_pair_memory_regression_rejects(key):
    values = arms()
    values[2][key] *= 1.051
    assert not summarize_abba(values)["timing_thresholds_passed"]


def test_tail_regression_not_hidden_by_median():
    values = arms()
    values[1]["wall_ms"][-4:] = [1200] * 4
    assert not summarize_abba(values)["timing_thresholds_passed"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_invalid_timing_rejected(bad):
    values = deepcopy(arms())
    values[0]["wall_ms"][0] = bad
    with pytest.raises(ValueError):
        summarize_abba(values)


def test_incomplete_replicate_rejected():
    values = arms()
    values[0]["wall_ms"].pop()
    with pytest.raises(ValueError):
        summarize_abba(values)


def test_relative_gain_alone_insufficient():
    values = arms()
    for arm in values:
        arm["wall_ms"] = [v / 10 for v in arm["wall_ms"]]
    assert not summarize_abba(values)["timing_thresholds_passed"]
