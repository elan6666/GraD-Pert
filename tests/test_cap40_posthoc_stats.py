import importlib.util
from pathlib import Path

import numpy as np
import pytest

P = Path(__file__).parents[1] / "scripts/v2/cap40_posthoc_stats.py"
spec = importlib.util.spec_from_file_location("posthoc_stats", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_streaming_centered_covariance_matches_large_offset_reference():
    x = 1e8 + np.random.default_rng(1).normal(size=(53, 7))
    a, b = m.Moments(7), m.Moments(7)
    a.add(x)
    for block in np.array_split(x, 9):
        b.add(block)
    np.testing.assert_allclose(a.m2, b.m2, atol=1e-6)
    assert a.result()["effective_rank"] == pytest.approx(b.result()["effective_rank"], abs=1e-7)


def test_rank_zero_and_singleton_are_distinct():
    a = m.Moments(3)
    a.add(np.ones((1, 3)))
    assert a.result()["effective_rank"] is None
    a.add(np.ones((4, 3)))
    assert a.result()["effective_rank"] == 0


def test_probability_regimes_and_chunk_invariance():
    uniform, shared, different = (m.Distribution(4) for _ in range(3))
    uniform.add(np.zeros((4, 4)))
    shared.add(np.tile([100, 0, 0, 0], (4, 1)))
    for row in np.eye(4) * 100:
        different.add(row[None])
    assert uniform.result()["H_sample"] == pytest.approx(np.log(4))
    assert uniform.result()["JS"] == pytest.approx(0)
    assert shared.result()["JS"] == pytest.approx(0)
    assert different.result()["JS"] == pytest.approx(np.log(4))
    x = np.random.default_rng(4).normal(size=(51, 17))
    a, b = m.Distribution(17), m.Distribution(17)
    a.add(x)
    for block in np.array_split(x, 8):
        b.add(block)
    for k in ["H_sample", "H_marginal", "JS", "N_eff"]:
        assert a.result()[k] == pytest.approx(b.result()[k])


def test_group_bootstrap_is_condition_equal_not_row_equal():
    r = [{"condition_id": "a", "v": 0}] * 9 + [{"condition_id": "b", "v": 10}]
    result = m.condition_summary(r, "v")
    assert result["condition_mean"] == 5
    assert result["row_mean"] == 1
    assert result["bootstrap_unit"] == "perturbation_condition"


def test_query_panels_cover_allowed_only_and_have_common_anchor():
    allowed = [i for i in range(5000) if i % 17 != 0]
    panels = m.query_panels(allowed, 1000, 42)
    assert all(len(set(p)) == 1000 for p in panels)
    assert set.union(*(set(p) for p in panels)) == set(allowed)
    assert len(set.intersection(*(set(p) for p in panels))) >= 256
    assert panels == m.query_panels(allowed, 1000, 42)


def test_balanced_sampling_preserves_contexts_and_uniqueness():
    records = [{"truth_row_id": str(i), "context_id": "a" if i < 9 else "b"} for i in range(10)]
    s = m.select_balanced(records, 4, 42)
    assert len({r["truth_row_id"] for r in s}) == 4
    assert {r["context_id"] for r in s} == {"a", "b"}


def test_nonfinite_is_rejected():
    with pytest.raises(ValueError):
        m.Moments(2).add(np.array([[1, np.nan]]))
    with pytest.raises(ValueError):
        m.Distribution(2).add(np.array([[1, np.inf]]))


def test_ibot_cell_means_differ_from_token_mean_and_omit_empty_cells():
    values = np.array([[1.0], [1.0], [1.0], [9.0]])
    out = m.cell_balanced_terms(values, np.array([0, 0, 0, 1]), 3)
    assert len(out) == 2  # third cell has no mask, never a zero denominator item
    assert out.mean() == 5
    assert values.mean() == 3
    with pytest.raises(ValueError):
        m.cell_balanced_terms(values, np.array([0, 0, 1, 3]), 3)
