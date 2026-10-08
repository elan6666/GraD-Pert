import numpy as np
import pytest

from scripts.analysis.split_half_significance import (
    blocked_null,
    control_pools,
    correlation_rows,
    draw_matched_controls,
    holm_adjust,
)


def test_joint_null_is_invariant_to_duplicated_resplits():
    block = {"similarity": np.eye(4)[None], "weights": np.full(4, 0.25)}
    once = blocked_null([block], permutations=199, seed=2)
    block["similarity"] = np.repeat(block["similarity"], 20, axis=0)
    twenty = blocked_null([block], permutations=199, seed=2)
    assert once == twenty


def test_null_never_exchanges_identities_across_batch_blocks():
    blocks = [
        {"similarity": np.full((2, 3, 3), value), "weights": np.full(3, 1 / 6)}
        for value in [0.2, 0.8]
    ]
    result = blocked_null(blocks, permutations=99, seed=1)
    assert result["observed"] == pytest.approx(0.5)
    assert result["p"] == 1
    assert result["excess"] == pytest.approx(0)
    assert np.allclose(result["null"], 0.5)


def test_control_draws_disjoint_and_exactly_match_batch_counts():
    batch = np.array(["a"] * 8 + ["b"] * 6)
    pools = control_pools(batch, 42)
    a, b = draw_matched_controls(pools, {"a": 180, "b": 120}, np.random.default_rng(4))
    assert len(a) == len(b) == 300
    assert np.intersect1d(a, b).size == 0
    for rows in [a, b]:
        assert np.count_nonzero(batch[rows] == "a") == 180
        assert np.count_nonzero(batch[rows] == "b") == 120
    pools = control_pools(np.array(["a"]), 42)
    assert draw_matched_controls(pools, {"a": 300}, np.random.default_rng(4)) is None


def test_shared_control_noise_can_raise_correlation_without_identity_signal():
    rng = np.random.default_rng(42)
    genes = 10000
    left, right = rng.normal(size=(2, genes))
    ca, cb = rng.normal(scale=3, size=(2, genes))
    shared = correlation_rows(left - ca, right - ca)
    separate = correlation_rows(left - ca, right - cb)
    assert shared > 0.85
    assert abs(separate) < 0.05


def test_holm_adjust_original_order_and_arbitrary_dependence_family():
    assert np.allclose(holm_adjust([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])
    with pytest.raises(ValueError):
        holm_adjust([-0.1])
