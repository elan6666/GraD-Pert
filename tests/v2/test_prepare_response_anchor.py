"""Synthetic contracts; no canonical data or server access."""

import numpy as np
import pytest

from scripts.v2.prepare_response_anchor import build_bank, projection, train_boundary


def fixture():
    entries = [
        dict(
            condition_id=f"g{p}",
            rows=list(range(p * 5, p * 5 + 5)),
            A=list(range(p * 5, p * 5 + 2)),
            B=list(range(p * 5 + 2, p * 5 + 5)),
        )
        for p in range(100)
    ]
    metadata = dict(
        row_ids=[str(i) for i in range(508)],
        context_ids=["k"] * 508,
        train_row_indices=list(range(500)),
        control_condition_id="ctrl",
    )
    plan = dict(
        splits=dict(selected_train=dict(eligible=entries)),
        controls=dict(k=dict(TA=[500, 501, 502, 503], TB=[504, 505, 506, 507])),
    )
    cache = np.random.default_rng(4).normal(size=(508, 300)).astype(np.float32)
    # Shared strong condition response makes fixed A/B projected S positive.
    for p in range(100):
        cache[p * 5 : p * 5 + 5] += np.random.default_rng(p).normal(size=300).astype(np.float32) * 4
    return entries, metadata, plan, cache


def test_deterministic_projection_and_normalization():
    x = np.random.default_rng(3).normal(size=(100, 300))
    a, q, _, rms, error, h = projection(x)
    b = projection(x)[0]
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and np.isclose(np.mean(a.astype(float) ** 2), 1)
    np.testing.assert_allclose(q.T @ q, np.eye(256) * 300 / 256, atol=1e-12)
    assert rms > 0 and error <= 0.25 and len(h) == 64


def test_zero_rms_and_nonfinite_fail():
    for x in (np.ones((100, 300)), np.full((100, 300), np.nan)):
        with pytest.raises(ValueError):
            projection(x)


def test_odd_n_context_slots_and_six_arrays():
    entries, metadata, plan, cache = fixture()
    rows = train_boundary(plan, metadata)[1]
    bank, _, diag = build_bank(
        cache,
        rows,
        entries,
        plan,
        metadata,
        np.arange(300, dtype=np.int64),
        [f"g{i}" for i in range(300)],
        [f"g{i}" for i in range(300)],
    )
    assert len(bank) == 6 and bank["control_slot_offsets"][-1] == 500
    assert np.all(np.diff(bank["control_slot_offsets"]) == 5)
    assert bank["control_expression"].shape == (
        8,
        300,
    )
    assert diag["raw_ownmask_projected_cross_half_S"] > 0


def test_forbidden_axis_and_numeric_rows_rejected():
    entries, metadata, plan, cache = fixture()
    rows = train_boundary(plan, metadata)[1]
    with pytest.raises(ValueError, match="axis"):
        build_bank(cache, rows, entries, plan, metadata, np.zeros(300, dtype=np.int64), [], [])
    with pytest.raises(ValueError, match="boundary"):
        build_bank(
            cache,
            rows[:-1],
            entries,
            plan,
            metadata,
            np.arange(300, dtype=np.int64),
            [f"g{i}" for i in range(300)],
            [f"g{i}" for i in range(300)],
        )
    metadata["train_row_indices"] = [0]
    with pytest.raises(ValueError, match="TRAIN"):
        train_boundary(plan, metadata)


def test_context_weighted_union_mean(monkeypatch):
    from scripts.v2 import prepare_response_anchor as module

    entries, metadata, plan, cache = fixture()
    # First odd group has context proportions 2/5 and 3/5; control union means differ.
    metadata["context_ids"][2:5] = ["l"] * 3
    metadata["context_ids"][504:508] = ["l"] * 4
    plan["controls"] = dict(
        k=dict(TA=[500, 501], TB=[502, 503]), l=dict(TA=[504, 505], TB=[506, 507])
    )
    rows = train_boundary(plan, metadata)[1]
    captured = {}
    original = module.projection

    def capture(delta):
        captured["delta"] = delta.copy()
        return original(delta)

    monkeypatch.setattr(module, "projection", capture)
    build_bank(
        cache,
        rows,
        entries,
        plan,
        metadata,
        np.arange(300, dtype=np.int64),
        [f"g{i}" for i in range(300)],
        [f"g{i}" for i in range(300)],
    )
    expected = (
        cache[:5].mean(0, dtype=np.float64)
        - 0.4 * cache[500:504].mean(0, dtype=np.float64)
        - 0.6 * cache[504:508].mean(0, dtype=np.float64)
    )
    np.testing.assert_allclose(captured["delta"][0], expected, atol=1e-14)
