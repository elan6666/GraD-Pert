"""Matched conditions/controls/contexts and exact independent sample counts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from grouped_view_diagnostics import GroupedViewBank, feature_statistics  # noqa: E402


def test_control_dedup_condition_pair_strata_and_context_separation():
    records = [
        dict(
            cell_line="Jurkat",
            cell_type="T",
            context_id="T::1",
            condition_id=p,
            control_row_id=c,
            truth_row_id=str(i),
        )
        for i, (p, c) in enumerate(
            [("A", "c1"), ("A", "c2"), ("B", "c3"), ("B", "c1"), ("B", "c1")]
        )
    ]
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    bank = GroupedViewBank()
    for view in ("global0", "global1", "local0", "local1"):
        bank.add("ssl2_response_cls", "student", view, records, x)
        bank.add(
            "ssl2_response_cls", "student", view, [{**records[0], "context_id": "T::2"}], x[:1]
        )
    result = bank.result()
    rows = {
        r["stratum"]: r
        for r in result["pairs"]
        if r["context_id"] == "T::1" and r["view_a"] == "global0" and r["view_b"] == "global1"
    }
    assert {k: v["pair_count"] for k, v in rows.items()} == {
        "same_sample": 4,
        "same_condition_different_control": 2,
        "different_condition_different_control": 3,
        "different_condition_same_control": 1,
    }
    assert rows["same_sample"]["cosine_pair_mean"] == 1
    assert rows["same_sample"]["independent_controls"] == 3
    assert {r["view_family"] for r in result["pairs"]} == {"GG", "GL", "LL"}
    assert all(r["conditions"] == 1 for r in result["features"] if r["context_id"] == "T::2")


def test_constant_features_centered_cosine_is_undefined_not_zero():
    bank = GroupedViewBank()
    rows = [dict(condition_id="A", control_row_id=str(i)) for i in range(3)]
    for view in ("global0", "local0"):
        bank.add("response", "student", view, rows, np.ones((3, 4)))
    r = next(r for r in bank.result()["pairs"] if r["stratum"] == "same_sample")
    assert r["cosine_pair_mean"] == 1 and r["centered_cosine_pair_mean"] is None
    assert r["centered_cosine_valid_pairs"] == 0
    assert feature_statistics(np.ones((3, 4)))["effective_rank"] == 0


def test_static_graph_counts_conditions_not_cell_replicates():
    bank = GroupedViewBank()
    records = [
        dict(condition_id=p, cell_line="Jurkat", context_id="static_graph") for p in ("A", "B")
    ]
    for view in ("global0", "global1", "local0", "local1"):
        bank.add(
            "ssl1_condition", "student", view, records, np.eye(2), unit="perturbation_condition"
        )
    r = next(r for r in bank.result()["pairs"] if r["stratum"] == "different_condition")
    assert r["independent_samples"] == 2 and r["pair_count"] == 1
    assert r["independent_controls"] == 0


def test_pair_layout_reuse_preserves_every_statistic_exactly():
    from grouped_view_diagnostics import GroupedViewBank

    rng = np.random.default_rng(19)
    bank = GroupedViewBank()
    records = [
        dict(
            cell_line="Jurkat",
            cell_type="T",
            context_id="b1",
            condition_id=f"p{i // 3}",
            control_row_id=f"c{i % 4}",
        )
        for i in range(18)
    ]
    for view in ("global0", "global1", "local0", "local1"):
        bank.add("SSL2", "student", view, records, rng.normal(size=(18, 8)))
    assert bank.result(reuse_pair_layout=True) == bank.result(reuse_pair_layout=False)
