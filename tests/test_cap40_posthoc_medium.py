"""Medium probes preserve row/control/gene identity and do not select by outcomes."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc import medium_schedule
from cap40_posthoc_queue import ordered_rows


def parent():
    return {
        "seed": 42,
        "schedule": [
            {
                "index": i,
                "condition_id": f"p{i}",
                "rows": [
                    {
                        "truth_row_id": f"t{i}-{j}",
                        "control_row_id": f"c{i}-{j}",
                        "context_id": f"T::{j % 2}",
                        "condition_id": f"p{i}",
                    }
                    for j in range((1, 2, 8)[i % 3])
                ],
                "queries": [1, 3, 5],
                "view_seed": 100 + i,
            }
            for i in range(12)
        ],
    }


def test_medium_all_conditions_parent_pairs_contexts_budget_and_no_mutation():
    p = parent()
    original = copy.deepcopy(p)
    schedule = medium_schedule(p, deep_conditions=6, gradient_conditions=3)
    assert p == original
    assert schedule == medium_schedule(p, deep_conditions=6, gradient_conditions=3)
    assert len(schedule) == 12
    assert sum(bool(g["deep_rows"]) for g in schedule) == 6
    assert sum(bool(g["gradient_rows"]) for g in schedule) == 3
    for before, after in zip(p["schedule"], schedule, strict=True):
        assert len(after["rows"]) == min(4, len(before["rows"]))
        assert after["queries"] == before["queries"] and after["view_seed"] == before["view_seed"]
        pairs = {(r["truth_row_id"], r["control_row_id"]) for r in before["rows"]}
        assert all((r["truth_row_id"], r["control_row_id"]) in pairs for r in after["rows"])
        assert {r["context_id"] for r in after["rows"]} == {r["context_id"] for r in before["rows"]}
        assert all(r["control_treatment"] == "control" for r in after["rows"])
        assert not after["gradient_rows"] or after["deep_rows"] == after["rows"]
        assert after["recipe_file"] is None  # Actual medium-size masks must be resealed.


@pytest.mark.parametrize("deep,grad,cells", [(0, 0, 4), (13, 2, 4), (4, 5, 4), (4, 2, 1)])
def test_invalid_medium_budgets_rejected(deep, grad, cells):
    with pytest.raises(ValueError):
        medium_schedule(parent(), cells=cells, deep_conditions=deep, gradient_conditions=grad)


def test_priority_order_does_not_mutate_canonical_comparison_order():
    q = {
        "rows": [{"arm": a} for a in ("DP", "FN", "FG", "FB", "PO")],
        "execution_order": ["PO", "DP", "FN", "FG", "FB"],
    }
    assert next(r["arm"] for r in ordered_rows(q)) == "PO"
    assert q["rows"][0]["arm"] == "DP"
    q["execution_order"][-1] = "PO"
    with pytest.raises(ValueError, match="exactly once"):
        ordered_rows(q)
