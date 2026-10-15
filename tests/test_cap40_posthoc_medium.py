"""Medium probes preserve row/control/gene identity and do not select by outcomes."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc import (
    METHOD_SHA,
    medium_cells,
    medium_schedule,
    population_identity,
    reuse_medium_schedule,
    validate_method_manifest,
)
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


def test_contexts_beyond_four_are_seeded_per_condition_not_a_fixed_prefix():
    universe = [f"T::{i:02d}" for i in range(20)]
    selected = []
    for p in range(12):
        rows = [
            {"condition_id": f"p{p}", "truth_row_id": f"t{p}-{i}", "context_id": c}
            for i, c in enumerate(universe)
        ]
        actual = medium_cells(rows, 4, 42)
        assert actual == medium_cells(list(reversed(rows)), 4, 42)
        selected.append({r["context_id"] for r in actual})
    assert len(set.union(*selected)) > 4
    assert any(x != set(universe[:4]) for x in selected)


def medium_parent():
    p = parent()
    p["schedule"] = medium_schedule(p, deep_conditions=6, gradient_conditions=3)
    p.update(analysis_scope="medium_cap40_stratified", conditions=len(p["schedule"]))
    for name, field in (
        ("rows", "rows"),
        ("deep_rows", "deep_rows"),
        ("gradient_rows", "gradient_rows"),
    ):
        p[name] = sum(len(g[field]) for g in p["schedule"])
    for name, field in (("deep_conditions", "deep_rows"), ("gradient_conditions", "gradient_rows")):
        p[name] = sum(bool(g[field]) for g in p["schedule"])
    return p


def test_new_arm_reuses_exact_population_and_only_reseals_views():
    p = medium_parent()
    before = copy.deepcopy(p)
    for g in p["schedule"]:
        g.update(recipe_file="old.json", recipe_sha256="old recipe")
    reused = reuse_medium_schedule(p)
    assert population_identity(reused) == population_identity(before["schedule"])
    assert p["schedule"][0]["recipe_file"] == "old.json"
    assert all(g["recipe_file"] is None and g["recipe_sha256"] is None for g in reused)
    reused[0]["rows"][0]["control_row_id"] = "different-control"
    assert population_identity(reused) != population_identity(before["schedule"])
    assert p["schedule"][0]["rows"] == before["schedule"][0]["rows"]


@pytest.mark.parametrize("change", ["count", "method", "config", "population", "unsealed"])
def test_new_method_cannot_reuse_stale_view_or_sampling_identity(change):
    p = medium_parent()
    p.update(
        method_sha="b" * 40,
        view_recipe_config_sha256="cfg",
        diagnostic_population_sha256=population_identity(p["schedule"]),
    )
    validate_method_manifest(p, "b" * 40, "cfg")
    if change == "count":
        p["rows"] += 1
        with pytest.raises(ValueError, match="counts"):
            reuse_medium_schedule(p)
        return
    if change == "method":
        p["method_sha"] = METHOD_SHA
    elif change == "config":
        p["view_recipe_config_sha256"] = "other"
    elif change == "population":
        p["schedule"][0]["queries"][0] += 10
    else:
        p.pop("view_recipe_config_sha256")
    with pytest.raises(ValueError):
        validate_method_manifest(p, "b" * 40, "cfg")
