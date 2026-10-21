"""Input interventions preserve identity and cannot borrow held-out expression."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
from representation_panel import panel_schedule, validate_panel_runtime


def parent():
    rows = []
    for p in range(6):
        records = [
            dict(
                condition_id=f"p{p}",
                control_row_id=f"c{i}",
                truth_row_id=f"t{p}-{i}",
                context_id="T::b1",
                batch="b1",
                cell_line="Jurkat",
                cell_type="T",
            )
            for i in range(4)
        ]
        rows.append(dict(condition_id=f"p{p}", rows=records, queries=list(range(10))))
    return dict(analysis_scope="medium_cap40_stratified", schedule=rows)


def runtime(p):
    rows = [r for group in p["schedule"] for r in group["rows"]]
    ids = sorted({r["truth_row_id"] for r in rows}) + [f"c{i}" for i in range(4)]
    lookup = {r["truth_row_id"]: r["condition_id"] for r in rows}
    return SimpleNamespace(
        allowed_expression_ids=list(range(10)),
        data=SimpleNamespace(
            train_row_indices=list(range(len(ids) - 4)),
            control_row_indices=list(range(len(ids) - 4, len(ids))),
            _row_index={name: i for i, name in enumerate(ids)},
            context_ids=["T::b1"] * len(ids),
            condition_ids=[lookup.get(name, "control") for name in ids],
        ),
    )


def test_panel_is_reproducible_crossed_and_does_not_mutate_parent():
    original = parent()
    before = copy.deepcopy(original)
    panel = panel_schedule(original, conditions=4)
    assert original == before
    assert panel == panel_schedule(original, conditions=4)
    assert panel["rows"] == 16 and panel["conditions"] == 4
    pools = [tuple(r["control_row_id"] for r in g["rows"]) for g in panel["schedule"]]
    assert len(set(pools)) == 1 and len(set(pools[0])) == 4
    assert len({g["view_seed"] for g in panel["schedule"]}) == 1
    validate_panel_runtime(panel, runtime(original))


def test_insufficient_contexts_fail_or_are_reported_without_duplicate_cells():
    original = parent()
    for g in original["schedule"]:
        for r in g["rows"]:
            r["control_row_id"] = "same"
    with pytest.raises(ValueError, match="sufficient"):
        panel_schedule(original)


@pytest.mark.parametrize("invalid", ["test_truth", "hidden_gene", "wrong_context", "noncontrol"])
def test_runtime_guard_rejects_leakage_or_context_mismatch(invalid):
    original = parent()
    panel = panel_schedule(original)
    rt = runtime(original)
    row = panel["schedule"][0]["rows"][0]
    if invalid == "test_truth":
        rt.data.train_row_indices.remove(rt.data._row_index[row["truth_row_id"]])
    elif invalid == "hidden_gene":
        rt.allowed_expression_ids = [1, 2]
    elif invalid == "wrong_context":
        rt.data.context_ids[rt.data._row_index[row["control_row_id"]]] = "other"
    else:
        rt.data.control_row_indices.remove(rt.data._row_index[row["control_row_id"]])
    with pytest.raises(ValueError):
        validate_panel_runtime(panel, rt)
