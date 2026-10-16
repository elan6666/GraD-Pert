"""Pure identity/grouping contracts; real model parity is gated on the server."""

import sys
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc import recipe, records_from_data


def data(rows, controls):
    ids = ["t0", "t1", "c0", "c1"]

    class Obs:
        def __init__(self):
            self.iloc = self

        def __getitem__(self, i):
            return {"cell_type": "T"}

    spec = N(perturbed_row_ids=rows, control_row_ids=controls, condition_ids=("p",) * len(rows))
    return N(
        row_ids=ids,
        _row_index={x: i for i, x in enumerate(ids)},
        context_ids=np.array(["T::A", "T::B", "T::A", "T::B"]),
        batch_ids=np.array(["A", "B", "A", "B"]),
        _adata=N(obs=Obs()),
        train_row_indices=[0, 1],
        training_batch_identity_specs=lambda **kw: [spec],
    )


def test_exact_rows_with_context_matched_controls():
    rows = records_from_data(data(("t1", "t0"), ("c1", "c0")))
    assert [r["truth_row_id"] for r in rows] == ["t1", "t0"]
    assert rows[0]["batch"] == "B"
    assert rows[0]["control_row_id"] == "c1"


@pytest.mark.parametrize(
    "rows,controls",
    [(("t0", "t0"), ("c0", "c0")), (("t0",), ("c0",)), (("t0", "t1"), ("c1", "c0"))],
)
def test_reject_duplicate_missing_and_mismatched_context(rows, controls):
    with pytest.raises(ValueError):
        records_from_data(data(rows, controls))


def test_recipe_binds_actual_mask_edges_gene_and_view_identity():
    g = N(
        ids=np.array([1, 2]),
        masked_positions=np.array([0]),
        neighbors=np.array([[1], [0]]),
        valid=np.array([[True], [True]]),
        sources=np.zeros((2, 1, 4), bool),
        context=None,
    )
    b = N(
        graph=g,
        query_positions=np.array([0, 1]),
        graph_views=(g,),
        cell_views=(N(positions=np.array([0]), mask=np.array([[True]])),),
    )
    a = recipe(b)
    b.cell_views[0].mask[0, 0] = False
    assert a != recipe(b)
    assert a["queries"] == [1, 2]


@pytest.mark.parametrize(
    "scope,arm",
    [
        ("full_cap40_train", "DP"),
        ("medium_cap40_stratified", "DP"),
        ("medium_cap40_stratified", "B0-DP"),
        ("medium_cap40_stratified", "DP2"),
        ("medium_cap40_stratified", "D0"),
    ],
)
def test_report_contains_all_figures_tables_and_hashes(tmp_path, scope, arm):
    from cap40_posthoc import sha, write
    from cap40_posthoc_report import render

    source = tmp_path / "input"
    original = tmp_path / "original"
    out = tmp_path / "report"
    source.mkdir()
    original.mkdir()
    moments = {"rows": 2, "effective_rank": 1.0, "relative_variance": 0.2}
    summary = {
        "arm": arm,
        "analysis": {"commit": "analysis"},
        "method_sha": "training",
        "checkpoint_epoch": 4,
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "common",
        "config_sha256": "config",
        "resolved_config_sha256": "resolved",
        "rows": 2,
        "conditions": 1,
        "deep_rows": 2,
        "scope": scope,
        "global_heads": {"teacher.ssl1_cls.view0": {"H_sample": 1.0, "JS": 0.2}},
        "global_features": {"student.main.response_cls": moments},
        "metrics": {"prediction_mse": {"condition_mean": 0.1}},
        "cls_decomposition": {},
        "elapsed_seconds": 1.0,
        "peak_allocated_mib": 2.0,
        "updates": 0,
        "limits": ["synthetic test"],
    }
    write(source / "summary.json", summary)
    write(
        source / "deep.json",
        [
            {
                "condition_id": "p",
                "joint": 1.0,
                "independent_truth_rows": 2,
                "independent_controls": 2,
                "loss_terms": {"prediction": 0.1},
                "heads": {},
                "features": {},
                "pairs": [
                    {
                        "condition_id": "p",
                        "student": "s",
                        "teacher": "t",
                        "CE": 1.2,
                        "H_pair": 1.0,
                        "KL": 0.2,
                    }
                ],
                "same_gene": [],
                "cross_view": [],
                "per_cell": [{"feature": "response", "condition_id": "p", **moments}],
            }
        ],
    )
    write(source / "gradients.json", [])
    for name in ("outputs_full.csv", "cls_within_condition.csv"):
        (source / name).write_text("prediction_mse,effect_scale_slope_no_intercept\n0.1,1\n")
    write(
        source / "COMPLETE.json",
        {
            "scope": scope,
            "state_unchanged": True,
            "outputs": {p.name: {"sha256": sha(p)} for p in source.iterdir()},
        },
    )
    write(original / "summary.json", {"checkpoint_sha256": "checkpoint"})
    write(original / "receipt.json", {"status": "synthetic_test"})
    for name in ("training", "heads", "pairs", "features", "outputs", "gradients", "metrics"):
        (original / (name + ".csv")).write_text("name,value\ntest,1\n")
    # Valid tiny PNG bytes; historical image copying must be byte preserving.
    import matplotlib.pyplot as plt

    for name in ("01_losses", "02_prototypes", "03_representations", "04_outputs_gradients"):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        fig.savefig(original / (name + ".png"))
        plt.close(fig)
    render(N(input=source, original=original, output=out))
    text = (out / "report.md").read_text()
    assert text.count("![") == 9
    for image in out.glob("*.png"):
        assert image.name in text
    receipt = __import__("json").loads((out / "receipt.json").read_text())
    for name, identity in receipt["outputs"].items():
        assert sha(out / name) == identity["sha256"]
    assert sha(out / "01_losses.png") == sha(original / "01_losses.png")
    assert "pair_summary.csv" in text and "outputs_full.csv" in text
    assert sum(line.startswith("## ") for line in text.splitlines()) == 13


@pytest.mark.parametrize(
    "flags,expected",
    [((True, True), [0, 1]), ((False, True), [1]), ((True, False), [0]), ((False, False), [])],
)
def test_node_call_index_preserves_actual_view_when_global_masks_are_empty(flags, expected):
    from cap40_posthoc import Collector

    views = [N(mask=np.ones((2, 3), dtype=bool) * flag) for flag in flags]
    b = N(
        graph=N(ids=np.array([1, 2])),
        graph_views=[
            N(ids=np.array([1, 2]), masked_positions=np.array([1])),
            N(ids=np.array([1, 2]), masked_positions=np.array([])),
        ],
        cell_views=views,
    )
    c = Collector(b, [], 42)
    assert [c.head_view_index("ssl2_node", i) for i in range(len(expected))] == expected
    with pytest.raises(ValueError):
        c.head_view_index("ssl2_node", len(expected))
    assert c.head_view_index("ssl1_node", 0) == 0
    assert c.head_view_index("ssl2_cls", 1) == 1
    assert c.graph_view_names[id(b.graph.ids)] == "main"
    assert c.graph_view_names[id(b.graph_views[0].ids)] == "view0"
    assert c.graph_view_names[id(b.graph_views[1].ids)] == "view1"


def test_gate_schedule_uses_all_four_mask_patterns_without_changing_formal_order():
    from cap40_posthoc import select_schedule

    schedule = list(range(300))
    assert select_schedule(schedule, 1, "0,22,58,295") == [0, 22, 58, 295]
    assert select_schedule(schedule) == schedule
    for indices in ("0,0", "-1", "300"):
        with pytest.raises(ValueError):
            select_schedule(schedule, 1, indices)
