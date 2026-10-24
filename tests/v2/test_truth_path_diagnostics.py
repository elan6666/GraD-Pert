"""Synthetic truth membership, absolute-expression reduction and sealed hook checks."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
import truth_path_diagnostics as truth
from cap40_posthoc_stats import digest_json


def fixture(monkeypatch):
    rows = [
        dict(
            truth_row_id=p,
            control_row_id="c0",
            condition_id=p,
            cell_line="Jurkat",
            cell_type="T",
            context_id="T::b",
            batch="b",
        )
        for p in ("a", "z")
    ]
    schedule = [
        dict(
            condition_id=r["condition_id"],
            context=["Jurkat", "T", "T::b", "b"],
            queries=[0, 1],
            rows=[{**r, "control_row_id": f"c{i}"} for i in range(4)],
        )
        for r in rows
    ]
    panel = dict(
        schedule=schedule, panel_sha256=digest_json(schedule), query_sha256=digest_json([0, 1])
    )
    ids = ["a", "z", "c0", "c1", "c2", "c3", "HOLDOUT"]
    d = SimpleNamespace(
        row_ids=ids,
        _row_index={r: i for i, r in enumerate(ids)},
        train_row_indices=(0, 1),
        control_row_indices=(2, 3, 4, 5),
        condition_ids=["a", "z", "ctrl", "ctrl", "ctrl", "ctrl", "val"],
        context_ids=["T::b"] * 7,
        batch_ids=["b"] * 7,
        _adata=SimpleNamespace(obs=pd.DataFrame({"cell_type": ["T"] * 7})),
        split=SimpleNamespace(control_condition_id="ctrl"),
        expression_gene_ids=["g0", "g1"],
    )
    reads = []

    def read(indices):
        reads.extend(indices)
        assert 6 not in indices
        return np.array([[2.0, 4.0] if i < 2 else [0.0, 0.0] for i in indices])

    d._read_expression_indices = read
    rt = SimpleNamespace(data=d, allowed_expression_ids=[0, 1])
    monkeypatch.setattr(truth, "records_from_data", lambda d: copy.deepcopy(rows))
    return panel, rt, rows, reads


def bank(panel, rt):
    return truth.TruthPathBank(panel, rt, required_shape=(1, 2, 4), query_count=2)


def test_reduce_predictions_before_error_and_absolute_expression():
    f = np.array([[0.0, 2.0], [4.0, 6.0]])
    b = f + 1
    c = np.zeros_like(f)
    result = truth.population_errors(f, b, c, [2.0, 4.0], [1])
    assert result["query_F_mse"] == 0
    assert result["query_B_minus_F_mse"] == 1
    assert result["query_control_copy_mse"] == 10
    # A per-cell MSE would be 4; prediction-minus-control scoring would change under this shift.
    shifted = truth.population_errors(f, b, c + 7, [2.0, 4.0], [1])
    assert shifted["query_F_mse"] == result["query_F_mse"]
    assert shifted["query_B_mse"] == result["query_B_mse"]


def test_scalar_only_train_population_and_small_sample_na(monkeypatch):
    panel, rt, _, reads = fixture(monkeypatch)
    audit = bank(panel, rt)
    values = dict(
        F={"prediction": torch.tensor([[0.0, 2.0], [4.0, 6.0], [0.0, 2.0], [4.0, 6.0]])},
        B={"prediction": torch.tensor([[1.0, 3.0], [5.0, 7.0], [1.0, 3.0], [5.0, 7.0]])},
        control=torch.zeros(4, 2),
        parity={"tolerance": 2e-6},
    )
    monkeypatch.setattr(truth, "collect_group", lambda *a: values)
    with pytest.raises(ValueError, match="incomplete"):
        audit.result()
    for g in panel["schedule"]:
        audit.add(rt, None, None, g)
    result = audit.result()
    assert result["rows"][0]["query_F_mse"] == 0
    assert result["rows"][0]["deg_F_mse"] is None
    assert result["rows"][0]["deg_unavailable_reason"] == "insufficient_train_cells_for_t_test"
    assert set(reads) == set(range(6))
    assert "truth_row_id" not in str(result)
    assert "control_row_id" not in str(result)
    with pytest.raises(ValueError, match="duplicate"):
        audit.add(rt, None, None, panel["schedule"][0])


@pytest.mark.parametrize(
    "change,match",
    [
        ("query", "query axis"),
        ("context", "metadata"),
        ("holdout", "cap40"),
        ("control", "membership"),
        ("cap", "cap40"),
    ],
)
def test_fail_closed_membership_context_query(monkeypatch, change, match):
    panel, rt, rows, _ = fixture(monkeypatch)
    if change == "query":
        rt.allowed_expression_ids = [0, 2]
    if change == "context":
        rt.data.batch_ids[0] = "other"
    if change == "holdout":
        rows[0]["truth_row_id"] = "HOLDOUT"
    if change == "control":
        rt.data.control_row_indices = (2, 3, 4)
    if change == "cap":
        # Duplicate record must fail before any expression access.
        rows.extend([rows[0]] * 40)
    with pytest.raises(ValueError, match=match):
        bank(panel, rt)


def test_unknown_or_reordered_group(monkeypatch):
    panel, rt, _, _ = fixture(monkeypatch)
    audit = bank(panel, rt)
    with pytest.raises(ValueError, match="reordered"):
        audit.add(rt, None, None, panel["schedule"][1])
    assert truth.train_deg(np.ones((1, 2)), np.ones((4, 2)), ["a", "b"], "p", "ctrl")[0] == []


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_scoring_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        truth.population_errors([[value, 0]], [[0, 0]], [[0, 0]], [0, 0], [])


def test_missing_matched_truth_is_explicit_na(monkeypatch):
    panel, rt, _, _ = fixture(monkeypatch)
    audit = bank(panel, rt)
    audit.populations.clear()
    values = dict(parity={"tolerance": 2e-6})
    monkeypatch.setattr(truth, "collect_group", lambda *a: values)
    for group in panel["schedule"]:
        audit.add(rt, None, None, group)
    assert all(r["status"] == "missing_matched_train_truth" for r in audit.result()["rows"])


def test_native_deg_helper_ties_non_dropout_and_nonfinite(monkeypatch):
    captured = {}
    scores = np.array([1.0, 1.0, 2.0])

    class Ann:
        def __init__(self, x, obs):
            captured["x"] = x.copy()
            self.uns = {}

    def rank(data, **kwargs):
        captured["kwargs"] = kwargs
        data.uns["rank_genes_groups"] = dict(
            names={"truth": np.array(["g2", "g0", "g1"])}, scores={"truth": scores}
        )

    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(AnnData=Ann))
    monkeypatch.setitem(
        sys.modules, "scanpy", SimpleNamespace(tl=SimpleNamespace(rank_genes_groups=rank))
    )
    t = np.array([[2.0, 0.0, 3.0], [4.0, 0.0, 5.0]])
    c = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    deg, reason = truth.train_deg(t, c, ["g0", "g1", "g2"], "p", "ctrl")
    # Native v2 non-dropout removes g1; no model predictions or holdout cells enter ranking.
    assert deg == [0, 2] and reason is None
    np.testing.assert_array_equal(captured["x"], np.concatenate((t, c)))
    assert captured["kwargs"]["method"] == "t-test"
    scores[0] = np.inf
    assert (
        truth.train_deg(t, c, ["g0", "g1", "g2"], "p", "ctrl")[1] == "nonfinite_train_t_test_scores"
    )


def test_fixed_query_subset_of_larger_allowed_axis(monkeypatch):
    panel, rt, _, _ = fixture(monkeypatch)
    queries = list(range(0, 2000, 2))
    for group in panel["schedule"]:
        group["queries"] = queries.copy()
    panel["query_sha256"] = digest_json(queries)
    panel["panel_sha256"] = digest_json(panel["schedule"])
    rt.allowed_expression_ids = list(range(4775))
    audit = truth.TruthPathBank(panel, rt, required_shape=(1, 2, 4))
    assert audit.queries == queries and len(audit.queries) == 1000
    altered = copy.deepcopy(panel)
    altered["schedule"][1]["queries"][:2] = queries[:2][::-1]
    altered["panel_sha256"] = digest_json(altered["schedule"])
    with pytest.raises(ValueError, match="query order"):
        truth.TruthPathBank(altered, rt, required_shape=(1, 2, 4))


@pytest.mark.parametrize(
    "flags,match",
    [
        ({"path_audit": True}, "separate outputs"),
        ({"teacher_signal_audit": True}, "separate outputs"),
        ({"checkpoint_role": "pretrain"}, "final-checkpoint"),
        ({"controls": 8}, "final-checkpoint"),
    ],
)
def test_hook_audits_mutually_exclusive_and_final_only(monkeypatch, flags, match):
    import run_representation_panel as runner

    args = SimpleNamespace(
        output=Path("out"),
        run_root=Path("run"),
        data_root=Path("data"),
        parent_manifest=Path("parent"),
        resource_queue=Path("queue"),
        parent_manifest_sha256="hash",
        resource_queue_sha256="hash",
        seed=1,
        conditions=32,
        controls=4,
        checkpoint_role="last",
        truth_path_audit=True,
        path_audit=False,
        teacher_signal_audit=False,
        **{},
    )
    for key, value in flags.items():
        setattr(args, key, value)
    monkeypatch.setattr(runner, "server_path", lambda p: None)
    monkeypatch.setattr(runner, "sha", lambda p: "hash")
    monkeypatch.setattr(runner, "read", lambda p: {})
    monkeypatch.setattr(runner, "panel_schedule", lambda *a, **kw: {"conditions": 32})
    with pytest.raises(ValueError, match=match):
        runner.execute(args)
