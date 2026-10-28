"""Synthetic read boundary, immutable population and endpoint adapter tests."""

import copy
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    from gradpert.training import data

    root = Path(data.__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts/v2"))
    file = Path(__file__).resolve().parents[2] / "scripts/v2/response_anchor_medium.py"
    spec = importlib.util.spec_from_file_location("staged_response_anchor_medium", file)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result


class _Backed:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.obs = SimpleNamespace(iloc=[{"cell_type": "T"} for _ in range(len(values))])

    @property
    def X(self):
        raise AssertionError("full backed X must never be exposed")

    def __getitem__(self, key):
        rows, cols = key
        assert isinstance(rows, slice) and rows.step is None
        assert np.array_equal(cols, [0, 2, 4])
        assert set(range(rows.start, rows.stop)) <= {0, 1, 4, 5, 6, 7}
        self.calls.append((rows.start, rows.stop, cols.tolist()))
        return SimpleNamespace(X=self.values[rows][:, cols])


def _runtime(native=False, nonfinite=False):
    values = np.arange(40, dtype=np.float32).reshape(8, 5)
    values[:, [1, 3]] = np.nan  # The excluded columns must never enter an exposed array.
    if nonfinite:
        values[4, 2] = np.inf
    data = SimpleNamespace(
        manifest=SimpleNamespace(n_expression_genes=5),
        row_ids=tuple(str(i) for i in range(8)),
        train_row_indices=(0, 1),
        control_row_indices=(4, 5, 6, 7),
        _row_index={str(i): i for i in range(8)},
        condition_ids=("p", "p", "val", "test", "control", "control", "control", "control"),
        context_ids=("k",) * 8,
        batch_ids=("b",) * 8,
        _adata=_Backed(values),
        _control_expression_cache=None,
        pipeline_stats=SimpleNamespace(control_cache_requested=False, yielded_batches=0),
    )

    def old(indices):
        raise AssertionError("old fullwidth reader must not be called")

    data._read_expression_indices = old
    if native:
        from gradpert.training.data import CanonicalTrainingData

        data.configure_training_expression_columns = (
            CanonicalTrainingData.configure_training_expression_columns.__get__(data)
        )
        data._read_expression_indices = CanonicalTrainingData._read_expression_indices.__get__(data)
    records = [
        dict(
            truth_row_id=str(i),
            control_row_id=str(4 + i),
            condition_id="p",
            context_id="k",
            cell_line="Jurkat",
            cell_type="T",
            batch="b",
        )
        for i in (0, 1)
    ]
    return SimpleNamespace(data=data, allowed_expression_ids=np.array([0, 2, 4])), records, values


@pytest.mark.parametrize("native", [False, True])
def test_allowed_query_values_bitexact_and_no_unselected_row_or_excluded_nan(module, native):
    runtime, records, values = _runtime(native)
    original = runtime.data._read_expression_indices
    receipt = {}
    with module.allowed_train_reads(runtime, records, controls_only=False, receipt=receipt):
        result = runtime.data._read_expression_indices([5, 0, 4, 0])
        np.testing.assert_array_equal(result[:, [0, 2, 4]], values[[5, 0, 4, 0]][:, [0, 2, 4]])
        assert result.dtype == np.float32 and np.isfinite(result).all()
        assert np.count_nonzero(result[:, [1, 3]]) == 0
        with pytest.raises(ValueError, match="whitelist"):
            runtime.data._read_expression_indices([2])
    assert runtime.data._read_expression_indices == original
    assert receipt["calls"] == 1 and receipt["rows"] == 4
    assert not receipt["physical_bytes_measured"]
    assert runtime.data._adata.calls


def test_control_only_panel_rejects_truth_read_and_restores_reader_on_error(module):
    runtime, records, _ = _runtime()
    original = runtime.data._read_expression_indices
    with (
        pytest.raises(ValueError, match="whitelist"),
        module.allowed_train_reads(runtime, records, controls_only=True, receipt={}),
    ):
        runtime.data._read_expression_indices([0])
    assert runtime.data._read_expression_indices is original


@pytest.mark.parametrize("failure", ["nonfinite", "cache", "context", "batchcollision"])
def test_read_fail_closed_without_fullwidth_fallback(module, failure):
    runtime, records, _ = _runtime(nonfinite=failure == "nonfinite")
    if failure == "cache":
        runtime.data._control_expression_cache = np.ones((1, 5))
    if failure == "context":
        records[0]["context_id"] = "wrong"
    if failure == "batchcollision":
        records[1]["batch"] = "other"
    original = runtime.data._read_expression_indices
    with (
        pytest.raises(ValueError),
        module.allowed_train_reads(runtime, records, controls_only=False, receipt={}),
    ):
        runtime.data._read_expression_indices([4])
    assert runtime.data._read_expression_indices is original


def test_parent_protocol_whitelist_preserves_physical_identity_and_missing_null(module):
    parent = {
        "data": {
            "canonical_sha256": "same",
            "loss_protocol": {"version": "v"},
            "training_row_selection": {"sha": "s"},
        },
        "schedule": [{"fixed": "unchanged"}],
    }
    identity = {"run_id": "new", "data": copy.deepcopy(parent["data"])}
    identity["data"]["response_anchor"] = {"manifest_sha256": "fixed", "weight": 0.03}
    approved = module.identity_diff(parent["data"], identity["data"])
    bound = module.bind_parent(parent, identity, approved)
    assert parent["data"].get("response_anchor") is None
    assert bound["schedule"] == parent["schedule"]
    assert not bound["training_protocol_binding"]["sampling_changed"]
    with pytest.raises(ValueError):
        module.bind_parent(parent, identity, [])
    identity["data"]["canonical_sha256"] = "different"
    with pytest.raises(ValueError):
        module.bind_parent(parent, identity, module.identity_diff(parent["data"], identity["data"]))
    assert module.identity_diff({}, {"a": None}) != []


def test_adapter_only_replaces_binding_and_restores_every_reference(module):
    runtime, records, _ = _runtime()
    seen = []

    @contextmanager
    def prepare(*args, **kwargs):
        seen.append(kwargs["purpose"])
        yield runtime

    def runtime_args(args):
        return None, None, None, prepare, Path("/native")

    cap = SimpleNamespace(runtime_args=runtime_args)
    panel = SimpleNamespace(
        runtime_args=runtime_args, panel_parent_binding=object(), panel_schedule=object()
    )
    old = (panel.runtime_args, panel.panel_parent_binding, panel.panel_schedule)
    receipts = []
    with (
        pytest.raises(RuntimeError, match="synthetic"),
        module.diagnostic_adapter(cap, panel, records, controls_only=False, read_receipts=receipts),
    ):
        with cap.runtime_args(None)[3](purpose="evaluation") as rt:
            assert np.isfinite(rt.data._read_expression_indices([4])).all()
        raise RuntimeError("synthetic")
    assert cap.runtime_args is runtime_args
    assert old == (panel.runtime_args, panel.panel_parent_binding, panel.panel_schedule)
    assert seen == ["evaluation"] and receipts[0]["calls"] == 1


def test_axis_hash_rejected_before_any_read_and_training_purpose_rejected(module):
    runtime, records, _ = _runtime()

    @contextmanager
    def prepare(**kwargs):
        yield runtime

    cap = SimpleNamespace(runtime_args=lambda _: (None, None, None, prepare, Path("/native")))
    with module.diagnostic_adapter(
        cap,
        None,
        records,
        controls_only=False,
        read_receipts=[],
        expected_axis={"count": 3, "sha256": "wrong"},
    ):
        with (
            pytest.raises(ValueError, match="before any numerical read"),
            cap.runtime_args(None)[3](purpose="evaluation"),
        ):
            pytest.fail("unsealed axis accepted")
        with (
            pytest.raises(ValueError, match="evaluation"),
            cap.runtime_args(None)[3](purpose="training"),
        ):
            pytest.fail("training runtime accepted")
    assert not runtime.data._adata.calls


def _panel():
    from cap40_posthoc_stats import digest_json

    schedule = []
    for context in range(16):
        for p in range(2):
            condition = f"p{context}_{p}"
            rows = [dict(control_row_id=f"c{context}_{i}") for i in range(4)]
            schedule.append(
                dict(
                    condition_id=condition,
                    context=["J", "T", f"k{context}", "b"],
                    queries=[0, 2, 4],
                    rows=rows,
                )
            )
    panel = dict(
        conditions=32,
        rows=128,
        controls_per_context=4,
        condition_context_groups=32,
        schedule=schedule,
        panel_sha256=digest_json(schedule),
    )
    parent = dict(
        schedule=[dict(condition_id=g["condition_id"], queries=[0, 2, 4]) for g in schedule]
    )
    return panel, parent


def test_exact_fixed_crossed_panel_no_new_sampling_and_order_corruption(module):
    from cap40_posthoc_stats import digest_json

    panel, parent = _panel()
    module.validate_panel(panel, parent)
    panel["schedule"][1]["rows"].reverse()
    panel["panel_sha256"] = digest_json(panel["schedule"])
    with pytest.raises(ValueError, match="ordered controls"):
        module.validate_panel(panel, parent)


def test_anchor_probe_na_never_numeric_zero(module):
    scope = dict(
        diagnostic_objective_scope={
            "unavailable_terms": {"response_anchor": "requires_complete_train_condition_bank"}
        },
        loss_terms={"ssl1": 1.0},
    )
    module.validate_scopes([scope], anchor=True)
    scope["loss_terms"]["response_anchor"] = 0.0
    with pytest.raises(ValueError):
        module.validate_scopes([scope], anchor=True)
    module.validate_scopes([{}], anchor=False)


def test_frozen_matched_recipe_checked_before_collector_and_class_restored(module):
    import torch
    from representation_panel import MatchedCellViews

    from gradpert.training.v2.objective import CellView, GraphView, TrainingBatch

    _runtime_instance, records, _ = _runtime()
    records = [{**records[0], "control_row_id": str(i)} for i in (4, 5, 6, 7)]
    graph = GraphView(
        torch.tensor([0, 2, 4]), *[torch.zeros(0, dtype=torch.long) for _ in range(6)]
    )
    batch = TrainingBatch(
        graph,
        torch.arange(3),
        torch.zeros(4, 3),
        torch.zeros(4, 3),
        torch.zeros(4, dtype=torch.long),
        (),
        (CellView(torch.tensor([0, 1]), torch.zeros(4, 2, dtype=torch.bool)),),
    )
    _, receipt = MatchedCellViews().bind(batch, records)
    expected = [dict(condition_id="p", **receipt)]
    cap = SimpleNamespace(runtime_args=lambda _: None)
    panel = SimpleNamespace(
        runtime_args=None,
        panel_parent_binding=None,
        panel_schedule=None,
        MatchedCellViews=MatchedCellViews,
    )
    with module.diagnostic_adapter(
        cap, panel, records, controls_only=True, read_receipts=[], expected_views=expected
    ):
        panel.MatchedCellViews().bind(batch, records)
        batch.cell_views[0].mask[0, 0] = True
        with pytest.raises(ValueError, match="before collection"):
            panel.MatchedCellViews().bind(batch, records)
    assert panel.MatchedCellViews is MatchedCellViews


def test_unapproved_sampling_fields_rejected_before_opening_inputs(module):
    spec = dict.fromkeys(
        (
            "name",
            "analysis_source",
            "native_source",
            "run_root",
            "run_manifest",
            "resolved_config",
            "config",
            "journal",
            "checkpoint",
            "data_root",
            "medium",
            "panel",
            "allowed_axis",
            "resource_policy",
            "resource_owner",
            "output",
            "gpu",
        )
    )
    spec["schema"] = module.SCHEMA
    spec["medium"] = {
        "parent": {},
        "performance": {},
        "population_sha256": "x",
        "identity_changes": [],
        "new_random_seed": 42,
    }
    with pytest.raises(ValueError, match="medium"):
        module.validate_spec(spec, server=False)


def _owner():
    return dict(
        schema="response-anchor-analysis-owner-v1",
        process=dict(
            pid=10,
            uid=1003,
            start_ticks="42",
            command_sha256="a" * 64,
            script_basename="run_response_anchor_postfit.py",
        ),
    )


@pytest.mark.parametrize(
    "bad", ["start_ticks", "command_sha256", "uid", "script_basename", "nonancestor"]
)
def test_owner_pid_reuse_identity_and_ancestry_fail_closed(module, bad):
    receipt = _owner()
    actual = dict(receipt["process"])
    if bad != "nonancestor":
        actual[bad] = "different"
    with pytest.raises(ValueError, match="ancestry"):
        module.verified_owner(
            receipt,
            lookup=lambda _: actual,
            parents={} if bad == "nonancestor" else {20: 10},
            pid=20,
            uid=1003,
        )


def test_sibling_snapshot_uses_same_live_owner_without_touching_admission(module, monkeypatch):
    import shared_gpu

    receipt = _owner()
    for worker in (20, 30):
        assert (
            module.verified_owner(
                receipt,
                lookup=lambda _: dict(receipt["process"]),
                parents={20: 10, 30: 10},
                pid=worker,
                uid=1003,
            )
            == 10
        )
    monkeypatch.setattr(module, "verified_owner", lambda _: 10)

    def original_snapshot(owner):
        return {"owner_pid": owner}

    monkeypatch.setattr(shared_gpu, "snapshot", original_snapshot)
    admit = shared_gpu.admit
    with module.resource_snapshots(receipt):
        assert shared_gpu.snapshot(20)["owner_pid"] == shared_gpu.snapshot(30)["owner_pid"] == 10
        assert shared_gpu.admit is admit
    assert shared_gpu.snapshot is original_snapshot
