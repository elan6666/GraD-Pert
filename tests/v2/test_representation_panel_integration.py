"""Synthetic native forwards for the frozen crossed-control diagnostic panel."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_relay_method import small_index
from test_unified_integration import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
from cap40_posthoc import Collector, main_response, static_graph, verify_cache
from capacity_probe import validation_state
from grouped_view_diagnostics import GroupedViewBank
from prototype_pair_diagnostics import PrototypePairBank
from representation_panel import MatchedCellViews, panel_schedule, validate_panel_runtime
from representation_panel import assemble_control_only as assemble
from response_stage_diagnostics import ResponseStageObserver
from run_representation_panel import (
    add_prototype_rows,
    collect_panel_views,
    grouped_main_result,
    prototype_result,
)

from gradpert.training.v2.optimizer import V2Optimizer


def native_runtime(arm="N0"):
    objective, _, raw, options = fixture(arm, checkpointed=False)
    objective.eval()
    ids = ["t0", "t1", "t2", "t3", "c0", "c1", "c2", "c3"]
    values = np.concatenate((raw.target_expression.numpy(), raw.control_expression.numpy()))
    reads = []

    def read_indices(indices):
        reads.extend(indices)
        assert all(i >= 4 for i in indices), "diagnostics must never read perturbed expression"
        return values[indices].copy()

    data = SimpleNamespace(
        _row_index={name: i for i, name in enumerate(ids)},
        train_row_indices=list(range(4)),
        control_row_indices=list(range(4, 8)),
        context_ids=["T::b1"] * 8,
        condition_ids=["g0", "g1", "g0", "g1"] + ["control"] * 4,
        anchors_by_condition={"g0": (0,), "g1": (1,)},
        manifest=SimpleNamespace(n_expression_genes=16),
        _read_expression_indices=read_indices,
        read_indices=reads,
    )
    rt = SimpleNamespace(
        objective=objective,
        options=options,
        data=data,
        index=small_index(16),
        allowed_expression_ids=np.arange(12),
        device=torch.device("cpu"),
        optimizer=V2Optimizer(objective.student, 1e-3, 0),
        generator=np.random.default_rng(11),
    )
    groups = []
    for p, ti in (("g0", "t0"), ("g1", "t1")):
        rows = [
            dict(
                condition_id=p,
                truth_row_id=ti,
                control_row_id=f"c{i}",
                context_id="T::b1",
                cell_type="T",
                cell_line="Jurkat",
                batch="b1",
            )
            for i in range(4)
        ]
        groups.append(dict(condition_id=p, rows=rows, queries=[0, 2, 4, 6, 8, 10]))
    panel = panel_schedule(
        dict(analysis_scope="medium_cap40_stratified", schedule=groups), conditions=2, controls=4
    )
    validate_panel_runtime(panel, rt)
    return rt, panel


@pytest.mark.parametrize("arm", ["N0", "CG1", "U24", "KL"])
def test_native_crossed_forward_queries_hooks_and_state_preservation(arm):
    torch.set_num_threads(1)
    rt, panel = native_runtime(arm)
    if arm == "KL":
        rt.objective.weights = (rt.objective.weights[0], (1, 1, 0.1))
        rt.objective.koleo_exclude_same_condition = True
    before = validation_state(rt)
    views, prototypes = GroupedViewBank(), PrototypePairBank()
    matched = MatchedCellViews()
    receipts = []
    with torch.random.fork_rng(), torch.no_grad():
        cached = static_graph(rt)
        for group in panel["schedule"]:
            batch = assemble(rt, group["rows"], group["queries"], group["view_seed"], deep=False)
            ids = batch.graph.ids[batch.query_positions]
            assert ids.tolist() == group["queries"]
            assert torch.count_nonzero(batch.truth) == 0
            assert torch.count_nonzero(batch.control) > 0
            verify_cache(rt, batch, cached)
            with ResponseStageObserver(
                rt.objective.student.response, gene_ids=ids, records=group["rows"]
            ) as observer:
                response = main_response(rt, batch, cached)
            assert observer.result()["cell_statistics"]
            # No ground-truth tensor enters encode_response or the main cache path.
            changed_truth = replace(batch, truth=batch.truth + 1000)
            other = main_response(rt, changed_truth, cached)
            for key in response:
                torch.testing.assert_close(response[key], other[key], rtol=0, atol=0)
            deep = assemble(rt, group["rows"], group["queries"], group["view_seed"], deep=True)
            deep, receipt = matched.bind(deep, group["rows"])
            receipts.append(receipt)
            scope = collect_panel_views(rt.objective, deep, group["rows"], 11, views, prototypes)
            if arm == "KL":
                assert scope["diagnostic_objective_scope"]["unavailable_terms"] == {
                    "ssl2_koleo": "requires_complete_global_population"
                }
                # Fixed-probe observation must not weaken the real objective.
                with pytest.raises(ValueError, match="unlike-condition neighbor"):
                    rt.objective(deep)
                rt.objective.pending.clear()
            else:
                assert scope == {}
    assert rt.data.read_indices
    assert before == validation_state(rt)
    assert views.entries
    proto = prototype_result(prototypes, rt.objective)
    assert proto["status"] == "completed"
    assert set(proto["head_status"].values()) == {"observed"}
    results = proto["pairs"]
    assert results
    assert len({r["matched_cell_views_sha256"] for r in receipts}) == 1
    for stratum in ("same_condition_different_control", "different_condition_same_control"):
        assert any(r["stratum"] == stratum and r["pair_count"] > 0 for r in results)


@pytest.mark.parametrize("arm", ["N0", "CG1"])
@pytest.mark.parametrize("phase", [0, 1])
def test_staged_endpoint_preserves_phase_and_completes_only_feature_views(arm, phase, tmp_path):
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint, save_checkpoint
    from gradpert.training.v2.staged import StagedObjective

    torch.set_num_threads(1)
    rt, panel = native_runtime(arm)
    obj = StagedObjective(
        rt.objective.student,
        pretrain_steps=1,
        loss_reduction="row_mean",
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
    )
    obj.stage_phase.fill_(phase)
    obj._apply_phase(phase)
    obj.eval()
    rt.objective = obj
    rt.optimizer = V2Optimizer(obj.student, 1e-3, 0)
    path = tmp_path / "endpoint.pt"
    save_checkpoint(
        path,
        obj,
        rt.optimizer,
        identity={"synthetic": True},
        progress={"epoch": 2 if phase else 1},
        generator=rt.generator,
    )
    obj.stage_phase.fill_(1 - phase)
    obj._apply_phase(1 - phase)
    load_evaluation_checkpoint(
        path, obj, training_identity={"synthetic": True}, checkpoint_sha256=sha256_file(path)
    )
    assert obj.training_stage == ("mse_finetune" if phase else "ssl_pretrain")
    group = panel["schedule"][0]
    batch = assemble(rt, group["rows"], group["queries"], group["view_seed"], deep=True)
    before = validation_state(rt)
    views, prototypes = GroupedViewBank(), PrototypePairBank()
    with torch.random.fork_rng(), torch.no_grad():
        collector = Collector(batch, group["rows"], 11, stats_backend="fast")
        collector.install(obj)
        try:
            obj(batch)
            collector.complete_feature_views()
            collector.add_to_view_bank(views)
            add_prototype_rows(collector, prototypes)
        finally:
            collector.close()
            obj.pending.clear()
    assert validation_state(rt) == before
    assert {e["branch"] for e in views.entries} == {"ssl1_condition", "ssl2_response_cls"}
    student_ssl2 = [
        e for e in views.entries if e["branch"] == "ssl2_response_cls" and e["role"] == "student"
    ]
    assert len(student_ssl2) == 4
    proto = prototype_result(prototypes, obj)
    if phase:
        assert proto["status"] == "unavailable:no_active_cls_distillation_head"
        assert all(s.startswith("unavailable:") for s in proto["head_status"].values())
        assert proto["pairs"] == []
        assert {e["role"] for e in views.entries} == {"student"}
    else:
        assert proto["status"] == "completed"
        assert set(proto["head_status"].values()) == {"observed"}
        assert proto["pairs"]


def test_main_bank_labels_interventions_and_rejects_augmentation():
    _, panel = native_runtime()
    bank = GroupedViewBank()
    for group in panel["schedule"]:
        bank.add("response_cls", "student", "main", group["rows"], np.eye(4))
    result = grouped_main_result(bank)
    assert result["pairs"]
    assert {r["view_family"] for r in result["pairs"]} == {"main/main"}
    bank.add("response_cls", "student", "global0", panel["schedule"][0]["rows"], np.eye(4))
    with pytest.raises(ValueError, match="augmented view"):
        grouped_main_result(bank)


def test_active_missing_prototypes_fail_closed_and_disabled_empty_is_explicit():
    rt, _ = native_runtime()
    bank = PrototypePairBank()
    with pytest.raises(ValueError, match="active prototype head"):
        prototype_result(bank, rt.objective)
    rt.objective.lambda1 = rt.objective.lambda2 = 0
    assert prototype_result(bank, rt.objective)["status"].startswith("unavailable:")
