"""CPU synthetic behavior tests; no canonical data, server, or native GPU run."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/v2"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "train40_probe_test_module", SCRIPTS / "response_anchor_gradient_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.options = SimpleNamespace(
            control_conditioned_graph=False, prediction_parameterization="softplus"
        )
        self.response = nn.Linear(3, 256)
        self.gene = nn.Parameter(torch.randn(256) * 0.02)
        self.condition = nn.Parameter(torch.randn(256) * 0.02)
        self.dropout = nn.Dropout(0.2)
        self.prediction = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
        self.response_anchor_readout = nn.Linear(256, 256)

    def encode_response(self, gene, control, condition, *, decode=True, **_kwargs):
        z = self.dropout(self.response(control)) + condition
        tokens = z[:, None, :] + gene[None, :, :]
        output = dict(response_cls=z, response_tokens=tokens)
        if decode:
            joint = torch.cat((tokens, condition[:, None, :].expand_as(tokens)), -1)
            score = self.prediction(joint).squeeze(-1)
            prediction = F.softplus(score.float())
            output.update(prediction=prediction, delta=prediction - control)
        return output


class ToyObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = ToyStudent()
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.pending = {}
        self.loss_reduction = "row_mean"
        self.prediction_error_power = 2
        self.prediction_reduction_override = "inherit"
        self.prediction_enabled = True
        self.population_response = False

    def _batch_graph(self, model, batch, prefix=None):
        return model.gene[None].expand(len(batch.graph.ids), -1), model.condition[None].expand(
            len(batch.control), -1
        )

    def _select_graph(self, graph, positions):
        return graph[positions]

    def response_metadata(self, _batch):
        return {}

    def commit_statistics(self, *_args):
        raise AssertionError("must be guarded")


class ToyOptimizer:
    def state_dict(self):
        return dict(steps=0)

    def step(self, *_args):
        raise AssertionError("must be guarded")

    def zero_grad(self, *_args):
        raise AssertionError("must be guarded")


def toy_runtime():
    from gradpert.training.v2.objective import TrainingBatch

    torch.manual_seed(5)
    objective = ToyObjective()
    graph = SimpleNamespace(ids=torch.tensor([0, 1, 2]))
    batch = TrainingBatch(
        graph,
        torch.arange(3),
        torch.rand(40, 3),
        torch.rand(40, 3),
        torch.zeros(40, dtype=torch.long),
        (),
        (),
    )
    pids = tuple(f"g{i}" for i in range(1322))
    selection, policy = dict(selected="frozen"), dict(allowed="frozen")
    data = SimpleNamespace(
        graph_gene_ids=pids,
        split=SimpleNamespace(
            split_content_sha256="split", train_conditions=pids, control_condition_id="ctrl"
        ),
        steps_per_epoch=lambda **_kw: 150,
    )
    controls = np.random.default_rng(1).normal(size=(40, 4775)).astype(np.float32)
    bank = SimpleNamespace(
        identity=dict(split_content_sha256="split", selection=selection, allowed_policy=policy),
        condition_ids=pids,
        allowed_gene_indices=np.arange(4775),
        condition_anchor_gene_indices=np.arange(1322),
        condition_targets=np.random.default_rng(2).normal(size=(1322, 256)).astype(np.float32),
        control_for_condition=lambda _: controls,
    )
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=ToyOptimizer(),
        data=data,
        options=SimpleNamespace(
            microbatch=80, response_anchor_seed=20261028, response_anchor_weight=0.03148168297045377
        ),
        device=torch.device("cpu"),
        generator=np.random.default_rng(1),
        batch_size=320,
        identity=dict(training_row_selection=selection, training_expression_policy=policy),
        index=SimpleNamespace(view=lambda ids, *_a, **_kw: SimpleNamespace(ids=torch.tensor(ids))),
        batches=lambda *_a: iter(()),
        validation_batches=lambda *_a: iter(()),
    )
    return runtime, bank, batch


def test_actual_native_helpers_two_inputs_gradients_and_state(tmp_path):
    import calibrate_response_anchor as native

    rt, bank, batch = toy_runtime()
    ctx = probe.bank_context(rt, bank)
    group = dict(condition_id="g0", bank_condition_index=0)
    state = probe.constant_state(rt)
    cpu_rng = torch.get_rng_state().clone()
    np_rng = copy.deepcopy(np.random.get_state())
    python_rng = random.getstate()
    with native.preserved_calibration(rt), probe.forbidden_updates(rt) as calls:
        row, arrays = probe.measure_case(rt, batch, ctx, group, 0, native.prediction_losses)
        assert row["calls"] == {"prediction": 1, "auxiliary": 1}
        assert row["modules"]["prediction"]["counts"]["prediction_only"] > 0
        assert row["modules"]["response_anchor_readout"]["counts"]["auxiliary_only"] > 0
        assert row["gradient"]["counts"]["shared_connected"] > 0
        assert row["gradient"]["shared_cosine"] is not None
        assert row["prediction_loss"] > 0 and row["auxiliary_loss"] > 0
        assert arrays["pred_cls"].shape == arrays["aux_cls"].shape == (256,)
        assert not np.array_equal(arrays["pred_cls"], arrays["aux_cls"])
        for block in row["decoder"]["local_sensitivity"].values():
            assert block["squared_l2_per_output_mean"] == pytest.approx(
                256 * block["coordinate_mean_square"], rel=2e-6
            )
        assert calls == {"autograd_grad": 3}
        probe.persist_case_scalar(tmp_path, 1, row, "synthetic-spec")
        assert probe.read(tmp_path / probe.case_scalar_path(1))["row"] == row
    assert probe.constant_state(rt) == state
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert random.getstate() == python_rng
    assert all(p.grad is None for p in rt.objective.parameters())


def test_same_recipe_repeats_but_third_rng_label_changes_not_pair_epoch():
    import calibrate_response_anchor as native

    rt, bank, batch = toy_runtime()
    ctx, group = probe.bank_context(rt, bank), dict(condition_id="g0", bank_condition_index=0)
    with native.preserved_calibration(rt):
        a, _ = probe.measure_case(rt, batch, ctx, group, 0, native.prediction_losses)
        b, _ = probe.measure_case(rt, batch, ctx, group, 0, native.prediction_losses)
        c, _ = probe.measure_case(rt, batch, ctx, group, 2, native.prediction_losses)
    a.pop("timings")
    b.pop("timings")
    assert a == b
    assert c["auxiliary_loss"] != a["auxiliary_loss"]
    assert c["pairing_epoch"] == a["pairing_epoch"] == 1


@pytest.mark.parametrize("operation", ["step", "zero_grad", "backward", "teacher"])
def test_updates_and_teacher_fail_closed(operation):
    rt, _, _ = toy_runtime()
    with probe.forbidden_updates(rt) as calls, pytest.raises(ValueError, match="forbidden probe"):
        if operation in ("step", "zero_grad"):
            getattr(rt.optimizer, operation)()
        elif operation == "backward":
            rt.objective.student.gene.sum().backward()
        else:
            rt.objective.teacher(torch.ones(1))
    assert sum(calls.values()) == 1


@pytest.mark.parametrize("kind", ["split", "selection", "allowed", "condition", "steps", "anchor"])
def test_readonly_context_checks_native_constructor_predicates(kind):
    rt, bank, _ = toy_runtime()
    if kind == "split":
        bank.identity["split_content_sha256"] = "wrong"
    elif kind == "selection":
        bank.identity["selection"] = {}
    elif kind == "allowed":
        bank.identity["allowed_policy"] = {}
    elif kind == "condition":
        rt.data.split.train_conditions = ()
    elif kind == "steps":
        rt.data.steps_per_epoch = lambda **_kw: 149
    else:
        bank.condition_anchor_gene_indices[0] = 1
    with pytest.raises(ValueError):
        probe.bank_context(rt, bank)
    assert not hasattr(probe.bank_context, "backward")


def test_observer_rejects_aux_decode_and_prediction_anchor_head():
    rt, _, batch = toy_runtime()
    with probe.observe_case(rt) as seen:
        seen["phase"] = "auxiliary"
        with pytest.raises(ValueError, match="decode"):
            rt.objective.student.encode_response(
                torch.zeros(3, 256), batch.control, torch.zeros(40, 256), decode=True
            )
        seen["phase"] = "prediction"
        with pytest.raises(ValueError, match="anchor readout"):
            rt.objective.student.response_anchor_readout(torch.zeros(256))
    assert not rt.objective.student.prediction[0]._forward_pre_hooks


def test_native_mean_group_loss_not_average_cell_losses():
    from gradpert.training.v2.response_anchor import group_mean_loss

    head = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1)
    z = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]], requires_grad=True)
    target = torch.zeros(1)
    group = group_mean_loss(z, target, head)
    cell = (head(z) - target).square().mean()
    assert group.item() == 0 and cell.item() == 1
    assert not torch.equal(
        torch.autograd.grad(group, z, retain_graph=True)[0], torch.autograd.grad(cell, z)[0]
    )


def test_native_pairing_cannot_be_replaced_by_context_only_pairing():
    from gradpert.training.controls import TrainingControlPairer

    pairer = TrainingControlPairer(run_seed=1)
    kw = dict(
        epoch=1,
        perturbed_row_ids=["a", "b"],
        context_ids=["k", "k"],
        control_pools={"k": ["c1", "c2", "c3"]},
    )
    assert pairer.pair_epoch(**kw) == pairer.pair_epoch(**kw)
    prediction, truth = torch.tensor([1.0, 3.0]), torch.tensor([0.0, 4.0])
    assert F.mse_loss(prediction, truth) != F.mse_loss(prediction, truth.flip(0))


def test_spectra_rank_ceiling_direction_and_mean_energy_without_cancellation():
    values = np.vstack([np.eye(256)[:8], -np.eye(256)[:8]])
    summary = probe.spectra(values)
    assert summary["raw"]["centered_rank_ceiling"] == 15
    assert summary["raw"]["effective_rank"] == pytest.approx(8.0)
    assert summary["raw"]["mean_squared_norm"] == 0
    assert probe.spectrum(np.ones((16, 256)))["effective_rank"] == 0
    assert probe.spectrum(np.zeros((2, 256)))["relative_variance"] is None
    with pytest.raises(ValueError):
        probe.spectrum(np.array([[np.nan, 1.0]]))


def test_amplitude_only_geometry_is_exposed_by_direction_sensitivity():
    scale = np.arange(1.0, 9.0)
    truth = scale[:, None] * np.array([[1.0, 2.0, 3.0]])
    cls = scale[:, None] * np.array([[1.0, 1.0]])
    result = probe.amplitude_rsa(truth, cls, np.arange(3), np.full((8, 1), -1), draws=8)
    assert result["correlations"]["raw"]["value"] == pytest.approx(1.0)
    # All true directions are identical; floating residuals must not be called useful geometry.
    assert result["condition_N"] == 8 and result["pair_N"] == 28
    small = probe.amplitude_rsa(truth[:2], cls[:2], np.arange(3), np.full((2, 1), -1), draws=2)
    assert small["correlations"]["raw"]["value"] is None


def test_pilot_full_plan_has_no_half_group_or_duplicate_and_resource_failure():
    metadata = dict(groups=[dict(condition_id=f"g{i}") for i in range(16)])
    plan = probe.case_plan(metadata)
    assert plan[:2] == [(0, 0), (1, 0)] and len(plan) == len(set(plan)) == 48
    assert len(plan[2:]) == 46
    budgets = dict(
        max_total_seconds=100,
        max_peak_reserved_bytes=1000,
        max_cpu_rss_bytes=1000,
        min_free_cuda_bytes=10,
    )
    metrics = dict(
        completed_cases=2,
        elapsed_seconds=3,
        peak_reserved_bytes=500,
        cpu_rss_bytes=500,
        case_seconds_sum=3,
        free_cuda_bytes=100,
    )
    assert probe.resource_gate(metrics, budgets) == 72
    with pytest.raises(RuntimeError, match="projected"):
        probe.resource_gate(dict(metrics, elapsed_seconds=5, case_seconds_sum=5), budgets)
    with pytest.raises(RuntimeError, match="headroom"):
        probe.resource_gate(dict(metrics, free_cuda_bytes=9), budgets)


def test_recover_missing_bank_mapping_is_na_but_known_mismatch_fails():
    bank = SimpleNamespace(identity={})
    result, rows = probe.recover_bank_slots(bank, {}, make_plan=None, native_digest=probe.digest)
    assert rows is None and result["status"] == "unknown"
    bank.identity["metadata_sha256"] = "wrong"
    with pytest.raises(ValueError, match="known bank metadata"):
        probe.recover_bank_slots(bank, {}, make_plan=None, native_digest=probe.digest)


def test_strict_complete40_selection_and_native_ids():
    from gradpert.hashing import sha256_json
    from gradpert.training.controls import TrainingControlPairer

    ps = [f"g{i}" for i in range(17)]
    truth = [f"t{i}-{j}" for i in range(17) for j in range(40)]
    row_ids = [*truth, "c1", "c2"]
    conditions = [p for p in ps for _ in range(40)] + ["ctrl", "ctrl"]
    context_ids = ["k"] * len(row_ids)

    class Obs:
        def __getitem__(self, _index):
            return {"cell_type": "T"}

    obs = SimpleNamespace(iloc=Obs())
    data = SimpleNamespace(
        train_row_indices=list(range(680)),
        condition_ids=conditions,
        row_ids=row_ids,
        context_ids=context_ids,
        control_pools={"k": ["c1", "c2"]},
        _row_index={x: i for i, x in enumerate(row_ids)},
        batch_ids=["b"] * len(row_ids),
        _adata=SimpleNamespace(obs=obs),
    )
    bank = SimpleNamespace(
        condition_ids=ps,
        control_slot_offsets=np.arange(18) * 40,
        control_slot_indices=np.arange(680),
        condition_anchor_gene_indices=np.arange(17),
    )
    groups, candidates = probe.selected_groups(
        data, bank, pairer=TrainingControlPairer(run_seed=1), native_digest=sha256_json
    )
    assert candidates == 17 and len(groups) == 16
    assert all(len(g["records"]) == 40 for g in groups)
    assert {r["control_row_id"] for g in groups for r in g["records"]} <= {"c1", "c2"}
    data.train_row_indices = data.train_row_indices[:600]
    with pytest.raises(ValueError, match="fewer than sixteen"):
        probe.selected_groups(
            data, bank, pairer=TrainingControlPairer(run_seed=1), native_digest=sha256_json
        )


def test_byte_fingerprint_detects_parameter_change_without_restoring_it():
    rt, _, _ = toy_runtime()
    before = probe.constant_state(rt)
    with torch.no_grad():
        rt.objective.student.gene.add_(1)
    assert probe.constant_state(rt) != before


def test_allowed_first_read_whitelist_and_optional_bank_identity():
    class Obs:
        def __getitem__(self, _index):
            return {"cell_type": "T"}

    class Data:
        row_ids: ClassVar[list[str]] = ["t", "c", "unselected"]
        condition_ids: ClassVar[list[str]] = ["g", "ctrl", "g"]
        context_ids = ["k"] * 3
        batch_ids = ["b"] * 3
        train_row_indices: ClassVar[list[int]] = [0, 2]
        control_row_indices: ClassVar[list[int]] = [1]
        _row_index: ClassVar[dict[str, int]] = {"t": 0, "c": 1, "unselected": 2}
        _adata = SimpleNamespace(obs=SimpleNamespace(iloc=Obs()))
        manifest = SimpleNamespace(n_expression_genes=5)
        pipeline_stats = SimpleNamespace()

        def configure_training_expression_columns(self, columns):
            self.columns = list(columns)

        def _read_expression_indices(self, indices):
            assert self.columns == [0, 1, 3]
            result = np.zeros((len(indices), 5), dtype=np.float32)
            result[:, self.columns] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            return result

    record = dict(
        truth_row_id="t",
        control_row_id="c",
        condition_id="g",
        cell_line="Jurkat",
        cell_type="T",
        context_id="k",
        batch="b",
    )
    rt = SimpleNamespace(data=Data(), allowed_expression_ids=np.array([0, 1, 3]))
    metadata = dict(groups=[dict(records=[record])], bank_slot_identity=dict(reason="missing"))
    bank = SimpleNamespace(control_expression=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    with probe.train_read_scope(rt, metadata, bank) as receipt:
        actual = rt.data._read_expression_indices([0, 1])
        assert actual.shape == (2, 5) and not actual[:, [2, 4]].any()
        assert receipt["bank_slot_row_ids_verified"] is False
        with pytest.raises(ValueError, match="whitelist"):
            rt.data._read_expression_indices([2])
    metadata["bank_slot_identity"] = dict(selected_control_row_ids=["c"], selected_bank_indices=[0])
    with probe.train_read_scope(
        SimpleNamespace(data=Data(), allowed_expression_ids=np.array([0, 1, 3])), metadata, bank
    ) as receipt:
        assert receipt["bank_slot_row_ids_verified"] is True
    bank.control_expression[0, 0] = 4
    with (
        pytest.raises(ValueError, match="expression mismatch"),
        probe.train_read_scope(
            SimpleNamespace(data=Data(), allowed_expression_ids=np.array([0, 1, 3])), metadata, bank
        ),
    ):
        pass


def test_spec_hash_protocol_output_and_checkpoint_checks(tmp_path, monkeypatch):
    source = tmp_path / "native"
    analysis = tmp_path / "analysis"
    source.mkdir()
    analysis.mkdir()

    def binding(path, value):
        path.write_text(value)
        return dict(path=str(path), sha256=probe.sha(path))

    publication = binding(tmp_path / "publication.json", "{}")
    config = binding(tmp_path / "config.json", "{}")
    checkpoint = binding(tmp_path / "checkpoint.pt", "synthetic checksum fixture")
    monkeypatch.setattr(probe, "CP_SHA", checkpoint["sha256"])
    document = dict(
        schema=probe.SCHEMA,
        analysis_source=dict(root=str(analysis), commit="b" * 40, publication=publication),
        endpoint=dict(
            name="R14-FRGA",
            source=dict(root=str(source), commit=probe.NATIVE_SHA, publication=publication),
            run_root=str(tmp_path / "run"),
            run_manifest=config,
            journal=config,
            config=config,
            checkpoint=checkpoint,
        ),
        data_root=str(tmp_path / "data"),
        resource_policy=config,
        budgets=dict(
            max_total_seconds=100,
            max_peak_reserved_bytes=1000,
            max_cpu_rss_bytes=1000,
            min_free_cuda_bytes=10,
        ),
        output=str(tmp_path / "probe"),
        protocol=copy.deepcopy(probe.PROTOCOL),
    )
    probe.validate_spec(document, server=False)
    bad = copy.deepcopy(document)
    bad["protocol"]["rows"] = 20
    with pytest.raises(ValueError, match="protocol"):
        probe.validate_spec(bad, server=False)
    bad = copy.deepcopy(document)
    bad["output"] = str(source / "outputs")
    with pytest.raises(ValueError, match="overlaps"):
        probe.validate_spec(bad, server=False)
    Path(checkpoint["path"]).write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        probe.validate_spec(document, server=False)


def test_all_module_clock_and_native_pilot_scalar_parity():
    import calibrate_response_anchor as native

    rt, bank, batch = toy_runtime()
    before = probe.constant_state(rt)
    rt.objective.student.options_extra_clock = 1
    assert probe.constant_state(rt) != before
    ctx, group = probe.bank_context(rt, bank), dict(condition_id="g0", bank_condition_index=0)
    with native.preserved_calibration(rt), probe.forbidden_updates(rt) as calls:
        reference = probe.native_reference_case(rt, batch, ctx, group, 0, native.prediction_losses)
        assert not calls
        row, _ = probe.measure_case(rt, batch, ctx, group, 0, native.prediction_losses)
        parity = probe.scalar_parity(reference, row)
        assert parity["passed"] and parity["max_abs"] == 0
        assert reference["extra_native_forwards"] == dict(prediction=1, auxiliary=1)
        with pytest.raises(RuntimeError, match="parity"):
            probe.scalar_parity(reference, dict(row, prediction_loss=row["prediction_loss"] + 1e-4))


def test_terminal_exact_file_closure_and_missing_required_fields(tmp_path, monkeypatch):
    document = dict(output=str(tmp_path))
    metadata = dict(groups=[dict(condition_id=f"g{i}") for i in range(16)])
    sealed = dict(metadata_sha256="metadata")
    monkeypatch.setattr(probe, "validate_sealed", lambda _s: (sealed, metadata))
    for name in probe.REQUIRED_OUTPUTS:
        probe.write_new(tmp_path / name, {})
    rows = [
        dict(condition_id=metadata["groups"][i]["condition_id"], aux_rng_epoch_label=e)
        for i, e in probe.case_plan(metadata)
    ]
    (tmp_path / "cases.json").write_text(probe.json.dumps(rows))
    for number, row in enumerate(rows, 1):
        probe.persist_case_scalar(tmp_path, number, row, probe.digest(document))
    parity = dict(
        passed=True,
        atol=2e-6,
        max_abs=0.0,
        reference=dict(
            state_before_sha256="same",
            state_after_sha256="same",
            extra_native_forwards=dict(prediction=1, auxiliary=1),
        ),
    )
    pilot = dict(
        status="passed",
        complete_groups=2,
        rows_per_group=40,
        replicates=1,
        remaining_cases=46,
        original_scalar_parity=[parity, parity],
    )
    (tmp_path / "PILOT.json").write_text(probe.json.dumps(pilot))
    result = dict(
        spec_digest=probe.digest(document),
        metadata_sha256="metadata",
        updates=0,
        state_unchanged=True,
        scientific_complete=False,
        forbidden_calls=dict(autograd_grad=144),
        native_forward_totals=dict(
            prediction=50,
            auxiliary=50,
            formal_cases_each_branch=48,
            extra_pilot_original_each_branch=2,
        ),
        output_sha256={name: probe.sha(tmp_path / name) for name in probe.expected_outputs()},
    )
    probe.write_new(tmp_path / "COMPLETE.json", result)
    assert probe.validate_result(document)["status"] == "validated"
    probe.write_new(tmp_path / "extra.json", {})
    with pytest.raises(ValueError, match="closure"):
        probe.validate_result(document)
    (tmp_path / "extra.json").unlink()
    case_path = tmp_path / probe.case_scalar_path(1)
    original_case = case_path.read_text()
    original_hash = result["output_sha256"][probe.case_scalar_path(1)]
    altered = probe.read(case_path)
    altered["row"]["prediction_loss"] = 999.0
    case_path.write_text(probe.json.dumps(altered))
    result["output_sha256"][probe.case_scalar_path(1)] = probe.sha(case_path)
    (tmp_path / "COMPLETE.json").write_text(probe.json.dumps(result))
    with pytest.raises(ValueError, match="exact aggregate row"):
        probe.validate_result(document)
    case_path.write_text(original_case)
    result["output_sha256"][probe.case_scalar_path(1)] = original_hash
    (tmp_path / "COMPLETE.json").write_text(probe.json.dumps(result))
    case_path.unlink()
    with pytest.raises(ValueError, match="closure"):
        probe.validate_result(document)
    case_path.write_text(original_case)
    changed = dict(result)
    changed.pop("state_unchanged")
    (tmp_path / "COMPLETE.json").write_text(probe.json.dumps(changed))
    with pytest.raises(ValueError, match="zero-update"):
        probe.validate_result(document)
    (tmp_path / "COMPLETE.json").write_text(probe.json.dumps(result))
    (tmp_path / "readout_spectrum.json").unlink()
    with pytest.raises(ValueError, match="closure"):
        probe.validate_result(document)


def test_sealed_metadata_modification_fails(tmp_path, monkeypatch):
    document = dict(output=str(tmp_path))
    monkeypatch.setattr(probe, "validate_spec", lambda _s: None)
    probe.write_new(tmp_path / "SPEC.json", document)
    probe.write_new(tmp_path / "METADATA.json", dict(genuine="members"))
    probe.write_new(
        tmp_path / "SEALED.json",
        dict(
            spec_digest=probe.digest(document),
            metadata_sha256=probe.sha(tmp_path / "METADATA.json"),
        ),
    )
    assert probe.validate_sealed(document)[1] == dict(genuine="members")
    (tmp_path / "METADATA.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        probe.validate_sealed(document)


@pytest.mark.parametrize(
    "exitcode,identity_kind", [(0, "valid"), (3, "valid"), (0, "wrong"), (0, "none")]
)
def test_dispatch_actual_synthetic_subprocess_owner_race_and_external_logs(
    tmp_path, monkeypatch, exitcode, identity_kind
):
    import shared_gpu

    output = tmp_path / "native-result"
    output.mkdir()
    document = dict(
        output=str(output),
        analysis_source=dict(root=str(tmp_path)),
        endpoint=dict(source=dict(root=str(tmp_path / "native-source"))),
    )
    monkeypatch.setattr(probe, "validate_sealed", lambda _s: ({}, {}))
    monkeypatch.setattr(probe, "verify_sources", lambda _s: dict(verified=True))
    monkeypatch.setattr(probe, "preload_analysis_dependencies", lambda _s: {})
    monkeypatch.setattr(probe, "validate_result", lambda _s: dict(status="validated"))
    monkeypatch.setattr(probe, "SERVER_PYTHON", sys.executable)
    lookups = {}

    def lookup(pid):
        lookups[pid] = lookups.get(pid, 0) + 1
        if pid != probe.os.getpid() and identity_kind == "none":
            return None
        if pid != probe.os.getpid() and lookups[pid] == 1:
            return None
        command_hash = "a" * 64
        if pid != probe.os.getpid() and identity_kind == "valid":
            command_hash = probe.hashlib.sha256(
                b"\0".join(probe.os.fsencode(a) for a in spawned[0]) + b"\0"
            ).hexdigest()
        return dict(
            pid=pid,
            uid=probe.os.getuid(),
            start_ticks="123",
            command_sha256=command_hash,
            script_basename="response_anchor_gradient_probe.py",
            ppid=probe.os.getpid(),
        )

    monkeypatch.setattr(shared_gpu, "process_info", lookup)
    original = subprocess.Popen
    spawned = []

    def spawn(command, **kwargs):
        spawned.append(command)
        assert kwargs["shell"] is False
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
        native = str(tmp_path / "native-source")
        assert kwargs["env"]["PYTHONPATH"] == probe.os.pathsep.join([native + "/src", native])
        return original(
            [
                sys.executable,
                "-c",
                f"import time; print('synthetic'); time.sleep(.25); raise SystemExit({exitcode})",
            ],
            **kwargs,
        )

    monkeypatch.setattr(probe.subprocess, "Popen", spawn)
    spec_file = dict(path=str(tmp_path / "input.json"), sha256="b" * 64)
    if exitcode or identity_kind != "valid":
        message = "child failed" if exitcode else "child identity failed"
        with pytest.raises(RuntimeError, match=message):
            probe.dispatch(document, spec_file=spec_file, gpu="0")
    else:
        assert probe.dispatch(document, spec_file=spec_file, gpu="0")["returncode"] == 0
    controller = output.with_name(output.name + "-dispatch")
    owner = probe.read(controller / "OWNER.json")
    launch = probe.read(controller / "LAUNCH.json")
    assert owner["process"]["pid"] == probe.os.getpid()
    assert (launch["child_identity"] is not None) == (identity_kind == "valid")
    assert lookups[launch["pid"]] >= 2
    assert (controller / "stdout.log").read_text() == "synthetic\n"
    assert not list(output.iterdir()) and len(spawned) == 1
    if exitcode or identity_kind != "valid":
        assert probe.read(controller / "FAILURE.json")["automatic_retry"] is False
    else:
        with pytest.raises(FileExistsError):
            probe.dispatch(document, spec_file=spec_file, gpu="0")
        assert not (controller / "FAILURE.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("uid", -1),
        ("pid", -1),
        ("ppid", -1),
        ("start_ticks", "bad"),
        ("script_basename", "wrong.py"),
    ],
)
def test_child_identity_requires_all_live_identity_components(key, value):
    command = ["python", "response_anchor_gradient_probe.py", "run"]
    observed = dict(
        pid=7,
        uid=probe.os.getuid(),
        ppid=5,
        start_ticks="123",
        script_basename="response_anchor_gradient_probe.py",
        command_sha256=probe.hashlib.sha256(
            b"\0".join(probe.os.fsencode(a) for a in command) + b"\0"
        ).hexdigest(),
    )
    observed[key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        probe.validated_child_identity(observed, pid=7, command=command, parent_pid=5)


def test_real_manifold_first_forward_is_telemetry_not_unknown_clock():
    from gradpert.modeling.v2.operators import ManifoldResidual

    rt, _, _ = toy_runtime()
    layer = ManifoldResidual(8, 2, nn.Identity())
    rt.objective.student.derived_backend_probe = layer
    before = probe.constant_state(rt)
    assert layer.resolved_sinkhorn_backend == "unused"
    layer(torch.randn(2, 3, 2, 8))
    assert layer.resolved_sinkhorn_backend == "native"
    assert probe.constant_state(rt) == before
    assert "native" in probe.derived_telemetry(rt).values()
    rt.objective.student.resolved_sinkhorn_backend = "unused"
    before = probe.constant_state(rt)
    rt.objective.student.resolved_sinkhorn_backend = "native"
    assert probe.constant_state(rt) != before


def test_actual_relay_randomize_order_restored_beyond_native_context():
    import calibrate_response_anchor as native

    from gradpert.modeling.v2.operators import RelayDeltaAttention

    rt, _, _ = toy_runtime()
    attention = RelayDeltaAttention(8, 2)
    rt.objective.student.actual_relay_probe = attention
    attention.randomize_order = False
    with native.preserved_calibration(rt):
        attention.randomize_order = True
    # Current calibration now preserves this flag too. Frozen R14 native code
    # did not; the analysis wrapper must still tolerate that historical context.
    assert attention.randomize_order is False
    attention.randomize_order = False
    before = probe.constant_state(rt)
    with probe.preserved_probe(rt, lambda runtime: contextlib.nullcontext()):
        attention.randomize_order = True
    assert attention.randomize_order is False
    assert probe.constant_state(rt) == before
    rt.objective.student.unknown_clock = 0
    with (
        pytest.raises(RuntimeError, match="unknown clocks"),
        probe.preserved_probe(rt, native.preserved_calibration),
    ):
        rt.objective.student.unknown_clock += 1


def test_isolated_original_native_without_geometry_runs_analysis_geometry(tmp_path):
    native_src = tmp_path / "native" / "src"
    evaluation = native_src / "gradpert/evaluation"
    evaluation.mkdir(parents=True)
    (evaluation.parent / "__init__.py").write_text("IDENTITY = 'frozen-original'\n")
    (evaluation / "__init__.py").write_text("")
    code = """
import importlib.util, pathlib, sys
import numpy as np
import gradpert, gradpert.evaluation
assert gradpert.IDENTITY == 'frozen-original'
assert importlib.util.find_spec('gradpert.evaluation.response_geometry') is None
native_file, native_path = gradpert.__file__, list(gradpert.__path__)
native_modules = {k: getattr(v, '__file__', None) for k, v in sys.modules.items()
                  if k == 'gradpert' or k.startswith('gradpert.')}
worker = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('isolated_train40_worker', worker)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
binding = {'analysis_source': {'root': str(worker.parents[2])}}
preloaded = probe.preload_analysis_dependencies(binding)
expected_helper = worker.parents[2] / 'src/gradpert/evaluation/response_geometry.py'
assert pathlib.Path(preloaded['response_geometry']['path']) == expected_helper
helper = probe.geometry_module()
assert helper.WeightedRanks.__module__ == helper.__name__
assert sys.modules[helper.__name__] is helper
probe.PROTOCOL = dict(probe.PROTOCOL, bootstrap_draws=2)
metadata = {'groups': [{'condition_id': f'g{i}', 'anchor_gene_index': i+100} for i in range(16)],
            'query_indices': [0, 1, 2]}
rng = np.random.default_rng(7)
truth, cls = rng.normal(size=(16, 3)), rng.normal(size=(16, 4))
arrays = {(i, e): dict(truth_delta=truth[i], prediction_delta=truth[i]+.01,
                       aux_cls=cls[i], pred_cls=cls[i]+.1, readout=cls[i]-.1)
          for i in range(16) for e in (0, 1, 2)}
result = probe.small_geometry(metadata, arrays)
assert set(result) == {'0', '1', '2'}
assert result['0']['response_RSA']['finite_pairs'] == 120
assert result['0']['amplitude_sensitivity']['condition_N'] == 16
assert result['0']['amplitude_sensitivity']['bootstrap_draws'] == 2
assert gradpert.__file__ == native_file and list(gradpert.__path__) == native_path
assert {k: getattr(v, '__file__', None) for k, v in sys.modules.items()
        if k == 'gradpert' or k.startswith('gradpert.')} == native_modules
assert 'gradpert.evaluation.response_geometry' not in sys.modules
print('isolated-analysis-geometry-and-native-namespace-ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPTS / "response_anchor_gradient_probe.py")],
        cwd=tmp_path,
        env=dict(probe.os.environ, PYTHONPATH=str(native_src)),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "isolated-analysis-geometry-and-native-namespace-ok"


def test_geometry_failure_keeps_scalar_cases_without_latent_resume_cache(tmp_path, monkeypatch):
    rows = [
        dict(
            condition_id=f"g{i}",
            bank_condition_index=i,
            aux_rng_epoch_label=0,
            prediction_loss=0.1 + i,
            auxiliary_loss=0.2 + i,
        )
        for i in range(2)
    ]
    for number, row in enumerate(rows, 1):
        probe.persist_case_scalar(tmp_path, number, row, "synthetic-spec")

    def failed_geometry(*_args):
        raise ImportError("synthetic late geometry failure")

    monkeypatch.setattr(probe, "small_geometry", failed_geometry)
    with pytest.raises(ImportError, match="late geometry"):
        probe.geometry_after_scalar_aggregate(tmp_path, rows, {}, {})
    assert probe.read(tmp_path / "cases.json") == rows
    assert [probe.read(tmp_path / probe.case_scalar_path(i))["row"] for i in (1, 2)] == rows
    assert not (tmp_path / "COMPLETE.json").exists()
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.pkl"))
    with pytest.raises(ValueError, match="fields"):
        probe.persist_case_scalar(tmp_path, 3, dict(rows[0], raw_latent=[[1, 2]]), "spec")
    with pytest.raises(ValueError, match="arrays"):
        probe.persist_case_scalar(tmp_path, 3, dict(rows[0], gradient=np.ones((2, 2))), "spec")
    with pytest.raises(ValueError, match="arrays"):
        probe.persist_case_scalar(tmp_path, 3, dict(rows[0], gradient=[[1, 2], [3, 4]]), "spec")


@pytest.mark.parametrize("mode", ["seal", "run", "dispatch"])
def test_dependency_preload_fails_before_model_or_gpu_or_child(tmp_path, monkeypatch, mode):
    from gradpert.training.v2 import runtime as native_runtime

    def fail_if_used(*_args, **_kwargs):
        raise AssertionError("GPU/model/subprocess must not start before dependency check")

    def missing_dependency(_spec):
        raise ImportError("missing analysis geometry")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe, "validate_spec", lambda _s: None)
    monkeypatch.setattr(probe, "validate_sealed", lambda _s: ({}, {}))
    monkeypatch.setattr(probe, "verify_sources", lambda _s: {})
    monkeypatch.setattr(probe, "preload_analysis_dependencies", missing_dependency)
    monkeypatch.setattr(native_runtime, "prepare_runtime", fail_if_used)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_used)
    monkeypatch.setattr(probe.subprocess, "Popen", fail_if_used)
    with pytest.raises(ImportError, match="analysis geometry"):
        if mode == "seal":
            probe.seal({}, spec_file={})
        elif mode == "run":
            probe.run({}, owner_binding={})
        else:
            probe.dispatch({}, spec_file={}, gpu="0")
