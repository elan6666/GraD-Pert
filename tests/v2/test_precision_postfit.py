"""Synthetic precision postfit identity and arithmetic contracts; no CUDA or scientific data."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_components import fixture
from test_response_anchor_postfit import m
from test_response_anchor_r15_postfit import retained_fixture
from test_restored_ffn import restored

from gradpert.config import load_experiment_config
from gradpert.execution.v2_training_stage import seal_training_stage
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.objective import JointObjective

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_precision_family import BASELINE, CHANGES, FAMILY, SCHEMA, generate  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return bound(path)


def bound(path):
    return dict(path=str(path), sha256=m.sha(path))


def precision_fixture(tmp_path, arm="PREC-SSL32"):
    old, references = retained_fixture(tmp_path / "retained")
    source, directory, data = (tmp_path / n for n in ("native", "queue", "data"))
    data.mkdir()
    baseline = source / BASELINE
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes((ROOT / BASELINE).read_bytes())
    matrix = generate(source, source / "configs/v2" / FAMILY)
    publication = write(source / "publication.json", {"published_commit": m.PRECISION_SOURCE})
    rows = []
    for leaf in matrix["rows"]:
        config = source / leaf["config"]
        resolved = load_experiment_config(config).model_dump(mode="json")
        root = tmp_path / "runs" / leaf["name"]
        plan = dict(
            config=str(config),
            config_sha256=leaf["sha256"],
            repository_root=str(source),
            source_commit=m.PRECISION_SOURCE,
            run_id=leaf["name"] + "-synthetic",
            run_root=str(root),
            test_root=str(tmp_path / "tests" / leaf["name"]),
            data_root=str(data),
            resolved_config=resolved,
        )
        rows.append(dict(name=leaf["name"], plan=plan))
        write(directory / f"{leaf['name']}.launch-plan.json", plan)
    queue = dict(
        schema="unified-ddp-queue-1",
        family=SCHEMA,
        mode=SCHEMA,
        source=str(source),
        source_commit=m.PRECISION_SOURCE,
        manifest=str(source / "configs/v2" / FAMILY / "manifest.json"),
        manifest_sha256=m.sha(source / "configs/v2" / FAMILY / "manifest.json"),
        output=str(directory),
        rows=rows,
        steps=10,
        preflight_only=True,
        postfit_mode="main_owned_precision_postfit",
        precision_training_budget=m.PRECISION_BUDGET,
        shared_gpu_policy={"synthetic_original_policy": True},
    )
    qb = write(directory / "queue.json", queue)
    plan = next(row["plan"] for row in rows if row["name"] == arm)
    run = Path(plan["run_root"])
    write(run / "launch.json", plan)
    write(run / "resolved_config.json", plan["resolved_config"])
    identity = dict(
        run_id=plan["run_id"],
        config_sha256=plan["config_sha256"],
        resolved_config_sha256=m.digest(plan["resolved_config"]),
        source=dict(commit=m.PRECISION_SOURCE, published_commit=m.PRECISION_SOURCE, dirty=False),
        data=dict(run_seed=1),
    )
    ib = write(run / "run_manifest.json", identity)
    # Hash-only placeholder, never deserialized or passed to a native model.
    cp = run / "fit/last.pt"
    cp.parent.mkdir()
    cp.write_bytes(b"synthetic-checkpoint-not-weights")
    last = dict(epoch=2, file="last.pt", sha256=m.sha(cp))
    jb = write(
        run / "fit/epoch_state.json",
        dict(identity=identity, epoch=2, budget=[2, 150], best=None, last=last),
    )
    write(
        run / "fit/history.json",
        [
            dict(
                epoch=i,
                optimizer_steps=i * 150,
                validation=dict(validation_mode="disabled", performed=False),
            )
            for i in (1, 2)
        ],
    )
    seal_training_stage(run, identity, 2)
    fit = dict(
        schema="precision-fit-complete-1",
        arm=arm,
        queue_sha256=qb["sha256"],
        source_commit=m.PRECISION_SOURCE,
        launch_plan_sha256=m.sha(directory / f"{arm}.launch-plan.json"),
        train_complete=dict(
            receipt=str(run / "TRAIN_COMPLETE.json"), sha256=m.sha(run / "TRAIN_COMPLETE.json")
        ),
        last=last,
        budget=m.PRECISION_BUDGET,
        scientific_complete=False,
        postfit_mode="main_owned_precision_postfit",
        test_status="deferred_to_main",
    )
    fb = write(directory / f"FIT_COMPLETE.{arm}.json", fit)
    state = dict(
        status="fit_complete",
        phase="awaiting_main_precision_postfit",
        active=None,
        scientific_complete=False,
        formal_fit_authorized=False,
        postfit_mode="main_owned_precision_postfit",
        test_status="deferred_to_main",
        precision_fits={arm: dict(receipt=fb["path"], sha256=fb["sha256"])},
    )
    # An immutable snapshot; later promotion may update the live controller's state.json.
    sb = write(tmp_path / "sealed-controller-state.json", state)
    endpoint = dict(
        name=arm,
        source=dict(root=str(source), commit=m.PRECISION_SOURCE, publication=publication),
        run_root=str(run),
        run_manifest=ib,
        journal=jb,
        config=bound(Path(plan["config"])),
        checkpoint=bound(cp),
    )
    binding = dict(
        family_manifest=bound(Path(queue["manifest"])),
        queue=qb,
        fit_complete=fb,
        controller_state=sb,
    )
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        analysis_source=old["analysis_source"],
        data_root=str(data),
        shared_policy=write(tmp_path / "policy.json", queue["shared_gpu_policy"]),
        output=str(tmp_path / "new-output"),
        validation=old["validation"],
        endpoints=[endpoint],
        bootstrap=old["bootstrap"],
        medium={},
        precision=binding,
        evaluation_precision="common_ri_native",
        retained_references={"R3-RI": references["R3-RI"]},
    )
    return spec, queue


@pytest.mark.parametrize("arm", m.PRECISION_ARMS)
def test_precision_exact_published_family_and_terminal_fit(tmp_path, arm):
    spec, queue = precision_fixture(tmp_path, arm)
    m.validate_spec(spec, server=False)
    assert (
        m.validate_precision_identity(spec["endpoints"][0], spec["precision"], server=False)
        == queue
    )
    assert [e["name"] for e in spec["endpoints"]] == [arm]
    assert spec["evaluation_precision"] == "common_ri_native"


@pytest.mark.parametrize(
    "case",
    [
        "arm",
        "source",
        "dirty",
        "config",
        "journal",
        "history",
        "budget",
        "fit",
        "fit_hash",
        "state",
        "state_active",
        "last",
        "resolved",
        "baseline",
        "test",
        "policy",
    ],
)
def test_precision_mutations_fail_before_any_model_or_baseline_forward(tmp_path, case):
    spec, _ = precision_fixture(tmp_path)
    endpoint, binding = spec["endpoints"][0], spec["precision"]
    if case == "arm":
        endpoint["name"] = "R3-RI"
    elif case == "source":
        endpoint["source"]["commit"] = "b" * 40
    elif case in ("dirty", "resolved"):
        value = json.loads(Path(endpoint["run_manifest"]["path"]).read_text())
        if case == "dirty":
            value["source"]["dirty"] = True
        else:
            value["resolved_config_sha256"] = "b" * 64
        endpoint["run_manifest"] = write(Path(endpoint["run_manifest"]["path"]), value)
    elif case == "config":
        Path(endpoint["config"]["path"]).write_text("changed")
    elif case in ("journal", "budget", "last"):
        value = json.loads(Path(endpoint["journal"]["path"]).read_text())
        if case == "journal":
            value["identity"]["run_id"] = "other"
        elif case == "budget":
            value["budget"] = [2, 149]
        else:
            value["last"]["epoch"] = 1
        endpoint["journal"] = write(Path(endpoint["journal"]["path"]), value)
    elif case == "history":
        path = Path(endpoint["run_root"]) / "fit/history.json"
        value = json.loads(path.read_text())
        value[-1]["optimizer_steps"] = 299
        write(path, value)
    elif case in ("fit", "fit_hash"):
        value = json.loads(Path(binding["fit_complete"]["path"]).read_text())
        value["budget"]["completed_updates"] = 299
        updated = write(Path(binding["fit_complete"]["path"]), value)
        if case == "fit":
            binding["fit_complete"] = updated
    elif case in ("state", "state_active"):
        value = json.loads(Path(binding["controller_state"]["path"]).read_text())
        if case == "state":
            value["precision_fits"][endpoint["name"]]["sha256"] = "b" * 64
        else:
            value["active"] = {"phase": "fit"}
        binding["controller_state"] = write(Path(binding["controller_state"]["path"]), value)
    elif case == "baseline":
        spec["endpoints"].append(copy.deepcopy(endpoint))
        spec["endpoints"][-1]["name"] = "R3-RI"
    elif case == "test":
        spec["validation"]["split"] = "test"
    elif case == "policy":
        spec["evaluation_precision"] = "native"
    with pytest.raises((ValueError, KeyError)):
        m.validate_spec(spec, server=False)


@pytest.mark.parametrize("location", ["queue", "run", "test", "other_run", "output"])
def test_precision_zero_pkl_checks_every_root_without_deleting(tmp_path, location):
    spec, queue = precision_fixture(tmp_path)
    selected, other = queue["rows"][:2]
    root = {
        "queue": queue["output"],
        "run": selected["plan"]["run_root"],
        "test": selected["plan"]["test_root"],
        "other_run": other["plan"]["run_root"],
        "output": spec["output"],
    }[location]
    polluted = Path(root) / "nested/evidence.pkl"
    polluted.parent.mkdir(parents=True, exist_ok=True)
    polluted.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="zero-PKL"):
        m.validate_spec(spec, server=False)
    assert polluted.read_bytes() == b"preserve"


@pytest.mark.parametrize("arm", m.PRECISION_ARMS)
def test_common_student_is_independent_and_owner_teacher_rng_remain_exact(arm):
    original = restored("BR")[0].student if arm == "PREC-FFN8" else fixture()[0]
    options = replace(
        original.options,
        **dict(CHANGES[arm], **({"ffn_variant": "gelu"} if arm == "PREC-FFN8" else {})),
    )
    objective = JointObjective(GraDPertV2(original.graph.embedding.weight.detach(), options)).eval()
    before = {key: value.clone() for key, value in objective.state_dict().items()}
    rng = torch.get_rng_state()
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        evaluation_precision=m.PRECISION_POLICY,
        endpoints=[dict(name=arm)],
    )
    student, metadata = m.evaluation_model(spec, objective, torch.device("cpu"))
    assert student is not objective.student
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in objective.state_dict().items():
        assert torch.equal(value, before[key])
    m.validate_precision_receipt(metadata["precision_policy"], arm)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with m.precision_context(spec, "cpu"):
            assert not torch.is_autocast_enabled("cpu")
            assert student.ssl2_cls(torch.randn(2, options.width)).dtype == torch.float32
        assert torch.is_autocast_enabled("cpu")
    assert not any(p.requires_grad for p in objective.teacher.parameters())


def test_historical_context_and_student_defaults_are_unchanged():
    model, _ = fixture()
    objective = SimpleNamespace(student=model)
    selected, metadata = m.evaluation_model(dict(schema=m.SCHEMA), objective, torch.device("cpu"))
    assert selected is model and metadata == {}
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        m.precision_context(dict(schema=m.SCHEMA), "cpu"),
    ):
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(
    "field", ["evaluation_precision", "training", "evaluation", "autocast_enabled", "backend"]
)
def test_precision_policy_receipt_cannot_be_relabeled(field):
    model, _ = fixture()
    objective = JointObjective(
        GraDPertV2(
            model.graph.embedding.weight.detach(),
            replace(model.options, ssl_head_precision="float32"),
        )
    ).eval()
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        evaluation_precision=m.PRECISION_POLICY,
        endpoints=[dict(name="PREC-SSL32")],
    )
    _, metadata = m.evaluation_model(spec, objective, torch.device("cpu"))
    value = metadata["precision_policy"]
    value[field] = "native"
    with pytest.raises(ValueError):
        m.validate_precision_receipt(value, "PREC-SSL32")


def test_observer_and_native_evaluator_share_separate_common_student_under_outer_autocast():
    from contextlib import contextmanager

    model, _ = fixture()
    objective = JointObjective(
        GraDPertV2(
            model.graph.embedding.weight.detach(),
            replace(model.options, prediction_precision="float32"),
        )
    ).eval()
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        evaluation_precision=m.PRECISION_POLICY,
        endpoints=[dict(name="PREC-PRED32")],
    )
    student, _ = m.evaluation_model(spec, objective, torch.device("cpu"))
    observed = []

    @contextmanager
    def observe(native, selected, data):
        observed.append(selected)
        yield
        observed.append("restored")

    def evaluate(selected, *args, **kwargs):
        assert selected is observed[0] is student
        assert selected is not objective.student
        assert not torch.is_autocast_enabled("cpu")
        return selected.ssl2_cls(torch.ones(2, selected.options.width)).dtype

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = m.evaluate_observed(
            spec,
            SimpleNamespace(evaluate=evaluate),
            student,
            SimpleNamespace(observe=observe),
            None,
            None,
            None,
            device=torch.device("cpu"),
        )
        assert torch.is_autocast_enabled("cpu")
    assert result == torch.float32 and observed[-1] == "restored"


def completed_precision_fixture(tmp_path, monkeypatch):
    """Real merge/compare/validation; only CUDA workers and costly statistics are synthetic."""
    import shutil

    from test_response_anchor_postfit import synthetic_arrays
    from test_response_anchor_r15_postfit import real_retained_fixture

    from gradpert.evaluation.response_geometry import compare_endpoints

    spec, _ = precision_fixture(tmp_path / "candidate")
    _, reference_binding, donor = real_retained_fixture(tmp_path / "reference")
    spec["retained_references"]["R3-RI"] = reference_binding
    root = Path(spec["output"])
    name = spec["endpoints"][0]["name"]
    endpoint = spec["endpoints"][0]
    sb = write(tmp_path / "INPUT_SPEC.json", spec)
    write(root / "SPEC.json", spec)
    owner = write(root / name / "DISPATCH_OWNER.json", dict(synthetic_owner=True))
    policy = dict(
        evaluation_precision=m.PRECISION_POLICY,
        training=dict(
            ffn_compute_precision="autocast",
            ssl_head_precision="float32",
            prediction_precision="autocast",
        ),
        evaluation=dict(
            ffn_compute_precision="autocast",
            ssl_head_precision="autocast",
            prediction_precision="autocast",
        ),
        autocast_enabled=False,
        autocast_dtype=None,
        parameter_dtype="float32",
        backend=dict(
            cuda_matmul_allow_tf32=False, float32_matmul_precision="highest", cudnn_allow_tf32=True
        ),
    )
    for i in (0, 1):
        worker = root / name / f"worker-{i}"
        shutil.copytree(donor / f"worker-{i}", worker)
        arrays = bound(worker / "PRIVATE_ENDPOINT.npz")
        native = json.loads((worker / "PRIVATE_NATIVE_RESULT.json").read_text())
        # Donor is the historical seen-only unit fixture. Add the required unseen
        # axis on both row and aggregate levels at this explicit synthetic boundary.
        native["metric_gene_groups"]["unseen_expression"] = copy.deepcopy(
            native["metric_gene_groups"]["seen_expression"]
        )
        native["metric_gene_groups"]["unseen_expression"]["gene_count"] = 225
        for row in native["conditions"]:
            for metrics in (row["metrics"], row["metric_gene_groups"]["seen_expression"]):
                deg = copy.deepcopy(metrics["results"])
                for metric in deg:
                    metric["metric_id"] = metric["metric_id"].replace("_all", "_deg")
                metrics["results"].extend(deg)
            row["metric_gene_groups"]["unseen_expression"] = copy.deepcopy(
                row["metric_gene_groups"]["seen_expression"]
            )
        from gradpert.training.v2.evaluation import merge_evaluation_shards

        native = merge_evaluation_shards(
            [native], ordered_conditions=tuple(r["condition_id"] for r in native["conditions"])
        )
        native_binding = write(worker / "PRIVATE_NATIVE_RESULT.json", native)
        receipt = json.loads((worker / "COMPLETE.json").read_text())
        inputs = [
            sb,
            spec["shared_policy"],
            spec["analysis_source"]["publication"],
            endpoint["source"]["publication"],
            owner,
            *[endpoint[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
            *spec["precision"].values(),
        ]
        receipt.update(
            endpoint=name,
            postfit_spec_sha256=m.digest(spec),
            endpoint_spec_sha256=m.digest(endpoint),
            source=dict(commit=m.PRECISION_SOURCE, dirty=False),
            analysis_source=dict(commit=spec["analysis_source"]["commit"], dirty=False),
            checkpoint_sha256=endpoint["checkpoint"]["sha256"],
            state_before=dict(
                student="a" * 64,
                teacher="b" * 64,
                objective_state="c" * 64,
                objective_buffers="d" * 64,
                objective_scalar_clocks="e" * 64,
            ),
            arrays=arrays,
            native_result=native_binding,
            summary_sha256=m.sha(worker / "SUMMARY.json"),
            resource_owner=owner,
            precision_policy=policy,
            input_files_sha256={str(Path(b["path"]).resolve()): b["sha256"] for b in inputs},
        )
        write(worker / "COMPLETE.json", receipt)
        (root / f"{name}-worker-{i}.log").write_text(
            "Synthetic worker boundary; no CUDA execution.\n"
        )
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path.resolve())
    original_validate = m.validate_spec
    monkeypatch.setattr(
        m, "validate_spec", lambda value, **kw: original_validate(value, server=False)
    )
    monkeypatch.setattr(m, "_source", lambda source: dict(commit=source["commit"], dirty=False))

    def statistics(a, b, **kwargs):
        assert kwargs == spec["bootstrap"]
        # Geometry numerics are covered independently; this fixture tests receipt closure.
        small = synthetic_arrays()
        result = compare_endpoints(small, small, draws=3, seed=kwargs["seed"])
        result.update(draws=kwargs["draws"], condition_N=445)
        for key in ("ownmask_Pearson_delta", "ownmask_MSE_delta", "RSA_delta"):
            row = result[key]
            row["valid_draws"] = kwargs["draws"] if row["CI95"] is not None else 0
            row["invalid_draws"] = kwargs["draws"] - row["valid_draws"]
        return result

    geometry = SimpleNamespace(
        compare_endpoints=statistics, summarize_endpoint=lambda **kw: dict(synthetic=True)
    )
    monkeypatch.setattr(m, "geometry_module", lambda _: geometry)
    m.compare(spec, spec_file=sb)
    return spec, sb, root


def test_precision_merge_comparison_readback_reuses_ri_without_forward(tmp_path, monkeypatch):
    spec, binding, root = completed_precision_fixture(tmp_path, monkeypatch)
    before = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    result = m.validate_precision_result(spec, spec_file=binding, server=False)
    assert result["conditions"] == 445 and result["updates"] == 0
    assert result["scientific_complete"] is False
    assert before == {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    assert not (root / "R3-RI").exists()


@pytest.mark.parametrize(
    "case", ["extra", "missing", "tamper", "policy", "state", "clock_proof", "counter", "pkl"]
)
def test_precision_terminal_closure_rejects_even_resealed_bad_worker(tmp_path, monkeypatch, case):
    spec, binding, root = completed_precision_fixture(tmp_path, monkeypatch)
    name = spec["endpoints"][0]["name"]
    if case == "extra":
        (root / "unexpected.json").write_text("{}")
    elif case == "missing":
        (root / name / "worker-0/SUMMARY.json").unlink()
    elif case == "tamper":
        (root / "COMPARISON.json").write_text("{}")
    elif case == "pkl":
        (root / "result.pkl").write_bytes(b"keep")
    else:
        worker = root / name / "worker-0/COMPLETE.json"
        value = json.loads(worker.read_text())
        if case == "policy":
            value["precision_policy"]["autocast_enabled"] = True
        elif case == "state":
            value["state_unchanged"] = False
        elif case == "clock_proof":
            del value["state_before"]["objective_scalar_clocks"]
        else:
            value["forbidden_calls"]["optimizer_step"] = 1
        write(worker, value)
        aggregate = root / name / "COMPLETE.json"
        receipt = json.loads(aggregate.read_text())
        receipt["worker_receipts_sha256"][0] = m.sha(worker)
        write(aggregate, receipt)
        terminal = root / "COMPLETE.json"
        receipt = json.loads(terminal.read_text())
        receipt["endpoint_receipts"][name] = m.sha(aggregate)
        receipt["outputs"][str(worker.relative_to(root))] = m.sha(worker)
        receipt["outputs"][str(aggregate.relative_to(root))] = m.sha(aggregate)
        write(terminal, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.validate_precision_result(spec, spec_file=binding, server=False)


@pytest.mark.parametrize(
    "case", ["valid", "na", "missing", "unseen", "count", "na_value", "nonfinite", "mse", "order"]
)
def test_precision_val18_counts_and_missing_values_are_truthful(case):
    keys = (
        "txpert_macro_pearson_delta_all",
        "txpert_macro_pearson_delta_deg",
        "trishift_pearson_delta_all",
        "trishift_pearson_delta_deg",
        "systema_pearson_all",
        "systema_pearson_deg",
    )
    metrics = [
        dict(metric_id=k, total_condition_count=445, finite_condition_count=445, macro_mean=0.2)
        for k in keys
    ]
    ids = [f"p{i}" for i in range(445)]
    native = dict(
        split="val",
        prediction_loss=1.0,
        metrics=copy.deepcopy(metrics),
        metric_gene_groups={
            "seen_expression": dict(gene_count=4775, metrics=copy.deepcopy(metrics)),
            "unseen_expression": dict(gene_count=225, metrics=copy.deepcopy(metrics)),
        },
        conditions=[
            dict(condition_id=p, control_row_ids_sha256="c-" + p, truth_row_ids_sha256="t-" + p)
            for p in ids
        ],
    )
    if case == "na":
        native["metrics"][0].update(finite_condition_count=0, macro_mean=None)
    elif case == "missing":
        native["metrics"].pop()
    elif case == "unseen":
        native["metric_gene_groups"]["unseen_expression"]["gene_count"] = 224
    elif case == "count":
        native["metrics"][0]["finite_condition_count"] = 446
    elif case == "na_value":
        native["metrics"][0].update(finite_condition_count=0, macro_mean=0.0)
    elif case == "nonfinite":
        native["metrics"][0]["macro_mean"] = float("nan")
    elif case == "mse":
        native["prediction_loss"] = float("nan")
    elif case == "order":
        native["conditions"][:2] = native["conditions"][1::-1]
    if case in ("valid", "na"):
        m.validate_precision_metrics(native, dict(conditions=ids))
    else:
        with pytest.raises(ValueError):
            m.validate_precision_metrics(native, dict(conditions=ids))


@pytest.mark.parametrize("mutation", ["temperature", "private_buffer"])
def test_precision_rejects_direct_original_objective_mutations_outside_update_methods(mutation):
    import numpy as np
    from capacity_probe import state_fingerprint, validation_state

    model, _ = fixture()
    objective = JointObjective(
        GraDPertV2(
            model.graph.embedding.weight.detach(),
            replace(model.options, prediction_precision="float32"),
        )
    ).eval()
    objective.register_buffer("private_clock", torch.zeros(()))
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=SimpleNamespace(state_dict=lambda: {}),
        generator=np.random.default_rng(1),
        device=torch.device("cpu"),
    )
    spec = dict(schema=m.PRECISION_SCHEMA)
    before = m.evaluation_protected_state(spec, runtime, validation_state, state_fingerprint)
    legacy = validation_state(runtime)
    # Neither mutation calls an optimizer, statistics method, Teacher, or backward.
    if mutation == "temperature":
        objective._temperature_updates += 1
    else:
        objective.private_clock.add_(1)
    assert legacy == validation_state(runtime)  # Demonstrates the original gap.
    with pytest.raises(ValueError, match="mutated protected state"):
        m.verify_evaluation_state(spec, runtime, before, validation_state, state_fingerprint)
    historical = m.evaluation_protected_state(
        dict(schema=m.SCHEMA), runtime, validation_state, state_fingerprint
    )
    assert historical == validation_state(runtime)
    assert "objective_buffers" not in historical  # Old profiles keep their exact state shape.
