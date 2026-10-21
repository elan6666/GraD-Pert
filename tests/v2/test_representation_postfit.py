"""Research completion retains native gates without consuming test evidence."""

import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as research  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from test_br400_group import eval_receipt  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402
from test_fgpo_group import training_gate  # noqa: E402


@pytest.fixture
def research_queue(new_queue):
    old, output = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / "configs/v2/representation_round1_20261021/manifest.json",
        Path(old["runtime"]),
        output.parent / "research",
        old["shared_gpu_policy"],
    )
    return q, Path(q["output"])


def test_research_exact_two_epoch_and_original_leaf_gates(research_queue):
    q, output = research_queue
    assert q["steps"] == 10 and q["family"] == family.SCHEMA
    assert list(medium.FAMILY_ARMS[family.SCHEMA]) == ["RS-B0", "RS-RI"]
    for row in q["rows"]:
        assert runner.expected_epochs(row["plan"]) == 2
        bad = copy.deepcopy(row["plan"])
        bad["resolved_config"]["model"]["parameters"]["microbatch"]["value"] = 79
        with pytest.raises(ValueError, match="exact representation"):
            runner.expected_epochs(bad)
        path = output / f"{row['name']}-preflight/receipt.json"
        atomic_json(path, training_gate(q, row))
        runner.validate_training_gate(q, row, runner.gate_entry(path))
    for binding in q["eval_bindings"]:
        row = next(r for r in q["rows"] if r["name"] == binding["row_name"])
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=128,
            reference_batch=128,
            config=Path(row["plan"]["config"]),
            publication=Path(row["plan"]["publication"]),
            publication_sha256=row["plan"]["publication_sha256"],
            data_root=Path(row["plan"]["data_root"]),
            output=Path(binding["output"]),
            shared_queue=output / "queue.json",
            row_name=row["name"],
        )
        assert validate_eval_binding(q, args, q["source_commit"]) == row


@pytest.mark.parametrize("fail", [None, "postfit-validation", "research-analysis"])
def test_controller_uses_validation_then_analysis_stops_for_design(
    research_queue, monkeypatch, fail
):
    q, output = research_queue
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "snapshot", lambda pid: dict(processes=[], gpus={}))
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    locks, calls = set(), []

    @contextlib.contextmanager
    def lock(path):
        assert path.name not in locks
        locks.add(path.name)
        try:
            yield
        finally:
            locks.remove(path.name)

    def stage(queue, row, name, command, gpu, state, guard):
        calls.append((row["name"], name))
        assert len([s for s in locks if s.startswith("v2-gpu-")]) == (
            0 if name == "research-analysis" else 2
        )
        assert name not in ("last", "best-last")
        if row["name"] == "RS-B0" and name == fail:
            raise RuntimeError("injected research failure")
        if name == "preflight":
            atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(q, row))
        elif name.startswith("eval-capacity"):
            binding = next(
                b for b in q["eval_bindings"] if b["row_name"] == row["name"] and b["gpu"] == gpu
            )
            receipt = eval_receipt(q, row, binding)
            receipt["cell_batch"] = 128
            atomic_json(Path(binding["output"]) / "receipt.json", receipt)

    def accept(_):
        atomic_json(
            output / "MAIN_ACCEPTANCE.json",
            {**read_json(output / "PREFLIGHT_COMPLETE.json"), "approved": True},
        )

    monkeypatch.setattr(runner, "lock", lock)
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner.time, "sleep", accept)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: {})
    monkeypatch.setattr(runner, "validate_complete", lambda p: pytest.fail("test finalizer used"))
    monkeypatch.setattr(research, "validate_research_complete", lambda *a: {"research": True})
    if fail:
        with pytest.raises(RuntimeError, match="injected research"):
            runner.execute(q, output)
        assert ("RS-RI", "fit") not in calls
    else:
        runner.execute(q, output)
        assert calls[6:] == [
            (arm, step)
            for arm in family.ARMS
            for step in ("fit", "postfit-validation", "research-analysis")
        ]
        result = read_json(output / "RESEARCH_COMPLETE.json")
        assert result["status"] == "research_complete"
        assert result["phase"] == "awaiting_main_analysis"
        assert result["scientific_complete"] is False and result["completed"] == {}
        assert set(result["research_completed"]) == set(family.ARMS)
    assert not (output / "COMPLETE.json").exists() and not locks


def test_validation_adapter_uses_only_val_and_no_test_finalizer(tmp_path, monkeypatch):
    from gradpert.evaluation import state
    from gradpert.execution import v2_checkpoint_eval as evaluation

    cp = dict(file="last.pt", sha256="a" * 64)
    plan = dict(
        config=str(tmp_path / "config"),
        runtime=str(tmp_path / "runtime"),
        run_root=str(tmp_path / "run"),
        data_root=str(tmp_path / "data"),
        resolved_config=dict(dataset_id="nadig_jurkat", data=dict(protocol_id="p")),
    )
    q = dict(output=str(tmp_path), shared_gpu_policy=dict(max_own_memory_fraction=0.9))
    row = dict(name="RS-B0", plan=plan)
    seen = {}
    monkeypatch.setattr(research, "context", lambda *a: dict(last=cp))
    monkeypatch.setattr(state, "prepare_evaluation_state", lambda **kw: seen.update(prepare=kw))
    monkeypatch.setattr(evaluation, "resolve_evaluation_plan", lambda a: {"split": a.split})
    monkeypatch.setattr(evaluation, "execute_evaluation_plan", lambda p: seen.update(evaluate=p))
    monkeypatch.setattr(research, "validate_validation", lambda *a: "validated")
    assert research.run_validation(q, row) == "validated"
    assert seen["prepare"]["validation_only"] is True
    assert seen["evaluate"] == dict(split="val", cpu_training_state=True, cuda_memory_fraction=0.9)


def test_research_report_renders_without_any_test_result(tmp_path):
    root = tmp_path / "RS-B0-medium"
    run = tmp_path / "run"
    row = dict(name="RS-B0", plan=dict(run_root=str(run), config_sha256="c" * 64))
    q = dict(output=str(tmp_path), source_commit="a" * 40)
    atomic_json(
        root / "analysis/summary.json",
        dict(
            global_features={
                "student.main.response_cls": dict(effective_rank=4.0, relative_variance=0.1)
            },
            global_heads={"teacher.ssl2_cls": dict(H=9.7)},
            metrics={},
            cls_decomposition={},
        ),
    )
    atomic_json(
        root / "analysis/deep.json", [dict(condition_id="p1", joint=2.0, loss_terms={"mse": 1.0})]
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(run / "fit/history.json", [dict(epoch=1, loss=2.0), dict(epoch=2, loss=1.0)])
    atomic_json(
        research.validation_path(q, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    research.render_medium_report(q, row, root)
    assert (root / "report/representation.png").stat().st_size > 1000
    text = (root / "report/REPORT_CN.md").read_text()
    assert "测试推迟" in text and "training.csv" in text
    assert not (run / "fit/last-test.json").exists()


def test_exact_epochs_keeps_historical_four_epoch_anchor():
    path = ROOT / family.BASELINE
    plan = dict(
        config=str(path),
        repository_root=str(ROOT),
        resolved_config=load_experiment_config(path).model_dump(mode="json"),
    )
    assert runner.expected_epochs(plan) == 4


def test_output_hashes_and_path_boundaries(tmp_path):
    p = tmp_path / "report.md"
    p.write_text("valid")
    seal = research.output_seal(tmp_path)
    research.validate_outputs(tmp_path, seal)
    p.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        research.validate_outputs(tmp_path, seal)
    with pytest.raises(ValueError, match="escaped"):
        research.validate_outputs(tmp_path, {"../outside": {"sha256": sha256_file(p)}})


@pytest.fixture
def validation_receipt(tmp_path, monkeypatch):
    plan = dict(run_root=str(tmp_path / "run"), config_sha256="c" * 64, runtime_sha256="d" * 64)
    q = dict(output=str(tmp_path), source_commit="a" * 40)
    row = dict(name="RS-B0", plan=plan)
    training = dict(identity={"run_id": "r"}, last=dict(sha256="b" * 64))
    monkeypatch.setattr(research, "context", lambda *a: training)
    evaluated = dict(
        training_run_root=plan["run_root"],
        checkpoint_sha256="b" * 64,
        checkpoint_epoch=2,
        checkpoint_role="last",
        split="val",
        config_sha256=plan["config_sha256"],
        runtime_sha256=plan["runtime_sha256"],
        output_root=str(research.validation_path(q, row)),
        gpu=["0", "1"],
        training_identity=training["identity"],
        evaluation_source=dict(commit=q["source_commit"]),
    )
    metrics = [
        dict(metric_id=k, finite_condition_count=3, total_condition_count=4, macro_mean=0.2)
        for k in runner.METRICS
    ]
    result = dict(
        split="val",
        prediction_loss=0.3,
        metrics=metrics,
        metric_gene_groups={
            k: dict(metrics=copy.deepcopy(metrics))
            for k in ("seen_expression", "unseen_expression")
        },
    )
    root = research.validation_path(q, row)
    atomic_json(root / "plan.json", evaluated)
    atomic_json(root / "COMPLETE.json", dict(plan=evaluated, result=result, zero_pkl=True))
    return q, row, root


@pytest.mark.parametrize("bad", [None, "split", "source", "partial", "count", "missing_group"])
def test_postfit_validation_rejects_test_partial_or_bad_na(validation_receipt, bad):
    q, row, root = validation_receipt
    value = read_json(root / "COMPLETE.json")
    if bad == "split":
        value["plan"]["split"] = "test"
    elif bad == "source":
        value["plan"]["evaluation_source"]["commit"] = "0" * 40
    elif bad == "partial":
        value["plan"]["diagnostic_condition_count"] = 3
    elif bad == "count":
        value["result"]["metrics"][0]["finite_condition_count"] = 0
    elif bad == "missing_group":
        value["result"]["metric_gene_groups"].pop("unseen_expression")
    atomic_json(root / "COMPLETE.json", value)
    if bad:
        with pytest.raises(ValueError):
            research.validate_validation(q, row)
    else:
        assert research.validate_validation(q, row)["split"] == "val"


def test_context_rejects_existing_test_completion(tmp_path, monkeypatch):
    plan = dict(run_root=str(tmp_path / "run"), resolved_config={})
    row = dict(name="RS-B0", plan=plan)
    q = dict(family=family.SCHEMA, rows=[row])
    root = Path(plan["run_root"])
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "resolved_config.json", {})
    monkeypatch.setattr(runner, "verify", lambda q: None)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: dict(last=dict(epoch=2)))
    assert research.context(q, row)["last"]["epoch"] == 2
    atomic_json(root / "COMPLETE.json", {})
    with pytest.raises(ValueError, match="test must remain deferred"):
        research.context(q, row)


@pytest.mark.parametrize("bad", [None, "panel", "parity", "endpoint", "state"])
def test_panel_completion_binds_population_state_and_actual_used_cache(tmp_path, monkeypatch, bad):
    from representation_panel import panel_schedule
    from test_representation_panel import parent

    p = tmp_path / "parent.json"
    atomic_json(p, parent())
    panel = panel_schedule(parent())
    row = dict(name="RS-B0", plan=dict(run_id="r", config_sha256="a" * 64))
    q = dict(
        output=str(tmp_path),
        source_commit="c" * 40,
        medium_analysis=dict(parent_manifest=str(p), parent_manifest_sha256=sha256_file(p)),
    )
    monkeypatch.setattr(research, "context", lambda *a: dict(last=dict(sha256="b" * 64)))
    root = tmp_path / "RS-B0-panel"
    atomic_json(root / "panel.json", panel)
    record = dict(
        status="completed",
        updates=0,
        method_sha=q["source_commit"],
        run_id="r",
        checkpoint_sha256="b" * 64,
        config_sha256="a" * 64,
        parent_manifest=str(p),
        parent_manifest_sha256=sha256_file(p),
        state_before={"model": "same"},
        state_after={"model": "same"},
        state_unchanged=True,
        panel_sha256=panel["panel_sha256"],
        panel_rows=panel["rows"],
        checkpoint_endpoint=dict(
            role="last", epoch=2, training_stage="joint", full_training_epochs=2
        ),
        parity=dict(
            used_path_passed=True,
            tolerance=2e-6,
            cell_chunk_used=False,
            graph_max_abs=0.0,
            response_max_abs=1e-7,
        ),
        outputs=research.output_seal(root),
    )
    if bad == "panel":
        record["panel_sha256"] = "incorrect"
    elif bad == "parity":
        record["parity"]["response_max_abs"] = 3e-6
    elif bad == "endpoint":
        record["checkpoint_endpoint"]["epoch"] = 4
    elif bad == "state":
        record["state_after"]["model"] = "changed"
    atomic_json(root / "COMPLETE.json", record)
    if bad:
        with pytest.raises(ValueError):
            research.validate_panel(q, row)
    else:
        assert research.validate_panel(q, row)["panel_sha256"] == panel["panel_sha256"]


@pytest.mark.parametrize("parent_cublas", [None, ":16:8"])
def test_panel_subprocess_has_explicit_deterministic_source_environment(
    tmp_path, monkeypatch, parent_cublas
):
    if parent_cublas is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", parent_cublas)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "wrong")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    q = dict(output=str(tmp_path), source=str(ROOT), source_commit="a" * 40)
    row = dict(name="RS-B0", plan={})
    atomic_json(tmp_path / "queue.json", q)
    monkeypatch.setattr(research, "validate_validation", lambda *a: {})
    monkeypatch.setattr(medium, "execute", lambda *a: None)
    monkeypatch.setattr(medium, "validate_medium_complete", lambda *a: {})
    monkeypatch.setattr(research, "validate_panel", lambda *a: {})
    monkeypatch.setattr(research, "validate_research_complete", lambda *a: {})
    command = ["python", "run_representation_panel.py"]
    monkeypatch.setattr(research, "panel_command", lambda *a: command)
    seen = {}

    def subprocess_run(actual, **kwargs):
        seen.update(command=actual, **kwargs)

    monkeypatch.setattr(research.subprocess, "run", subprocess_run)
    research.run_analysis(q, row)
    assert seen["command"] == command and seen["cwd"] == str(ROOT)
    assert seen["check"] is True
    for key, expected in dict(
        CUDA_VISIBLE_DEVICES="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(ROOT / "src"),
    ).items():
        assert seen["env"][key] == expected
    assert research.os.environ.get("CUBLAS_WORKSPACE_CONFIG") == parent_cublas
    assert research.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
