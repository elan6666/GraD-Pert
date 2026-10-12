"""Frozen family, actual native leaf bindings, and guarded sequential dispatch."""

from __future__ import annotations

import contextlib
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_br400_group as generator  # noqa: E402
import run_br400_group as runner  # noqa: E402

MANIFEST = ROOT / "configs/v2/br400_best_last_20261012/manifest.json"
PROTECTED = {
    "student",
    "teacher",
    "centers",
    "optimizer",
    "gradients",
    "numpy_rng",
    "python_rng",
    "torch_rng",
    "pending_statistics",
    "auxiliary_statistics",
    "module_modes",
}


def policy():
    return {
        "schema": "shared-gpu-admission-3",
        "devices": [{"gpu_index": 0, "gpu_uuid": "GPU-a"}, {"gpu_index": 1, "gpu_uuid": "GPU-b"}],
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "max_foreign_processes_per_gpu": 1,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
        "authorization_reference": "2026-10-12-user-scbutterfly-coexistence",
    }


def test_generated_family_only_authorized_recipe_changes():
    matrix = generator.verify_manifest(ROOT, MANIFEST)
    assert matrix["priority"] == ["BR", "F1", "N1", "GM1"]
    assert matrix["preflight_priority"] == ["GM1", "F1", "BR", "N1"]
    for row in matrix["rows"]:
        path = ROOT / row["config"]
        raw = yaml.safe_load(path.read_text())
        config = load_experiment_config(path)
        assert config.training.monitor == "val/joint_loss" and config.training.monitor_mode == "min"
        assert (
            config.training.max_epochs.value == 4 and config.training.train_batch_size.value == 400
        )
        assert config.training.eval_batch_size.value == 300
        original = yaml.safe_load((ROOT / generator.PARENTS[row["name"]][0]).read_text())
        for key in ("microbatch", "world_size", "accumulation", "validation_mode"):
            raw["model"]["parameters"][key] = original["model"]["parameters"][key]
        for key in (
            "train_batch_size",
            "eval_batch_size",
            "max_epochs",
            "formal_run_policy",
            "monitor",
            "monitor_mode",
        ):
            raw["training"][key] = original["training"][key]
        assert raw == original


@pytest.mark.parametrize("batch", [True, 0, 1, 301])
def test_generator_invalid_common_eval_chunk_rejected(batch):
    with pytest.raises(ValueError):
        generator.configurations(ROOT, batch)


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    runtime, publication = tmp_path / "runtime.json", tmp_path / "publication.json"
    atomic_json(publication, {"published_commit": "c" * 40})
    atomic_json(
        runtime,
        {
            "repository_root": str(ROOT),
            "data_root": str(tmp_path / "data"),
            "runs_root": str(tmp_path / "runs"),
            "publication_receipt": str(publication),
            "publication_sha256": sha256_file(publication),
        },
    )
    monkeypatch.setattr(runner, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "inspect_source_identity",
        lambda *a, **k: SimpleNamespace(commit="c" * 40, dirty=False),
    )
    counter = 0

    def resolve(args):
        nonlocal counter
        counter += 1
        return {
            "run_id": f"fresh{counter}",
            "run_root": str(tmp_path / "runs" / f"fresh{counter}"),
            "test_root": str(tmp_path / "runs" / f"fresh{counter}-test"),
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "resolved_config": load_experiment_config(args.config).model_dump(mode="json"),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "repository_root": str(ROOT),
            "source_commit": "c" * 40,
            "data_root": str(tmp_path / "data"),
            "publication": str(publication),
            "publication_sha256": sha256_file(publication),
            "gpu": "0,1",
            "seed": 1,
            "postfit_policy": "deferred",
        }

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    output = tmp_path / "queue"
    shared_policy = policy()
    if getattr(request, "param", 0.85) == 0.90:
        shared_policy.update(
            additional_command_basenames=["run_totalvi_source_validation_development.py"],
            authorization_reference="2026-10-12-user-shared-memory90-coexistence",
            max_own_memory_fraction=0.90,
        )
    queue = runner.prepare(ROOT, MANIFEST, runtime, output, shared_policy)
    return queue, output


@pytest.mark.parametrize("prepared", [0.85, 0.90], indirect=True)
def test_fit_and_test_use_exact_sealed_policy_fraction(prepared, monkeypatch):
    import gradpert.execution.v2_deferred_postfit as native

    queue, output = prepared
    fraction = queue["shared_gpu_policy"]["max_own_memory_fraction"]
    calls = []
    monkeypatch.setattr(native, "run_deferred_postfit", lambda plan, **kw: calls.append((plan, kw)))
    path = output / "BR.launch-plan.json"
    monkeypatch.setattr(sys, "argv", ["test", str(path)])
    exec(runner.evaluation_command(path)[2], {})
    assert calls == [(queue["rows"][0]["plan"], {"gpu": "0,1", "cuda_memory_fraction": fraction})]
    assert all(row["plan"]["cuda_memory_fraction"] == fraction for row in queue["rows"])
    queue["rows"][0]["plan"]["cuda_memory_fraction"] = 0.70
    with pytest.raises(ValueError):
        runner.verify(queue)


def training_receipt(queue, row):
    before = {key: "a" * 64 for key in PROTECTED}
    receipt = {
        "kind": "preflight_only",
        "status": "passed",
        "steps_requested": 10,
        "steps_completed": 10,
        "resume_checkpoint_sha256": "resume",
        "inference_exercised": True,
        "inference_shape": [300, 5000],
        "source": {
            "commit": queue["source_commit"],
            "published_commit": queue["source_commit"],
            "dirty": False,
            "publication_receipt_sha256": row["plan"]["publication_sha256"],
        },
        "config_sha256": row["plan"]["config_sha256"],
        "data_root": row["plan"]["data_root"],
        "data": {"run_seed": 1, **{key: "a" * 64 for key in runner.DATA_HASHES}},
        "inference_control_manifest_sha256": "a" * 64,
        "ordered_control_row_ids_sha256": sha256_json(list(range(300))),
        "inference_expression_gene_order_sha256": "a" * 64,
        "gpu": "0,1",
        "world_size": 2,
        "peak_allocated_bytes": 1,
        "last_terms": {"gradient_norm": 1.0, "joint_loss": 1.0},
        "shared_gpu_policy": queue["shared_gpu_policy"],
        "shared_queue_sha256": sha256_file(Path(queue["output"]) / "queue.json"),
        "evaluation_cell_batch": 300,
        "joint_validation_check": {
            "status": "passed",
            "validation_mode": "joint_only",
            "split": "val",
            "finite": True,
            "joint_loss": 1.0,
            "batch_count": 1,
            "batch_size": 100,
            "scope": "one_complete_native_validation_batch_engineering_only",
            "full_validation": False,
            "state_unchanged": {key: True for key in PROTECTED},
            "before_sha256": before,
            "after_sha256": copy.deepcopy(before),
        },
    }

    _architecture, options = V2Options.parse_parameters(
        load_experiment_config(row["plan"]["config"]).model.parameters
    )
    receipt["data"].update(
        graph_manifest_sha256=options.graph_manifest_sha256, genept_sha256=options.genept_sha256
    )
    return receipt


def eval_receipt(queue, row, binding):
    architecture, options = V2Options.parse_parameters(
        load_experiment_config(row["plan"]["config"]).model.parameters
    )
    controls = list(range(300))
    receipt = {
        "kind": "eval_batch_capacity",
        "status": "passed",
        "source": {
            "commit": queue["source_commit"],
            "published_commit": queue["source_commit"],
            "dirty": False,
            "formal_eligible": True,
            "publication_receipt_sha256": row["plan"]["publication_sha256"],
        },
        "config_sha256": row["plan"]["config_sha256"],
        "data_root": row["plan"]["data_root"],
        "gpu": binding["gpu"],
        "cell_batch": 300,
        "reference_batch": 128,
        "shared_queue_sha256": sha256_file(Path(queue["output"]) / "queue.json"),
        "shared_gpu_policy": queue["shared_gpu_policy"],
        "inference_shape": [300, 5000],
        "architecture": architecture.payload(),
        "evaluation_query_count": options.eval_query_count,
        "runtime_purpose": "evaluation",
        "only_student_on_cuda": True,
        "split": "val",
        "no_truth_access": True,
        "no_scientific_scores": True,
        "comparison": {
            "allclose": True,
            "atol": 2e-6,
            "rtol": 2e-6,
            "max_abs_error": 0.0,
            "max_scaled_error": 0.0,
            "mean_abs_error": 0.0,
        },
        "inference_seconds": 1.0,
        "reference_seconds": 1.0,
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 1,
        "model_state_unchanged": True,
        "rng_unchanged": True,
        "data": {
            key: "a" * 64
            for key in (
                "canonical_sha256",
                "split_sha256",
                "expression_gene_order_sha256",
                "graph_gene_order_sha256",
            )
        },
        "ordered_control_row_ids": controls,
        "ordered_control_row_ids_sha256": sha256_json(controls),
    }
    receipt["data"].update(
        graph_manifest_sha256=options.graph_manifest_sha256, genept_sha256=options.genept_sha256
    )
    for name in (
        "inference_control_manifest_sha256",
        "reference_prediction_sha256",
        "prediction_sha256",
        "model_state_before_sha256",
        "model_state_after_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
    ):
        receipt[name] = "a" * 64
    return receipt


def test_real_native_binding_and_eval_leaf_interoperation(prepared):
    from shared_gpu import validate_probe_binding

    queue, output = prepared
    runner.verify(queue)
    assert queue["schema"] == "unified-ddp-queue-1" and queue["family"] == "br400-best-last-1"
    state = {
        "queue_sha256": sha256_file(output / "queue.json"),
        "source_commit": queue["source_commit"],
        "preflights": {},
    }
    for row in queue["rows"]:
        pref_path = output / f"{row['name']}-preflight/receipt.json"
        pref_path.parent.mkdir()
        atomic_json(pref_path, training_receipt(queue, row))
        state["preflights"][row["name"]] = runner.gate_entry(pref_path)
        atomic_json(output / "state.json", state)
        args = SimpleNamespace(
            config=Path(row["plan"]["config"]),
            data_root=Path(row["plan"]["data_root"]),
            publication=Path(row["plan"]["publication"]),
            publication_sha256=row["plan"]["publication_sha256"],
            shared_queue=output / "queue.json",
            output=output / f"{row['name']}-preflight",
            steps=10,
            gpu="0,1",
        )
        validate_probe_binding(queue, args, queue["source_commit"], "preflight_only", 2)
        for binding in (b for b in queue["eval_bindings"] if b["row_name"] == row["name"]):
            path = Path(binding["output"]) / "receipt.json"
            path.parent.mkdir()
            atomic_json(path, eval_receipt(queue, row, binding))
            result = runner.validate_eval_gate(queue, row, binding, runner.gate_entry(path))
            assert result["gpu"] == binding["gpu"] and result["cell_batch"] == 300
            command = runner.eval_capacity_command(queue, row, binding)
            assert command[command.index("--cell-batch") + 1] == "300"
            assert command[command.index("--gpu") + 1] == binding["gpu"]


@pytest.mark.parametrize(
    "change",
    [
        "teacher",
        "optimizer",
        "hash",
        "extra_batches",
        "not_joint",
        "inference_chunk",
        "missing_state",
    ],
)
def test_joint_gate_rejects_incomplete_or_mutating_proof(prepared, change):
    queue, output = prepared
    row = queue["rows"][0]
    receipt = training_receipt(queue, row)
    joint = receipt["joint_validation_check"]
    if change in ("teacher", "optimizer"):
        joint["state_unchanged"][change] = False
    elif change == "hash":
        joint["after_sha256"]["teacher"] = "b" * 64
    elif change == "extra_batches":
        joint["batch_count"] = 2
    elif change == "not_joint":
        joint["validation_mode"] = "disabled"
    elif change == "inference_chunk":
        receipt["evaluation_cell_batch"] = 128
    else:
        del joint["state_unchanged"]["numpy_rng"]
    path = output / "receipt.json"
    atomic_json(path, receipt)
    with pytest.raises(ValueError):
        runner.validate_training_gate(queue, row, runner.gate_entry(path))


@pytest.mark.parametrize(
    "field", ["data_root", "runtime_sha256", "publication_sha256", "cuda_memory_fraction"]
)
def test_native_plan_binding_drift_rejected(prepared, field):
    queue, _ = prepared
    queue["rows"][0]["plan"][field] = "drift"
    with pytest.raises(ValueError):
        runner.verify(queue)


def execution_mocks(prepared, monkeypatch, *, fail=None, approval=True):
    queue, output = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)

    @contextlib.contextmanager
    def locked(path):
        yield

    monkeypatch.setattr(runner, "lock", locked)
    monkeypatch.setattr(runner, "snapshot", lambda pid: {"processes": [], "gpus": {}})
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    calls = []

    def run(q, row, stage, command, gpu, state, guard):
        guard()
        calls.append((row["name"], stage, command, gpu))
        if (row["name"], stage) == fail:
            raise RuntimeError("native OOM preserved")
        if stage == "preflight":
            assert "--joint-validation-check" in command
            path = output / f"{row['name']}-preflight/receipt.json"
            path.parent.mkdir()
            atomic_json(path, training_receipt(queue, row))
        elif stage.startswith("eval-capacity"):
            binding = next(
                b
                for b in queue["eval_bindings"]
                if b["row_name"] == row["name"] and b["gpu"] == gpu
            )
            path = Path(binding["output"]) / "receipt.json"
            path.parent.mkdir()
            atomic_json(path, eval_receipt(queue, row, binding))

    monkeypatch.setattr(runner, "run_stage", run)
    monkeypatch.setattr(runner, "validate_training_stage", lambda plan: {"epoch": 4})
    monkeypatch.setattr(
        runner,
        "validate_complete",
        lambda plan: {"run_id": plan["run_id"], "roles": ["best", "last"]},
    )

    def wait(seconds):
        assert seconds == 5
        if not approval:
            raise KeyboardInterrupt("manual approval pending")
        ready = read_json(output / "PREFLIGHT_COMPLETE.json")
        atomic_json(output / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})

    monkeypatch.setattr(runner.time, "sleep", wait)
    return queue, output, calls


def test_all_gates_before_barrier_then_sequential_fit_best_last(prepared, monkeypatch):
    queue, output, calls = execution_mocks(prepared, monkeypatch)
    runner.execute(queue, output)
    assert [(name, stage) for name, stage, _, _ in calls[:12]] == [
        (name, stage)
        for name in generator.PREFLIGHT_PRIORITY
        for stage in ("preflight", "eval-capacity-g0", "eval-capacity-g1")
    ]
    assert [(name, stage) for name, stage, _, _ in calls[12:]] == [
        (name, stage) for name in generator.PRIORITY for stage in ("fit", "best-last")
    ]
    for _, stage, command, gpu in calls[12:]:
        assert gpu == "0,1"
        if stage == "best-last":
            assert (
                "run_deferred_postfit" in command[2]
                and "cuda_memory_fraction=p['cuda_memory_fraction']" in command[2]
            )
    ready = read_json(output / "PREFLIGHT_COMPLETE.json")
    assert len(ready["preflights"]) == 4 and len(ready["eval_capacity"]) == 8
    assert read_json(output / "COMPLETE.json")["scientific_complete"]


@pytest.mark.parametrize(
    "failed", [("GM1", "preflight"), ("F1", "eval-capacity-g1"), ("BR", "fit"), ("F1", "best-last")]
)
def test_any_oom_stops_without_later_arm_or_fallback(prepared, monkeypatch, failed):
    queue, output, calls = execution_mocks(prepared, monkeypatch, fail=failed)
    with pytest.raises(RuntimeError, match="OOM"):
        runner.execute(queue, output)
    assert calls[-1][:2] == failed
    assert not read_json(output / "FAILURE.json")["scientific_complete"]
    assert not (output / "COMPLETE.json").exists()


def test_acceptance_is_manual_and_cannot_be_created_by_controller(prepared, monkeypatch):
    queue, output, calls = execution_mocks(prepared, monkeypatch, approval=False)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(queue, output)
    assert len(calls) == 12 and not (output / "MAIN_ACCEPTANCE.json").exists()


def test_active_guard_fatal_cleans_only_captured_owned_identities(prepared, monkeypatch):
    queue, _output = prepared
    row = queue["rows"][0]
    alive = {"pid": 101, "uid": os.getuid(), "start_ticks": "1", "command_sha256": "exact"}
    monkeypatch.setattr(runner, "parent_map", lambda: {101: os.getpid(), 102: 101})
    monkeypatch.setattr(runner, "process_info", lambda pid: {**alive, "pid": pid})
    monkeypatch.setattr(runner, "_capture_process_identity", lambda child: alive)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=101, poll=lambda: None)
    )
    killed = []
    monkeypatch.setattr(runner, "terminate_tree", lambda pid, info: killed.append(pid))
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("owned budget exceeded")
        return {"processes": []}

    with pytest.raises(RuntimeError, match="budget"):
        runner.run_stage(queue, row, "fit", ["native"], "0,1", {}, guard)
    assert set(killed) == {101, 102}


@pytest.mark.parametrize("bad", [None, "missing_role", "missing_metric", "wrong_best", "pkl"])
def test_real_native_best_last_completion_same_sha_and_failure_boundaries(prepared, bad):
    from gradpert.execution.v2_training_stage import training_stage_evidence

    queue, _ = prepared
    plan = queue["rows"][0]["plan"]
    root = Path(plan["run_root"])
    fit = root / "fit"
    fit.mkdir(parents=True)
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "resolved_config.json", plan["resolved_config"])
    source = {
        "commit": plan["source_commit"],
        "published_commit": plan["source_commit"],
        "dirty": False,
    }
    identity = {
        "source": source,
        "run_id": plan["run_id"],
        "config_sha256": plan["config_sha256"],
        "resolved_config_sha256": sha256_json(plan["resolved_config"]),
    }
    atomic_json(root / "run_manifest.json", identity)
    checkpoint = fit / "epoch-0004.pt"
    checkpoint.write_bytes(b"synthetic same-content best and last checkpoint")
    selected = {
        "file": checkpoint.name,
        "sha256": sha256_file(checkpoint),
        "epoch": 4,
        "joint_loss": 1.0,
    }
    atomic_json(
        fit / "epoch_state.json",
        {
            "identity": identity,
            "epoch": 4,
            "budget": [4, 125],
            "selection_metric": "joint_loss",
            "best": selected,
            "last": selected,
        },
    )
    history = [
        {"epoch": epoch, "validation": {"split": "val", "joint_loss": float(5 - epoch)}}
        for epoch in range(1, 5)
    ]
    if bad == "wrong_best":
        history[2]["validation"]["joint_loss"] = 0.1
    atomic_json(fit / "history.json", history)
    training = training_stage_evidence(root, identity, 4)
    atomic_json(root / "TRAIN_COMPLETE.json", training)
    metrics = [
        {
            "metric_id": metric_id,
            "macro_mean": 0.5,
            "finite_condition_count": 1,
            "total_condition_count": 1,
        }
        for metric_id in runner.METRICS
    ]
    for role in ("best", "last"):
        result = {
            "split": "test",
            "prediction_loss": 0.5,
            "metrics": copy.deepcopy(metrics),
            "metric_gene_groups": {
                group: {"metrics": copy.deepcopy(metrics)}
                for group in ("seen_expression", "unseen_expression")
            },
        }
        if bad == "missing_metric" and role == "best":
            result["metrics"].pop()
        independent = root / f"postfit-{role}/COMPLETE.json"
        independent.parent.mkdir()
        evaluation = {
            "training_identity": identity,
            "checkpoint_sha256": selected["sha256"],
            "checkpoint_role": role,
            "checkpoint_epoch": 4,
            "training_run_root": str(root),
            "split": "test",
            "output_root": str(independent.parent),
            "evaluation_source": source,
        }
        atomic_json(
            independent,
            {"plan": evaluation, "result": result, "evaluation_environment": {}, "zero_pkl": True},
        )
        atomic_json(
            fit / f"{role}-test.json",
            {
                "identity": {
                    "training": identity,
                    "checkpoint": selected,
                    "role": role,
                    "evaluation": {"source": source, "environment": {}},
                },
                "result": result,
                "independent_evaluation_receipt": str(independent),
                "independent_evaluation_sha256": sha256_file(independent),
            },
        )
    runner.finalize_deferred_run(plan, training)
    if bad == "missing_role":
        (fit / "best-test.json").unlink()
    if bad == "pkl":
        (root / "unexpected.pkl").write_text("synthetic")
    if bad is not None:
        with pytest.raises((ValueError, FileNotFoundError)):
            runner.validate_complete(plan)
    else:
        completed = runner.validate_complete(plan)
        assert [row["role"] for row in completed["roles"]] == ["best", "last"]
        assert (
            completed["roles"][0]["checkpoint_sha256"] == completed["roles"][1]["checkpoint_sha256"]
        )
        assert completed["roles"][1]["checkpoint_epoch"] == 4
        assert set(completed["test_sha256"]) == {"best", "last"}


def test_resealed_eval_data_drift_rejected_by_actual_leaf(prepared):
    queue, output = prepared
    row, binding = queue["rows"][0], queue["eval_bindings"][0]
    pref_path = output / f"{row['name']}-preflight/receipt.json"
    pref_path.parent.mkdir()
    atomic_json(pref_path, training_receipt(queue, row))
    state = {
        "queue_sha256": sha256_file(output / "queue.json"),
        "source_commit": queue["source_commit"],
        "preflights": {row["name"]: runner.gate_entry(pref_path)},
    }
    atomic_json(output / "state.json", state)
    receipt = eval_receipt(queue, row, binding)
    receipt["data"]["canonical_sha256"] = "b" * 64
    path = Path(binding["output"]) / "receipt.json"
    path.parent.mkdir()
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match=r"data|preflight"):
        runner.validate_eval_gate(queue, row, binding, runner.gate_entry(path))


def test_cross_arm_resealed_data_drift_rejected_at_common_barrier(prepared, monkeypatch):
    queue, output, _calls = execution_mocks(prepared, monkeypatch, approval=False)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(queue, output)
    state = read_json(output / "state.json")
    name = "F1"
    path = Path(state["preflights"][name]["receipt"])
    receipt = read_json(path)
    receipt["data"]["canonical_sha256"] = "b" * 64
    atomic_json(path, receipt)
    state["preflights"][name] = runner.gate_entry(path)
    for gpu in ("0", "1"):
        path = Path(state["eval_capacity"][f"{name}-g{gpu}"]["receipt"])
        leaf = read_json(path)
        leaf["data"]["canonical_sha256"] = "b" * 64
        atomic_json(path, leaf)
        state["eval_capacity"][f"{name}-g{gpu}"] = runner.gate_entry(path)
    atomic_json(output / "state.json", state)
    row = next(row for row in queue["rows"] if row["name"] == name)
    binding = next(
        item for item in queue["eval_bindings"] if item["row_name"] == name and item["gpu"] == "0"
    )
    # The actual leaf accepts the coherently re-sealed local pair; the family
    # barrier must still reject disagreement with the other authorized arms.
    runner.validate_eval_gate(queue, row, binding, state["eval_capacity"]["F1-g0"])
    with pytest.raises(ValueError, match="common data"):
        runner.validate_common_gates(queue, state)


CHECKPOINT_MANIFEST = ROOT / "configs/v2/br400_best_last_checkpoint_20261012/manifest.json"


def test_checkpoint_family_only_adds_common_optin_and_preserves_old_bytes():
    old = generator.verify_manifest(ROOT, MANIFEST)
    checkpoint = generator.verify_manifest(ROOT, CHECKPOINT_MANIFEST)
    assert "checkpoint_genept" not in old
    assert checkpoint["checkpoint_genept"] is True
    assert checkpoint["preflight_priority"] == ["F1", "GM1", "BR", "N1"]
    assert checkpoint["priority"] == old["priority"] == ["BR", "F1", "N1", "GM1"]
    for old_row, new_row in zip(old["rows"], checkpoint["rows"], strict=True):
        old_path, new_path = ROOT / old_row["config"], ROOT / new_row["config"]
        before = old_path.read_bytes()
        original = yaml.safe_load(before)
        updated = yaml.safe_load(new_path.read_text())
        assert "checkpoint_genept_mapping" not in original["model"]["parameters"]
        extra = updated["model"]["parameters"].pop("checkpoint_genept_mapping")
        assert extra == {
            "value": True,
            "source": "project_preregistered",
            "reference": generator.REFERENCE,
        }
        assert updated == original
        assert sha256_file(old_path) == old_row["sha256"]
        assert before == old_path.read_bytes()
        old_arch, _ = V2Options.parse_parameters(load_experiment_config(old_path).model.parameters)
        new_arch, _ = V2Options.parse_parameters(load_experiment_config(new_path).model.parameters)
        assert "checkpoint_genept_mapping" not in old_arch.payload()
        payload = new_arch.payload()
        assert payload.pop("checkpoint_genept_mapping") is True
        assert payload == old_arch.payload()


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_checkpoint_generator_rejects_non_boolean_optin(flag):
    with pytest.raises(ValueError, match="explicit bool"):
        generator.configurations(ROOT, checkpoint_genept=flag)


def test_old_default_configs_and_metadata_remain_identical():
    old = read_json(MANIFEST)
    assert generator.metadata() == {key: value for key, value in old.items() if key != "rows"}
    expected = generator.configurations(ROOT)
    for row in old["rows"]:
        assert expected[row["name"]] == yaml.safe_load((ROOT / row["config"]).read_text())


def test_checkpoint_runner_reads_verified_manifest_priority(prepared, monkeypatch):
    queue, output, calls = execution_mocks(prepared, monkeypatch, approval=False)
    matrix = generator.verify_manifest(ROOT, CHECKPOINT_MANIFEST)
    queue["manifest"] = str(CHECKPOINT_MANIFEST)
    queue["manifest_sha256"] = sha256_file(CHECKPOINT_MANIFEST)
    by_name = {row["name"]: row for row in matrix["rows"]}
    for row in queue["rows"]:
        spec = by_name[row["name"]]
        row["plan"]["config"] = str(ROOT / spec["config"])
        row["plan"]["config_sha256"] = spec["sha256"]
        row["plan"]["resolved_config"] = load_experiment_config(row["plan"]["config"]).model_dump(
            mode="json"
        )
        atomic_json(output / f"{row['name']}.launch-plan.json", row["plan"])
    atomic_json(output / "queue.json", queue)
    runner.verify(queue)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(queue, output)
    assert [name for name, stage, _, _ in calls if stage == "preflight"] == [
        "F1",
        "GM1",
        "BR",
        "N1",
    ]
    assert not (output / "MAIN_ACCEPTANCE.json").exists()
