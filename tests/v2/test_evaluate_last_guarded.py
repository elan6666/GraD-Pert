"""Independent native last evaluation preserves the best/last training protocol."""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.execution import train_entry
from gradpert.execution import v2_checkpoint_eval as native
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import evaluate_last_guarded as evaluator  # noqa: E402


def policy():
    return {
        "schema": "shared-gpu-admission-3",
        "devices": [{"gpu_index": 0, "gpu_uuid": "GPU-a"}, {"gpu_index": 1, "gpu_uuid": "GPU-b"}],
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "additional_command_basenames": ["run_totalvi_source_validation_development.py"],
        "max_foreign_processes_per_gpu": 1,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.9,
        "authorization_reference": "2026-10-12-user-shared-memory90-coexistence",
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(native, "SERVER_ROOT", tmp_path)
    science = tmp_path / "scientific"
    science.mkdir()
    controller = tmp_path / "controller"
    script = controller / "scripts/v2/evaluate_last_guarded.py"
    script.parent.mkdir(parents=True)
    script.write_text("immutable published controller")
    monkeypatch.setattr(evaluator, "__file__", str(script))
    config = science / "gradpert_v2/nadig_jurkat.yaml"
    config.parent.mkdir()
    config.write_text(
        (
            ROOT / "configs/v2/br400_best_last_checkpoint_20261012/BR/gradpert_v2/nadig_jurkat.yaml"
        ).read_text()
    )
    resolved = load_experiment_config(config).model_dump(mode="json")
    runs = tmp_path / "runs"
    run = runs / "BR-original"
    fit = run / "fit"
    fit.mkdir(parents=True)
    checkpoint = fit / "epoch-0004.pt"
    checkpoint.write_bytes(b"epoch4 checkpoint")
    (fit / "last.pt").write_bytes(checkpoint.read_bytes())
    data = tmp_path / "data"
    data.mkdir()
    source_commit = "6a8c612bd7733d2f03b5c041aeff63c5f6ec8339"

    def runtime_for(source, commit):
        pub = tmp_path / (source.name + "-publication.json")
        atomic_json(pub, {"published_commit": commit})
        path = tmp_path / (source.name + "-runtime.json")
        atomic_json(
            path,
            {
                "repository_root": str(source),
                "data_root": str(data),
                "runs_root": str(runs),
                "publication_receipt": str(pub),
                "publication_sha256": sha256_file(pub),
            },
        )
        return path, pub

    oldruntime, oldpub = runtime_for(science, source_commit)
    newruntime, _ = runtime_for(controller, "c" * 40)
    plan = {
        "repository_root": str(science),
        "source_commit": source_commit,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "resolved_config": resolved,
        "runtime": str(oldruntime),
        "runtime_sha256": sha256_file(oldruntime),
        "publication": str(oldpub),
        "publication_sha256": sha256_file(oldpub),
        "data_root": str(data),
        "run_id": "BR-original",
        "run_root": str(run),
        "test_root": str(runs / "BR-original-test"),
        "gpu": "0,1",
        "seed": 1,
    }
    identity = {
        "source": {"commit": source_commit, "published_commit": source_commit, "dirty": False},
        "config_sha256": plan["config_sha256"],
        "resolved_config_sha256": sha256_json(resolved),
        "data": {"run_seed": 1, "canonical_sha256": "a" * 64, "split_sha256": "b" * 64},
    }
    atomic_json(run / "launch.json", plan)
    atomic_json(run / "resolved_config.json", resolved)
    atomic_json(run / "run_manifest.json", identity)
    selected = {"epoch": 4, "file": checkpoint.name, "sha256": sha256_file(checkpoint)}
    atomic_json(
        fit / "epoch_state.json",
        {"identity": identity, "budget": [4, 250], "epoch": 4, "best": selected, "last": selected},
    )
    atomic_json(fit / "history.json", [{"epoch": i} for i in range(1, 5)])
    atomic_json(run / "TRAIN_COMPLETE.json", training_stage_evidence(run, identity, 4))
    parent = tmp_path / "oldqueue"
    parent.mkdir()
    launch = parent / "BR.launch-plan.json"
    atomic_json(launch, plan)

    def inspect(source, **kwargs):
        receipt = read_json(Path(kwargs["publication_receipt"]))
        assert (
            sha256_file(Path(kwargs["publication_receipt"]))
            == kwargs["expected_publication_receipt_sha256"]
        )
        payload = {
            "repository_root": str(source),
            "commit": receipt["published_commit"],
            "dirty": False,
            "published_commit": receipt["published_commit"],
        }
        return SimpleNamespace(**payload, payload=lambda: payload)

    monkeypatch.setattr(evaluator, "inspect_source_identity", inspect)
    monkeypatch.setattr(native, "inspect_source_identity", inspect)
    monkeypatch.setattr(train_entry, "repository_root", lambda: science)

    def resolve(plan, training, output):
        args = evaluator.native_args(plan, training, output)
        for key in ("config", "runtime", "training_run_root", "checkpoint", "output_root"):
            args[key] = Path(args[key])
        result = native.resolve_evaluation_plan(argparse.Namespace(**args))
        result.update(cpu_training_state=True, cuda_memory_fraction=0.9)
        return result

    monkeypatch.setattr(evaluator, "resolve_native", resolve)
    output = tmp_path / "new-last-controller"
    queue = evaluator.prepare(controller, newruntime, launch, policy(), output)
    return queue, output


def materialize(queue):
    ep = queue["evaluation_plan"]
    root = Path(ep["output_root"])
    root.mkdir()
    atomic_json(root / "plan.json", ep)
    order = ["conditionA", "conditionB", "conditionC"]
    shards = []
    for index in range(2):
        rows = []
        for condition in order[index::2]:
            metrics = {
                "condition_id": condition,
                "results": [
                    {"metric_id": metric, "value": 0.5, "reason": None, "gene_count": 5000}
                    for metric in sorted(evaluator.METRICS)
                ],
            }
            controls = [f"cell-{i}" for i in range(300)]
            truth = ["truth1", "truth2"]
            rows.append(
                {
                    "condition_id": condition,
                    "loss": 1.0,
                    "metrics": metrics,
                    "metric_gene_groups": {
                        "seen_expression": metrics,
                        "unseen_expression": metrics,
                    },
                    "control_row_ids": controls,
                    "control_row_ids_sha256": sha256_json(controls),
                    "truth_row_ids": truth,
                    "truth_row_ids_sha256": sha256_json(truth),
                }
            )
        result = {
            "split": "test",
            "selection_gene_ids": None,
            "control_manifest_sha256": "d" * 64,
            "reference_sha256": "e" * 64,
            "query_recipe": {"name": "ordered_disjoint_blocks", "query_count": 5000},
            "conditions": rows,
            "metrics": [],
            "prediction_loss": 1.0,
            "metric_gene_groups": {
                g: {"gene_count": 5000, "gene_ids_sha256": "f" * 64, "metrics": []}
                for g in ("seen_expression", "unseen_expression")
            },
            "expression_exposure": {"checkpoint_epoch": 4},
        }
        shard = {
            "frozen_condition_order": order,
            "evaluation_environment": {"lock_file_sha256": "1" * 64},
            "result": result,
            "runtime_measurement": {
                "condition_count": len(rows),
                "cell_batch": 300,
                "cpu_training_state": True,
                "cuda_memory_fraction": 0.9,
            },
        }
        atomic_json(root / f"worker-{index}.json", shard)
        shards.append(shard)
    merged = evaluator.merge_evaluation_shards(
        [s["result"] for s in shards], ordered_conditions=tuple(order)
    )
    merged["expression_exposure"] = shards[0]["result"]["expression_exposure"]
    atomic_json(
        root / "COMPLETE.json",
        {
            "plan": ep,
            "zero_pkl": True,
            "evaluation_environment": shards[0]["evaluation_environment"],
            "result": merged,
            "worker_runtime_measurements": [s["runtime_measurement"] for s in shards],
        },
    )


def test_real_native_last_resolution_and_real_shard_metrics(prepared):
    queue, _output = prepared
    assert evaluator.verify(queue, fresh=True) == queue["plan"]
    assert queue["training"]["test_roles"] == ["best", "last"]
    assert queue["evaluation_plan"]["checkpoint_role"] == "last"
    command = evaluator.child_command(queue)
    assert "execute_evaluation_plan" in command[2] and "resolve_evaluation_plan" in command[2]
    compile(command[2], "native-last-child", "exec")
    assert "assert p" not in command[2]
    assert "finalize_deferred_run" not in command[2] and "run_deferred_postfit" not in command[2]
    materialize(queue)
    completed = evaluator.validate_complete(queue)
    assert len(completed["result"]["metrics"]) == 6
    assert completed["checkpoint_epoch"] == 4
    assert not (Path(queue["plan"]["run_root"]) / "COMPLETE.json").exists()
    assert not (Path(queue["plan"]["run_root"]) / "fit/last-test.json").exists()


@pytest.mark.parametrize(
    "key", ["config", "runtime", "publication", "checkpoint", "journal", "controller_script"]
)
def test_input_seal_drift(prepared, key):
    queue, _ = prepared
    plan = queue["plan"]
    path = {
        "config": Path(plan["config"]),
        "runtime": Path(plan["runtime"]),
        "publication": Path(plan["publication"]),
        "checkpoint": Path(plan["run_root"]) / "fit/last.pt",
        "journal": Path(plan["run_root"]) / "fit/epoch_state.json",
        "controller_script": Path(queue["controller"]["script"]["path"]),
    }[key]
    path.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        evaluator.verify(queue, fresh=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_role", "best"),
        ("checkpoint_epoch", 3),
        ("gpu", ["0"]),
        ("data_root", "/data/elsewhere"),
        ("cuda_memory_fraction", 1.0),
        ("diagnostic_only", True),
    ],
)
def test_native_scope_drift(prepared, field, value):
    queue, _ = prepared
    queue = copy.deepcopy(queue)
    queue["evaluation_plan"][field] = value
    with pytest.raises(ValueError, match="contract"):
        evaluator.verify(queue, fresh=True)


@pytest.mark.parametrize("failure", ["metric", "shard", "pkl", "training_pkl", "failure", "counts"])
def test_native_completion_rejects_missing_metrics_shards_and_pkl(prepared, failure):
    queue, _ = prepared
    materialize(queue)
    root = Path(queue["evaluation_plan"]["output_root"])
    if failure == "metric":
        receipt = read_json(root / "COMPLETE.json")
        receipt["result"]["metrics"].pop()
        atomic_json(root / "COMPLETE.json", receipt)
    elif failure == "counts":
        shard = read_json(root / "worker-0.json")
        shard["runtime_measurement"]["condition_count"] = 1
        atomic_json(root / "worker-0.json", shard)
    elif failure == "shard":
        (root / "worker-1.json").unlink()
    elif failure == "pkl":
        (root / "unwanted.pkl").write_bytes(b"x")
    elif failure == "training_pkl":
        (Path(queue["plan"]["run_root"]) / "unwanted.pkl").write_bytes(b"x")
    else:
        atomic_json(root / "FAILURE.json", {"error": "failed"})
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.validate_complete(queue)


def execute_mocks(prepared, monkeypatch, code=0):
    queue, _ = prepared
    locks = []
    killed = []
    alive = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", evaluator.ALLOCATOR)

    @contextlib.contextmanager
    def locked(path):
        locks.append(path.name)
        yield

    monkeypatch.setattr(evaluator, "lock", locked)
    monkeypatch.setattr(evaluator, "snapshot", lambda pid: {"processes": []})
    monkeypatch.setattr(evaluator, "admit", lambda *a: None)
    info = {"pid": 201, "uid": os.getuid(), "start_ticks": "123", "command_sha256": "exact"}
    monkeypatch.setattr(
        evaluator, "process_info", lambda pid: {**info, "pid": pid} if alive else None
    )
    monkeypatch.setattr(evaluator, "parent_map", lambda: {201: os.getpid(), 202: 201})
    monkeypatch.setattr(evaluator, "_capture_process_identity", lambda child: info)

    class Child:
        pid = 201

        def poll(self):
            return None if alive else code

        def wait(self, timeout=None):
            return code

        def terminate(self):
            nonlocal alive
            alive = False

        kill = terminate

    def launch(command, **kw):
        assert command == evaluator.child_command(queue)
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kw["env"]["PYTHONPATH"] == str(Path(queue["plan"]["repository_root"]) / "src")
        assert kw["cwd"] == queue["plan"]["repository_root"]
        return Child()

    monkeypatch.setattr(evaluator.subprocess, "Popen", launch)

    def release(seconds):
        nonlocal alive
        assert seconds == 5
        alive = False
        if code == 0:
            materialize(queue)

    monkeypatch.setattr(evaluator.time, "sleep", release)
    monkeypatch.setattr(evaluator, "terminate_tree", lambda pid, expected: killed.append(pid))
    return locks, killed


def test_exit0_real_complete_and_existing_dual_leases(prepared, monkeypatch):
    queue, output = prepared
    locks, killed = execute_mocks(prepared, monkeypatch)
    evaluator.execute(queue, output)
    assert read_json(output / "COMPLETE.json")["scientific_complete"]
    assert locks == ["controller.lock", "v2-gpu-0.lock", "v2-gpu-1.lock"] and killed == []


def test_guard_failure_exact_owned_cleanup(prepared, monkeypatch):
    queue, output = prepared
    _, killed = execute_mocks(prepared, monkeypatch)
    calls = 0

    def admission(*a):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("physical reserve exhausted")

    monkeypatch.setattr(evaluator, "admit", admission)
    with pytest.raises(RuntimeError, match="reserve"):
        evaluator.execute(queue, output)
    assert set(killed) == {201, 202}
    assert not (output / "COMPLETE.json").exists()


def test_nonzero_no_retry(prepared, monkeypatch):
    queue, output = prepared
    execute_mocks(prepared, monkeypatch, code=1)
    with pytest.raises(RuntimeError, match="no retry"):
        evaluator.execute(queue, output)
    assert read_json(output / "FAILURE.json")["exit_code"] == 1


@pytest.mark.parametrize("change", ["visible_cuda", "fraction", "uuid"])
def test_cpu_and_policy_safety_before_launch(prepared, monkeypatch, change):
    queue, output = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", evaluator.ALLOCATOR)
    if change == "visible_cuda":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    else:
        queue = copy.deepcopy(queue)
        if change == "fraction":
            queue["shared_gpu_policy"]["max_own_memory_fraction"] = 1.0
        else:
            queue["shared_gpu_policy"]["devices"][1]["gpu_uuid"] = "GPU-a"
        queue["policy_sha256"] = sha256_json(queue["shared_gpu_policy"])
        atomic_json(output / "queue.json", queue)
    monkeypatch.setattr(
        evaluator.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError):
        evaluator.execute(queue, output)


def test_resealed_native_source_payload_drift_rejected(prepared):
    queue, _output = prepared
    queue = copy.deepcopy(queue)
    queue["evaluation_plan"]["evaluation_source"]["tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract"):
        evaluator.verify(queue, fresh=True)


def test_matching_shards_and_merged_receipt_still_require_ordered_control_hash(prepared):
    queue, _output = prepared
    materialize(queue)
    root = Path(queue["evaluation_plan"]["output_root"])
    shard = read_json(root / "worker-0.json")
    shard["result"]["conditions"][0]["control_row_ids_sha256"] = "0" * 64
    atomic_json(root / "worker-0.json", shard)
    receipt = read_json(root / "COMPLETE.json")
    receipt["result"]["conditions"][0]["control_row_ids_sha256"] = "0" * 64
    atomic_json(root / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="ordered row"):
        evaluator.validate_complete(queue)


def test_existing_interrupted_native_best_outputs_are_sealed_and_preserved(prepared):
    previous, _output = prepared
    root = Path(previous["plan"]["run_root"])
    partial = root / "postfit-best"
    partial.mkdir()
    historical = partial / "plan.json"
    atomic_json(historical, {"checkpoint_role": "best", "old_interrupted": True})
    digest = sha256_file(historical)
    controller = previous["controller"]
    output = Path(previous["output"]).parent / "fresh-last-after-partial"
    queue = evaluator.prepare(
        Path(controller["repository_root"]),
        Path(controller["runtime"]["path"]),
        Path(previous["launch"]["path"]),
        previous["shared_gpu_policy"],
        output,
    )
    assert str(historical.resolve()) in [entry["path"] for entry in queue["original_receipts"]]
    materialize(queue)
    evaluator.verify(queue, fresh=False)
    evaluator.validate_complete(queue)
    assert sha256_file(historical) == digest
    assert not (root / "fit/last-test.json").exists()
    assert not (root / "COMPLETE.json").exists()
    atomic_json(historical, {"changed": True})
    with pytest.raises(ValueError, match="sealed"):
        evaluator.verify(queue, fresh=False)
