"""Original epoch4 BR data stay immutable until its native capped evaluation."""

from __future__ import annotations

import contextlib
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import evaluate_completed_br as evaluator  # noqa: E402


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


def publication(path, commit):
    atomic_json(
        path,
        {
            "schema_version": "source-publication-receipt-v1",
            "repository": "https://github.com/elan6666/GraD-Pert.git",
            "remote_url": "https://github.com/elan6666/GraD-Pert.git",
            "remote_ref": "refs/heads/main",
            "published_commit": commit,
            "tree_sha256": "a" * 64,
            "verification_method": "git_ls_remote_live",
            "verified_unix": 1000.0,
        },
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "SERVER_ROOT", tmp_path)
    science = tmp_path / "scientific-b37"
    config = science / evaluator.BR_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text((ROOT / evaluator.BR_CONFIG).read_text())
    runs = tmp_path / "runs"
    run = runs / "original-BR"
    fit = run / "fit"
    fit.mkdir(parents=True)
    checkpoint = fit / "epoch-0004.pt"
    checkpoint.write_bytes(b"synthetic-complete-training-checkpoint")
    (fit / "last.pt").write_bytes(checkpoint.read_bytes())
    source_publication = tmp_path / "scientific-publication.json"
    controller_publication = tmp_path / "controller-publication.json"
    publication(source_publication, evaluator.SCIENTIFIC_COMMIT)
    publication(controller_publication, "c" * 40)
    runtime = tmp_path / "original-runtime.json"
    data_root = tmp_path / "scientific-data"
    data_root.mkdir()
    atomic_json(
        runtime,
        {
            "repository_root": str(science),
            "data_root": str(data_root),
            "runs_root": str(runs),
            "publication_receipt": str(source_publication),
            "publication_sha256": sha256_file(source_publication),
        },
    )
    plan = {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "resolved_config": yaml.safe_load(config.read_text()),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "source_commit": evaluator.SCIENTIFIC_COMMIT,
        "repository_root": str(science),
        "data_root": str(data_root),
        "run_id": "original-BR",
        "run_root": str(run),
        "test_root": str(runs / "original-BR-test"),
        "gpu": "0,1",
        "seed": 1,
        "publication": str(source_publication),
        "publication_sha256": sha256_file(source_publication),
        "genept_receipt": None,
        "genept_sha256": None,
        "postfit_policy": "deferred",
        "cuda_memory_fraction": 0.85,
        "resource_schedule": {"fit_fraction": 0.85, "capped_eval_fraction": 0.25},
    }
    atomic_json(run / "launch.json", plan)
    atomic_json(run / "resolved_config.json", plan["resolved_config"])
    identity = {
        "config_sha256": plan["config_sha256"],
        "source": {
            "commit": evaluator.SCIENTIFIC_COMMIT,
            "published_commit": evaluator.SCIENTIFIC_COMMIT,
            "dirty": False,
        },
        "run_seed": 1,
        "data_root": str(data_root),
    }
    atomic_json(run / "run_manifest.json", identity)
    last = {"epoch": 4, "file": checkpoint.name, "sha256": sha256_file(checkpoint)}
    atomic_json(
        fit / "epoch_state.json",
        {"identity": identity, "epoch": 4, "budget": [4, 250], "last": last, "best": None},
    )
    atomic_json(fit / "history.json", [{"epoch": epoch} for epoch in range(1, 5)])
    atomic_json(run / "TRAIN_COMPLETE.json", training_stage_evidence(run, identity, 4))
    parent = tmp_path / "retired-original-queue"
    parent.mkdir()
    launch = parent / "BR.launch-plan.json"
    atomic_json(launch, plan)
    monkeypatch.setattr(evaluator, "ORIGINAL_PLAN", launch)

    def inspect(source, **kwargs):
        receipt = read_json(Path(kwargs["publication_receipt"]))
        assert receipt["repository"] == kwargs["expected_repository"]
        assert (
            sha256_file(Path(kwargs["publication_receipt"]))
            == kwargs["expected_publication_receipt_sha256"]
        )
        return SimpleNamespace(
            dirty=False, commit=evaluator.SCIENTIFIC_COMMIT if Path(source) == science else "c" * 40
        )

    monkeypatch.setattr(evaluator, "inspect_source_identity", inspect)
    output = tmp_path / "fresh-evaluation-controller"
    queue = evaluator.prepare(launch, output, policy(), controller_publication, last["sha256"])
    return queue, output


def test_real_training_stage_original_command_and_seals(prepared):
    queue, output = prepared
    assert evaluator.verify(queue, fresh=True) == queue["plan"]
    command = evaluator.child_command(queue)
    assert command == [
        sys.executable,
        str(Path(queue["plan"]["repository_root"]) / "scripts/v2/run_unified_ddp.py"),
        "--evaluate",
        queue["launch"]["path"],
        "--attempt",
        "1",
    ]
    assert "--uncapped" not in command and "--queue" not in command
    assert queue["eval_fraction"] == 0.25
    assert queue["training"]["epoch"] == 4 and queue["training"]["test_roles"] == ["last"]
    assert read_json(output / "queue.json") == queue


@pytest.mark.parametrize(
    "file", ["launch", "checkpoint", "runtime", "config", "publication", "history"]
)
def test_sealed_native_input_drift_rejected(prepared, file):
    queue, _ = prepared
    plan = queue["plan"]
    path = {
        "launch": Path(queue["launch"]["path"]),
        "checkpoint": Path(plan["run_root"]) / "fit/last.pt",
        "runtime": Path(plan["runtime"]),
        "config": Path(plan["config"]),
        "publication": Path(plan["publication"]),
        "history": Path(plan["run_root"]) / "fit/history.json",
    }[file]
    path.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        evaluator.verify(queue, fresh=True)


@pytest.mark.parametrize(
    "artifact",
    [
        "postfit-last-attempt1",
        "fit/last-test.json",
        "COMPLETE.json",
        "POSTFIT_ATTEMPT1_FAILURE.json",
        "test_root",
    ],
)
def test_original_eval_outputs_must_remain_fresh(prepared, artifact):
    queue, _ = prepared
    plan = queue["plan"]
    path = Path(plan["test_root"]) if artifact == "test_root" else Path(plan["run_root"]) / artifact
    path.mkdir() if artifact in ("postfit-last-attempt1", "test_root") else path.write_text("{}")
    with pytest.raises(FileExistsError):
        evaluator.verify(queue, fresh=True)


@pytest.mark.parametrize("digest", ["short", "0" * 64])
def test_full_exact_checkpoint_sha_required(prepared, digest):
    queue, _ = prepared
    with pytest.raises(ValueError, match=r"SHA256|checkpoint"):
        evaluator.scientific_contract(queue["plan"], digest)


def test_source_policy_and_protected_root_rejected(prepared, monkeypatch):
    queue, _ = prepared
    wrong = copy.deepcopy(queue)
    wrong["shared_gpu_policy"]["max_own_memory_fraction"] = 0.9
    wrong["policy_sha256"] = sha256_json(wrong["shared_gpu_policy"])
    with pytest.raises(ValueError):
        evaluator.verify(wrong, fresh=True)
    wrong = copy.deepcopy(queue)
    wrong["output"] = str(Path(wrong["plan"]["run_root"]) / "forbidden-controller")
    with pytest.raises(ValueError, match="preserve"):
        evaluator.verify(wrong, fresh=True)
    monkeypatch.setattr(
        evaluator,
        "inspect_source_identity",
        lambda *a, **k: SimpleNamespace(dirty=True, commit=evaluator.SCIENTIFIC_COMMIT),
    )
    with pytest.raises(ValueError, match="source"):
        evaluator.verify(queue, fresh=True)


def execute_mocks(prepared, monkeypatch, code=0, materialize=None):
    queue, _output = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", evaluator.ALLOCATOR)
    locks, killed = [], []

    @contextlib.contextmanager
    def locked(path):
        locks.append(path.name)
        yield

    monkeypatch.setattr(evaluator, "lock", locked)
    monkeypatch.setattr(evaluator, "snapshot", lambda pid: {"processes": [], "gpus": {}})
    monkeypatch.setattr(evaluator, "admit", lambda *a: None)
    alive = True
    info = {"pid": 201, "uid": os.getuid(), "start_ticks": "123", "command_sha256": "exact"}
    monkeypatch.setattr(evaluator, "parent_map", lambda: {201: os.getpid(), 202: 201})
    monkeypatch.setattr(
        evaluator, "process_info", lambda pid: {**info, "pid": pid} if alive else None
    )
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

    def launch(command, **kwargs):
        assert command == evaluator.child_command(queue)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["PYTHONPATH"] == str(Path(queue["plan"]["repository_root"]) / "src")
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == evaluator.ALLOCATOR
        return Child()

    monkeypatch.setattr(evaluator.subprocess, "Popen", launch)

    def release(seconds):
        nonlocal alive
        assert seconds == 5
        alive = False
        if materialize is not None:
            materialize()

    monkeypatch.setattr(evaluator.time, "sleep", release)
    monkeypatch.setattr(evaluator, "terminate_tree", lambda pid, expected: killed.append(pid))
    return locks, killed


def test_native_exit0_completed_validation_and_same_leases(prepared, monkeypatch):
    queue, output = prepared
    locks, killed = execute_mocks(prepared, monkeypatch)
    validated = []

    def completed(plan):
        assert plan == queue["plan"]
        validated.append(plan["run_id"])
        return {
            "run_id": plan["run_id"],
            "metrics": [1] * 6,
            "metric_gene_groups": {"seen_expression": {}, "unseen_expression": {}},
        }

    monkeypatch.setattr(evaluator, "validate_completed", completed)
    evaluator.execute(queue, output)
    receipt = read_json(output / "COMPLETE.json")
    assert receipt["scientific_complete"] and receipt["exit_code"] == 0
    assert validated == ["original-BR"] and killed == []
    assert locks == ["controller.lock", "v2-gpu-0.lock", "v2-gpu-1.lock"]


def test_nonzero_native_exit_never_validates_or_retries(prepared, monkeypatch):
    queue, output = prepared
    execute_mocks(prepared, monkeypatch, code=1)
    monkeypatch.setattr(
        evaluator, "validate_completed", lambda *a: pytest.fail("nonzero is not success")
    )
    with pytest.raises(RuntimeError, match="no retry"):
        evaluator.execute(queue, output)
    assert read_json(output / "FAILURE.json")["exit_code"] == 1
    assert not (output / "COMPLETE.json").exists()


def test_active_resource_guard_terminates_only_exact_owned_tree(prepared, monkeypatch):
    queue, output = prepared
    _, killed = execute_mocks(prepared, monkeypatch)
    count = 0

    def admission(*a):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("physical free reserve exhausted")

    monkeypatch.setattr(evaluator, "admit", admission)
    monkeypatch.setattr(
        evaluator, "validate_completed", lambda *a: pytest.fail("guard failure cannot complete")
    )
    with pytest.raises(RuntimeError, match="reserve"):
        evaluator.execute(queue, output)
    assert set(killed) == {201, 202}
    assert not read_json(output / "FAILURE.json")["scientific_complete"]


def test_visible_controller_cuda_rejected_before_child(prepared, monkeypatch):
    queue, output = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        evaluator.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError, match="CPU"):
        evaluator.execute(queue, output)


def test_real_completed_validator_rejects_missing_native_receipts(prepared, monkeypatch):
    queue, output = prepared
    execute_mocks(prepared, monkeypatch)
    with pytest.raises(ValueError, match="terminal receipt"):
        evaluator.execute(queue, output)
    assert not (output / "COMPLETE.json").exists()


def test_real_completed_validator_requires_all18_metrics_and_zero_pkl(prepared, monkeypatch):
    import run_unified_group

    queue, output = prepared
    plan = queue["plan"]
    root = Path(plan["run_root"])
    metrics = [
        {
            "metric_id": metric_id,
            "macro_mean": 0.25,
            "finite_condition_count": 10,
            "total_condition_count": 10,
        }
        for metric_id in run_unified_group.METRICS
    ]
    result = {
        "metrics": metrics,
        "metric_gene_groups": {
            group: {"metrics": copy.deepcopy(metrics)}
            for group in ("seen_expression", "unseen_expression")
        },
    }

    def materialize():
        atomic_json(root / "COMPLETE.json", {"zero_pkl": True})
        atomic_json(root / "fit/last-test.json", {"result": result})

    execute_mocks(prepared, monkeypatch, materialize=materialize)
    monkeypatch.setattr(
        run_unified_group,
        "collect_run",
        lambda run: [
            {
                "role": "last",
                "status": "complete",
                "checkpoint_epoch": 4,
                "training_sha": evaluator.SCIENTIFIC_COMMIT,
                "evaluation_sha": evaluator.SCIENTIFIC_COMMIT,
                "config_sha256": plan["config_sha256"],
            }
        ],
    )
    evaluator.execute(queue, output)
    complete = read_json(output / "COMPLETE.json")
    assert len(complete["completed"]["metrics"]) == 6
    assert (
        sum(len(group["metrics"]) for group in complete["completed"]["metric_gene_groups"].values())
        == 12
    )
    assert complete["completed"]["complete_sha256"] == sha256_file(root / "COMPLETE.json")
    (root / "unexpected.pkl").write_text("synthetic")
    with pytest.raises(ValueError, match="PKL"):
        run_unified_group.validate_completed(plan)


def test_fresh_prepare_cannot_reuse_controller_root(prepared):
    queue, output = prepared
    with pytest.raises(ValueError, match="fresh"):
        evaluator.prepare(
            Path(queue["launch"]["path"]),
            output,
            queue["shared_gpu_policy"],
            Path(queue["controller"]["publication"]["path"]),
            queue["checkpoint_sha256"],
        )


def test_recycled_child_identity_never_signals_replacement(prepared, monkeypatch):
    queue, output = prepared
    execute_mocks(prepared, monkeypatch)
    current = {
        "pid": 201,
        "uid": os.getuid(),
        "start_ticks": "recycled",
        "command_sha256": "new-command",
    }
    monkeypatch.setattr(evaluator, "process_info", lambda pid: current)
    refused = []

    def terminate(pid, expected):
        assert expected["start_ticks"] == "123"
        assert expected["command_sha256"] == "exact"
        refused.append(pid)
        raise RuntimeError("refuse recycled process ID")

    monkeypatch.setattr(evaluator, "terminate_tree", terminate)
    with pytest.raises(RuntimeError, match="identity"):
        evaluator.execute(queue, output)
    failure = read_json(output / "FAILURE.json")
    assert refused == [201]
    assert failure["cleanup_errors"] == ["refuse recycled process ID"]
    assert not failure["scientific_complete"]
