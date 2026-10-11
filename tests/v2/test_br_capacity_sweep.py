from __future__ import annotations

import contextlib
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_br_capacity_sweep as sweep  # noqa: E402


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


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.json"
    publication = tmp_path / "publication.json"
    atomic_json(publication, {"commit": "new-capacity-source"})
    atomic_json(
        runtime,
        {
            "repository_root": str(ROOT),
            "runs_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "publication_receipt": str(publication),
            "publication_sha256": sha256_file(publication),
        },
    )
    monkeypatch.setattr(sweep, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(
        sweep,
        "inspect_source_identity",
        lambda *a, **k: SimpleNamespace(dirty=False, commit="new-capacity-source"),
    )
    count = 0

    def resolve(args):
        nonlocal count
        count += 1
        return {
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "source_commit": "new-capacity-source",
            "repository_root": str(ROOT),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "resolved_config": yaml.safe_load(args.config.read_text()),
            "publication": str(publication),
            "publication_sha256": sha256_file(publication),
            "data_root": str(tmp_path / "data"),
            "run_root": str(tmp_path / f"run{count}"),
            "test_root": str(tmp_path / f"test{count}"),
            "run_id": f"fresh{count}",
            "gpu": "0,1",
            "seed": 1,
            "postfit_policy": "deferred",
        }

    monkeypatch.setattr(sweep, "resolve_plan", resolve)
    output = tmp_path / "queue"
    queue = sweep.prepare(
        ROOT, ROOT / "configs/v2/br_capacity_20261012/manifest.json", runtime, output, policy()
    )
    return queue, output


def test_prepare_exact_native_fresh_binding(prepared):
    queue, output = prepared
    assert queue["schema"] == "unified-ddp-queue-1"
    assert queue["source_commit"] == "new-capacity-source"
    assert len(queue["rows"]) == len({r["plan"]["run_id"] for r in queue["rows"]}) == 53
    assert read_json(output / "queue.json") == queue
    sweep.verify(queue)


@pytest.mark.parametrize("field", ["runtime", "publication", "config"])
def test_sealed_file_drift(prepared, field, tmp_path):
    queue, _ = prepared
    if field == "runtime":
        path = Path(queue[field])
    else:
        path = tmp_path / f"{field}.json"
        row = queue["rows"][0]["plan"]
        path.write_text(Path(row[field]).read_text())
        row[field] = str(path)
    path.write_text("{}")
    with pytest.raises(ValueError):
        sweep.verify(queue)


def test_reject_scope_and_identity(prepared):
    queue, _ = prepared
    bad = copy.deepcopy(queue)
    bad["rows"][1]["plan"]["run_id"] = bad["rows"][0]["plan"]["run_id"]
    with pytest.raises(ValueError, match="binding"):
        sweep.verify(bad)
    bad = copy.deepcopy(queue)
    bad["rows"][0]["plan"]["source_commit"] = "old-b37"
    with pytest.raises(ValueError, match="binding"):
        sweep.verify(bad)


def test_manifest_only_two_entries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = read_json(ROOT / "configs/v2/br_capacity_20261012/manifest.json")
    original["rows"] = copy.deepcopy(original["rows"])
    for relative in [original["parent"], *(r["config"] for r in original["rows"])]:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text())
    row = original["rows"][1]
    config_path = source / row["config"]
    raw = yaml.safe_load(config_path.read_text())
    raw["model"]["parameters"]["accumulation"]["value"] = 3
    config_path.write_text(yaml.safe_dump(raw))
    row["sha256"] = sha256_file(config_path)
    manifest = source / "manifest.json"
    atomic_json(manifest, original)
    with pytest.raises(ValueError, match="only micro/global"):
        sweep.manifest_contract(source, manifest)


def test_selection_first_oom_refines_without_repeat():
    results = []
    visited = []
    while (micro := sweep.next_candidate(results)) is not None:
        visited.append(micro)
        results.append({"microbatch": micro, "status": "passed" if micro <= 108 else "oom"})
    assert visited[:7] == [48, 56, 64, 80, 96, 112, 104]
    assert len(set(visited)) == len(visited)
    assert max(r["microbatch"] for r in results if r["status"] == "passed") == 108
    assert min(r["microbatch"] for r in results if r["status"] == "oom") == 112
    assert sweep.next_candidate([{"microbatch": 48, "status": "oom"}]) is None
    assert (
        sweep.next_candidate([{"microbatch": m, "status": "passed"} for m in sweep.COARSE]) is None
    )


def successful_receipt(queue, row):
    plan = row["plan"]
    return {
        "kind": "preflight_only",
        "status": "passed",
        "steps_requested": 10,
        "steps_completed": 10,
        "resume_checkpoint_sha256": "checkpoint",
        "inference_exercised": True,
        "inference_shape": [300, 5000],
        "source": {
            "dirty": False,
            "commit": plan["source_commit"],
            "published_commit": plan["source_commit"],
            "publication_receipt_sha256": plan["publication_sha256"],
        },
        "config_sha256": plan["config_sha256"],
        "data_root": plan["data_root"],
        "data": {"run_seed": 1},
        "world_size": 2,
        "peak_allocated_bytes": 1,
        "last_terms": {"gradient_norm": 1.0, "ssl1": 2.0, "ssl2": 3.0, "prediction": 4.0},
        "shared_gpu_policy": queue["shared_gpu_policy"],
        "shared_queue_sha256": sha256_file(Path(queue["output"]) / "queue.json"),
        "end_to_end_training_cells_per_second": 20.0,
        "startup_seconds_until_training": 2.0,
        "peak_reserved_bytes": 2,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"steps_completed": 11},
        {"inference_shape": [300, 4999]},
        {"resume_checkpoint_sha256": None},
        {"last_terms": {"gradient_norm": float("nan")}},
        {"shared_queue_sha256": "drift"},
    ],
)
def test_exact_native_receipt_rejects_invalid_evidence(prepared, change):
    queue, output = prepared
    row = queue["rows"][0]
    path = output / "receipt.json"
    receipt = successful_receipt(queue, row)
    receipt.update(change)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        sweep.validate_receipt(queue, row, {"receipt": str(path), "sha256": sha256_file(path)})


def test_receipt_pass_and_native_shared_binding(prepared):
    from shared_gpu import validate_probe_binding

    queue, output = prepared
    row = queue["rows"][0]
    path = output / "receipt.json"
    atomic_json(path, successful_receipt(queue, row))
    sweep.validate_receipt(queue, row, {"receipt": str(path), "sha256": sha256_file(path)})
    args = SimpleNamespace(
        config=Path(row["plan"]["config"]),
        steps=10,
        gpu="0,1",
        data_root=Path(row["plan"]["data_root"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        output=output / "m48-preflight",
        shared_queue=output / "queue.json",
    )
    validate_probe_binding(queue, args, queue["source_commit"], "preflight_only", 2)
    args.output = output / "other"
    with pytest.raises(ValueError):
        validate_probe_binding(queue, args, queue["source_commit"], "preflight_only", 2)


def test_oom_requires_bound_failed_native_receipt(prepared):
    queue, output = prepared
    row = queue["rows"][0]
    failed = output / "failure"
    failed.mkdir()
    receipt = successful_receipt(queue, row)
    receipt.update(status="failed", error_type="RuntimeError", error="unrelated failure")
    path = failed / "rank-0-failure.json"
    atomic_json(path, receipt)
    assert not sweep.is_oom(row, failed)
    receipt.update(error_type="OutOfMemoryError", error="CUDA out of memory")
    atomic_json(path, receipt)
    assert sweep.is_oom(row, failed)
    receipt["config_sha256"] = "other"
    atomic_json(path, receipt)
    with pytest.raises(ValueError):
        sweep.is_oom(row, failed)


def execution_mocks(monkeypatch, queue):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", sweep.ALLOCATOR)
    locks = []

    @contextlib.contextmanager
    def locked(path):
        locks.append(path)
        yield

    monkeypatch.setattr(sweep, "lock", locked)
    monkeypatch.setattr(sweep, "snapshot", lambda pid: {"processes": [], "gpus": {}})
    monkeypatch.setattr(sweep, "admit", lambda p, o: None)
    return locks


def test_execute_leases_ceiling_and_fastest_separate(prepared, monkeypatch):
    queue, output = prepared
    locks = execution_mocks(monkeypatch, queue)
    starts = []

    def candidate(q, row, directory, guard):
        guard()
        starts.append(row["microbatch"])
        return {
            "microbatch": row["microbatch"],
            "status": "passed",
            "throughput": 1000 - row["microbatch"],
        }

    monkeypatch.setattr(sweep, "run_candidate", candidate)
    sweep.execute(queue, output)
    state = read_json(output / "COMPLETE.json")
    assert starts == sweep.COARSE
    assert [p.name for p in locks] == ["controller.lock", "v2-gpu-0.lock", "v2-gpu-1.lock"]
    assert state["highest_passed"] == 256 and state["fastest"]["microbatch"] == 48
    assert state["bounded_ceiling_not_maximum"] and not state["scientific_complete"]


def test_guard_failure_stops_queue_before_launch(prepared, monkeypatch):
    queue, output = prepared
    execution_mocks(monkeypatch, queue)

    def fail(*args):
        raise RuntimeError("UUID drift")

    monkeypatch.setattr(sweep, "admit", fail)
    monkeypatch.setattr(sweep, "run_candidate", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="UUID"):
        sweep.execute(queue, output)
    assert read_json(output / "FAILURE.json")["results"] == []


def test_native_command_and_active_guard_terminates_only_owned(prepared, monkeypatch):
    queue, output = prepared
    row = queue["rows"][0]
    info = {"pid": 200, "uid": os.getuid(), "start_ticks": "1", "command_sha256": "own"}
    monkeypatch.setattr(sweep, "parent_map", lambda: {200: os.getpid(), 201: 200})
    monkeypatch.setattr(sweep, "process_info", lambda pid: {**info, "pid": pid})
    monkeypatch.setattr(sweep, "_capture_process_identity", lambda child: info)
    calls = []

    class Child:
        pid = 200

        def poll(self):
            return None

    def spawn(command, **kwargs):
        assert "--shared-queue" in command and "--resource-owner-pid" in command
        assert command[command.index("--steps") + 1] == "10"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == sweep.ALLOCATOR
        return Child()

    monkeypatch.setattr(sweep.subprocess, "Popen", spawn)
    monkeypatch.setattr(sweep, "terminate_tree", lambda pid, identity: calls.append(pid))
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("owned cap exceeded")
        return {"processes": []}

    with pytest.raises(RuntimeError, match="cap"):
        sweep.run_candidate(queue, row, output, guard)
    assert set(calls) == {200, 201}


def test_mixed_oom_and_independent_failure_is_fatal(prepared):
    queue, output = prepared
    row = queue["rows"][0]
    failed = output / "mixed"
    failed.mkdir()
    receipt = successful_receipt(queue, row)
    receipt.update(status="failed", error_type="OutOfMemoryError", error="CUDA out of memory")
    atomic_json(failed / "rank-0-failure.json", receipt)
    receipt.update(error_type="ValueError", error="binding/source identity failed")
    atomic_json(failed / "rank-1-failure.json", receipt)
    assert not sweep.is_oom(row, failed)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_native_candidate_finishes_and_classifies_after_rank_release(prepared, monkeypatch, code):
    queue, output = prepared
    row = queue["rows"][0]
    alive = True
    info = {"pid": 200, "uid": os.getuid(), "start_ticks": "1", "command_sha256": "owned"}
    monkeypatch.setattr(sweep, "parent_map", lambda: {200: os.getpid(), 201: 200})
    monkeypatch.setattr(sweep, "process_info", lambda pid: {**info, "pid": pid} if alive else None)
    monkeypatch.setattr(sweep, "_capture_process_identity", lambda child: info)

    class Child:
        pid = 200

        def poll(self):
            return None if alive else code

        def wait(self):
            return code

    def spawn(command, **kwargs):
        native = output / "m48-preflight"
        native.mkdir()
        receipt = successful_receipt(queue, row)
        if code:
            receipt.update(
                status="failed",
                error_type="OutOfMemoryError" if code == 1 else "ValueError",
                error="CUDA out of memory" if code == 1 else "bad identity",
            )
            atomic_json(native / "rank-0-failure.json", receipt)
        atomic_json(native / "receipt.json", receipt)
        return Child()

    monkeypatch.setattr(sweep.subprocess, "Popen", spawn)

    def release(seconds):
        nonlocal alive
        assert seconds == 5
        alive = False

    monkeypatch.setattr(sweep.time, "sleep", release)
    monkeypatch.setattr(
        sweep, "terminate_tree", lambda *a: pytest.fail("released trees need no signal")
    )
    if code == 2:
        with pytest.raises(RuntimeError, match="non-OOM"):
            sweep.run_candidate(queue, row, output, lambda: {"processes": []})
    else:
        result = sweep.run_candidate(queue, row, output, lambda: {"processes": []})
        assert result["status"] == ("passed" if code == 0 else "oom")
        assert read_json(output / "m48.exit.json") == result
        assert (output / "m48-preflight/receipt.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "data_root",
        "runtime",
        "runtime_sha256",
        "publication",
        "publication_sha256",
        "repository_root",
    ],
)
def test_plan_runtime_binding_drift_rejected(prepared, field, tmp_path):
    queue, _ = prepared
    queue["rows"][0]["plan"][field] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="binding"):
        sweep.verify(queue)


def test_real_runtime_without_source_root_is_accepted(prepared):
    queue, _ = prepared
    runtime = read_json(Path(queue["runtime"]))
    assert "source_root" not in runtime
    assert runtime["repository_root"] == queue["source"]
    assert {"data_root", "runs_root", "publication_receipt", "publication_sha256"} <= runtime.keys()
    sweep.verify(queue)


def test_real_runtime_repository_root_mismatch_is_rejected(prepared, tmp_path):
    queue, _ = prepared
    path = Path(queue["runtime"])
    runtime = read_json(path)
    runtime["repository_root"] = str(tmp_path / "wrong-source")
    atomic_json(path, runtime)
    queue["runtime_sha256"] = sha256_file(path)
    for row in queue["rows"]:
        row["plan"]["runtime_sha256"] = queue["runtime_sha256"]
    with pytest.raises(ValueError, match="runtime source binding"):
        sweep.verify(queue)
