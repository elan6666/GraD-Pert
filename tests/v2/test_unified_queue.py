"""Synthetic queue tests: launch authority, barrier, lane isolation and terminal evidence."""

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
spec = importlib.util.spec_from_file_location(
    "unified_queue", ROOT / "scripts/v2/run_unified_group.py"
)
assert spec and spec.loader
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
MANIFEST = ROOT / "configs/v2/unified_first_20261010/manifest.json"
SHA = "a" * 40


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(queue, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", queue.ALLOCATOR)
    runtime, publication = tmp_path / "runtime.json", tmp_path / "publication.json"
    publication.write_text("{}")
    runtime.write_text(json.dumps({"publication_receipt": str(publication)}))
    calls = []

    def resolver(args):
        calls.append(args)
        run_id = f"fresh-{len(calls)}"
        return {
            "resolved_config": load_experiment_config(args.config).model_dump(mode="json"),
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "publication": str(publication),
            "publication_sha256": sha256_file(publication),
            "data_root": str(tmp_path / "data"),
            "run_root": str(tmp_path / run_id),
            "test_root": str(tmp_path / (run_id + "-test")),
            "run_id": run_id,
            "repository_root": str(ROOT),
            "source_commit": SHA,
            "gpu": args.gpu,
            "seed": args.seed,
        }

    planned = queue.prepare_queue(MANIFEST, runtime, ROOT, "0,1", resolver=resolver)
    directory = tmp_path / "queue"
    directory.mkdir()
    atomic_json(directory / "queue.json", planned)
    for row in planned["rows"]:
        atomic_json(directory / f"{row['name']}.launch-plan.json", row["plan"])
    return planned, directory, calls


def write_preflight(plan, output, **changes):
    output.mkdir(parents=True)
    receipt = {
        "kind": "preflight_only",
        "status": "passed",
        "steps_requested": 10,
        "steps_completed": 10,
        "resume_checkpoint_sha256": "b" * 64,
        "inference_exercised": True,
        "inference_shape": [300, 5000],
        "source": {
            "dirty": False,
            "commit": SHA,
            "published_commit": SHA,
            "publication_receipt_sha256": plan["publication_sha256"],
        },
        "config_sha256": plan["config_sha256"],
        "data_root": plan["data_root"],
        "data": {"run_seed": 1},
        "world_size": 1,
        "gpu": plan["gpu"],
        "peak_allocated_bytes": 1024,
        "last_terms": {"gradient_norm": 0.5},
        **changes,
    }
    path = output / "receipt.json"
    atomic_json(path, receipt)
    return {"receipt": str(path), "sha256": sha256_file(path)}


def test_prepare_assigns_exact_authorized_rows_to_two_independent_lanes(prepared):
    planned, _, calls = prepared
    assert [r["name"] for r in planned["rows"]] == list(queue.ARMS)
    assert [r["lane"] for r in planned["rows"]] == ["0", "1"] * 5
    assert len({r["plan"]["run_id"] for r in planned["rows"]}) == 10
    assert all(call.seed == 1 and call.gpu in ("0", "1") for call in calls)
    assert all(not Path(r["plan"]["run_root"]).exists() for r in planned["rows"])


@pytest.mark.parametrize("value", ["0", "0,0", "1,2", "0,1,2", "0, 1"])
def test_gpu_topology_cannot_expand_or_duplicate(value):
    with pytest.raises(ValueError, match="independent"):
        queue.parse_gpus(value)


@pytest.mark.parametrize(
    "output,gpu,expected",
    [
        ("0, 512\n1, 30000", "0", True),
        ("0, 513\n1, 0", "0", False),
        ("1, 0", "0", False),
        ("0, -1", "0", False),
        ("0, invalid", "0", False),
    ],
)
def test_idle_check_is_per_selected_gpu(output, gpu, expected):
    assert queue.gpu_idle(output, gpu) is expected


def test_child_environment_resets_ddp_and_pins_allocator_source_and_device(monkeypatch):
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_PORT"):
        monkeypatch.setenv(key, "2")
    monkeypatch.setenv("PYTHONPATH", "unrelated-source")
    env = queue.child_environment("/immutable-source", "1")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTORCH_ALLOC_CONF"] == queue.ALLOCATOR
    assert env["PYTHONPATH"] == "/immutable-source/src"
    assert all(key not in env for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_PORT"))


def test_preflight_is_direct_single_process_with_ten_updates(prepared):
    planned, directory, _ = prepared
    command = queue.probe_command(planned["rows"][1], directory)
    assert "torch.distributed.run" not in command
    assert command[command.index("--gpu") + 1] == "1"
    assert command[command.index("--steps") + 1] == "10"


@pytest.mark.parametrize(
    "changes",
    [
        {"steps_completed": 9},
        {"steps_requested": 128},
        {"kind": "integration_only"},
        {"inference_exercised": False},
        {"inference_shape": [299, 5000]},
        {"gpu": "1"},
        {"world_size": 2},
        {"sequence_checkpoint_disabled_diagnostic_only": True},
        {"resume_checkpoint_sha256": None},
    ],
)
def test_preflight_cannot_admit_short_wrong_gpu_diagnostic_or_unrestored_run(prepared, changes):
    planned, directory, _ = prepared
    plan = planned["rows"][0]["plan"]
    entry = write_preflight(plan, directory / "probe", **changes)
    with pytest.raises(ValueError):
        queue.validate_short_preflight(plan, entry)


def test_preflight_publication_identity_is_bound_to_saved_plan(prepared):
    planned, directory, _ = prepared
    plan = planned["rows"][0]["plan"]
    source = {
        "dirty": False,
        "commit": SHA,
        "published_commit": SHA,
        "publication_receipt_sha256": "different",
    }
    entry = write_preflight(plan, directory / "probe", source=source)
    with pytest.raises(ValueError, match="full-path"):
        queue.validate_short_preflight(plan, entry)


def test_verify_seal_checks_current_publication_and_config(prepared):
    planned, _, _ = prepared
    observed = []

    def inspector(*args, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(commit=SHA, dirty=False)

    queue.verify_seal(planned, source_inspector=inspector)
    assert observed[0]["formal"] is True
    planned["rows"][0]["plan"]["gpu"] = "1"
    with pytest.raises(ValueError, match="identity"):
        queue.verify_seal(planned, source_inspector=inspector)


def test_dirty_or_wrong_current_source_cannot_execute(prepared):
    planned, _, _ = prepared
    for identity in (
        SimpleNamespace(commit="b" * 40, dirty=False),
        SimpleNamespace(commit=SHA, dirty=True),
    ):
        with pytest.raises(ValueError, match="published"):
            queue.verify_seal(
                planned, source_inspector=lambda *a, identity=identity, **kw: identity
            )


def fake_runner(events, *, fail=None):
    event_lock = threading.Lock()
    active_gpus = set()

    def runner(row, directory, stage, state, cancel):
        gpu = row["lane"]
        with event_lock:
            assert gpu not in active_gpus
            active_gpus.add(gpu)
            events.append((stage, row["name"], "start", gpu))
            if stage == "formal":
                assert len(state.values["preflights"]) == 10
        try:
            time.sleep(0.002)
            if fail == (stage, row["name"]):
                raise RuntimeError("synthetic CUDA failure")
            if stage == "preflight":
                write_preflight(row["plan"], directory / (row["name"] + "-preflight"))
            with event_lock:
                events.append((stage, row["name"], "finish", gpu))
            return True
        finally:
            with event_lock:
                active_gpus.remove(gpu)

    return runner


def test_all_ten_gates_precede_formal_and_each_lane_remains_sequential(prepared):
    planned, directory, _ = prepared
    events = []
    queue.execute_queue(
        planned,
        directory,
        child_runner=fake_runner(events),
        seal_validator=lambda q: None,
        completion_validator=lambda p: {"run_id": p["run_id"]},
    )
    first_formal = next(i for i, e in enumerate(events) if e[0] == "formal")
    assert sum(e[:1] == ("preflight",) and e[2] == "finish" for e in events[:first_formal]) == 10
    for gpu in ("0", "1"):
        actual = [e[1] for e in events if e[0] == "formal" and e[2] == "start" and e[3] == gpu]
        assert actual == [r["name"] for r in planned["rows"] if r["lane"] == gpu]
        preflights = [
            e[1] for e in events if e[0] == "preflight" and e[2] == "start" and e[3] == gpu
        ]
        expected = [
            planned["rows"][i]["name"]
            for i in queue.PREFLIGHT_PRIORITY
            if planned["rows"][i]["lane"] == gpu
        ]
        assert preflights == expected
    complete = json.loads((directory / "COMPLETE.json").read_text())
    assert complete["status"] == "complete" and len(complete["completed"]) == 10
    assert not (directory / "FAILURE.json").exists()


def test_single_preflight_failure_cancels_later_launches_and_never_trains(prepared):
    planned, directory, _ = prepared
    events = []
    with pytest.raises(RuntimeError, match="no automatic retry"):
        queue.execute_queue(
            planned,
            directory,
            child_runner=fake_runner(events, fail=("preflight", "N0")),
            seal_validator=lambda q: None,
        )
    assert not any(e[0] == "formal" for e in events)
    assert not (directory / "COMPLETE.json").exists()
    failed = json.loads((directory / "FAILURE.json").read_text())
    assert failed["status"] == "failed" and failed["failures"][0]["row"] == "N0"
    digest = sha256_file(directory / "FAILURE.json")
    with pytest.raises(ValueError, match="immutable"):
        queue.execute_queue(planned, directory, seal_validator=lambda q: None)
    assert sha256_file(directory / "FAILURE.json") == digest


def test_preflight_only_then_formal_reuses_exact_saved_run_ids(prepared):
    planned, directory, _ = prepared
    events = []
    runner = fake_runner(events)
    queue.execute_queue(
        planned, directory, "preflight", child_runner=runner, seal_validator=lambda q: None
    )
    assert not (directory / "COMPLETE.json").exists()
    assert json.loads((directory / "state.json").read_text())["status"] == "preflight_complete"
    ids = [r["plan"]["run_id"] for r in planned["rows"]]
    queue.execute_queue(
        planned,
        directory,
        "formal",
        child_runner=runner,
        seal_validator=lambda q: None,
        completion_validator=lambda p: {"run_id": p["run_id"]},
    )
    complete = json.loads((directory / "COMPLETE.json").read_text())
    assert {v["run_id"] for v in complete["completed"].values()} == set(ids)


def test_formal_requires_all_preflights_and_their_sealed_index(prepared):
    planned, directory, _ = prepared
    with pytest.raises(FileNotFoundError):
        queue.execute_queue(planned, directory, "formal", seal_validator=lambda q: None)
    assert (directory / "FAILURE.json").exists()


def test_changed_preflight_index_cannot_start_formal(prepared):
    planned, directory, _ = prepared
    queue.execute_queue(
        planned, directory, "preflight", child_runner=fake_runner([]), seal_validator=lambda q: None
    )
    atomic_json(directory / "preflight-index.json", {"rows": []})
    with pytest.raises(ValueError, match="seal changed"):
        queue.execute_queue(planned, directory, "formal", seal_validator=lambda q: None)


def test_child_starts_only_after_gpu_and_row_leases_with_isolated_environment(
    prepared, monkeypatch
):
    planned, directory, _ = prepared
    row = planned["rows"][0]
    state = queue.QueueState(directory, planned)
    observed = []
    monkeypatch.setattr(queue, "verify_seal", lambda q: None)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: "0, 0\n1, 30000")

    def popen(command, **kwargs):
        observed.append((command, kwargs))
        assert len(kwargs["pass_fds"]) == 2
        for descriptor in kwargs["pass_fds"]:
            os.fstat(descriptor)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == queue.ALLOCATOR
        return SimpleNamespace(pid=12345, wait=lambda: 0)

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    assert queue.run_child(row, directory, "preflight", state, threading.Event())
    assert len(observed) == 1
    exit_receipt = json.loads((directory / "N0-preflight.exit.json").read_text())
    assert exit_receipt["exit_code"] == 0 and exit_receipt["gpu"] == "0"
    assert state.values["lanes"]["0"]["child_pid"] is None


def test_gpu_became_occupied_after_lease_preserves_work_and_never_spawns(prepared, monkeypatch):
    planned, directory, _ = prepared
    snapshots = iter(["0, 0\n1, 0", "0, 1000\n1, 0"])
    monkeypatch.setattr(queue, "verify_seal", lambda q: None)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: next(snapshots))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("GPU was occupied"))
    with pytest.raises(RuntimeError, match="existing work preserved"):
        queue.run_child(
            planned["rows"][0],
            directory,
            "preflight",
            queue.QueueState(directory, planned),
            threading.Event(),
        )


def test_saved_formal_plan_cannot_be_changed_between_preflight_and_fit(prepared, monkeypatch):
    planned, directory, _ = prepared
    queue.execute_queue(
        planned, directory, "preflight", child_runner=fake_runner([]), seal_validator=lambda q: None
    )
    atomic_json(directory / "N0.launch-plan.json", {"run_id": "substituted"})
    monkeypatch.setattr(queue, "verify_seal", lambda q: None)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: "0, 0\n1, 0")
    monkeypatch.setattr(
        queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("plan was substituted")
    )
    with pytest.raises(ValueError, match="launch plan changed"):
        queue.run_child(
            planned["rows"][0],
            directory,
            "formal",
            queue.QueueState(directory, planned),
            threading.Event(),
        )


def test_existing_formal_root_is_never_implicitly_resumed(prepared, monkeypatch):
    planned, directory, _ = prepared
    Path(planned["rows"][0]["plan"]["run_root"]).mkdir()
    monkeypatch.setattr(queue, "verify_seal", lambda q: None)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: "0, 0\n1, 0")
    monkeypatch.setattr(
        queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("run already exists")
    )
    with pytest.raises(FileExistsError, match="resume"):
        queue.run_child(
            planned["rows"][0],
            directory,
            "formal",
            queue.QueueState(directory, planned),
            threading.Event(),
        )


def terminal_fixture(plan):
    root = Path(plan["run_root"])
    (root / "fit").mkdir(parents=True)
    atomic_json(root / "COMPLETE.json", {"epoch": 6})
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "resolved_config.json", plan["resolved_config"])
    metrics = [
        {
            "metric_id": name,
            "macro_mean": 0.1,
            "finite_condition_count": 10,
            "total_condition_count": 12,
        }
        for name in sorted(queue.METRICS)
    ]
    result = {
        "metrics": metrics,
        "metric_gene_groups": {
            name: {"metrics": metrics, "gene_count": 100}
            for name in ("seen_expression", "unseen_expression")
        },
    }
    atomic_json(root / "fit/last-test.json", {"result": result})
    rows = [
        {
            "role": "last",
            "status": "complete",
            "checkpoint_epoch": 6,
            "training_sha": SHA,
            "evaluation_sha": SHA,
            "config_sha256": plan["config_sha256"],
        }
    ]
    return root, result, rows


def test_completion_requires_all_eighteen_metrics_and_effective_counts(prepared):
    planned, _, _ = prepared
    plan = planned["rows"][0]["plan"]
    _, _, rows = terminal_fixture(plan)
    receipt = queue.validate_completed(plan, collector=lambda root: rows)
    assert len(receipt["metrics"]) == 6
    assert len(receipt["metric_gene_groups"]) == 2


@pytest.mark.parametrize(
    "corruption",
    ["missing_group", "missing_metric", "counts", "pkl", "wrong_epoch", "wrong_source"],
)
def test_corrupt_terminal_evidence_is_not_success(prepared, corruption):
    planned, _, _ = prepared
    plan = planned["rows"][0]["plan"]
    root, result, rows = terminal_fixture(plan)
    if corruption == "missing_group":
        result["metric_gene_groups"].pop("unseen_expression")
    elif corruption == "missing_metric":
        result["metrics"].pop()
    elif corruption == "counts":
        result["metrics"][0]["finite_condition_count"] = 13
    elif corruption == "pkl":
        (root / "forbidden.pkl").write_bytes(b"not scientifically valid")
    elif corruption == "wrong_epoch":
        rows[0]["checkpoint_epoch"] = 5
    else:
        rows[0]["evaluation_sha"] = "b" * 40
    atomic_json(root / "fit/last-test.json", {"result": result})
    with pytest.raises(ValueError):
        queue.validate_completed(plan, collector=lambda _: rows)
