"""A completed BR is imported once; untouched peers keep the frozen b37 protocol."""

from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_unified_ddp as ddp  # noqa: E402
import run_unified_group  # noqa: E402


def dynamic_policy():
    return {
        "schema": "shared-gpu-admission-3",
        "devices": [{"gpu_index": 0, "gpu_uuid": "GPU-a"}, {"gpu_index": 1, "gpu_uuid": "GPU-b"}],
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "max_foreign_processes_per_gpu": 1,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
        "authorization_reference": ddp.RESOURCE_POLICY_AUTHORIZATION,
    }


@pytest.fixture
def completed_parent(tmp_path, monkeypatch):
    source, parent_root = tmp_path / "b37-source", tmp_path / ddp.COMPLETED_BR_PARENT_NAME
    source.mkdir()
    parent_root.mkdir()
    monkeypatch.setattr(ddp, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(ddp, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(ddp, "process_identity", lambda pid: None)
    # The original recovery chain is already tested in test_unified_ddp_queue;
    # this fixture isolates the new mode while retaining the native queue/state fields.
    monkeypatch.setattr(ddp, "verify_recovery", lambda parent: None)
    monkeypatch.setattr(
        ddp,
        "inspect_source_identity",
        lambda root, **kwargs: SimpleNamespace(
            dirty=False, commit=ddp.RECOVERY_SOURCE if Path(root) == source else "c" * 40
        ),
    )
    old_pub, new_pub, runtime = (
        source / "publication.json",
        tmp_path / "new-publication.json",
        tmp_path / "runtime.json",
    )
    atomic_json(old_pub, {"published_commit": ddp.RECOVERY_SOURCE})
    atomic_json(new_pub, {"published_commit": "c" * 40})
    runs, data = tmp_path / "runs", tmp_path / "data"
    atomic_json(
        runtime,
        {
            "repository_root": str(source),
            "data_root": str(data),
            "runs_root": str(runs),
            "publication_receipt": str(old_pub),
            "publication_sha256": sha256_file(old_pub),
        },
    )
    parent = {
        "schema": ddp.SCHEMA,
        "source": str(source),
        "source_commit": ddp.RECOVERY_SOURCE,
        "manifest": str(source / "manifest.json"),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "steps": 10,
        "initial_preflight": ["F1", "BR", "N1"],
        "rows": [],
        "deferred_arms": [],
        "mode": ddp.RECOVERY_MODE,
        "recovery": {"schema": ddp.RECOVERY_SCHEMA_V2, "ancestry": []},
        "shared_gpu_policy": dynamic_policy(),
    }
    matrix = {"schema": "ffn-four-1", "epochs": 4, "rows": []}
    state = {
        "schema": ddp.SCHEMA,
        "status": "running",
        "phase": "dual_gpu_fit",
        "pid": 9001,
        "completed": {},
        "events": [{"event": "start", "stage": "fit", "name": "BR", "pid": 9002}],
        "active": {"fit": {"stage": "fit", "name": "BR", "pid": 9002}},
        "preflights": {},
        "plans": {},
    }
    finals = {}
    for name in ddp.DEPENDENCY_ARMS:
        relative = f"configs/v2/ffn_four_20261011/{name}/gradpert_v2/nadig_jurkat.yaml"
        config = source / relative
        config.parent.mkdir(parents=True)
        config.write_bytes((ROOT / relative).read_bytes())
        plan = {
            "run_id": f"old-{name}",
            "run_root": str(runs / f"old-{name}"),
            "test_root": str(runs / f"old-{name}-test"),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "resolved_config": load_experiment_config(config).model_dump(mode="json"),
            "repository_root": str(source),
            "source_commit": ddp.RECOVERY_SOURCE,
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "publication": str(old_pub),
            "publication_sha256": sha256_file(old_pub),
            "data_root": str(data),
            "seed": 1,
            "gpu": "0,1",
            "postfit_policy": "deferred",
        }
        row = {"name": name, "plan": plan}
        parent["rows"].append(row)
        matrix["rows"].append({"name": name, "config": relative, "sha256": sha256_file(config)})
        receipt = parent_root / f"{name}-preflight/receipt.json"
        receipt.parent.mkdir()
        atomic_json(
            receipt,
            {
                "kind": "preflight_only",
                "status": "passed",
                "steps_requested": 10,
                "steps_completed": 10,
                "resume_checkpoint_sha256": "restore-sha",
                "inference_exercised": True,
                "inference_shape": [300, 5000],
                "source": {
                    "commit": ddp.RECOVERY_SOURCE,
                    "published_commit": ddp.RECOVERY_SOURCE,
                    "dirty": False,
                    "publication_receipt_sha256": plan["publication_sha256"],
                },
                "config_sha256": plan["config_sha256"],
                "data_root": plan["data_root"],
                "data": {"run_seed": 1},
                "gpu": "0,1",
                "world_size": 2,
                "peak_allocated_bytes": 40,
                "peak_reserved_bytes": 40,
                "last_terms": {"gradient_norm": 1.0},
                "shared_gpu_policy": dynamic_policy(),
            },
        )
        entry = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
        state["preflights"][name] = entry
        final = ddp.finalized_plan(row, entry, 100, dynamic_policy())
        finals[name] = final
        path = parent_root / f"{name}.launch-plan.json"
        atomic_json(path, final)
        state["plans"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "fit_fraction": 0.65,
        }
    atomic_json(source / "manifest.json", matrix)
    parent["manifest_sha256"] = sha256_file(source / "manifest.json")
    monkeypatch.setattr(
        ddp, "matrix_contract", lambda *args: (matrix, ddp.DEPENDENCY_ARMS, ["F1", "BR", "N1"])
    )
    parent_path = parent_root / "queue.json"
    atomic_json(parent_path, parent)
    parent_sha = sha256_file(parent_path)
    monkeypatch.setattr(ddp, "COMPLETED_BR_PARENT_SHA", parent_sha)
    state["queue_sha256"] = parent_sha
    atomic_json(parent_root / "state.json", state)
    br = Path(finals["BR"]["run_root"])
    (br / "fit").mkdir(parents=True)
    atomic_json(br / "launch.json", finals["BR"])
    atomic_json(br / "resolved_config.json", finals["BR"]["resolved_config"])
    identity = {
        "source": {
            "commit": ddp.RECOVERY_SOURCE,
            "published_commit": ddp.RECOVERY_SOURCE,
            "dirty": False,
        },
        "config_sha256": finals["BR"]["config_sha256"],
    }
    atomic_json(br / "run_manifest.json", identity)
    checkpoint = br / "fit/epoch-0004.pt"
    checkpoint.write_bytes(b"synthetic epoch4 complete checkpoint")
    (br / "fit/last.pt").write_bytes(checkpoint.read_bytes())
    monkeypatch.setattr(ddp, "COMPLETED_BR_CHECKPOINT_SHA", sha256_file(checkpoint))
    atomic_json(
        br / "fit/epoch_state.json",
        {
            "identity": identity,
            "epoch": 4,
            "budget": [4, 250],
            "best": None,
            "last": {"file": checkpoint.name, "sha256": sha256_file(checkpoint), "epoch": 4},
        },
    )
    atomic_json(br / "fit/history.json", [{"epoch": epoch} for epoch in range(1, 5)])
    atomic_json(br / "TRAIN_COMPLETE.json", training_stage_evidence(br, identity, 4))
    independent = br / "postfit-last-attempt1/COMPLETE.json"
    independent.parent.mkdir()
    atomic_json(independent, {"zero_pkl": True})
    metrics = [
        {
            "metric_id": metric,
            "macro_mean": 0.25,
            "finite_condition_count": 1,
            "total_condition_count": 1,
        }
        for metric in run_unified_group.METRICS
    ]
    result = {
        "metrics": metrics,
        "metric_gene_groups": {
            group: {"metrics": copy.deepcopy(metrics)}
            for group in ("seen_expression", "unseen_expression")
        },
    }
    atomic_json(
        br / "fit/last-test.json",
        {
            "result": result,
            "independent_evaluation_receipt": str(independent),
            "independent_evaluation_sha256": sha256_file(independent),
        },
    )
    atomic_json(br / "COMPLETE.json", {"epoch": 4, "zero_pkl": True})
    monkeypatch.setattr(ddp, "COMPLETED_BR_COMPLETE_SHA", sha256_file(br / "COMPLETE.json"))
    monkeypatch.setattr(
        run_unified_group,
        "collect_run",
        lambda root: [
            {
                "role": "last",
                "status": "complete",
                "checkpoint_epoch": 4,
                "training_sha": ddp.RECOVERY_SOURCE,
                "evaluation_sha": ddp.RECOVERY_SOURCE,
                "config_sha256": finals["BR"]["config_sha256"],
            }
        ],
    )
    resolved = []

    def resolver(_source, row, _runtime):
        name = row["name"]
        assert name in ("F1", "N1")
        resolved.append(name)
        plan = copy.deepcopy(next(r["plan"] for r in parent["rows"] if r["name"] == name))
        plan.update(
            run_id=f"fresh-{name}",
            run_root=str(runs / f"fresh-{name}"),
            test_root=str(runs / f"fresh-{name}-test"),
        )
        return plan

    monkeypatch.setattr(ddp, "_resolve_recovery_fresh", resolver)
    return SimpleNamespace(
        parent=parent,
        root=parent_root,
        path=parent_path,
        state=state,
        finals=finals,
        br=br,
        output=tmp_path / "continued",
        new_pub=new_pub,
        resolved=resolved,
        prepare=lambda: ddp.prepare_completed_br(
            parent_path, tmp_path / "continued", dynamic_policy(), new_pub
        ),
    )


def test_completed_import_fresh_ids_unchanged_plans_and_old_state(completed_parent):
    f = completed_parent
    original = {path: path.read_bytes() for path in f.root.rglob("*.json")}
    queue = f.prepare()
    assert f.resolved == ["F1", "N1"]
    assert [row["plan"]["run_id"] for row in queue["rows"]] == ["old-BR", "fresh-F1", "fresh-N1"]
    assert "recovery" not in queue and "resume" not in queue["completed_br_continuation"]
    assert queue["source_commit"] == ddp.RECOVERY_SOURCE
    for name, seed in queue["completed_br_continuation"]["seeded"].items():
        assert (
            read_json(seed["plan"]["path"])["resource_schedule"]
            == f.finals[name]["resource_schedule"]
        )
        assert seed["preflight"] == f.state["preflights"][name]
    assert all(path.read_bytes() == payload for path, payload in original.items())
    ddp.verify(queue)
    assert [
        row["name"] for row in ddp._dependency_contract(queue, f.output / "queue.json")["rows"]
    ] == ["BR", "F1", "N1"]


@pytest.mark.parametrize(
    "change",
    [
        "no_terminal",
        "checkpoint",
        "lasttest",
        "missing_metrics",
        "pkl",
        "peer_exists",
        "live_parent",
        "preflight",
        "plan",
        "runtime",
    ],
)
def test_completed_continuation_rejects_terminal_or_parent_drift(
    completed_parent, monkeypatch, change
):
    f = completed_parent
    if change == "no_terminal":
        (f.br / "COMPLETE.json").unlink()
    elif change == "checkpoint":
        (f.br / "fit/last.pt").write_bytes(b"drift")
    elif change == "lasttest":
        (f.br / "fit/last-test.json").write_text("{}")
    elif change == "missing_metrics":
        path = f.br / "fit/last-test.json"
        item = read_json(path)
        item["result"]["metrics"].pop()
        atomic_json(path, item)
    elif change == "pkl":
        (f.br / "unexpected.pkl").write_text("synthetic")
    elif change == "peer_exists":
        Path(f.parent["rows"][1]["plan"]["run_root"]).mkdir()
    elif change == "live_parent":
        monkeypatch.setattr(ddp, "process_identity", lambda pid: {"pid": pid})
    elif change == "preflight":
        Path(f.state["preflights"]["F1"]["receipt"]).write_text("{}")
    elif change == "plan":
        Path(f.state["plans"]["F1"]["path"]).write_text("{}")
    else:
        Path(f.parent["runtime"]).write_text("{}")
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        f.prepare()


def test_continuation_verifies_after_new_peer_run_progress(completed_parent):
    f = completed_parent
    queue = f.prepare()
    Path(queue["rows"][1]["plan"]["run_root"]).mkdir()
    ddp.verify(queue)
    assert not Path(f.parent["rows"][1]["plan"]["run_root"]).exists()


def test_post_seal_drift_and_reused_id_rejected(completed_parent, monkeypatch):
    f = completed_parent
    queue = f.prepare()
    path = f.br / "fit/last-test.json"
    path.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        ddp.verify(queue)


def execution(completed_parent, monkeypatch, *, fail_eval=False):
    f = completed_parent
    queue = f.prepare()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)

    @contextlib.contextmanager
    def locked(path):
        yield

    monkeypatch.setattr(ddp, "lock", locked)
    monkeypatch.setattr(
        ddp, "snapshot", lambda pid: {"processes": [], "gpus": {}, "observed_unix": 1.0}
    )
    monkeypatch.setattr(ddp, "admit", lambda *args: None)
    monkeypatch.setattr(
        ddp,
        "_capture_process_identity",
        lambda child: {"pid": child.pid, "start_ticks": "1", "command_sha256": "own"},
    )
    monkeypatch.setattr(ddp, "_terminate_active_child", lambda active: None)
    calls = []

    class Child:
        pid = 10000

        def __init__(self, command):
            self.command = command
            self.pid += len(calls)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 1 if fail_eval and "--evaluate" in self.command else 0

    def spawn(command, **kwargs):
        assert kwargs["env"]["PYTHONPATH"] == str(Path(queue["source"]) / "src")
        assert "--resume" not in command and "--uncapped" not in command
        if "--evaluate" in command:
            assert command[1] == str(Path(queue["source"]) / "scripts/v2/run_unified_ddp.py")
            assert command[command.index("--attempt") + 1] == "1"
        calls.append(command)
        return Child(command)

    monkeypatch.setattr(ddp.subprocess, "Popen", spawn)
    monkeypatch.setattr(ddp, "validate_training_stage", lambda plan: {"epoch": 4})
    # Original BR evidence has already been verified above; retain its validation
    # while abstracting peer native training/materialization from CPU dispatch tests.
    real_verify = ddp.verify
    monkeypatch.setattr(ddp, "verify", lambda q: None if q is queue else real_verify(q))
    monkeypatch.setattr(ddp, "validate_completed", lambda plan: {"run_id": plan["run_id"]})
    atomic_json(f.output / "COEXISTENCE_ACCEPTANCE.json", {"approved": True})
    monkeypatch.setattr(ddp, "validate_coexistence_acceptance", lambda q, ready, acceptance: None)
    return queue, calls


def test_execute_imports_br_once_and_caps_peers_in_order(completed_parent, monkeypatch):
    f = completed_parent
    queue, calls = execution(f, monkeypatch)
    ddp.execute(queue, f.output)
    assert len(calls) == 4
    assert [Path(command[-1]).stem for command in calls if "--evaluate" not in command] == [
        "F1.launch-plan",
        "N1.launch-plan",
    ]
    assert [
        Path(command[command.index("--evaluate") + 1]).stem
        for command in calls
        if "--evaluate" in command
    ] == ["F1.launch-plan", "N1.launch-plan"]
    state = read_json(f.output / "COMPLETE.json")
    assert set(state["completed"]) == set(ddp.DEPENDENCY_ARMS)
    assert [event["name"] for event in state["events"] if event["event"] == "import_completed"] == [
        "BR"
    ]
    assert state["scientific_source_commit"] == ddp.RECOVERY_SOURCE
    assert state["controller_source"]["commit"] == "c" * 40


def test_eval_oom_has_no_uncapped_retry_or_later_peer(completed_parent, monkeypatch):
    f = completed_parent
    queue, calls = execution(f, monkeypatch, fail_eval=True)
    atomic_json(Path(queue["rows"][1]["plan"]["run_root"]).parent / "dummy.json", {})
    failure = Path(queue["rows"][1]["plan"]["run_root"]) / "POSTFIT_ATTEMPT1_FAILURE.json"
    failure.parent.mkdir()
    atomic_json(failure, {"oom": True})
    # The artifact appears after dispatch in real execution; our CPU dispatcher
    # model keeps only the failure file and permits its synthetic fit root.
    original_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda path: False if path == failure.parent else original_exists(path)
    )
    with pytest.raises(RuntimeError, match="eval failed"):
        ddp.execute(queue, f.output)
    assert len(calls) == 2
    assert all("N1.launch-plan.json" not in " ".join(command) for command in calls)
    assert not (f.output / "COMPLETE.json").exists()


def test_readonly_dependency_binding_from_another_controller_checkout(
    completed_parent, monkeypatch, tmp_path
):
    f = completed_parent
    queue = f.prepare()
    current = tmp_path / "gm1-source/scripts/v2/run_unified_ddp.py"
    monkeypatch.setattr(ddp, "__file__", str(current))
    ddp.verify(queue)
    atomic_json(
        f.output / "state.json",
        {
            "pid": 777,
            "status": "running",
            "phase": "waiting_coexistence_acceptance",
            "queue_sha256": sha256_file(f.output / "queue.json"),
        },
    )
    monkeypatch.setattr(ddp, "process_identity", lambda pid: {"pid": pid} if pid == 777 else None)
    monkeypatch.setattr(
        ddp,
        "process_command",
        lambda pid: (
            b"python\0"
            + str(ROOT / "scripts/v2/run_unified_ddp.py").encode()
            + b"\0--queue\0"
            + str(f.output / "queue.json").encode()
            + b"\0"
        ),
    )
    dependency = ddp.seal_dependency(f.output / "queue.json")
    assert [row["name"] for row in dependency["rows"]] == list(ddp.DEPENDENCY_ARMS)
    assert dependency["source_commit"] == ddp.RECOVERY_SOURCE


def test_execution_cannot_use_different_source_or_root(completed_parent, monkeypatch, tmp_path):
    f = completed_parent
    queue = f.prepare()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)
    monkeypatch.setattr(
        ddp, "__file__", str(tmp_path / "different-source/scripts/v2/run_unified_ddp.py")
    )
    with pytest.raises(ValueError, match="execution source"):
        ddp.execute(queue, f.output)
    copied = tmp_path / "copy-root"
    copied.mkdir()
    atomic_json(copied / "queue.json", queue)
    with pytest.raises(ValueError, match="execution source"):
        ddp.execute(queue, copied)


def test_reused_reserved_peer_id_is_rejected(completed_parent, monkeypatch):
    f = completed_parent
    monkeypatch.setattr(
        ddp,
        "_resolve_recovery_fresh",
        lambda source, row, runtime: copy.deepcopy(
            next(item["plan"] for item in f.parent["rows"] if item["name"] == row["name"])
        ),
    )
    with pytest.raises(ValueError, match="fresh native IDs"):
        f.prepare()


def test_preflight_data_mismatch_rejected_even_with_resealed_receipt(completed_parent):
    f = completed_parent
    entry = f.state["preflights"]["F1"]
    path = Path(entry["receipt"])
    receipt = read_json(path)
    receipt["data_root"] = str(f.root / "wrong-data")
    atomic_json(path, receipt)
    entry["sha256"] = sha256_file(path)
    atomic_json(f.root / "state.json", f.state)
    with pytest.raises(ValueError, match="data"):
        f.prepare()


def test_completed_br_acceptance_barrier_cannot_self_approve(completed_parent, monkeypatch):
    f = completed_parent
    queue, calls = execution(f, monkeypatch)
    (f.output / "COEXISTENCE_ACCEPTANCE.json").unlink()

    def stop_wait(seconds):
        assert seconds == 5
        raise KeyboardInterrupt("synthetic manual acceptance remains pending")

    monkeypatch.setattr(ddp.time, "sleep", stop_wait)
    with pytest.raises(KeyboardInterrupt):
        ddp.execute(queue, f.output)
    assert calls == []
    assert (f.output / "PREFLIGHT_COMPLETE.json").exists()
    assert not (f.output / "COEXISTENCE_ACCEPTANCE.json").exists()
