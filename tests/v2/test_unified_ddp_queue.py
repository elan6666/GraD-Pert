"""Frozen DDP matrix and resource admission retain the scientific protocol."""

import contextlib
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.data._io import atomic_json, read_json
from gradpert.execution import v2_training_stage
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_unified_ddp as ddp  # noqa: E402
from generate_unified_ddp import PRIORITY, configs, verify_manifest  # noqa: E402
from generate_unified_group import baseline  # noqa: E402
from run_unified_ddp import fit_budget, probe  # noqa: E402


def test_spawn_identity_waits_through_transient_unreadable_samples(monkeypatch):
    identity = {"pid": 123, "command_sha256": "cmd", "start_ticks": "456"}
    samples = iter([None, None, identity])
    monkeypatch.setattr(ddp, "process_identity", lambda pid: next(samples))
    child = SimpleNamespace(pid=123, poll=lambda: None)

    assert ddp._capture_process_identity(child) == identity


def test_spawn_identity_fails_safely_if_child_exits_before_identity(monkeypatch):
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: None)
    monkeypatch.setattr(ddp, "terminate_tree", lambda *_: pytest.fail("unsafe termination"))
    child = SimpleNamespace(pid=123, poll=lambda: 7)

    with pytest.raises(RuntimeError, match="exited before its process identity"):
        ddp._capture_process_identity(child)


def test_cleanup_never_calls_tree_terminator_without_identity(monkeypatch):
    monkeypatch.setattr(ddp, "terminate_tree", lambda *_: pytest.fail("unsafe termination"))
    child = SimpleNamespace(pid=123)

    ddp._terminate_active_child((child, {}, {"identity": None}))


def test_spawn_identity_timeout_stops_only_the_direct_child(monkeypatch):
    monkeypatch.setattr(ddp, "CHILD_IDENTITY_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: None)
    monkeypatch.setattr(ddp, "terminate_tree", lambda *_: pytest.fail("unsafe termination"))
    calls = []

    class Child:
        pid = 123

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate-direct-child")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return -15

    child = Child()

    with pytest.raises(RuntimeError, match="timed out reading process identity"):
        ddp._capture_process_identity(child)
    assert calls == ["terminate-direct-child", ("wait", ddp.CHILD_STOP_TIMEOUT_SECONDS)]


def test_spawn_identity_timeout_kills_direct_child_if_terminate_times_out(monkeypatch):
    monkeypatch.setattr(ddp, "CHILD_IDENTITY_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: None)
    calls = []

    class Child:
        pid = 123

        def __init__(self):
            self.waits = 0

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate-direct-child")

        def kill(self):
            calls.append("kill-direct-child")

        def wait(self, timeout):
            self.waits += 1
            calls.append(("wait", timeout))
            if self.waits == 1:
                raise ddp.subprocess.TimeoutExpired("child", timeout)
            return -9

    with pytest.raises(RuntimeError, match="timed out reading process identity"):
        ddp._capture_process_identity(Child())
    assert calls == [
        "terminate-direct-child",
        ("wait", ddp.CHILD_STOP_TIMEOUT_SECONDS),
        "kill-direct-child",
        ("wait", ddp.CHILD_STOP_TIMEOUT_SECONDS),
    ]


def test_spawn_identity_rejects_identity_for_another_pid(monkeypatch):
    monkeypatch.setattr(
        ddp,
        "process_identity",
        lambda _pid: {"pid": 999, "command_sha256": "cmd", "start_ticks": "456"},
    )
    calls = []
    child = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        terminate=lambda: calls.append("terminate-direct-child"),
        wait=lambda timeout: calls.append(("wait", timeout)),
    )

    with pytest.raises(RuntimeError, match="identity is incomplete or mismatched"):
        ddp._capture_process_identity(child)
    assert calls == ["terminate-direct-child", ("wait", ddp.CHILD_STOP_TIMEOUT_SECONDS)]


def test_spawn_identity_does_not_accept_recycled_pid_after_child_exit(monkeypatch):
    monkeypatch.setattr(
        ddp,
        "process_identity",
        lambda pid: {"pid": pid, "command_sha256": "other", "start_ticks": "999"},
    )
    monkeypatch.setattr(ddp, "terminate_tree", lambda *_: pytest.fail("unsafe termination"))
    child = SimpleNamespace(pid=123, poll=lambda: 7)

    with pytest.raises(RuntimeError, match="exited before its process identity"):
        ddp._capture_process_identity(child)


def test_all_ten_configs_restore_dual_accumulation_and_only_change_execution():
    manifest = verify_manifest(ROOT, ROOT / "configs/v2/unified_ddp_20261010/manifest.json")
    assert tuple(manifest["priority"]) == ("N0", "CG1", "U24", "MR1", "P1")
    reference = baseline(ROOT, 32)
    for row in manifest["rows"]:
        cfg = load_experiment_config(ROOT / row["config"])
        _, options = V2Options.parse_parameters(cfg.model.parameters)
        assert (options.microbatch, options.world_size, options.accumulation) == (68, 2, 2)
        assert cfg.training.train_batch_size.value == 272
        assert cfg.training.eval_batch_size.value == 32
        value = yaml.safe_load((ROOT / row["config"]).read_text())
        for key in ("world_size", "accumulation", "microbatch", *row["changes"]):
            value["model"]["parameters"][key] = reference["model"]["parameters"][key]
        value["training"]["train_batch_size"] = reference["training"]["train_batch_size"]
        assert value == reference
    assert len(manifest["rows"]) == 10 and len(PRIORITY) == 5


def test_cg_and_population_config_are_loadable_with_two_rank_accumulation():
    values = configs(ROOT, 48)
    for name in ("CG1", "P1"):
        from gradpert.config.schema import ExperimentConfig

        cfg = ExperimentConfig.model_validate(values[name])
        _, options = V2Options.parse_parameters(cfg.model.parameters)
        assert options.world_size == options.accumulation == 2


def test_resource_admission_keeps_margin_and_no_unmeasured_overlap():
    assert fit_budget(50, 100) == 0.65
    assert fit_budget(60, 100) == 1.0
    with pytest.raises(ValueError):
        fit_budget(0, 100)


def test_probe_uses_two_ranks_and_exactly_ten_steps(tmp_path):
    plan = {
        "repository_root": "/data/yilangliu/source",
        "config": "cfg",
        "data_root": "data",
        "publication": "pub",
        "publication_sha256": "hash",
    }
    command = probe({"name": "CG1", "plan": plan}, tmp_path)
    assert command[1:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "/data/yilangliu/source/scripts/v2/capacity_probe.py",
    ]
    assert command[command.index("--steps") + 1] == "10"
    assert command[command.index("--gpu") + 1] == "0,1"


def test_manifest_does_not_accept_order_or_unrelated_method_change(tmp_path):
    import shutil

    from generate_unified_ddp import generate
    from generate_unified_group import PARENT

    parent = tmp_path / PARENT
    parent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / PARENT, parent)
    directory = tmp_path / "configs/v2/ddp"
    generate(tmp_path, directory, 48)
    path = directory / "manifest.json"
    value = json.loads(path.read_text())
    changed = copy.deepcopy(value)
    changed["priority"][:2] = list(reversed(changed["priority"][:2]))
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="priority"):
        verify_manifest(tmp_path, path)


@pytest.fixture
def dependency_parent(tmp_path, monkeypatch):
    """Real on-disk final epoch/checkpoint/18-metric receipts, no GPU runtime."""
    verify_queue = ddp.verify
    monkeypatch.setattr(ddp, "SERVER_ROOT", tmp_path)
    root = tmp_path / "parent"
    root.mkdir()
    source = tmp_path / "old-source"
    source.mkdir()
    manifest = source / "manifest.json"
    atomic_json(manifest, {"schema": "ffn-four-1"})
    parent = {
        "schema": ddp.SCHEMA,
        "source": str(source),
        "source_commit": "old-published-sha",
        "manifest": str(manifest),
        "rows": [],
    }
    for name in ddp.DEPENDENCY_ARMS:
        config = source / name / "gradpert_v2/nadig_jurkat.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("{}")
        parent["rows"].append(
            {
                "name": name,
                "plan": {
                    "run_id": f"old-{name}",
                    "config": str(config),
                    "config_sha256": sha256_file(config),
                    "repository_root": str(source),
                    "source_commit": parent["source_commit"],
                    "run_root": str(tmp_path / f"run-{name}"),
                    "test_root": str(tmp_path / f"test-{name}"),
                    "resolved_config": {"training": {"max_epochs": {"value": 4}}},
                },
            }
        )
    path = root / "queue.json"
    atomic_json(path, parent)
    controller = {"pid": 123, "command_sha256": "parent-command", "start_ticks": "456"}
    state = {
        "schema": ddp.SCHEMA,
        "pid": 123,
        "queue_sha256": sha256_file(path),
        "status": "running",
        "phase": "dual_gpu_fit",
        "plans": {},
        "completed": {},
    }
    atomic_json(root / "state.json", state)
    # Source publication is a separate queue.verify contract, exercised by existing tests.
    monkeypatch.setattr(ddp, "verify", lambda _queue: None)
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: controller)
    monkeypatch.setattr(
        ddp,
        "process_command",
        lambda _pid: f"python\0{source}/scripts/v2/run_unified_ddp.py\0--queue\0{path}\0".encode(),
    )
    dependency = ddp.seal_dependency(path)
    validations = []
    original_validate = ddp.validate_completed

    def validate(plan):
        validations.append(plan["run_id"])
        return original_validate(
            plan,
            collector=lambda _root: [
                {
                    "role": "last",
                    "status": "complete",
                    "checkpoint_epoch": 4,
                    "training_sha": parent["source_commit"],
                    "evaluation_sha": parent["source_commit"],
                    "config_sha256": plan["config_sha256"],
                }
            ],
        )

    monkeypatch.setattr(ddp, "validate_completed", validate)
    monkeypatch.setattr(
        v2_training_stage,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            training=SimpleNamespace(max_epochs=SimpleNamespace(value=4))
        ),
    )

    def complete():
        from run_unified_group import METRICS

        metrics = [
            {
                "metric_id": metric,
                "macro_mean": 0.1,
                "finite_condition_count": 3,
                "total_condition_count": 3,
            }
            for metric in sorted(METRICS)
        ]
        for row in parent["rows"]:
            plan = {**row["plan"], "cuda_memory_fraction": 0.65, "resource_schedule": {}}
            launch = root / f"{row['name']}.launch-plan.json"
            atomic_json(launch, plan)
            state["plans"][row["name"]] = {"path": str(launch), "sha256": sha256_file(launch)}
            run = Path(plan["run_root"])
            (run / "fit").mkdir(parents=True)
            checkpoint = run / "fit/last.pt"
            checkpoint.write_bytes(b"sealed synthetic checkpoint")
            identity = {
                "source": {
                    "commit": parent["source_commit"],
                    "published_commit": parent["source_commit"],
                    "dirty": False,
                },
                "config_sha256": plan["config_sha256"],
            }
            atomic_json(run / "launch.json", plan)
            atomic_json(run / "resolved_config.json", plan["resolved_config"])
            atomic_json(run / "run_manifest.json", identity)
            atomic_json(run / "fit/history.json", [{"epoch": epoch} for epoch in range(1, 5)])
            atomic_json(
                run / "fit/epoch_state.json",
                {
                    "identity": identity,
                    "epoch": 4,
                    "budget": [4],
                    "best": None,
                    "last": {"epoch": 4, "file": "last.pt", "sha256": sha256_file(checkpoint)},
                },
            )
            atomic_json(
                run / "TRAIN_COMPLETE.json",
                v2_training_stage.training_stage_evidence(run, identity, 4),
            )
            atomic_json(run / "COMPLETE.json", {"status": "complete"})
            atomic_json(
                run / "fit/last-test.json",
                {
                    "result": {
                        "metrics": metrics,
                        "metric_gene_groups": {
                            group: {"metrics": metrics}
                            for group in ("seen_expression", "unseen_expression")
                        },
                    }
                },
            )
            state["completed"][row["name"]] = validate(plan)
        validations.clear()
        state.update(status="complete", phase="complete")
        atomic_json(root / "state.json", state)
        atomic_json(root / "COMPLETE.json", state)

    return SimpleNamespace(
        root=root,
        path=path,
        parent=parent,
        state=state,
        dependency=dependency,
        controller=controller,
        complete=complete,
        validations=validations,
        verify_queue=verify_queue,
    )


def test_dependency_running_then_complete_validates_all_science_before_gpu_lease(
    dependency_parent, tmp_path, monkeypatch
):
    fixture = dependency_parent
    child = tmp_path / "child"
    child.mkdir()
    queue = {"schema": ddp.SCHEMA, "dependency": fixture.dependency, "source_commit": "new-sha"}
    atomic_json(child / "queue.json", queue)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)
    monkeypatch.setattr(ddp, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(ddp, "snapshot", lambda *_: pytest.fail("GPU snapshot before dependency"))
    monkeypatch.setattr(ddp, "gpu_idle", lambda *_: pytest.fail("GPU idle before dependency"))
    monkeypatch.setattr(ddp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unexpected launch"))
    leases = []

    @contextlib.contextmanager
    def lock(path):
        leases.append(path)
        if path.name.startswith("v2-gpu"):
            assert fixture.validations == ["old-BR", "old-F1", "old-N1"]
            assert read_json(child / "state.json")["dependency_completed"]["rows"]
            raise RuntimeError("synthetic GPU lease boundary")
        yield

    def sleep(seconds):
        assert seconds == 20
        assert leases == [child / "controller.lock"]
        assert not fixture.validations
        assert read_json(child / "state.json")["phase"] == "waiting_dependency"
        fixture.complete()

    monkeypatch.setattr(ddp, "lock", lock)
    monkeypatch.setattr(ddp.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="synthetic GPU lease boundary"):
        ddp.execute(queue, child)
    assert len(leases) == 2


def test_dependency_terminal_complete_allows_exited_controller(dependency_parent, monkeypatch):
    fixture = dependency_parent
    fixture.complete()
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: None)
    fixture.dependency = ddp.seal_dependency(fixture.path)
    assert fixture.dependency["controller_identity"] is None
    state = {}
    ddp.wait_dependency(fixture.dependency, state, lambda: None)
    assert fixture.validations == ["old-BR", "old-F1", "old-N1"]
    assert state["dependency_completed"]["complete_sha256"]


@pytest.mark.parametrize("corruption", ["hash", "source", "scope", "duplicate_run"])
def test_dependency_rejects_identity_or_scope_drift(dependency_parent, corruption):
    fixture = dependency_parent
    if corruption == "source":
        fixture.dependency["source_commit"] = "new-sha"
    else:
        if corruption == "scope":
            fixture.parent["rows"][2]["name"] = "GM1"
        elif corruption == "duplicate_run":
            fixture.parent["rows"][2]["plan"]["run_id"] = "old-BR"
        else:
            fixture.parent["extra"] = True
        atomic_json(fixture.path, fixture.parent)
        if corruption != "hash":
            fixture.dependency["queue_sha256"] = sha256_file(fixture.path)
    with pytest.raises(ValueError, match="dependency"):
        ddp.verify_dependency(fixture.dependency)


@pytest.mark.parametrize("failure", ["receipt", "state", "missing", "recycled"])
def test_dependency_failure_never_takes_gpu_lease_or_launches(
    dependency_parent, tmp_path, monkeypatch, failure
):
    fixture = dependency_parent
    if failure == "receipt":
        atomic_json(fixture.root / "FAILURE.json", {"error": "fit failed"})
    elif failure == "state":
        fixture.state["status"] = "failed"
        atomic_json(fixture.root / "state.json", fixture.state)
    else:
        monkeypatch.setattr(
            ddp,
            "process_identity",
            lambda _pid: (
                None if failure == "missing" else {**fixture.controller, "start_ticks": "recycled"}
            ),
        )
    child = tmp_path / "child"
    child.mkdir()
    queue = {"schema": ddp.SCHEMA, "dependency": fixture.dependency}
    atomic_json(child / "queue.json", queue)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)

    @contextlib.contextmanager
    def lock(path):
        assert path == child / "controller.lock", "GPU lease attempted during dependency wait"
        yield

    monkeypatch.setattr(ddp, "lock", lock)
    monkeypatch.setattr(ddp, "snapshot", lambda *_: pytest.fail("GPU snapshot attempted"))
    monkeypatch.setattr(ddp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launch attempted"))
    with pytest.raises(RuntimeError, match="dependency"):
        ddp.execute(queue, child)
    evidence = read_json(child / "FAILURE.json")
    assert evidence["status"] == "failed"
    assert evidence["dependency_observed"]["state_sha256"]


@pytest.mark.parametrize(
    "corruption", ["checkpoint", "checkpoint_path", "epochs", "pkl", "metrics", "plan"]
)
def test_dependency_complete_rejects_bad_scientific_receipts(dependency_parent, corruption):
    fixture = dependency_parent
    fixture.complete()
    row = fixture.parent["rows"][0]
    run = Path(row["plan"]["run_root"])
    if corruption == "checkpoint":
        (run / "fit/last.pt").write_bytes(b"changed")
    elif corruption == "checkpoint_path":
        (run / "fit/last.pt").rename(run / "fit/other.pt")
        (run / "fit/last.pt").write_bytes(b"wrong last role alias")
        journal = read_json(run / "fit/epoch_state.json")
        journal["last"]["file"] = "other.pt"
        atomic_json(run / "fit/epoch_state.json", journal)
        atomic_json(
            run / "TRAIN_COMPLETE.json",
            v2_training_stage.training_stage_evidence(run, read_json(run / "run_manifest.json"), 4),
        )
    elif corruption == "epochs":
        journal = read_json(run / "fit/epoch_state.json")
        journal["epoch"] = 3
        atomic_json(run / "fit/epoch_state.json", journal)
    elif corruption == "pkl":
        (fixture.root / "stale.pkl").write_bytes(b"forbidden")
    elif corruption == "metrics":
        result = read_json(run / "fit/last-test.json")
        result["result"]["metrics"] = []
        atomic_json(run / "fit/last-test.json", result)
    else:
        path = fixture.root / "BR.launch-plan.json"
        plan = read_json(path)
        plan["source_commit"] = "new-sha"
        atomic_json(path, plan)
        fixture.state["plans"]["BR"]["sha256"] = sha256_file(path)
        atomic_json(fixture.root / "state.json", fixture.state)
        atomic_json(fixture.root / "COMPLETE.json", fixture.state)
    with pytest.raises(ValueError):
        ddp.wait_dependency(fixture.dependency, {}, lambda: None)


def test_non_gm1_cannot_seal_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(ddp, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(ddp, "matrix_contract", lambda *_: ({}, ("BR", "F1", "N1"), []))
    monkeypatch.setattr(ddp, "resolve_plan", lambda *_: pytest.fail("resolved unauthorized scope"))
    with pytest.raises(ValueError, match="only GM1"):
        ddp.prepare(
            tmp_path,
            tmp_path / "manifest",
            tmp_path / "runtime",
            tmp_path / "output",
            dependency_queue=tmp_path / "parent/queue.json",
        )


def test_graph_mhc_matrix_is_exactly_one_initial_preflight():
    matrix, priority, initial = ddp.matrix_contract(
        ROOT, ROOT / "configs/v2/graph_mhc_four_20261011/manifest.json"
    )
    assert matrix["schema"] == "graph-mhc-four-1"
    assert priority == ("GM1",)
    assert initial == ["GM1"]


def test_gm1_verify_keeps_new_source_independent_and_rejects_dependency_drift(
    dependency_parent, tmp_path, monkeypatch
):
    fixture = dependency_parent
    manifest_path = ROOT / "configs/v2/graph_mhc_four_20261011/manifest.json"
    row = read_json(manifest_path)["rows"][0]
    runtime, publication = tmp_path / "runtime.json", tmp_path / "publication.json"
    atomic_json(runtime, {})
    atomic_json(publication, {})
    queue = {
        "schema": ddp.SCHEMA,
        "source": str(ROOT),
        "source_commit": "new-published-sha",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "initial_preflight": ["GM1"],
        "steps": 10,
        "dependency": fixture.dependency,
        "rows": [
            {
                "name": "GM1",
                "plan": {
                    "source_commit": "new-published-sha",
                    "gpu": "0,1",
                    "seed": 1,
                    "postfit_policy": "deferred",
                    "config": str(ROOT / row["config"]),
                    "config_sha256": row["sha256"],
                    "publication": str(publication),
                    "publication_sha256": sha256_file(publication),
                    "resolved_config": load_experiment_config(ROOT / row["config"]).model_dump(
                        mode="json"
                    ),
                },
            }
        ],
    }
    monkeypatch.setattr(
        ddp,
        "inspect_source_identity",
        lambda *_a, **_kw: SimpleNamespace(commit="new-published-sha", dirty=False),
    )
    fixture.verify_queue(queue)
    assert fixture.dependency["source_commit"] == "old-published-sha"
    fixture.parent["changed"] = True
    atomic_json(fixture.path, fixture.parent)
    with pytest.raises(ValueError, match="dependency queue hash changed"):
        fixture.verify_queue(queue)


@pytest.mark.parametrize(
    "names,policy,expected",
    [
        (["GM1"], None, True),
        (["GM1"], {"schema": "legacy"}, True),
        (["GM1"], {"schema": ddp.DUAL_SCHEMA}, True),
        (["BR", "F1", "N1"], {"schema": ddp.DUAL_SCHEMA}, True),
        (["N0", "CG1"], None, False),
        (["BR", "F1", "N1"], {"schema": "legacy"}, False),
    ],
)
def test_preflight_acceptance_preserves_old_scope_and_always_includes_gm1(names, policy, expected):
    queue = {"rows": [{"name": name} for name in names], "shared_gpu_policy": policy}
    assert ddp.needs_preflight_acceptance(queue) is expected


def test_exclusive_gm1_stops_after_preflight_until_main_acceptance(tmp_path, monkeypatch):
    directory = tmp_path / "queue"
    directory.mkdir()
    plan = {
        "run_id": "fresh-GM1",
        "run_root": str(tmp_path / "fresh-GM1"),
        "test_root": str(tmp_path / "fresh-GM1-test"),
    }
    queue = {
        "schema": ddp.SCHEMA,
        "source": str(tmp_path),
        "source_commit": "new-sha",
        "rows": [{"name": "GM1", "plan": plan}],
        "initial_preflight": ["GM1"],
    }
    atomic_json(directory / "queue.json", queue)
    monkeypatch.setattr(ddp, "SERVER_ROOT", tmp_path)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)
    monkeypatch.setattr(ddp, "verify", lambda _queue: None)

    @contextlib.contextmanager
    def lock(_path):
        yield

    monkeypatch.setattr(ddp, "lock", lock)
    monkeypatch.setattr(ddp, "gpu_processes_owned", lambda: True)
    monkeypatch.setattr(ddp, "snapshot", lambda *_: pytest.fail("shared snapshot without policy"))
    monkeypatch.setattr(ddp, "probe", lambda *_: ["synthetic-preflight"])
    monkeypatch.setattr(ddp, "child_environment", lambda *_: {})
    monkeypatch.setattr(ddp, "validate_preflight", lambda *_: None)
    monkeypatch.setattr(ddp, "finalized_plan", lambda *_: {**plan, "cuda_memory_fraction": 0.65})
    monkeypatch.setattr(
        ddp.subprocess,
        "check_output",
        lambda command, **_kw: "0, 0\n1, 0" if "index,memory.used" in command[1] else "1000\n1000",
    )
    monkeypatch.setattr(
        ddp,
        "_capture_process_identity",
        lambda child: {"pid": child.pid, "command_sha256": "preflight", "start_ticks": "123"},
    )
    launches = []

    def launch(command, **_kwargs):
        launches.append(command)
        assert command == ["synthetic-preflight"], "formal launch bypassed acceptance"
        receipt = directory / "GM1-preflight/receipt.json"
        receipt.parent.mkdir()
        atomic_json(receipt, {"status": "passed"})
        return SimpleNamespace(pid=123, wait=lambda **_kw: 0)

    monkeypatch.setattr(ddp.subprocess, "Popen", launch)

    def wait(seconds):
        assert seconds == 5
        state = read_json(directory / "state.json")
        assert state["phase"] == "waiting_coexistence_acceptance"
        ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
        assert ready["scientific_complete"] is False and ready["formal_started"] is False
        assert list(ready["preflights"]) == ["GM1"]
        raise RuntimeError("synthetic stop at acceptance barrier")

    monkeypatch.setattr(ddp.time, "sleep", wait)
    with pytest.raises(RuntimeError, match="synthetic stop at acceptance barrier"):
        ddp.execute(queue, directory)
    assert launches == [["synthetic-preflight"]]


@pytest.fixture
def failed_ffn_parent(tmp_path, monkeypatch):
    """A stopped BR at a committed epoch boundary, with all three real CPU-valid probes."""
    resolve_cpu = ddp._resolve_recovery_fresh
    source, root = tmp_path / "b37-source", tmp_path / "failed-queue"
    source.mkdir()
    root.mkdir()
    monkeypatch.setattr(ddp, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(ddp, "LEASE_ROOT", tmp_path / "leases")
    monkeypatch.setattr(ddp, "process_identity", lambda _pid: None)
    monkeypatch.setattr(
        ddp,
        "inspect_source_identity",
        lambda *_a, **_kw: SimpleNamespace(commit=ddp.RECOVERY_SOURCE, dirty=False),
    )
    publication, runtime = source / "publication.json", tmp_path / "runtime.json"
    atomic_json(publication, {})
    atomic_json(runtime, {})
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
    }
    matrix = {"schema": "ffn-four-1", "epochs": 4, "rows": []}
    for name in ddp.DEPENDENCY_ARMS:
        original_config = (
            ROOT / f"configs/v2/ffn_four_20261011/{name}/gradpert_v2/nadig_jurkat.yaml"
        )
        config = source / name / "gradpert_v2/nadig_jurkat.yaml"
        config.parent.mkdir(parents=True)
        config.write_bytes(original_config.read_bytes())
        matrix["rows"].append(
            {"name": name, "config": str(config.relative_to(source)), "sha256": sha256_file(config)}
        )
        parent["rows"].append(
            {
                "name": name,
                "plan": {
                    "run_id": f"original-{name}",
                    "run_root": str(tmp_path / f"original-{name}"),
                    "test_root": str(tmp_path / f"original-{name}-test"),
                    "config": str(config),
                    "config_sha256": sha256_file(config),
                    "resolved_config": load_experiment_config(config).model_dump(mode="json"),
                    "repository_root": str(source),
                    "source_commit": ddp.RECOVERY_SOURCE,
                    "runtime": str(runtime),
                    "runtime_sha256": sha256_file(runtime),
                    "publication": str(publication),
                    "publication_sha256": sha256_file(publication),
                    "data_root": str(tmp_path / "canonical-data"),
                    "seed": 1,
                    "gpu": "0,1",
                    "postfit_policy": "deferred",
                },
            }
        )
    atomic_json(source / "manifest.json", matrix)
    parent["manifest_sha256"] = sha256_file(source / "manifest.json")
    monkeypatch.setattr(
        ddp, "matrix_contract", lambda *_: (matrix, ddp.DEPENDENCY_ARMS, ["F1", "BR", "N1"])
    )
    atomic_json(root / "queue.json", parent)
    state = {
        "schema": ddp.SCHEMA,
        "pid": 9001,
        "queue_sha256": sha256_file(root / "queue.json"),
        "status": "failed",
        "phase": "failed",
        "completed": {},
        "active": {},
        "events": [{"event": "start", "name": "BR", "stage": "fit", "pid": 9002}],
        "plans": {},
        "preflights": {},
    }
    finals = {}
    for row in parent["rows"]:
        name, plan = row["name"], row["plan"]
        receipt = root / f"{name}-preflight/receipt.json"
        receipt.parent.mkdir()
        atomic_json(
            receipt,
            {
                "kind": "preflight_only",
                "status": "passed",
                "steps_completed": 10,
                "resume_checkpoint_sha256": "restored-checkpoint-sha",
                "inference_exercised": True,
                "inference_shape": [300, 100],
                "source": {
                    "commit": ddp.RECOVERY_SOURCE,
                    "published_commit": ddp.RECOVERY_SOURCE,
                    "dirty": False,
                },
                "config_sha256": plan["config_sha256"],
                "data_root": plan["data_root"],
                "data": {"run_seed": 1},
                "world_size": 2,
                "peak_allocated_bytes": 60,
                "peak_reserved_bytes": 60,
                "last_terms": {"gradient_norm": 1.0},
            },
        )
        state["preflights"][name] = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
        final = ddp.finalized_plan(row, state["preflights"][name], 100)
        path = root / f"{name}.launch-plan.json"
        atomic_json(path, final)
        state["plans"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "fit_fraction": final["cuda_memory_fraction"],
        }
        finals[name] = final
    atomic_json(root / "state.json", state)
    atomic_json(root / "FAILURE.json", state)
    br_plan = finals["BR"]
    br = Path(br_plan["run_root"])
    (br / "fit").mkdir(parents=True)
    atomic_json(br / "launch.json", br_plan)
    atomic_json(br / "resolved_config.json", br_plan["resolved_config"])
    identity = {
        "source": {
            "commit": ddp.RECOVERY_SOURCE,
            "published_commit": ddp.RECOVERY_SOURCE,
            "dirty": False,
        },
        "environment": {"original": True},
        "config_sha256": br_plan["config_sha256"],
        "resolved_config_sha256": sha256_json(br_plan["resolved_config"]),
        "run_id": br_plan["run_id"],
    }
    atomic_json(br / "run_manifest.json", identity)
    atomic_json(br / "fit/history.json", [{"epoch": 1}])
    checkpoint = br / "fit/epoch-0001.pt"
    checkpoint.write_bytes(b"synthetic complete optimizer teacher center RNG checkpoint")
    (br / "fit/last.pt").write_bytes(checkpoint.read_bytes())
    params, training = (
        br_plan["resolved_config"]["model"]["parameters"],
        br_plan["resolved_config"]["training"],
    )
    journal = {
        "identity": identity,
        "epoch": 1,
        "budget": [4, 250],
        "best": None,
        "selection_metric": "final_epoch",
        "last": {"epoch": 1, "file": "epoch-0001.pt", "sha256": sha256_file(checkpoint)},
        "contract": {
            "schedule": {
                key: training["scheduler"]["value"][key]
                for key in ("max_lr", "min_lr", "warmup_fraction")
            },
            "teacher_start": params["teacher_start"]["value"],
            "teacher_end": params["teacher_end"]["value"],
            "microbatch": params["microbatch"]["value"],
            "bf16": True,
            "selection_metric": "final_epoch",
        },
    }
    atomic_json(br / "fit/epoch_state.json", journal)
    resolved_names = []

    def resolver(_source, row, _runtime):
        name = row["name"]
        resolved_names.append(name)
        assert name in ("F1", "N1"), "BR must preserve its original resolved identity"
        base = copy.deepcopy(next(r["plan"] for r in parent["rows"] if r["name"] == name))
        base.update(
            run_id=f"fresh-{name}",
            run_root=str(tmp_path / f"fresh-{name}"),
            test_root=str(tmp_path / f"fresh-{name}-test"),
        )
        return base

    monkeypatch.setattr(ddp, "_resolve_recovery_fresh", resolver)
    ddp.verify(parent)
    return SimpleNamespace(
        root=root,
        source=source,
        parent=parent,
        state=state,
        br=br,
        journal=journal,
        finals=finals,
        resolved_names=resolved_names,
        prepare=lambda: ddp.prepare_recovery(root / "queue.json", tmp_path / "recovery"),
        resolve_cpu=resolve_cpu,
    )


def test_recovery_only_br_preserves_original_plans_and_reuses_preflights(failed_ffn_parent):
    fixture = failed_ffn_parent
    original_bytes = {str(path): path.read_bytes() for path in fixture.root.rglob("*.json")}
    original_launch = (fixture.br / "launch.json").read_bytes()
    queue = fixture.prepare()
    assert fixture.resolved_names == ["F1", "N1"]
    assert [row["plan"]["run_id"] for row in queue["rows"]] == [
        "original-BR",
        "fresh-F1",
        "fresh-N1",
    ]
    assert queue["rows"][0]["plan"] == fixture.parent["rows"][0]["plan"]
    assert queue["source_commit"] == ddp.RECOVERY_SOURCE
    assert queue["recovery"]["resume"]["budget"] == [4, 250]
    assert queue["recovery"]["resume"]["checkpoint"]["path"].endswith("epoch-0001.pt")
    assert queue["recovery"]["resume"]["last_role"]["path"].endswith("last.pt")
    assert (fixture.br / "launch.json").read_bytes() == original_launch
    for path, payload in original_bytes.items():
        assert Path(path).read_bytes() == payload
    for name, seed in queue["recovery"]["seeded"].items():
        assert seed["preflight"] == fixture.state["preflights"][name]
        actual = read_json(seed["plan"]["path"])
        assert actual["resource_schedule"] == fixture.finals[name]["resource_schedule"]
        assert actual["cuda_memory_fraction"] == fixture.finals[name]["cuda_memory_fraction"]
    backup = queue["recovery"]["checkpoint_backup"]
    assert sha256_file(Path(backup["path"])) == queue["recovery"]["resume"]["checkpoint"]["sha256"]


@pytest.mark.parametrize(
    "corruption",
    [
        "live",
        "not_failed",
        "other_arm_started",
        "old_f1_root",
        "budget",
        "identity",
        "contract",
        "epoch",
        "checkpoint",
        "alias",
        "no_journal",
        "partial_epoch",
        "probe",
        "source",
        "fresh_id",
        "scope",
    ],
)
def test_recovery_rejects_unowned_or_uncommitted_resume(failed_ffn_parent, monkeypatch, corruption):
    fixture = failed_ffn_parent
    if corruption == "live":
        monkeypatch.setattr(ddp, "process_identity", lambda pid: {"pid": pid})
    elif corruption in ("not_failed", "other_arm_started"):
        if corruption == "not_failed":
            fixture.state["status"] = "running"
        else:
            fixture.state["events"].append(
                {"event": "start", "stage": "fit", "name": "F1", "pid": 9999}
            )
        atomic_json(fixture.root / "state.json", fixture.state)
        atomic_json(fixture.root / "FAILURE.json", fixture.state)
    elif corruption == "old_f1_root":
        Path(fixture.parent["rows"][1]["plan"]["run_root"]).mkdir()
    elif corruption in ("budget", "identity", "contract", "epoch", "partial_epoch"):
        journal = copy.deepcopy(fixture.journal)
        if corruption == "budget":
            journal["budget"] = [4, 251]
        elif corruption == "identity":
            journal["identity"]["environment"] = {"changed": True}
        elif corruption == "contract":
            journal["contract"]["teacher_end"] = 0.9
        elif corruption == "epoch":
            journal["epoch"] = 2
        else:
            journal["epoch"] = 0
        atomic_json(fixture.br / "fit/epoch_state.json", journal)
    elif corruption in ("checkpoint", "alias", "no_journal"):
        path = (
            fixture.br
            / "fit"
            / {"checkpoint": "epoch-0001.pt", "alias": "last.pt", "no_journal": "epoch_state.json"}[
                corruption
            ]
        )
        if corruption == "no_journal":
            path.unlink()
        else:
            path.write_bytes(b"changed")
    elif corruption == "probe":
        path = Path(fixture.state["preflights"]["BR"]["receipt"])
        receipt = read_json(path)
        receipt["steps_completed"] = 9
        atomic_json(path, receipt)
    elif corruption == "source":
        fixture.parent["source_commit"] = "new-scientific-sha"
        atomic_json(fixture.root / "queue.json", fixture.parent)
    elif corruption == "scope":
        fixture.parent["rows"][2]["name"] = "GM1"
        atomic_json(fixture.root / "queue.json", fixture.parent)
    else:
        monkeypatch.setattr(
            ddp,
            "_resolve_recovery_fresh",
            lambda _s, row, _r: copy.deepcopy(
                next(r["plan"] for r in fixture.parent["rows"] if r["name"] == row["name"])
            ),
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        fixture.prepare()


@pytest.mark.parametrize("drift", ["journal", "checkpoint", "plan", "preflight", "metadata"])
def test_recovery_seal_rejects_drift_before_dispatch(failed_ffn_parent, drift):
    fixture = failed_ffn_parent
    queue = fixture.prepare()
    if drift in ("journal", "checkpoint"):
        path = Path(queue["recovery"]["resume"][drift]["path"])
        if drift == "journal":
            value = read_json(path)
            value["selection_start_epoch"] = 999
            atomic_json(path, value)
        else:
            path.write_bytes(b"new checkpoint")
    elif drift == "plan":
        path = Path(queue["recovery"]["seeded"]["BR"]["plan"]["path"])
        value = read_json(path)
        value["cuda_memory_fraction"] = 0.3
        atomic_json(path, value)
    elif drift == "preflight":
        path = Path(queue["recovery"]["seeded"]["BR"]["preflight"]["receipt"])
        path.write_text("{}")
    else:
        queue["recovery"]["unexpected_mode"] = True
    with pytest.raises(ValueError):
        ddp.verify(queue)


def test_recovery_command_and_cpu_resolver_keep_old_scientific_source(
    failed_ffn_parent, monkeypatch
):
    fixture = failed_ffn_parent
    queue = fixture.prepare()
    br_command = ddp.recovery_fit_command(
        queue, queue["rows"][0], Path(queue["recovery"]["seeded"]["BR"]["plan"]["path"])
    )
    assert br_command[-1] == "--resume"
    assert br_command[6] == str(fixture.source / "scripts/v2/distributed_train.py")
    for row in queue["rows"][1:]:
        command = ddp.recovery_fit_command(queue, row, Path("fresh.launch-plan.json"))
        assert "--resume" not in command and "execute_plan" in command[2]
    from run_unified_group import child_environment

    environment = child_environment(str(fixture.source), "0,1")
    assert environment["PYTHONPATH"] == str(fixture.source / "src")
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    calls = []

    def resolve(command, **kwargs):
        calls.append((command, kwargs))
        return json.dumps(queue["rows"][1]["plan"])

    monkeypatch.setattr(ddp.subprocess, "check_output", resolve)
    matrix_row = {"config": "F1/gradpert_v2/nadig_jurkat.yaml"}
    plan = fixture.resolve_cpu(fixture.source, matrix_row, Path(queue["runtime"]))
    assert plan["source_commit"] == ddp.RECOVERY_SOURCE
    assert calls[0][1]["cwd"] == fixture.source
    assert calls[0][1]["env"]["PYTHONPATH"] == str(fixture.source / "src")
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][0][-2:] == [str(fixture.source / matrix_row["config"]), queue["runtime"]]


def test_recovery_executor_reuses_probes_and_resumes_only_br(failed_ffn_parent, monkeypatch):
    fixture = failed_ffn_parent
    queue = fixture.prepare()
    root = Path(queue["recovery"]["output"])
    original_launch = (fixture.br / "launch.json").read_bytes()
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)
    monkeypatch.setattr(ddp, "probe", lambda *_: pytest.fail("recovery must reuse passed probes"))
    monkeypatch.setattr(ddp, "snapshot", lambda *_: pytest.fail("exclusive has no shared snapshot"))
    monkeypatch.setattr(ddp, "gpu_processes_owned", lambda: True)
    monkeypatch.setattr(ddp, "validate_training_stage", lambda _plan: {})
    monkeypatch.setattr(ddp, "validate_completed", lambda plan: {"run_id": plan["run_id"]})
    monkeypatch.setattr(
        ddp,
        "_capture_process_identity",
        lambda child: {"pid": child.pid, "start_ticks": str(child.pid), "command_sha256": "cmd"},
    )
    launches = []

    @contextlib.contextmanager
    def lock(_path):
        yield

    monkeypatch.setattr(ddp, "lock", lock)

    def launch(command, **kwargs):
        launches.append(command)
        assert kwargs["cwd"] == str(fixture.source)
        assert kwargs["env"]["PYTHONPATH"] == str(fixture.source / "src")
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == ddp.ALLOCATOR
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in ("0", "0,1")
        if "--resume" in command:
            assert len(launches) == 1
            assert command[6] == str(fixture.source / "scripts/v2/distributed_train.py")
        elif command[1] == "-c":
            actual = read_json(command[-1])
            Path(actual["run_root"]).mkdir()

        def finish():
            if "--resume" in command:
                journal = read_json(fixture.br / "fit/epoch_state.json")
                journal["epoch"] = 2
                atomic_json(fixture.br / "fit/epoch_state.json", journal)
                # Match immutable b37 lifecycle pruning after the next epoch commit.
                (fixture.br / "fit/epoch-0001.pt").unlink()
            return 0

        return SimpleNamespace(pid=10000 + len(launches), wait=finish, poll=lambda: None)

    monkeypatch.setattr(ddp.subprocess, "Popen", launch)
    ddp.execute(queue, root)
    assert len(launches) == 6
    assert sum("--resume" in command for command in launches) == 1
    assert read_json(root / "COMPLETE.json")["status"] == "complete"
    assert list(read_json(root / "state.json")["preflights"]) == ["BR", "F1", "N1"]
    assert (fixture.br / "launch.json").read_bytes() == original_launch
    assert read_json(root / "RECOVERY_BR_STARTED.json")["source_commit"] == ddp.RECOVERY_SOURCE
    assert not (fixture.br / "fit/epoch-0001.pt").exists()
    backup = Path(queue["recovery"]["checkpoint_backup"]["path"])
    assert sha256_file(backup) == queue["recovery"]["resume"]["checkpoint"]["sha256"]
    ddp.verify(queue)
    backup.write_bytes(b"corrupted preserved input")
    with pytest.raises(ValueError, match="immutable checkpoint backup"):
        ddp.verify(queue)


@pytest.mark.parametrize("self_process", [True, False])
def test_recovery_proc_scan_excludes_only_its_own_preparation_process(
    failed_ffn_parent, tmp_path, monkeypatch, self_process
):
    fixture = failed_ffn_parent
    pid = os.getpid() if self_process else os.getpid() + 100000
    proc = tmp_path / str(pid) / "cmdline"
    proc.parent.mkdir()
    proc.write_bytes(
        f"python\0new/run_unified_ddp.py\0--recover-from\0{fixture.root}/queue.json\0".encode()
    )
    original_glob = Path.glob

    def glob(path, pattern, **kwargs):
        if str(path) == "/proc" and pattern == "[0-9]*/cmdline":
            return iter([proc])
        return original_glob(path, pattern, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    if self_process:
        assert fixture.prepare()["mode"] == ddp.RECOVERY_MODE
    else:
        with pytest.raises(ValueError, match="descendant is still live"):
            fixture.prepare()


def test_fresh_queue_still_rejects_existing_br_root(failed_ffn_parent, monkeypatch):
    fixture = failed_ffn_parent
    queue = fixture.prepare()
    queue.pop("mode")
    recovery = queue.pop("recovery")
    root = Path(recovery["output"])
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", ddp.ALLOCATOR)
    monkeypatch.setattr(ddp, "probe", lambda *_: pytest.fail("synthetic probes already seeded"))
    monkeypatch.setattr(
        ddp.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("existing run launch")
    )
    # Only bypass probe production in this default-path regression; freshness remains real.
    queue["initial_preflight"] = []
    monkeypatch.setattr(ddp, "verify", lambda *_: None)

    @contextlib.contextmanager
    def lock(_path):
        yield

    monkeypatch.setattr(ddp, "lock", lock)
    monkeypatch.setattr(ddp, "needs_preflight_acceptance", lambda *_: False)
    # The default path would run probes; prove the same existing-root guard directly
    # after simulating those already-passed probes through the seeded recovery state.
    atomic = ddp.atomic_json

    def record(path, value):
        if path == root / "state.json":
            for name, seed in recovery["seeded"].items():
                value["preflights"][name] = seed["preflight"]
                value["plans"][name] = seed["plan"]
        atomic(path, value)

    monkeypatch.setattr(ddp, "atomic_json", record)
    with pytest.raises(FileExistsError, match="fresh fit cannot replace prior run"):
        ddp.execute(queue, root)


def test_recovery_cli_selects_only_recovery_preparation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unified_ddp.py", "--recover-from", "old/queue.json", "--output", "new/queue"],
    )
    monkeypatch.setattr(
        ddp, "prepare_recovery", lambda original, output: calls.append((original, output))
    )
    monkeypatch.setattr(ddp, "prepare", lambda *_a, **_kw: pytest.fail("ordinary prepare selected"))
    ddp.main()
    assert calls == [(Path("old/queue.json"), Path("new/queue"))]


@pytest.mark.parametrize(
    "extra",
    [
        ["--manifest", "manifest.json"],
        ["--runtime", "runtime.json"],
        ["--queue", "queue.json"],
        ["--evaluate", "plan.json"],
        ["--dependency-queue", "dependency.json"],
        ["--shared-policy", "policy.json"],
        ["--attempt", "2"],
        ["--uncapped"],
    ],
)
def test_recovery_cli_rejects_mixed_modes(monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unified_ddp.py", "--recover-from", "old/queue.json", "--output", "new/queue", *extra],
    )
    monkeypatch.setattr(ddp, "prepare_recovery", lambda *_a: pytest.fail("mixed recovery mode"))
    with pytest.raises(SystemExit) as error:
        ddp.main()
    assert error.value.code == 2


@pytest.mark.parametrize("field", ["mode", "__resume_flag"])
def test_queue_verification_rejects_unknown_recovery_mode_or_scientific_resume_flags(
    failed_ffn_parent, field
):
    queue = copy.deepcopy(failed_ffn_parent.parent)
    if field == "mode":
        queue["mode"] = "resume_everything"
    else:
        queue["rows"][1]["plan"][field] = True
    with pytest.raises(ValueError, match=r"recovery mode|scientific plan flag"):
        ddp.verify(queue)
