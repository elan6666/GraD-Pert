"""Frozen DDP matrix and resource admission retain the scientific protocol."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

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
