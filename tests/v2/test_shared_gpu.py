"""Bounded coexistence preserves foreign jobs and the default exclusive gate."""

import copy
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import shared_gpu as shared  # noqa: E402
from run_unified_ddp import probe  # noqa: E402


@pytest.fixture
def policy():
    return {
        "schema": shared.SCHEMA,
        "gpu_index": 1,
        "gpu_uuid": "GPU-1",
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
    }


@pytest.fixture
def observed():
    return {
        "gpus": {
            0: {"uuid": "GPU-0", "free_mib": 3000},
            1: {"uuid": "GPU-1", "free_mib": 2500},
        },
        "processes": [
            {"owned": True},
            {
                "owned": False,
                "pid": 123,
                "gpu_uuid": "GPU-1",
                "memory_mib": 716,
                "identity": {
                    "uid": 1003,
                    "script_basename": "run_scbutterfly_archived_predictions.py",
                },
            },
        ],
    }


def test_known_low_memory_foreign_job_and_its_departure_are_admitted(policy, observed):
    shared.admit(policy, observed)
    observed["processes"] = []
    shared.admit(policy, observed)


@pytest.mark.parametrize("field,value", [("memory_mib", 1025), ("gpu_uuid", "GPU-0")])
def test_foreign_growth_or_other_device_is_rejected(policy, observed, field, value):
    observed["processes"][1][field] = value
    with pytest.raises(RuntimeError):
        shared.admit(policy, observed)


@pytest.mark.parametrize("field,value", [("uid", 1004), ("script_basename", "other.py")])
def test_other_users_or_programs_are_not_whitelisted(policy, observed, field, value):
    observed["processes"][1]["identity"][field] = value
    with pytest.raises(RuntimeError, match="unapproved"):
        shared.admit(policy, observed)


def test_two_foreign_jobs_and_missing_process_identity_fail_closed(policy, observed):
    duplicate = copy.deepcopy(observed["processes"][1])
    duplicate.update(pid=124, memory_mib=1)
    observed["processes"].append(duplicate)
    with pytest.raises(RuntimeError, match="one-job"):
        shared.admit(policy, observed)
    observed["processes"].pop()
    observed["processes"][1]["identity"] = None
    with pytest.raises(RuntimeError, match="unapproved"):
        shared.admit(policy, observed)


def test_physical_headroom_and_uuid_are_checked(policy, observed):
    observed["gpus"][0]["free_mib"] = 2047
    with pytest.raises(RuntimeError, match="reserve"):
        shared.admit(policy, observed)
    observed["gpus"][0]["free_mib"] = 3000
    policy["gpu_uuid"] = "GPU-other"
    with pytest.raises(RuntimeError, match="identity"):
        shared.admit(policy, observed)


def test_policy_cannot_silently_expand_the_resource_allowance(policy):
    policy["max_foreign_memory_mib"] = 4096
    with pytest.raises(ValueError):
        shared.validate_policy(policy)


def test_descendant_check_handles_recycled_or_unrelated_and_cycle_pids():
    assert shared.descendant(12, 10, {12: 11, 11: 10})
    assert not shared.descendant(12, 10, {12: 11, 11: 12})
    assert not shared.descendant(12, 10, {12: 99})


def test_termination_only_signals_verified_current_user_descendants(monkeypatch):
    def info(pid):
        return {
            "pid": pid,
            "uid": os.getuid() if pid in (10, 11) else os.getuid() + 1,
            "start_ticks": "1",
            "command_sha256": "hash",
        }

    signals = []
    monkeypatch.setattr(shared, "process_info", info)
    monkeypatch.setattr(shared, "parent_map", lambda: {10: 1, 11: 10, 12: 1, 13: 10})
    monkeypatch.setattr(shared.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(shared.time, "sleep", lambda _: None)
    shared.terminate_tree(10, info(10))
    assert {pid for pid, _ in signals} == {10, 11}
    with pytest.raises(RuntimeError, match="recycled"):
        shared.terminate_tree(10, {**info(10), "start_ticks": "2"})


def test_shared_probe_is_bound_to_sealed_queue_and_owner_pid(tmp_path):
    row = {
        "name": "N0",
        "plan": {
            "repository_root": "/data/yilangliu/source",
            "config": "cfg",
            "data_root": "data",
            "publication": "pub",
            "publication_sha256": "hash",
        },
    }
    ordinary = probe(row, tmp_path)
    command = probe(row, tmp_path, shared=True)
    assert command[: len(ordinary)] == ordinary
    assert command[-4:] == [
        "--shared-queue",
        str(tmp_path / "queue.json"),
        "--resource-owner-pid",
        str(os.getpid()),
    ]


def test_shared_preflight_rejects_changed_config_output_or_update_protocol(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("sealed config")
    args = SimpleNamespace(
        config=config,
        steps=10,
        gpu="0,1",
        data_root=tmp_path / "data",
        publication=tmp_path / "publication.json",
        publication_sha256="publication-hash",
        output=tmp_path / "N0-preflight",
        shared_queue=tmp_path / "queue.json",
    )
    queue = {
        "schema": "unified-ddp-queue-1",
        "source_commit": "commit",
        "rows": [
            {
                "name": "N0",
                "plan": {
                    "config": str(config),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "data_root": str(args.data_root),
                    "publication": str(args.publication),
                    "publication_sha256": args.publication_sha256,
                },
            }
        ],
    }
    shared.validate_probe_binding(queue, args, "commit", "preflight_only", 2)
    for changes in ({"steps": 128}, {"output": tmp_path / "other"}, {"gpu": "1,0"}):
        with pytest.raises(ValueError):
            shared.validate_probe_binding(
                queue, SimpleNamespace(**{**vars(args), **changes}), "commit", "preflight_only", 2
            )
    config.write_text("changed config")
    with pytest.raises(ValueError):
        shared.validate_probe_binding(queue, args, "commit", "preflight_only", 2)
