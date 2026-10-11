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


@pytest.fixture
def dual_policy():
    return {
        "schema": shared.DUAL_SCHEMA,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
        "devices": [
            {
                "gpu_index": i,
                "gpu_uuid": f"GPU-{i}",
                "foreign_identity": {
                    "pid": 123 + i,
                    "uid": 1003,
                    "start_ticks": str(51 + i),
                    "command_sha256": str(i) * 64,
                    "script_basename": "run_scbutterfly_archived_predictions.py",
                },
            }
            for i in (0, 1)
        ],
    }


@pytest.fixture
def dual_observed(dual_policy):
    return {
        "gpus": {i: {"uuid": f"GPU-{i}", "free_mib": 3000} for i in (0, 1)},
        "processes": [{"owned": True}]
        + [
            {
                "owned": False,
                "pid": d["foreign_identity"]["pid"],
                "gpu_uuid": d["gpu_uuid"],
                "memory_mib": 720,
                "identity": copy.deepcopy(d["foreign_identity"]),
            }
            for d in dual_policy["devices"]
        ],
    }


def test_dual_current_jobs_and_their_departure_are_admitted(dual_policy, dual_observed):
    shared.admit(dual_policy, dual_observed)
    dual_observed["processes"].pop()
    shared.admit(dual_policy, dual_observed)
    dual_observed["processes"] = []
    shared.admit(dual_policy, dual_observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("uid", 1004),
        ("start_ticks", "999"),
        ("command_sha256", "a" * 64),
        ("script_basename", "other.py"),
    ],
)
def test_dual_recycled_changed_and_unapproved_jobs_fail_closed(
    dual_policy, dual_observed, field, value
):
    dual_observed["processes"][1]["identity"][field] = value
    with pytest.raises(RuntimeError, match="foreign"):
        shared.admit(dual_policy, dual_observed)


@pytest.mark.parametrize("failure", ["growth", "second_job", "uuid", "headroom"])
def test_dual_limits_cannot_be_bypassed(dual_policy, dual_observed, failure):
    if failure == "growth":
        dual_observed["processes"][1]["memory_mib"] = 1025
    elif failure == "second_job":
        dual_observed["processes"].append(copy.deepcopy(dual_observed["processes"][1]))
    elif failure == "uuid":
        dual_observed["gpus"][0]["uuid"] = "GPU-changed"
    else:
        dual_observed["gpus"][0]["free_mib"] = 2047
    with pytest.raises(RuntimeError):
        shared.admit(dual_policy, dual_observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_foreign_memory_mib", 4096),
        ("min_free_memory_mib", 0),
        ("max_own_memory_fraction", 1.0),
    ],
)
def test_dual_policy_does_not_expand_memory_boundaries(dual_policy, field, value):
    dual_policy[field] = value
    with pytest.raises(ValueError):
        shared.validate_policy(dual_policy)


@pytest.fixture
def dynamic_policy():
    return {
        "schema": shared.DYNAMIC_SCHEMA,
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "max_foreign_processes_per_gpu": 1,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
        "authorization_reference": shared.SCBUTTERFLY_AUTHORIZATION,
        "devices": [{"gpu_index": i, "gpu_uuid": f"GPU-{i}"} for i in (0, 1)],
    }


@pytest.fixture
def dynamic_observed(dual_observed):
    result = copy.deepcopy(dual_observed)
    result["gpus"][0]["total_mib"] = result["gpus"][1]["total_mib"] = 32607
    result["processes"][0] = {
        "owned": True,
        "gpu_uuid": "GPU-0",
        "memory_mib": 15150,
    }
    return result


def test_dynamic_same_account_program_can_restart_and_leave(dynamic_policy, dynamic_observed):
    shared.admit(dynamic_policy, dynamic_observed)
    process = dynamic_observed["processes"][1]
    process["pid"] = process["identity"]["pid"] = 999
    process["identity"].update(start_ticks="1000", command_sha256="f" * 64)
    shared.admit(dynamic_policy, dynamic_observed)
    dynamic_observed["processes"] = dynamic_observed["processes"][:1]
    shared.admit(dynamic_policy, dynamic_observed)


@pytest.mark.parametrize(
    "field,value", [("uid", 1002), ("uid", 1004), ("script_basename", "other.py"), ("pid", 999)]
)
def test_dynamic_unknown_user_script_or_mismatched_pid_rejected(
    dynamic_policy, dynamic_observed, field, value
):
    dynamic_observed["processes"][1]["identity"][field] = value
    with pytest.raises(RuntimeError, match="unapproved"):
        shared.admit(dynamic_policy, dynamic_observed)


def test_dynamic_missing_identity_fails_closed(dynamic_policy, dynamic_observed):
    dynamic_observed["processes"][1]["identity"] = None
    with pytest.raises(RuntimeError, match="unapproved"):
        shared.admit(dynamic_policy, dynamic_observed)


@pytest.mark.parametrize("failure", ["growth", "second_job", "uuid", "headroom", "own_sum"])
def test_dynamic_resource_limits_remain_strict(dynamic_policy, dynamic_observed, failure):
    if failure == "growth":
        dynamic_observed["processes"][1]["memory_mib"] = 1025
    elif failure == "second_job":
        duplicate = copy.deepcopy(dynamic_observed["processes"][1])
        duplicate["pid"] = duplicate["identity"]["pid"] = 999
        dynamic_observed["processes"].append(duplicate)
    elif failure == "uuid":
        dynamic_observed["gpus"][0]["uuid"] = "GPU-changed"
    elif failure == "headroom":
        dynamic_observed["gpus"][0]["free_mib"] = 2047
    else:
        dynamic_observed["processes"].append(
            {"owned": True, "gpu_uuid": "GPU-0", "memory_mib": 13000}
        )
    with pytest.raises(RuntimeError):
        shared.admit(dynamic_policy, dynamic_observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("foreign_uid", 1004),
        ("command_basename", "other.py"),
        ("max_foreign_processes_per_gpu", 2),
        ("max_foreign_processes_per_gpu", True),
        ("max_foreign_memory_mib", 2048),
        ("min_free_memory_mib", 0),
        ("max_own_memory_fraction", 1.0),
        ("authorization_reference", "unapproved"),
    ],
)
def test_dynamic_policy_cannot_expand_scope(dynamic_policy, field, value):
    dynamic_policy[field] = value
    with pytest.raises(ValueError, match="authorization"):
        shared.validate_policy(dynamic_policy)


@pytest.mark.parametrize("field,value", [("memory_mib", -1), ("memory_mib", None)])
def test_dynamic_bad_process_memory_does_not_bypass_gate(
    dynamic_policy, dynamic_observed, field, value
):
    dynamic_observed["processes"][1][field] = value
    with pytest.raises(RuntimeError, match="telemetry"):
        shared.admit(dynamic_policy, dynamic_observed)


def test_dynamic_budget_uses_owned_memory_across_all_children(dynamic_policy, dynamic_observed):
    dynamic_observed["processes"] = [
        {"owned": True, "gpu_uuid": "GPU-0", "memory_mib": 20000},
        {"owned": True, "gpu_uuid": "GPU-0", "memory_mib": 7000},
    ]
    shared.admit(dynamic_policy, dynamic_observed)
    dynamic_observed["processes"][1]["memory_mib"] = 8000
    with pytest.raises(RuntimeError, match="own GPU workload"):
        shared.admit(dynamic_policy, dynamic_observed)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 32608, True])
def test_dynamic_invalid_physical_free_memory_is_rejected(dynamic_policy, dynamic_observed, value):
    dynamic_observed["gpus"][0]["free_mib"] = value
    with pytest.raises(RuntimeError, match="telemetry"):
        shared.admit(dynamic_policy, dynamic_observed)


def test_formal_coexistence_requires_exact_preflight_acceptance():
    from run_unified_ddp import validate_coexistence_acceptance

    queue = {"source_commit": "a" * 40}
    ready = {
        "queue_sha256": "b" * 64,
        "preflights": {
            "BR": {"sha256": "c" * 64},
            "F1": {"sha256": "d" * 64},
            "N1": {"sha256": "e" * 64},
        },
    }
    acceptance = {
        "decision": "allow_formal",
        "queue_sha256": ready["queue_sha256"],
        "source_commit": queue["source_commit"],
        "preflight_sha256": {n: r["sha256"] for n, r in ready["preflights"].items()},
    }
    validate_coexistence_acceptance(queue, ready, acceptance)
    for key in acceptance:
        changed = copy.deepcopy(acceptance)
        changed[key] = None
        with pytest.raises(ValueError, match="coexistence acceptance"):
            validate_coexistence_acceptance(queue, ready, changed)


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
