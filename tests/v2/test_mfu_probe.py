"""Restore transactions and actual fixed native updates, never GPU success mocks."""

import copy
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mfu_probe import (
    TrainBatches,
    aggregate_ranks,
    capture_state,
    digest,
    frozen_clock,
    restore_state,
    run_probe,
)
from test_relay_method import relay_training_fixture

from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer


def setup():
    objective, batch = relay_training_fixture(checkpointed=True)
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=V2Optimizer(objective.student, 0.001, 0),
        data=SimpleNamespace(),
        index=SimpleNamespace(),
        device=torch.device("cpu"),
        options=SimpleNamespace(microbatch=2),
        generator=np.random.default_rng(85),
    )
    # Real nonempty moments and Muon split gradients before S0.
    optimizer_step(
        objective, runtime.optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
    )
    return runtime, Source(runtime, batch)


class Source:
    def __init__(self, runtime, batch):
        self.runtime, self.batch, self.offset = runtime, batch, 0
        self.last_identity = {}

    def state_dict(self):
        return dict(offset=self.offset)

    def load_state_dict(self, state):
        self.offset = state["offset"]

    def prepare(self):
        pass

    def close(self):
        pass

    def __next__(self):
        value = (
            self.runtime.generator.random()
            + random.random()
            + np.random.random()
            + torch.rand(()).item()
        ) * 0.001
        self.last_identity = dict(rows=[f"train{self.offset}"])
        self.offset += 1
        return replace(self.batch, control=self.batch.control + value)


def clock():
    return dict(
        policy="freeze_terminal_parent_last_accepted_lr_ema_diagnostic_only",
        learning_rate=0.001,
        teacher_momentum=0.99,
        formal_continuation=False,
    )


def test_full_protocol_exact_replay_restore_with_native_updates_and_all_rng():
    runtime, source = setup()
    before = capture_state(runtime, source)
    result = run_probe(runtime, source, clock())
    assert result["status"] == "completed_partial_coverage", result
    assert (result["warmup_updates"], result["counted_updates"], result["timed_updates"]) == (
        3,
        10,
        10,
    )
    assert (
        result["warmed_restore"] and result["paired_final_state_exact"] and result["final_restore"]
    )
    assert result["mfu"] is result["hfu"] is None
    assert result["ledger"]["covered_executed_flops"] > 0
    assert result["ledger"]["unsupported_calls"] > 0
    assert all(r["end_to_end_seconds"] >= r["update_seconds"] > 0 for r in result["timing"])
    assert digest(before) == digest(capture_state(runtime, source))
    assert "step" not in vars(runtime.optimizer)


def test_snapshot_restores_grad_none_modes_pending_optimizer_and_independent_snapshot():
    runtime, source = setup()
    runtime.objective.pending = {"probe": [torch.ones(2)]}
    original = capture_state(runtime, source)
    original_digest = digest(original)
    runtime.objective.pending["probe"][0].add_(1)
    runtime.objective.eval()
    runtime.objective._temperature_updates = 88
    runtime.optimizer.training_stage = "injected"
    runtime.optimizer.stage_reset_count = 99
    for p in runtime.objective.parameters():
        p.grad = None
    for _, parts in runtime.optimizer.views:
        for part in parts:
            part.grad = None
    torch.rand(4)
    random.random()
    np.random.random()
    runtime.generator.random()
    source.offset = 9
    restore_state(runtime, source, original)
    assert digest(capture_state(runtime, source)) == original_digest
    runtime.objective.pending.clear()
    batch = next(source)
    optimizer_step(
        runtime.objective,
        runtime.optimizer,
        batch,
        microbatch=2,
        lr=0.001,
        momentum=0.99,
        bf16=False,
    )
    assert digest(original) == original_digest  # opt load must not alias saved moments


def test_exception_in_count_restores_initial_and_uninstalls_observer():
    runtime, source = setup()
    before = capture_state(runtime, source)
    original = runtime.objective.commit_statistics
    calls = 0

    def injected(momentum):
        nonlocal calls
        calls += 1
        original(momentum)
        if calls == 5:
            raise RuntimeError("injected_count_failure")

    runtime.objective.commit_statistics = injected
    result = run_probe(runtime, source, clock())
    assert result["status"] == "failed" and "injected_count_failure" in result["error"]
    assert result["final_restore"]
    assert runtime.objective.commit_statistics is injected
    assert digest(capture_state(runtime, source)) == digest(before)
    assert "step" not in vars(runtime.optimizer)


def test_deadline_is_incomplete_restored_without_inventing_samples():
    runtime, source = setup()
    before = capture_state(runtime, source)
    result = run_probe(runtime, source, clock(), continue_together=lambda allowed: False)
    assert result["status"] == "incomplete_soft_deadline"
    assert result["timed_updates"] == result["counted_updates"] == 0
    assert result["final_restore"] and result["mfu"] is None
    assert digest(capture_state(runtime, source)) == digest(before)
    assert aggregate_ranks([result])["status"] == "incomplete_or_failed"


def test_changed_timing_input_fails_and_restores_instead_of_pairing_counts():
    runtime, source = setup()
    before = capture_state(runtime, source)
    original = source.load_state_dict
    restores = 0

    def wrong_cursor(state):
        nonlocal restores
        original(state)
        restores += 1
        if restores == 1:
            source.batch = replace(source.batch, control=source.batch.control + 0.2)

    source.load_state_dict = wrong_cursor
    result = run_probe(runtime, source, clock())
    assert result["status"] == "failed" and "identity mismatch" in result["error"]
    assert result["timed_updates"] == 0 and result["mfu"] is None
    assert result["final_restore"]
    assert digest(before) == digest(capture_state(runtime, source))


def test_terminal_parent_clock_uses_final_accepted_index_and_rejects_unfinished():
    journal = dict(
        budget=[2, 150],
        epoch=2,
        contract=dict(
            schedule=dict(max_lr=0.001, min_lr=0.00001, warmup_fraction=0.16),
            teacher_start=0.99,
            teacher_end=1,
        ),
    )
    result = frozen_clock(journal, 300)
    assert result["accepted_update_index"] == 299
    assert result["learning_rate"] == 0.00001 and result["teacher_momentum"] == 1
    assert result["formal_continuation"] is False
    with pytest.raises(ValueError, match="terminal parent"):
        frozen_clock(journal, 299)


def test_aggregate_uses_slow_rank_and_global_cells_once():
    result = dict(
        status="completed_partial_coverage",
        final_restore=True,
        timing=[
            dict(input_sha256=str(i), end_to_end_seconds=2.0, global_cells=320) for i in range(10)
        ],
        ledger=dict(covered_executed_flops=100),
    )
    other = copy.deepcopy(result)
    for row in other["timing"]:
        row["end_to_end_seconds"] = 4.0
    aggregate = aggregate_ranks([result, other])
    assert aggregate["median_seconds"] == 4 and aggregate["global_cells_per_second"] == 80
    assert aggregate["covered_flops"] == 200 and aggregate["mfu"] is aggregate["hfu"] is None
    other["timing"][0]["input_sha256"] = "different"
    with pytest.raises(ValueError, match="rank TRAIN inputs"):
        aggregate_ranks([result, other])


def test_train_cursor_seeks_raw_rows_without_consuming_view_rng(monkeypatch):
    from gradpert.training.v2 import views

    seen = []
    raw = [
        SimpleNamespace(
            control_row_ids=(f"c{i}",), perturbed_row_ids=(f"t{i}",), condition_ids=("A",)
        )
        for i in range(5)
    ]
    runtime = SimpleNamespace(
        purpose="training",
        options=SimpleNamespace(training_mode="joint", max_conditions=0),
        data=SimpleNamespace(
            iter_train_epoch=lambda **kw: iter(raw),
            training_batch_identity_specs=lambda **kw: tuple(raw),
        ),
        device="cpu",
        batch_size=1,
        index=None,
        generator=np.random.default_rng(18),
        allowed_expression_ids=None,
        au_context_indices=lambda row: None,
    )

    def assemble(row, index, options, rng, **kw):
        seen.append(row.control_row_ids)
        return float(rng.random())

    monkeypatch.setattr(views, "assemble_batch", assemble)
    source = TrainBatches(runtime)
    manifest = source.seal()
    assert len(manifest) == 13
    assert [(row["epoch"], row["offset"]) for row in manifest][-1] == (2, 3)
    source.load_state_dict(dict(epoch=0, offset=3))
    source.prepare()
    assert seen == []
    assert next(source) == np.random.default_rng(18).random()
    assert seen == [("c3",)]
    assert source.state_dict() == dict(epoch=0, offset=4)
    raw[4] = SimpleNamespace(
        control_row_ids=("changed",), perturbed_row_ids=("t4",), condition_ids=("A",)
    )
    with pytest.raises(ValueError, match="sealed diagnostic manifest"):
        next(source)


def test_restore_failure_never_claims_success():
    runtime, source = setup()
    original = source.load_state_dict

    def broken_restore(state):
        original(state)
        source.offset += 1

    source.load_state_dict = broken_restore
    result = run_probe(runtime, source, clock(), continue_together=lambda allowed: False)
    assert result["status"] == "failed_restore"
    assert not result["final_restore"] and result["mfu"] is result["hfu"] is None


def test_requires_explicit_clock_empty_pending_and_no_background_prefetch():
    runtime, source = setup()
    with pytest.raises(ValueError, match="explicit frozen"):
        run_probe(runtime, source, dict(clock(), learning_rate=0))
    runtime.objective.pending = {"stale": [torch.ones(1)]}
    with pytest.raises(ValueError, match="empty pending"):
        run_probe(runtime, source, clock())
    runtime.purpose = "training"
    runtime.options.training_mode = "joint"
    runtime.data.system_options = SimpleNamespace(background_prefetch=True)
    with pytest.raises(ValueError, match="background I/O"):
        TrainBatches(runtime)


def test_original_native_checkpoint_and_terminal_receipt_identity_not_reinterpreted(tmp_path):
    import json
    from pathlib import Path

    from mfu_probe import validate_parent

    from gradpert.config import load_experiment_config
    from gradpert.execution.v2_training_stage import training_stage_evidence
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint

    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/relay_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(config_path)
    runtime, _ = setup()
    manifest = dict(
        source=dict(commit="a" * 40, published_commit="a" * 40, dirty=False),
        config_sha256=sha256_file(config_path),
        resolved_config_sha256=sha256_json(config.model_dump(mode="json")),
        run_id="synthetic-native-parent",
    )
    fit = tmp_path / "fit"
    fit.mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    epochs = int(config.training.max_epochs.value)
    history = [dict(epoch=i) for i in range(1, epochs + 1)]
    checkpoint = fit / "epoch-last.pt"
    save_checkpoint(
        checkpoint,
        runtime.objective,
        runtime.optimizer,
        identity=manifest,
        progress=dict(history=history),
        generator=runtime.generator,
    )
    sha = sha256_file(checkpoint)
    journal = dict(
        identity=manifest,
        epoch=epochs,
        budget=[epochs, 1],
        best=None,
        selection_metric="final_epoch",
        last=dict(file=checkpoint.name, sha256=sha, epoch=epochs),
    )
    (fit / "epoch_state.json").write_text(json.dumps(journal))
    (fit / "history.json").write_text(json.dumps(history))
    receipt = training_stage_evidence(tmp_path, manifest, epochs)
    (tmp_path / "TRAIN_COMPLETE.json").write_text(json.dumps(receipt))
    verified, _, path = validate_parent(tmp_path, config, sha)
    assert verified == manifest and path == checkpoint
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(
            checkpoint,
            runtime.objective,
            runtime.optimizer,
            identity=dict(manifest, run_id="latest-source-is-not-original"),
            generator=runtime.generator,
        )
    with pytest.raises(ValueError, match="terminal native last"):
        validate_parent(tmp_path, config, "b" * 64)
    (tmp_path / "TRAIN_COMPLETE.json").write_text(json.dumps(dict(receipt, epoch=epochs - 1)))
    with pytest.raises(ValueError, match="terminal native training receipt"):
        validate_parent(tmp_path, config, sha)


def test_output_roots_are_protected_in_both_directions_and_across_symlinks(tmp_path):
    from mfu_probe import protect_output

    original = tmp_path / "native"
    original.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        protect_output(original / "new", [original])
    with pytest.raises(ValueError, match="overlaps"):
        protect_output(tmp_path / "new-parent", [tmp_path / "new-parent/future-data"])
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        protect_output(alias / "new", [original])
    with pytest.raises(ValueError, match="already exists"):
        protect_output(original, [])
    protect_output(tmp_path / "fresh", [original])


def test_input_closure_detects_content_and_symlink_retargeting(tmp_path):
    from mfu_probe import input_bindings, verify_inputs

    original = tmp_path / "input.json"
    original.write_text("original")
    binding = input_bindings([original])
    verify_inputs(binding)
    original.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(binding)
    other = tmp_path / "same-bytes.json"
    other.write_text("changed")
    alias = tmp_path / "link.json"
    alias.symlink_to(original)
    binding = input_bindings([alias])
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(binding)


def test_real_original_file_leases_and_kernel_owner_parser_require_both(tmp_path):
    import os

    from mfu_probe import verify_lease_records
    from run_group import lock

    leases = [tmp_path / f"v2-gpu-{i}.lock" for i in (0, 1)]
    from contextlib import ExitStack

    with ExitStack() as stack:
        for path in leases:
            stack.enter_context(lock(path))
            with pytest.raises(RuntimeError, match="another process owns lease"), lock(path):
                pass
        # Linux kernel-record parser unit contract on macOS. Actual file-lock
        # exclusion above uses the original helper and the real local kernel.
        records = []
        for i, path in enumerate(leases):
            stat = path.stat()
            device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
            records.append(f"{i}: FLOCK ADVISORY WRITE {os.getpid()} {device} 0 EOF")
        verify_lease_records("\n".join(records), os.getpid(), leases)
        with pytest.raises(ValueError, match="both original GPU leases"):
            verify_lease_records(records[0], os.getpid(), leases)
        with pytest.raises(ValueError, match="both original GPU leases"):
            verify_lease_records("\n".join(records), os.getpid() + 1, leases)


def test_resource_rejection_during_count_stops_and_restores_actual_native_state():
    runtime, source = setup()
    before = capture_state(runtime, source)
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("original resource gate rejected foreign growth")

    result = run_probe(runtime, source, clock(), resource_guard=guard)
    assert result["status"] == "failed" and result["final_restore"]
    assert result["warmup_updates"] == 3 and result["counted_updates"] == 1
    assert result["timed_updates"] == 0 and result["mfu"] is None
    assert digest(before) == digest(capture_state(runtime, source))
    assert "step" not in vars(runtime.optimizer)


def test_resource_binding_hash_target_and_unbound_gpu_fail_closed(tmp_path):
    import json

    from mfu_probe import ResourceGuard

    from gradpert.hashing import sha256_file

    path = tmp_path / "owner.json"
    path.write_text(
        json.dumps(dict(schema="mfu-resource-owner-binding-1", owner=dict(pid=1), binding={}))
    )
    with pytest.raises(ValueError, match="binding hash"):
        ResourceGuard(path, "a" * 64, 1, {})
    with pytest.raises(ValueError, match="exact diagnostic target"):
        ResourceGuard(path, sha256_file(path), 2, {})
    runtime, source = setup()
    runtime.device = torch.device("cuda")
    with pytest.raises(ValueError, match="bound resource guard"):
        run_probe(runtime, source, clock())


@pytest.mark.parametrize(
    "failure",
    [
        "owner_exit",
        "start_ticks",
        "wrong_user",
        "nonancestor",
        "policy_mutation",
        "missing_lease",
        "foreign_growth",
    ],
)
def test_resource_guard_check_wiring_rejects_each_owner_policy_lease_boundary(
    tmp_path, monkeypatch, failure
):
    """Linux process/telemetry fixtures; real original admit and lease parser.

    This is not a live Linux/SSH/CUDA admission receipt. The other lease test
    independently verifies real local-kernel lock exclusion with the same helper.
    """
    import json
    import os
    from pathlib import Path

    import mfu_probe as probe
    import shared_gpu as shared

    from gradpert.hashing import sha256_file

    owner = dict(
        pid=999991,
        uid=os.getuid(),
        ppid=1,
        start_ticks="123",
        command_sha256="a" * 64,
        script_basename="guardian.py",
    )
    policy = dict(
        schema=shared.SCHEMA,
        gpu_index=1,
        gpu_uuid="GPU-1",
        foreign_uid=1003,
        command_basename="run_scbutterfly_archived_predictions.py",
        max_foreign_memory_mib=1024,
        min_free_memory_mib=2048,
        max_own_memory_fraction=0.85,
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    spec = dict(owner=owner, policy=dict(path=str(policy_path), sha256=sha256_file(policy_path)))
    spec_path = tmp_path / "owner.json"
    spec_path.write_text(json.dumps(spec))
    # Constructor server-path/hash/target checks have separate negatives. Here
    # isolate check() without pretending tempfile paths are on the server.
    guard = probe.ResourceGuard.__new__(probe.ResourceGuard)
    guard.path, guard.sha256, guard.spec = spec_path, sha256_file(spec_path), spec
    guard.policy_path, guard.policy = policy_path, policy
    observed = dict(
        gpus={0: dict(uuid="GPU-0", free_mib=3000), 1: dict(uuid="GPU-1", free_mib=3000)},
        processes=[],
    )
    monkeypatch.setattr(shared, "process_info", lambda pid: owner)
    monkeypatch.setattr(shared, "parent_map", lambda: {os.getpid(): owner["pid"]})
    monkeypatch.setattr(shared, "snapshot", lambda pid: observed)
    records = []
    for i in (0, 1):
        path = tmp_path / f"v2-gpu-{i}.lock"
        path.touch()
        stat = path.stat()
        key = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        records.append(f"{i}: FLOCK ADVISORY WRITE {owner['pid']} {key} 0 EOF")
    monkeypatch.setattr(probe, "LEASE_ROOT", tmp_path)
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: (
            "\n".join(records) if str(self) == "/proc/locks" else read_text(self, *a, **k)
        ),
    )
    guard.check()
    if failure == "owner_exit":
        monkeypatch.setattr(shared, "process_info", lambda pid: None)
    elif failure == "start_ticks":
        monkeypatch.setattr(shared, "process_info", lambda pid: dict(owner, start_ticks="reused"))
    elif failure == "wrong_user":
        monkeypatch.setattr(os, "getuid", lambda: owner["uid"] + 1)
    elif failure == "nonancestor":
        monkeypatch.setattr(shared, "parent_map", dict)
    elif failure == "policy_mutation":
        policy_path.write_text("changed-policy")
    elif failure == "missing_lease":
        records.pop()
    else:
        observed["processes"].append(
            dict(
                owned=False,
                pid=123,
                gpu_uuid="GPU-1",
                memory_mib=1025,
                identity=dict(uid=1003, script_basename=policy["command_basename"]),
            )
        )
    with pytest.raises((ValueError, RuntimeError)):
        guard.check()
