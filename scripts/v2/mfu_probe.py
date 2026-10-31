"""Isolated TRAIN-only full-update timing with partial FLOPs and exact restore."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
import math
import os
import subprocess
import time
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mfu_ledger import ArithmeticLedger, counting_regions

WARMUP_UPDATES = 3
MEASURED_UPDATES = 10
WINDOW_SECONDS = 15 * 60
LEASE_ROOT = Path("/data/yilangliu/GraD-Pert/runtime")


def measured_update_budget(value: int) -> int:
    """Accept an explicit bounded integer, never bool or a coerced numeric value."""
    if type(value) is not int or not 1 <= value <= MEASURED_UPDATES:
        raise ValueError("measured updates must be an integer from 1 to 10")
    return value


def protect_output(output: Path, protected: list[Path]) -> None:
    """A new diagnostic directory cannot contain, or be inside, any input root."""
    target = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("diagnostic output already exists")
    for root in protected:
        resolved = root.resolve()
        if target.is_relative_to(resolved) or resolved.is_relative_to(target):
            raise ValueError(f"diagnostic output overlaps protected input root: {root}")


def input_bindings(paths: list[Path]) -> list[dict[str, str]]:
    from gradpert.hashing import sha256_file

    return [
        dict(
            path=str(path.absolute()),
            resolved_path=str(path.resolve(strict=True)),
            sha256=sha256_file(path),
        )
        for path in paths
    ]


def verify_inputs(bindings: list[dict[str, str]]) -> None:
    from gradpert.hashing import sha256_file

    for item in bindings:
        path = Path(item["path"])
        if (
            str(path.resolve(strict=True)) != item["resolved_path"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError("immutable diagnostic input changed before closure")


def verify_lease_records(records: str, owner_pid: int, leases: list[Path]) -> None:
    """Read Linux kernel FLOCK ownership, not the existence of lock files."""
    held = set()
    for line in records.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        if fields[4] != str(owner_pid):
            continue
        major, minor, inode = fields[5].split(":")
        held.add((int(major, 16), int(minor, 16), int(inode)))
    for path in leases:
        if path.is_symlink():
            raise ValueError("GPU lease path must not be a symlink")
        stat = path.stat()
        if (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) not in held:
            raise ValueError("resource owner does not hold both original GPU leases")


class ResourceGuard:
    """Bind a live CPU owner, both existing leases and the unchanged admission policy."""

    def __init__(self, path: Path, expected_sha256: str, owner_pid: int, binding: dict[str, Any]):
        from shared_gpu import validate_policy

        from gradpert.hashing import sha256_file

        if sha256_file(path) != expected_sha256:
            raise ValueError("resource owner binding hash differs")
        spec = json.loads(path.read_text())
        if (
            spec.get("schema") != "mfu-resource-owner-binding-1"
            or spec.get("binding") != binding
            or spec["owner"]["pid"] != owner_pid
        ):
            raise ValueError("resource owner binding differs from the exact diagnostic target")
        self.path, self.sha256, self.spec = path, expected_sha256, spec
        self.policy_path = Path(spec["policy"]["path"]).resolve(strict=True)
        if not self.policy_path.is_relative_to("/data/yilangliu"):
            raise ValueError("resource policy must remain on the authorized server")
        if sha256_file(self.policy_path) != spec["policy"]["sha256"]:
            raise ValueError("original resource policy hash differs")
        self.policy = json.loads(self.policy_path.read_text())
        validate_policy(self.policy)

    def check(self) -> None:
        from shared_gpu import admit, descendant, parent_map, process_info, snapshot

        from gradpert.hashing import sha256_file

        if (
            sha256_file(self.path) != self.sha256
            or sha256_file(self.policy_path) != self.spec["policy"]["sha256"]
        ):
            raise ValueError("sealed resource owner or original policy changed")
        owner = self.spec["owner"]
        if process_info(owner["pid"]) != owner or owner["uid"] != os.getuid():
            raise ValueError("resource owner exited, changed identity or belongs to another user")
        if not descendant(os.getpid(), owner["pid"], parent_map()):
            raise ValueError("diagnostic must be a descendant of its bound CPU resource owner")
        verify_lease_records(
            Path("/proc/locks").read_text(),
            owner["pid"],
            [LEASE_ROOT / f"v2-gpu-{gpu}.lock" for gpu in (0, 1)],
        )
        # This is the existing allowlist/foreign-job/free-memory/own-memory
        # checker. No allowance is inferred from the new diagnostic binding.
        admit(self.policy, snapshot(owner["pid"]))


def clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def digest(value: Any) -> str:
    # This existing helper supports dataclasses (actual TrainingBatch), NumPy
    # arrays and all tensor dtypes. Never called inside a timed update.
    from update_parity import tree_digest

    return tree_digest(value)


def state_hash_differences(
    counted: dict[str, Any], timed: dict[str, Any], *, max_entries: int = 64
) -> dict[str, Any]:
    """Failure-only bounded names and hashes; never serialize snapshot values.

    Top-level hashes identify every changed protected component. Nested leaf
    paths localize parameter/moment/RNG or plain-attribute differences without
    exporting tensors, scalars, attribute strings or RNG data. A visit budget
    bounds nested traversal; truncation is explicit, never interpreted as parity.
    """
    if type(max_entries) is not int or not 1 <= max_entries <= 256:
        raise ValueError("state difference entry limit must be an integer from 1 to 256")
    missing = object()
    components: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    visits = 0
    truncated = False

    def name(key: Any) -> str:
        # Snapshot keys are parameter/module names or optimizer state indices.
        # Bound path text too; unsupported key types are represented by hashes.
        if type(key) in (str, int):
            text = str(key)
            return text if len(text) <= 192 else text[:128] + "#sha256:" + digest(key)
        return type(key).__name__ + "#sha256:" + digest(key)

    def hashed(value: Any) -> str | None:
        return None if value is missing else digest(value)

    def record(path: list[str], left: Any, right: Any) -> None:
        nonlocal truncated
        if len(entries) >= max_entries:
            truncated = True
            return
        entries.append(
            dict(
                path=path,
                counted_sha256=hashed(left),
                timed_sha256=hashed(right),
                counted_present=left is not missing,
                timed_present=right is not missing,
                counted_type=None if left is missing else type(left).__name__,
                timed_type=None if right is missing else type(right).__name__,
            )
        )

    def walk(path: list[str], left: Any, right: Any) -> None:
        nonlocal visits, truncated
        if truncated:
            return
        visits += 1
        if visits > 4096 or len(path) > 32:
            truncated = True
            return
        if (
            type(left) is type(right)
            and dataclasses.is_dataclass(left)
            and not isinstance(left, type)
        ):
            for field in dataclasses.fields(left):
                walk([*path, field.name], getattr(left, field.name), getattr(right, field.name))
                if truncated:
                    break
        elif type(left) is type(right) and isinstance(left, dict):
            for key in sorted(left.keys() | right.keys(), key=repr):
                walk([*path, name(key)], left.get(key, missing), right.get(key, missing))
                if truncated:
                    break
        elif type(left) is type(right) and isinstance(left, (tuple, list)):
            for index in range(max(len(left), len(right))):
                walk(
                    [*path, str(index)],
                    left[index] if index < len(left) else missing,
                    right[index] if index < len(right) else missing,
                )
                if truncated:
                    break
        elif hashed(left) != hashed(right):
            record(path, left, right)

    for key in sorted(counted.keys() | timed.keys(), key=repr):
        left, right = counted.get(key, missing), timed.get(key, missing)
        before, after = hashed(left), hashed(right)
        if before != after:
            components.append(dict(path=[name(key)], counted_sha256=before, timed_sha256=after))
            walk([name(key)], left, right)
    return dict(
        schema="gradpert-native-state-hash-differences-1",
        changed_components=components,
        differences=entries,
        max_entries=max_entries,
        max_nested_visits=4096,
        max_path_depth=32,
        nested_visits=min(visits, 4096),
        truncated=truncated,
        values_exported=False,
        interpretation="localization only; no state component is excluded from exact parity",
    )


def plain_attributes(owner: Any) -> dict[str, Any]:
    return {k: v for k, v in vars(owner).items() if type(v) in (bool, int, float, str, type(None))}


def restore_attributes(owner: Any, state: dict[str, Any]) -> None:
    for name in plain_attributes(owner).keys() - state.keys():
        delattr(owner, name)
    for name, value in state.items():
        setattr(owner, name, value)


def part_gradients(optimizer: Any) -> list[list[Any]]:
    return [[clone_tree(part.grad) for part in parts] for _, parts in optimizer.views]


def capture_state(runtime: Any, batches: Any) -> dict[str, Any]:
    """Independent CPU copies including state omitted by resumable checkpoints."""
    from gradpert.training.checkpoint import _rng_state

    objective, optimizer = runtime.objective, runtime.optimizer
    return dict(
        objective=clone_tree(objective.state_dict()),
        optimizer=clone_tree(optimizer.state_dict()),
        gradients={n: clone_tree(p.grad) for n, p in objective.named_parameters()},
        requires_grad={n: p.requires_grad for n, p in objective.named_parameters()},
        part_gradients=part_gradients(optimizer),
        pending=clone_tree(objective.pending),
        module_flags={n: plain_attributes(m) for n, m in objective.named_modules()},
        optimizer_flags=plain_attributes(optimizer),
        data_flags=plain_attributes(runtime.data),
        index_flags=plain_attributes(runtime.index),
        pipeline_stats=clone_tree(getattr(runtime.data, "pipeline_stats", None)),
        rng=clone_tree(_rng_state()),
        generator=clone_tree(runtime.generator.bit_generator.state),
        cursor=clone_tree(batches.state_dict()),
    )


def restore_state(runtime: Any, batches: Any, state: dict[str, Any]) -> None:
    """Restore at every exit, without writing the parent checkpoint/run."""
    from gradpert.training.checkpoint import _restore_rng_state

    objective, optimizer = runtime.objective, runtime.optimizer
    batches.close()
    objective.load_state_dict(state["objective"], strict=True)
    # Native optimizer load can keep references to supplied tensors; clone the
    # snapshot again so subsequent updates cannot mutate the saved S0/S3 copy.
    optimizer.load_state_dict(clone_tree(state["optimizer"]))
    for name, parameter in objective.named_parameters():
        parameter.requires_grad_(state["requires_grad"][name])
        grad = state["gradients"][name]
        parameter.grad = None if grad is None else grad.to(parameter).clone()
    if len(optimizer.views) != len(state["part_gradients"]):
        raise ValueError("Muon split gradient topology changed")
    for (_, parts), gradients in zip(optimizer.views, state["part_gradients"], strict=True):
        for part, grad in zip(parts, gradients, strict=True):
            part.grad = None if grad is None else grad.to(part).clone()
    objective.pending = {
        key: [item.to(next(objective.parameters()).device).clone() for item in values]
        for key, values in state["pending"].items()
    }
    for name, module in objective.named_modules():
        restore_attributes(module, state["module_flags"][name])
    restore_attributes(optimizer, state["optimizer_flags"])
    restore_attributes(runtime.data, state["data_flags"])
    restore_attributes(runtime.index, state["index_flags"])
    if state["pipeline_stats"] is not None:
        runtime.data.pipeline_stats = clone_tree(state["pipeline_stats"])
    runtime.generator.bit_generator.state = clone_tree(state["generator"])
    batches.load_state_dict(state["cursor"])
    _restore_rng_state(state["rng"])
    if digest(capture_state(runtime, batches)) != digest(state):
        raise ValueError("exact full-state restoration failed")


def frozen_clock(journal: dict[str, Any], completed_steps: int) -> dict[str, Any]:
    """Freeze the last accepted native LR/EMA, never extrapolate a cosine."""
    from gradpert.config.step_schedule import EndpointLRWarmupCosine

    epochs, per_epoch = journal["budget"]
    total = epochs * per_epoch
    if completed_steps != total or journal["epoch"] != epochs or total < 1:
        raise ValueError("MFU requires the completed terminal parent clock")
    contract = journal["contract"]
    schedule = EndpointLRWarmupCosine(**contract["schedule"])
    index = completed_steps - 1
    start, end = contract["teacher_start"], contract["teacher_end"]
    if not 0 < start <= end <= 1:
        raise ValueError("invalid parent Teacher momentum endpoints")
    momentum = end - (end - start) * (1 + math.cos(math.pi * index / max(1, total - 1))) / 2
    return dict(
        policy="freeze_terminal_parent_last_accepted_lr_ema_diagnostic_only",
        parent_completed_updates=completed_steps,
        accepted_update_index=index,
        formal_horizon=total,
        learning_rate=schedule.at_step(index, total)["learning_rate"],
        teacher_momentum=momentum,
        formal_continuation=False,
    )


class TrainBatches:
    """Replay a raw TRAIN cursor before view assembly; no skipped view RNG draws."""

    def __init__(self, runtime: Any) -> None:
        if runtime.purpose != "training" or runtime.options.training_mode != "joint":
            raise ValueError("MFU currently supports native joint TRAIN-only runtimes")
        if getattr(getattr(runtime.data, "system_options", None), "background_prefetch", False):
            raise ValueError("exact MFU transactions currently require no background I/O prefetch")
        self.runtime, self.epoch, self.offset = runtime, 0, 0
        self.stream = None
        self.last_identity: dict[str, Any] = {}
        self.expected_rows: dict[tuple[int, int], dict[str, Any]] | None = None

    def seal(self, measured_updates: int = MEASURED_UPDATES) -> list[dict[str, Any]]:
        """Freeze warmup plus the requested measured TRAIN batches, without expressions."""
        remaining = WARMUP_UPDATES + measured_update_budget(measured_updates)
        r = self.runtime
        manifest = []
        epoch = 0
        while remaining:
            specs = r.data.training_batch_identity_specs(
                epoch=epoch,
                batch_size=r.batch_size,
                max_unique_conditions=0
                if r.options.max_conditions == 0
                else min(r.options.max_conditions, r.batch_size),
            )
            if not specs:
                raise ValueError("cannot seal empty TRAIN epoch")
            for offset, raw in enumerate(specs[:remaining], 1):
                manifest.append(
                    dict(
                        epoch=epoch,
                        offset=offset,
                        control=list(raw.control_row_ids),
                        truth=list(raw.perturbed_row_ids),
                        conditions=list(raw.condition_ids),
                    )
                )
            remaining -= min(remaining, len(specs))
            epoch += 1
        self.expected_rows = {(r["epoch"], r["offset"]): r for r in manifest}
        return manifest

    def state_dict(self) -> dict[str, int]:
        return dict(epoch=self.epoch, offset=self.offset)

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.close()
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.last_identity = {}

    def close(self) -> None:
        if self.stream is not None:
            close = getattr(self.stream, "close", None)
            if close is not None:
                close()
            self.stream = None

    def prepare(self) -> None:
        if self.stream is not None:
            return
        r = self.runtime
        self.stream = r.data.iter_train_epoch(
            epoch=self.epoch,
            device=r.device,
            batch_size=r.batch_size,
            max_unique_conditions=0
            if r.options.max_conditions == 0
            else min(r.options.max_conditions, r.batch_size),
        )
        # Seeking does not assemble views or advance the view RNG; its cost is
        # recorded as restore/seek overhead, outside measured samples.
        for _ in range(self.offset):
            try:
                next(self.stream)
            except StopIteration as error:
                raise ValueError("sealed TRAIN cursor exceeds its epoch") from error

    def __next__(self):
        from gradpert.training.v2.views import assemble_batch

        r = self.runtime
        r.data.control_only_training = False
        self.prepare()
        try:
            raw = next(self.stream)
        except StopIteration:
            self.close()
            self.epoch += 1
            self.offset = 0
            self.prepare()
            try:
                raw = next(self.stream)
            except StopIteration as error:
                raise ValueError("empty TRAIN epoch") from error
        self.offset += 1
        self.last_identity = dict(
            epoch=self.epoch,
            offset=self.offset,
            control=list(raw.control_row_ids),
            truth=list(raw.perturbed_row_ids),
            conditions=list(raw.condition_ids),
        )
        if (
            self.expected_rows is not None
            and self.expected_rows.get((self.epoch, self.offset)) != self.last_identity
        ):
            raise ValueError("actual TRAIN rows differ from the sealed diagnostic manifest")
        return assemble_batch(
            raw,
            r.index,
            r.options,
            r.generator,
            allowed_expression_ids=r.allowed_expression_ids,
            context_index=r.au_context_indices(raw),
        )


def run_probe(
    runtime: Any,
    batches: Any,
    clock: dict[str, Any],
    *,
    synchronize=lambda: None,
    continue_together=lambda allowed: allowed,
    resource_guard=None,
    monotonic=time.monotonic,
    window_seconds: float = WINDOW_SECONDS,
    measured_updates: int = MEASURED_UPDATES,
) -> dict[str, Any]:
    """Execute native updates, restoring S3 before timing and S0 on every exit."""
    from gradpert.training.checkpoint import _rng_state
    from gradpert.training.v2.engine import optimizer_step, slice_cells

    measured_updates = measured_update_budget(measured_updates)
    if runtime.device.type == "cuda" and resource_guard is None:
        raise ValueError("GPU probe requires its bound resource guard")
    if not 0 < window_seconds <= WINDOW_SECONDS:
        raise ValueError("diagnostic window must be positive and at most 15 minutes")
    if (
        clock.get("formal_continuation") is not False
        or clock.get("policy") != "freeze_terminal_parent_last_accepted_lr_ema_diagnostic_only"
        or not math.isfinite(clock["learning_rate"])
        or clock["learning_rate"] <= 0
        or not 0 < clock["teacher_momentum"] <= 1
    ):
        raise ValueError("explicit frozen diagnostic clock required")
    if runtime.objective.pending:
        raise ValueError(
            "MFU must start at a completed native update with empty pending statistics"
        )
    result: dict[str, Any] = dict(
        status="running",
        scientific_complete=False,
        mfu=None,
        hfu=None,
        warmup_updates=0,
        requested_measured_updates=measured_updates,
        counted_updates=0,
        timed_updates=0,
        actual_samples=0,
        clock=clock,
        count=[],
        timing=[],
        final_restore=False,
    )
    ledger = ArithmeticLedger()
    started = monotonic()
    initial = capture_state(runtime, batches)
    # Restore time is variable. Reserve at least 30 s, or four times the initial
    # snapshot cost; an indivisible update can cross a soft deadline.
    reserve = max(30.0, 4 * (monotonic() - started))
    result["initial_state_sha256"] = digest(initial)
    result["restore_reserve_seconds"] = reserve
    result["soft_window_seconds"] = window_seconds

    def admitted() -> bool:
        if resource_guard is not None:
            resource_guard()
        return continue_together(monotonic() - started < window_seconds - reserve)

    def update(batch):
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if world > 1 else 0
        conditions = batch.condition_index if world > 1 else None
        cells = len(batch.control)
        if world > 1:
            batch = slice_cells(batch, cells * rank // world, cells * (rank + 1) // world)
        terms = optimizer_step(
            runtime.objective,
            runtime.optimizer,
            batch,
            microbatch=runtime.options.microbatch,
            lr=clock["learning_rate"],
            momentum=clock["teacher_momentum"],
            bf16=runtime.device.type == "cuda",
            global_condition_index=conditions,
        )
        if not all(math.isfinite(v) for v in terms.values()):
            raise FloatingPointError("nonfinite native update metric")
        return cells

    try:
        for _ in range(WARMUP_UPDATES):
            if not admitted():
                result["status"] = "incomplete_soft_deadline"
                return result
            update(next(batches))
            synchronize()
            result["warmup_updates"] += 1
        warmed = capture_state(runtime, batches)
        result["warmed_state_sha256"] = digest(warmed)
        with counting_regions(runtime, ledger):
            for i in range(measured_updates):
                if not admitted():
                    result["status"] = "incomplete_soft_deadline"
                    return result
                ledger.update = i
                batch = next(batches)
                identity = dict(
                    input_sha256=digest(batch),
                    rows=clone_tree(batches.last_identity),
                    rng_before_sha256=digest((_rng_state(), runtime.generator.bit_generator.state)),
                )
                with ledger:
                    cells = update(batch)
                synchronize()
                result["count"].append(dict(**identity, global_cells=cells))
                result["counted_updates"] += 1
        counted_final = capture_state(runtime, batches)
        result["counted_final_state_sha256"] = digest(counted_final)
        # Close all contexts first, then restore. No counter/profiler/hooks are
        # active during the independently replayed timing window.
        restore_state(runtime, batches, warmed)
        result["warmed_restore"] = True
        batches.prepare()
        # Raw cursor seeking can increment I/O diagnostics without consuming
        # view RNG. Exclude that restore overhead from the replayed counters.
        restore_attributes(runtime.data, warmed["data_flags"])
        if warmed["pipeline_stats"] is not None:
            runtime.data.pipeline_stats = clone_tree(warmed["pipeline_stats"])
        for i in range(measured_updates):
            if not admitted():
                result["status"] = "incomplete_soft_deadline"
                return result
            # The RNG digest and all CPU copies occur outside the timer. View
            # RNG advances in next(batch), so compare it after materialization.
            synchronize()
            tick = monotonic()
            batch = next(batches)
            synchronize()
            ready = monotonic()
            cells = update(batch)
            synchronize()
            stop = monotonic()
            # Hash input after the update, outside timing; the native engine
            # must not mutate its batch. The paired final state checks RNG too.
            identity = dict(input_sha256=digest(batch), rows=clone_tree(batches.last_identity))
            if any(identity[k] != result["count"][i][k] for k in identity):
                raise ValueError("count/timing TRAIN input or view identity mismatch")
            result["timing"].append(
                dict(
                    **identity,
                    global_cells=cells,
                    end_to_end_seconds=stop - tick,
                    data_wait_seconds=ready - tick,
                    update_seconds=stop - ready,
                )
            )
            result["timed_updates"] += 1
            result["actual_samples"] += 1
        timed_final = capture_state(runtime, batches)
        result["timed_final_state_sha256"] = digest(timed_final)
        if result["timed_final_state_sha256"] != result["counted_final_state_sha256"]:
            result["paired_final_state_exact"] = False
            # This is outside both measurement windows, after observers exit.
            # Preserve the original strict failure even if forensics itself fails.
            try:
                result["state_hash_differences"] = state_hash_differences(
                    counted_final, timed_final
                )
            except Exception as error:
                result["state_hash_differences_error_type"] = type(error).__name__
            raise ValueError("count/timing complete update state or RNG differs")
        result["paired_final_state_exact"] = True
        durations = [row["end_to_end_seconds"] for row in result["timing"]]
        result["local_median_seconds"] = float(np.median(durations))
        result["local_p95_seconds"] = float(np.quantile(durations, 0.95))
        result["local_cells_per_second"] = sum(r["global_cells"] for r in result["timing"]) / sum(
            durations
        )
        result["status"] = "completed_partial_coverage"
        return result
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        return result
    finally:
        result["ledger"] = ledger.payload()
        try:
            restore_state(runtime, batches, initial)
            synchronize()
            result["final_restore"] = True
        except Exception as error:
            result.update(
                status="failed_restore",
                restore_error_type=type(error).__name__,
                restore_error=str(error)[:1000],
            )
        result["window_elapsed_seconds"] = monotonic() - started
        result["soft_window_exceeded"] = result["window_elapsed_seconds"] > window_seconds


def aggregate_ranks(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return dict(
            status="incomplete_or_failed",
            mfu=None,
            hfu=None,
            rank_results=results,
            scientific_complete=False,
        )
    budgets = [measured_update_budget(r.get("requested_measured_updates")) for r in results]
    if len(set(budgets)) != 1:
        raise ValueError("rank requested measured-update budgets differ")
    measured_updates = budgets[0]
    for result in results:
        for field, rows in (("counted_updates", "count"), ("timed_updates", "timing")):
            count = result.get(field)
            if (
                type(count) is not int
                or not 0 <= count <= measured_updates
                or not isinstance(result.get(rows), list)
                or len(result[rows]) != count
            ):
                raise ValueError("rank actual sample counts differ from recorded samples or budget")
        actual = result.get("actual_samples")
        if type(actual) is not int or actual != result["timed_updates"]:
            raise ValueError("rank actual sample count differs from timing samples")
        if result["status"] == "completed_partial_coverage" and (
            type(result.get("warmup_updates")) is not int
            or result["warmup_updates"] != WARMUP_UPDATES
            or result["counted_updates"] != measured_updates
            or actual != measured_updates
        ):
            raise ValueError("completed rank requires exactly the requested measured samples")
    if any(
        r["status"] != "completed_partial_coverage" or r["final_restore"] is not True
        for r in results
    ):
        return dict(
            status="incomplete_or_failed",
            requested_measured_updates=measured_updates,
            mfu=None,
            hfu=None,
            rank_results=results,
            scientific_complete=False,
        )
    for i in range(measured_updates):
        if len({r["timing"][i]["input_sha256"] for r in results}) != 1:
            raise ValueError("rank TRAIN inputs differ before sharding")
        if any(r["count"][i]["input_sha256"] != r["timing"][i]["input_sha256"] for r in results):
            raise ValueError("rank count/timing TRAIN inputs differ")
    durations = [
        max(r["timing"][i]["end_to_end_seconds"] for r in results) for i in range(measured_updates)
    ]
    return dict(
        status="completed_partial_coverage",
        requested_measured_updates=measured_updates,
        actual_samples=measured_updates,
        scientific_complete=False,
        mfu=None,
        hfu=None,
        rank_results=results,
        slow_rank_step_seconds=durations,
        median_seconds=float(np.median(durations)),
        p95_seconds=float(np.quantile(durations, 0.95)),
        global_cells_per_second=sum(r["global_cells"] for r in results[0]["timing"])
        / sum(durations),
        covered_flops=sum(r["ledger"]["covered_executed_flops"] for r in results),
        covered_tflops_per_second=sum(r["ledger"]["covered_executed_flops"] for r in results)
        / sum(durations)
        / 1e12,
        timing_scope="short isolated engineering diagnostic with paired complete-update samples; "
        "synchronized slow-rank wall includes TRAIN preparation; "
        "resource admission/hash/copy/restore excluded; not sustained or continuous wall "
        "throughput; no profiler; partial arithmetic only; not full MFU/HFU",
    )


def validate_parent(root: Path, config: Any, checkpoint_sha256: str) -> tuple[dict, dict, Path]:
    from gradpert.execution.v2_training_stage import training_stage_evidence
    from gradpert.hashing import sha256_json

    manifest = json.loads((root / "run_manifest.json").read_text())
    journal = json.loads((root / "fit/epoch_state.json").read_text())
    if (root / "FAILURE.json").exists() or journal["identity"] != manifest:
        raise ValueError("failed parent or mismatched native journal identity")
    if manifest["resolved_config_sha256"] != sha256_json(config.model_dump(mode="json")):
        raise ValueError("parent resolved configuration differs")
    expected = training_stage_evidence(root, manifest, int(config.training.max_epochs.value))
    if json.loads((root / "TRAIN_COMPLETE.json").read_text()) != expected:
        raise ValueError("terminal native training receipt changed")
    selected = journal["last"]
    if selected["sha256"] != checkpoint_sha256:
        raise ValueError("requested checkpoint is not the terminal native last")
    checkpoint = (root / "fit" / selected["file"]).resolve(strict=True)
    if not checkpoint.is_relative_to((root / "fit").resolve(strict=True)):
        raise ValueError("parent checkpoint escaped its fit root")
    return manifest, journal, checkpoint


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "data-root",
        "output",
        "training-root",
        "native-source",
        "publication",
        "native-publication",
        "resource-binding",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "publication-sha256",
        "native-publication-sha256",
        "checkpoint-sha256",
        "gpu",
        "resource-binding-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resource-owner-pid", type=int, required=True)
    parser.add_argument(
        "--measured-updates",
        type=lambda value: measured_update_budget(int(value)),
        default=MEASURED_UPDATES,
        help="paired counted/timed updates, 1..10 (default: 10); short engineering diagnostic",
    )
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2 or args.gpu != "0,1" or rank not in (0, 1):
        parser.error("use torchrun two ranks on the approved physical GPUs 0,1")
    for name in ("data_root", "output", "training_root", "native_source", "resource_binding"):
        if not getattr(args, name).resolve().is_relative_to("/data/yilangliu"):
            parser.error(f"{name} must remain under /data/yilangliu")
    repository = Path(__file__).resolve().parents[2]
    protect_output(
        args.output, [args.data_root, args.training_root, args.native_source, repository]
    )
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("launch requires PYTORCH_ALLOC_CONF=expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import gradpert
    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import load_checkpoint
    from gradpert.training.v2.distributed import primary_call
    from gradpert.training.v2.runtime import prepare_runtime

    native_root = Path(inspect.getfile(gradpert)).resolve().parents[2]
    if native_root != args.native_source.resolve(strict=True):
        raise ValueError(
            "native import root differs; place original native src first in PYTHONPATH"
        )
    if not repository.is_relative_to("/data/yilangliu"):
        raise ValueError("instrumentation checkout must remain under /data/yilangliu")
    config = load_experiment_config(args.config)
    source = inspect_source_identity(
        repository,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    native = inspect_source_identity(
        native_root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.native_publication,
        expected_publication_receipt_sha256=args.native_publication_sha256,
    )
    manifest, journal, checkpoint = validate_parent(
        args.training_root.resolve(strict=True), config, args.checkpoint_sha256
    )
    if (
        manifest["source"]["commit"] != native.commit
        or manifest["source"]["dirty"]
        or manifest["source"]["tree_sha256"] != native.tree_sha256
    ):
        raise ValueError("original checkpoint training source differs from the native runtime")
    if manifest["config_sha256"] != sha256_file(args.config):
        raise ValueError("parent raw configuration hash differs")
    guard = ResourceGuard(
        args.resource_binding,
        args.resource_binding_sha256,
        args.resource_owner_pid,
        dict(
            output=str(args.output.resolve()),
            config_sha256=sha256_file(args.config),
            native_source_commit=native.commit,
            instrumentation_source_commit=source.commit,
            training_root=str(args.training_root.resolve()),
            checkpoint_sha256=args.checkpoint_sha256,
            training_manifest_sha256=sha256_file(args.training_root / "run_manifest.json"),
        ),
    )
    guard.check()
    bindings = input_bindings(
        [
            args.config,
            args.publication,
            args.native_publication,
            args.training_root / "run_manifest.json",
            args.training_root / "TRAIN_COMPLETE.json",
            args.training_root / "fit/epoch_state.json",
            args.training_root / "fit/history.json",
            checkpoint,
            args.resource_binding,
            guard.policy_path,
        ]
    )
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("MFU requires idle GPUs; existing work is preserved")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(guard.policy["max_own_memory_fraction"], 0)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=2))
    receipt: dict[str, Any] = dict(
        schema="gradpert-native-mfu-diagnostic-1",
        status="running",
        mfu=None,
        hfu=None,
        scientific_complete=False,
        instrumentation_source=source.payload(),
        native_source=native.payload(),
        training_manifest=manifest,
        checkpoint_sha256=args.checkpoint_sha256,
        config_sha256=sha256_file(args.config),
        world_size=world,
        requested_measured_updates=args.measured_updates,
        resource_binding_sha256=args.resource_binding_sha256,
        resource_owner=guard.spec["owner"],
        original_resource_policy=guard.policy,
        immutable_inputs=bindings,
    )
    try:
        primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
        receipt["environment"] = inspect_environment(native_root, device_name="cuda:0").payload()
        for field in ("lock_file_sha256", "torch_version", "cuda_runtime"):
            if receipt["environment"][field] != manifest["environment"][field]:
                raise ValueError(f"original native environment differs: {field}")
        hardware = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,clocks.sm,power.draw,power.limit",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if hardware.returncode:
            raise RuntimeError("native hardware readback failed")
        receipt["hardware_before"] = hardware.stdout
        primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=config.training.run_seeds[0],
            device=torch.device("cuda:0"),
        ) as runtime:
            if runtime.identity != manifest["data"]:
                raise ValueError("runtime data/split/expression-exclusion identity differs")
            if (
                journal["contract"]["bf16"] is not True
                or journal["contract"]["microbatch"] != runtime.options.microbatch
            ):
                raise ValueError("parent native BF16/microbatch execution contract differs")
            load_checkpoint(
                checkpoint,
                runtime.objective,
                runtime.optimizer,
                identity=manifest,
                generator=runtime.generator,
            )
            architecture = runtime.objective.student.options
            if (
                architecture.relay_kernel != "eager"
                or architecture.weighted_gram_backend != "eager"
                or architecture.short_graph_kernel
            ):
                raise ValueError("first partial MFU probe supports native eager KDA only")
            clock = frozen_clock(journal, runtime.optimizer.steps)

            def continue_together(allowed):
                value = torch.tensor(int(allowed), device=runtime.device)
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
                return bool(value.item())

            batches = TrainBatches(runtime)
            receipt["train_batch_manifest"] = batches.seal(args.measured_updates)
            receipt["train_batch_manifest_sha256"] = digest(receipt["train_batch_manifest"])
            receipt["startup_seconds"] = time.monotonic() - process_started
            primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
            local = run_probe(
                runtime,
                batches,
                clock,
                synchronize=torch.cuda.synchronize,
                continue_together=continue_together,
                resource_guard=guard.check,
                measured_updates=args.measured_updates,
            )
            local["rank"] = rank
            ledger_path = args.output / f"rank-{rank}-ledger.json"
            atomic_json(ledger_path, local["ledger"])
            local["ledger_artifact"] = dict(
                file=ledger_path.name,
                sha256=sha256_file(ledger_path),
                bytes=ledger_path.stat().st_size,
            )
            local["ledger"] = {
                key: value
                for key, value in local["ledger"].items()
                if key not in ("rows", "forward_calls")
            }
            atomic_json(args.output / f"rank-{rank}.json", local)
            gathered: list[Any] = [None] * world
            torch.distributed.all_gather_object(gathered, local)
            receipt.update(aggregate_ranks(gathered))
        guard.check()
        verify_inputs(bindings)
        closed_source = inspect_source_identity(
            repository,
            formal=True,
            expected_repository=config.source_code.repository,
            publication_receipt=args.publication,
            expected_publication_receipt_sha256=args.publication_sha256,
        )
        closed_native = inspect_source_identity(
            native_root,
            formal=True,
            expected_repository=config.source_code.repository,
            publication_receipt=args.native_publication,
            expected_publication_receipt_sha256=args.native_publication_sha256,
        )
        if (
            closed_source.payload() != source.payload()
            or closed_native.payload() != native.payload()
        ):
            raise ValueError(
                "native or instrumentation source changed after full-state restoration"
            )
        receipt["input_source_closure_verified"] = True
        receipt["zero_pkl"] = not any(args.output.rglob("*.pkl"))
        if not receipt["zero_pkl"]:
            raise RuntimeError("diagnostic run root violates zero-PKL postcondition")
        receipt["checkpoint_unchanged"] = sha256_file(checkpoint) == args.checkpoint_sha256
        if not receipt["checkpoint_unchanged"]:
            raise RuntimeError("original parent checkpoint changed")
        primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
        if receipt["status"] != "completed_partial_coverage":
            raise RuntimeError("diagnostic incomplete or failed; inspect retained receipts")
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        with suppress(Exception):
            atomic_json(args.output / f"rank-{rank}-failure.json", receipt)
        raise
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
