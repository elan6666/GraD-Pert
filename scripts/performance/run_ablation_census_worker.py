#!/usr/bin/env python3
"""Run one bounded, training-only stage for one frozen ablation-matrix row.

The worker deliberately reuses ``run_native_experiment`` and stops from the
native ``GraDPertStepEngine.train_step`` immediately after the frozen step
budget.  It cannot run P3, validation, test evaluation, or a scientific epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import ablation_performance_census as census  # noqa: E402

GIB = 1024**3
SUPPORTED_STAGES = ("p1_capacity", "p2_timing", "diagnostic_profile")


class BoundedCensusComplete(RuntimeError):
    """Signal a successful stop immediately after the final bounded step."""


class TrainingOnlyAccessError(RuntimeError):
    """Raised if the bounded worker reaches validation or test truth."""


class WorkerGateError(RuntimeError):
    """Raised when a frozen worker safety or capacity gate fails."""


@dataclass(frozen=True)
class RuntimeModules:
    torch: Any
    native_execution: Any
    engine_class: type[Any]


@dataclass
class WorkerState:
    batches: list[Any] = field(default_factory=list)
    steps: list[dict[str, object]] = field(default_factory=list)
    timing_samples_ms: list[float] = field(default_factory=list)
    evaluation: dict[str, object] = field(
        default_factory=lambda: {
            "real_canonical_evaluation_constructor_count": 0,
            "guard_bindings": [],
            "validation_cache_requested": False,
            "validation_cache_materialized": False,
            "validation_callback_count": 0,
            "truth_access_attempts": [],
        }
    )
    stage_observer_failures: list[dict[str, object]] = field(default_factory=list)
    primary_failure: BaseException | None = None
    teardown_failures: list[dict[str, str]] = field(default_factory=list)
    profiler_trace: Path | None = None
    profiler_table: Path | None = None


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--stage-id", choices=SUPPORTED_STAGES, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--development-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--genept-preflight-receipt", type=Path)
    parser.add_argument(
        "--genept-preflight-receipt-sha256",
        type=_sha256_argument,
    )
    parser.add_argument("--minimum-gpu-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-gpu-free-bytes", type=int, default=4 * GIB)
    parser.add_argument("--maximum-idle-gpu-utilization-percent", type=float, default=5.0)
    parser.add_argument("--maximum-idle-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--minimum-disk-free-bytes", type=int, default=20 * GIB)
    parser.add_argument("--minimum-host-available-bytes", type=int, default=16 * GIB)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _failure_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_genept_preflight(
    args: argparse.Namespace,
    *,
    binding: Any,
) -> tuple[Path | None, str | None]:
    receipt = args.genept_preflight_receipt
    expected_sha256 = args.genept_preflight_receipt_sha256
    if binding.genept_preflight_required:
        if receipt is None or expected_sha256 is None:
            raise WorkerGateError("GenePT matrix row requires a hash-pinned preflight receipt")
        resolved = Path(receipt).resolve(strict=True)
        if _sha256_file(resolved) != expected_sha256:
            raise WorkerGateError("GenePT preflight receipt SHA-256 differs")
        return resolved, expected_sha256
    if receipt is not None or expected_sha256 is not None:
        raise WorkerGateError("non-GenePT matrix row cannot accept a GenePT preflight receipt")
    return None, None


def _host_available_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None


def _nvidia_rows(fields: Sequence[str], *, compute_apps: bool = False) -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise WorkerGateError("nvidia-smi is required for physical-GPU preflight")
    flag = "--query-compute-apps" if compute_apps else "--query-gpu"
    completed = subprocess.run(
        [executable, f"{flag}={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WorkerGateError(f"nvidia-smi preflight failed: {completed.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise WorkerGateError("nvidia-smi returned a malformed row")
        rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _physical_gpu_preflight(args: argparse.Namespace, attempt_root: Path) -> dict[str, object]:
    if not args.device.startswith("cuda:"):
        raise WorkerGateError("performance census requires an explicit cuda:N device")
    try:
        logical_index = int(args.device.split(":", 1)[1])
    except ValueError as error:
        raise WorkerGateError("performance census device must be cuda:N") from error
    gpu_fields = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total")
    app_fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    gpu_rows = _nvidia_rows(gpu_fields)
    app_rows = _nvidia_rows(app_fields, compute_apps=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [value.strip() for value in visible.split(",")] if visible else []
    if selectors and logical_index >= len(selectors):
        raise WorkerGateError("logical device is outside CUDA_VISIBLE_DEVICES")
    selector = selectors[logical_index] if selectors else str(logical_index)
    matches = [
        row
        for row in gpu_rows
        if row["index"] == selector
        or row["uuid"].startswith(selector)
        or selector.startswith(row["uuid"])
    ]
    if len(matches) != 1:
        raise WorkerGateError("logical device does not resolve to one physical GPU")
    selected = matches[0]
    competing = [
        row
        for row in app_rows
        if row["gpu_uuid"] == selected["uuid"] and row["pid"] != str(os.getpid())
    ]
    disk = shutil.disk_usage(attempt_root)
    host_available = _host_available_bytes()
    predicates = {
        "no_competing_compute_processes": not competing,
        "gpu_utilization_at_most_limit": float(selected["utilization.gpu"])
        <= args.maximum_idle_gpu_utilization_percent,
        "gpu_memory_used_at_most_limit": int(selected["memory.used"])
        <= args.maximum_idle_gpu_memory_mib,
        "disk_free_at_least_limit": disk.free >= args.minimum_disk_free_bytes,
        "host_available_at_least_limit": host_available is not None
        and host_available >= args.minimum_host_available_bytes,
    }
    return {
        "schema_version": "nadig-vnext-performance-resource-preflight-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "selected_physical_gpu": selected,
        "competing_compute_processes": competing,
        "disk_free_bytes": disk.free,
        "host_available_bytes": host_available,
        "predicates": predicates,
    }


class _TrainingOnlyEvaluationGuard:
    def __init__(self, state: dict[str, object], *, split_name: str) -> None:
        self._state = state
        self._split_name = split_name

    def __enter__(self) -> _TrainingOnlyEvaluationGuard:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def configure_expression_cache(self, *, enabled: bool) -> float:
        self._state["validation_cache_requested"] = bool(enabled)
        self._state["validation_cache_materialized"] = False
        return 0.0

    def __getattr__(self, name: str) -> object:
        attempts = self._state["truth_access_attempts"]
        assert isinstance(attempts, list)
        attempts.append(f"{self._split_name}.{name}")
        raise TrainingOnlyAccessError(
            f"training-only census attempted evaluator access: {self._split_name}.{name}"
        )


def _evaluation_guard_factory(
    state: dict[str, object],
) -> Callable[..., _TrainingOnlyEvaluationGuard]:
    def factory(*_: object, split_name: str, **__: object) -> _TrainingOnlyEvaluationGuard:
        bindings = state["guard_bindings"]
        assert isinstance(bindings, list)
        bindings.append(split_name)
        return _TrainingOnlyEvaluationGuard(state, split_name=split_name)

    return factory


def _training_only_evidence(state: Mapping[str, object]) -> dict[str, object]:
    attempts = [str(value) for value in state["truth_access_attempts"]]
    return {
        "scope": "performance_training_only",
        "real_canonical_evaluation_constructor_count": int(
            state["real_canonical_evaluation_constructor_count"]
        ),
        "validation_cache_materialized": bool(state["validation_cache_materialized"]),
        "validation_callback_count": int(state["validation_callback_count"]),
        "validation_accessed": bool(state["validation_cache_materialized"])
        or int(state["real_canonical_evaluation_constructor_count"]) > 0
        or any(value.startswith("val.") for value in attempts),
        "test_truth_accessed": any(value.startswith("test.") for value in attempts),
        "truth_access_attempts": attempts,
        "guard_bindings": [str(value) for value in state["guard_bindings"]],
        "validation_cache_requested": bool(state["validation_cache_requested"]),
    }


def ordered_batch_identity(batch: Any, *, global_step: int) -> Any:
    condition_ids = tuple(str(value) for value in batch.condition_ids)
    anchors = batch.anchors_by_condition
    active_anchor_ids = [
        [str(anchor) for anchor in anchors[condition_id]] for condition_id in condition_ids
    ]
    identity = census.OrderedBatchIdentity.create(
        global_step=global_step,
        row_ids=[str(value) for value in batch.perturbed_row_ids],
        condition_ids=condition_ids,
        control_row_ids=[str(value) for value in batch.control_row_ids],
        active_anchor_ids=active_anchor_ids,
        actual_batch_size=len(condition_ids),
        unique_condition_count=len(set(condition_ids)),
    )
    if identity.actual_batch_size != census.EXACT_TRAIN_BATCH_SIZE:
        raise WorkerGateError("performance census observed a non-256 training batch")
    return identity


def _metrics_payload(metrics: Any) -> dict[str, object]:
    if is_dataclass(metrics):
        return dict(asdict(metrics))
    if isinstance(metrics, Mapping):
        return {str(key): value for key, value in metrics.items()}
    return dict(vars(metrics))


def _cuda_telemetry(torch: Any, device: Any) -> dict[str, object]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    memory_stats = torch.cuda.memory_stats(device)
    return {
        "allocated_gpu_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_gpu_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu_free_bytes": int(free_bytes),
        "gpu_total_bytes": int(total_bytes),
        "cuda_retry_oom_counters": {
            str(key): int(value)
            for key, value in memory_stats.items()
            if "retry" in str(key).lower() or "oom" in str(key).lower()
        },
    }


def _stop_profiler(
    profiler: Any | None,
    *,
    evidence_root: Path,
    state: WorkerState,
) -> None:
    if profiler is None:
        return
    try:
        profiler.stop()
        evidence_root.mkdir(parents=True, exist_ok=True)
        trace = evidence_root / "torch-profiler-trace.json"
        table = evidence_root / "torch-profiler-table.txt"
        profiler.export_chrome_trace(str(trace))
        table.write_text(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=200),
            encoding="utf-8",
        )
        state.profiler_trace = trace
        state.profiler_table = table
    except BaseException as error:
        state.teardown_failures.append({"stage": "profiler", **_failure_payload(error)})


def _execute_bounded_native(
    args: argparse.Namespace,
    *,
    binding: Any,
    attempt_root: Path,
    runtime: RuntimeModules,
    resource_preflight: Mapping[str, object],
    genept_preflight: tuple[Path | None, str | None],
) -> WorkerState:
    protocol = census.STAGE_PROTOCOLS[args.stage_id]
    state = WorkerState()
    native_run_root = attempt_root / "native-run"
    device = runtime.torch.device(args.device)
    original_train_step = runtime.engine_class.train_step
    original_evaluator = runtime.native_execution.CanonicalEvaluationData
    original_validation = runtime.native_execution.evaluate_validation_macro_delta
    atomic_observer: Any | None = None
    if protocol.heavy_capacity_instrumentation:
        atomic_observer = census.AtomicStageObserver(
            attempt_root / "stage-progress.json",
            {
                "variant_id": binding.variant_id,
                "stage_id": args.stage_id,
                "matrix_sha256": binding.matrix_sha256,
                "config_sha256": binding.config_sha256,
            },
        )
    profiler: Any | None = None
    profiler_stopped = False

    def stage_callback(event: Any, engine: Any) -> None:
        if atomic_observer is None or event.status == "failure":
            return
        telemetry = {"stage_event": event.payload(), **_cuda_telemetry(runtime.torch, device)}
        if event.status == "entered":
            atomic_observer.entered(event.phase_id, telemetry)
        else:
            atomic_observer.completed(event.phase_id, telemetry)

    def bounded_train_step(engine: Any, batch: Any, *, global_step: int) -> Any:
        nonlocal profiler
        if len(state.steps) >= protocol.total_steps:
            raise AssertionError("bounded census requested an N+1 training step")
        identity = ordered_batch_identity(batch, global_step=global_step)
        state.batches.append(identity)
        if not state.steps:
            runtime.torch.cuda.reset_peak_memory_stats(device)
            if protocol.torch_profiler_enabled:
                schedule = protocol.profiler_schedule
                assert schedule is not None
                profiler = runtime.torch.profiler.profile(
                    activities=[
                        runtime.torch.profiler.ProfilerActivity.CPU,
                        runtime.torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=runtime.torch.profiler.schedule(**schedule, repeat=1),
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.start()
        engine.stage_observer = stage_callback if atomic_observer is not None else None
        try:
            metrics = original_train_step(engine, batch, global_step=global_step)
        except BaseException:
            failures = getattr(engine, "stage_observer_failures", [])
            if failures:
                state.stage_observer_failures.extend(dict(value) for value in failures)
            raise
        runtime.torch.cuda.synchronize(device)
        if profiler is not None:
            profiler.step()
        metrics_payload = _metrics_payload(metrics)
        step_payload = {
            "global_step": global_step,
            "phase": ("warmup" if len(state.steps) < protocol.warmup_steps else "measured"),
            "batch_identity_sha256": identity.sha256,
            "metrics": metrics_payload,
            "resource": _cuda_telemetry(runtime.torch, device),
            "view_stats": getattr(engine, "last_view_stats", None),
        }
        state.steps.append(step_payload)
        if len(state.steps) > protocol.warmup_steps and protocol.timing_acceptance:
            value = float(metrics_payload["step_wall_ms"])
            if not math.isfinite(value) or value <= 0:
                raise WorkerGateError("native step timing is not finite and positive")
            state.timing_samples_ms.append(value)
        failures = getattr(engine, "stage_observer_failures", [])
        if failures:
            state.stage_observer_failures.extend(dict(value) for value in failures)
            raise WorkerGateError("native engine stage observer failed")
        if len(state.steps) == protocol.total_steps:
            raise BoundedCensusComplete(
                "bounded census completed immediately after its final training step"
            )
        return metrics

    def reject_validation(*_: object, **__: object) -> object:
        state.evaluation["validation_callback_count"] = (
            int(state.evaluation["validation_callback_count"]) + 1
        )
        raise TrainingOnlyAccessError("bounded census reached validation callback")

    runtime.engine_class.train_step = bounded_train_step
    runtime.native_execution.CanonicalEvaluationData = _evaluation_guard_factory(state.evaluation)
    runtime.native_execution.evaluate_validation_macro_delta = reject_validation
    try:
        runtime.native_execution.run_native_experiment(
            config_path=Path(binding.config_path),
            data_root=args.data_root,
            run_root=native_run_root,
            run_id=f"{binding.variant_id}-{args.stage_id}-{attempt_root.name}",
            run_seed=binding.run_seed,
            mode="pilot",
            device_name=args.device,
            repository_root=args.repository_root,
            formal=False,
            development_commit=args.development_commit,
            genept_preflight_receipt=genept_preflight[0],
            genept_preflight_receipt_sha256=genept_preflight[1],
        )
        state.primary_failure = RuntimeError(
            "bounded census unexpectedly reached validation/test lifecycle"
        )
    except BoundedCensusComplete:
        pass
    except BaseException as error:
        state.primary_failure = error
    finally:
        try:
            if profiler is not None and not profiler_stopped:
                _stop_profiler(
                    profiler,
                    evidence_root=attempt_root / "profile-evidence",
                    state=state,
                )
                profiler_stopped = True
        finally:
            runtime.engine_class.train_step = original_train_step
            runtime.native_execution.CanonicalEvaluationData = original_evaluator
            runtime.native_execution.evaluate_validation_macro_delta = original_validation

    training_only = _training_only_evidence(state.evaluation)
    try:
        census.require_training_only_evidence(training_only)
    except BaseException as error:
        if state.primary_failure is None:
            state.primary_failure = error
    capacity = _capacity_evidence(args, protocol=protocol, state=state)
    capacity_predicates = capacity["predicates"]
    assert isinstance(capacity_predicates, dict)
    if state.primary_failure is None and not all(
        bool(value) for value in capacity_predicates.values()
    ):
        state.primary_failure = WorkerGateError("runtime capacity predicate failed")
    if atomic_observer is not None:
        atomic_observer.finalize(
            result={
                "observed_step_count": len(state.steps),
                "resource_preflight": dict(resource_preflight),
            },
            primary_failure=state.primary_failure,
            teardown_failures=state.teardown_failures,
        )
    return state


def _capacity_evidence(
    args: argparse.Namespace,
    *,
    protocol: Any,
    state: WorkerState,
) -> dict[str, object]:
    resources = [step["resource"] for step in state.steps]
    typed = [value for value in resources if isinstance(value, dict)]
    total_bytes = max((int(value["gpu_total_bytes"]) for value in typed), default=0)
    minimum_free = min((int(value["gpu_free_bytes"]) for value in typed), default=0)
    required_free = max(
        args.minimum_gpu_free_bytes,
        math.ceil(total_bytes * args.minimum_gpu_headroom_fraction),
    )
    retry_oom = max(
        (
            sum(int(counter) for counter in value["cuda_retry_oom_counters"].values())
            for value in typed
        ),
        default=0,
    )
    predicates = {
        "exact_observed_step_count": len(state.steps) == protocol.total_steps,
        "zero_cuda_allocation_retries_or_ooms": retry_oom == 0,
        "gpu_free_bytes_at_least_required_headroom": minimum_free >= required_free,
    }
    return {
        "minimum_gpu_free_bytes": minimum_free,
        "gpu_total_bytes": total_bytes,
        "required_gpu_free_bytes": required_free,
        "cuda_retry_or_oom_counter_max": retry_oom,
        "predicates": predicates,
    }


def _build_stage_receipt(
    args: argparse.Namespace,
    *,
    binding: Any,
    attempt_root: Path,
    resource_preflight: Mapping[str, object],
    state: WorkerState,
) -> dict[str, object]:
    protocol = census.STAGE_PROTOCOLS[args.stage_id]
    capacity = _capacity_evidence(args, protocol=protocol, state=state)
    preflight_predicates = resource_preflight.get("predicates")
    if not isinstance(preflight_predicates, dict):
        raise WorkerGateError("resource preflight lacks predicates")
    capacity_predicates = capacity["predicates"]
    assert isinstance(capacity_predicates, dict)
    if state.primary_failure is None and (
        not all(bool(value) for value in preflight_predicates.values())
        or not all(bool(value) for value in capacity_predicates.values())
    ):
        state.primary_failure = WorkerGateError("resource/capacity predicate failed")
    if state.teardown_failures and state.primary_failure is None:
        state.primary_failure = WorkerGateError("worker teardown failed")
    training_only = _training_only_evidence(state.evaluation)
    batches = [batch.payload() for batch in state.batches]
    trace_sha = _sha256_file(state.profiler_trace) if state.profiler_trace else None
    table_sha = _sha256_file(state.profiler_table) if state.profiler_table else None
    return {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "binding": binding.payload(),
        "stage_id": args.stage_id,
        "protocol": protocol.payload(),
        "attempt_root": str(attempt_root),
        "native_run_root": str(attempt_root / "native-run"),
        "development_commit": args.development_commit,
        "repository_root": str(args.repository_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "device": args.device,
        "genept_preflight": {
            "required": binding.genept_preflight_required,
            "receipt_path": (
                str(args.genept_preflight_receipt.resolve())
                if args.genept_preflight_receipt is not None
                else None
            ),
            "receipt_sha256": args.genept_preflight_receipt_sha256,
        },
        "status": "complete" if state.primary_failure is None else "failed",
        "training_only_evidence": training_only,
        "instrumentation": {
            "timing_acceptance": protocol.timing_acceptance,
            "heavy_capacity_instrumentation": protocol.heavy_capacity_instrumentation,
            "torch_profiler_enabled": protocol.torch_profiler_enabled,
            "step_timer": "native_train_step_cuda_synchronized_step_wall_ms",
            "stage_observer": (
                "atomic_per_native_phase" if protocol.heavy_capacity_instrumentation else "off"
            ),
        },
        "attempted_batch_count": len(state.batches),
        "completed_step_count": len(state.steps),
        "observed_step_count": len(state.steps),
        "batches": batches,
        "batch_sequence_sha256": census.batch_sequence_sha256(state.batches),
        "timing_samples_ms": state.timing_samples_ms,
        "timing_summary_ms": (
            census.timing_summary(state.timing_samples_ms) if state.timing_samples_ms else None
        ),
        "steps": state.steps,
        "resource_preflight": dict(resource_preflight),
        "capacity_evidence": capacity,
        "stage_evidence": {
            "atomic_progress_receipt": (
                str(attempt_root / "stage-progress.json")
                if protocol.heavy_capacity_instrumentation
                else None
            ),
            "stage_observer_failures": state.stage_observer_failures,
        },
        "torch_profiler_trace_sha256": trace_sha,
        "torch_profiler_table_sha256": table_sha,
        "primary_failure": (
            None if state.primary_failure is None else _failure_payload(state.primary_failure)
        ),
        "teardown_failures": state.teardown_failures,
    }


def _load_runtime() -> RuntimeModules:
    import torch

    import gradpert.execution.native as native_execution
    from gradpert.training.step import GraDPertStepEngine

    return RuntimeModules(
        torch=torch,
        native_execution=native_execution,
        engine_class=GraDPertStepEngine,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.minimum_gpu_headroom_fraction < 1:
        raise WorkerGateError("minimum GPU headroom fraction must be in (0, 1)")
    numeric = (
        args.minimum_gpu_free_bytes,
        args.maximum_idle_gpu_utilization_percent,
        args.maximum_idle_gpu_memory_mib,
        args.minimum_disk_free_bytes,
        args.minimum_host_available_bytes,
    )
    if any(value < 0 for value in numeric):
        raise WorkerGateError("worker resource thresholds must be nonnegative")
    if args.stage_id not in SUPPORTED_STAGES:
        raise WorkerGateError("worker cannot execute P3 or an unknown stage")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    binding = census.bind_matrix_variant(
        args.matrix,
        repository_root=args.repository_root,
        expected_matrix_sha256=args.expected_matrix_sha256,
        variant_id=args.variant_id,
    )
    genept_preflight = _resolve_genept_preflight(args, binding=binding)
    attempt_root = census.claim_fresh_attempt_root(
        args.census_root,
        variant_id=binding.variant_id,
        stage_id=args.stage_id,
    )
    receipt_path = attempt_root / "stage-receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": "nadig-vnext-performance-stage-v1",
            "variant_id": binding.variant_id,
            "config_sha256": binding.config_sha256,
            "matrix_sha256": binding.matrix_sha256,
            "stage_id": args.stage_id,
            "protocol": census.STAGE_PROTOCOLS[args.stage_id].payload(),
            "status": "running",
        },
    )
    state = WorkerState()
    try:
        resource_preflight = _physical_gpu_preflight(args, attempt_root)
        predicates = resource_preflight["predicates"]
        assert isinstance(predicates, dict)
        if not all(bool(value) for value in predicates.values()):
            raise WorkerGateError("physical GPU/host/disk preflight failed")
        state = _execute_bounded_native(
            args,
            binding=binding,
            attempt_root=attempt_root,
            runtime=_load_runtime(),
            resource_preflight=resource_preflight,
            genept_preflight=genept_preflight,
        )
    except BaseException as error:
        resource_preflight = locals().get(
            "resource_preflight",
            {
                "schema_version": "nadig-vnext-performance-resource-preflight-v1",
                "predicates": {},
                "failure": _failure_payload(error),
            },
        )
        state.primary_failure = error
    receipt = _build_stage_receipt(
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=resource_preflight,
        state=state,
    )
    _atomic_json(receipt_path, receipt)
    output = {
        "attempt_root": str(attempt_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "status": receipt["status"],
    }
    if args.as_json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(f"{output['status']}: {receipt_path}")
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
