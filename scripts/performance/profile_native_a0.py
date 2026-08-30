#!/usr/bin/env python3
"""Bounded, training-only profiler for the exact successor A0 native path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradpert.execution.system_resources import (  # noqa: E402
    host_available_memory_bytes,
)

GIB = 1024**3
EXACT_A0_GRAPH_NODE_COUNT = 2809
EXACT_A0_LOCAL_NODE_BUDGET = 1404
EXACT_A0_PROTOCOL_ID = "within_cell_unseen_single"
EXACT_A0_RUNTIME_GRAPH_ROOT = "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"


class ProfileComplete(RuntimeError):
    """Internal stop raised immediately after the final bounded optimizer step."""


class ProfileGateError(RuntimeError):
    """A hash, resource, or lifecycle predicate rejected the profiling run."""


class EvaluationAccessError(RuntimeError):
    """The training-only profiler attempted to enter an evaluation surface."""


def _sha256_argument(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a 64-character lowercase SHA-256")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--development-commit", required=True)
    parser.add_argument("--phase", choices=("capacity", "profile", "timing"), required=True)
    parser.add_argument("--expected-config-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-canonical-data-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-split-content-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-source-h5ad-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-source-registry-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-graph-gene-order-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-topology-content-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--minimum-gpu-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-gpu-free-bytes", type=int, default=4 * GIB)
    parser.add_argument("--maximum-idle-gpu-utilization-percent", type=float, default=5.0)
    parser.add_argument("--maximum-idle-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--minimum-disk-free-bytes", type=int, default=20 * GIB)
    parser.add_argument("--minimum-host-available-bytes", type=int, default=16 * GIB)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _parameter(config: Any, name: str) -> object:
    try:
        return config.model.parameters[name].value
    except KeyError as error:
        raise ValueError(f"A0 config lacks required parameter: {name}") from error


def _require_reference_a0(
    args: argparse.Namespace,
    config_path: Path,
) -> tuple[Any, dict[str, object]]:
    from gradpert.config import NativeArchitectureOptions, load_experiment_config
    from gradpert.contracts import CanonicalDataManifest, SplitManifest
    from gradpert.data import DatasetLayout
    from gradpert.hashing import sha256_file
    from gradpert.pilots.vnext_graph_axis import load_vnext_graph_topology

    observed_config_sha = sha256_file(config_path)
    if observed_config_sha != args.expected_config_sha256:
        raise ProfileGateError("A0 config SHA-256 differs from the launch contract")
    config = load_experiment_config(config_path)
    if (
        config.model_id != "gradpert_b2"
        or config.dataset_id != "nadig_jurkat"
        or config.model.family != "native_learned"
        or config.data.protocol_id != EXACT_A0_PROTOCOL_ID
    ):
        raise ProfileGateError("performance profiling accepts only native Nadig Jurkat A0")
    architecture = NativeArchitectureOptions.from_parameters(config.model.parameters)
    expected_architecture = {
        "graph_axis_policy": "recomputed_hvg_union_candidate_targets",
        "graph_hvg_count": 512,
        "graph_sources": ("string", "go"),
        "graph_encoder_family": "multi_source_sparse_transformer",
        "string_weight_mode": "selection_only",
        "local_view_builder": "ring_induced",
        "local_view_count": 8,
        "local_view_node_budget_ratio_numerator": 1,
        "local_view_node_budget_ratio_denominator": 2,
        "local_anchor_mask_view_ratio_numerator": 0,
        "local_anchor_mask_view_ratio_denominator": 1,
        "gene_feature_mode": "learned_id",
        "decoder_mode": "additive",
    }
    observed_architecture = {name: getattr(architecture, name) for name in expected_architecture}
    if (
        observed_architecture != expected_architecture
        or architecture.legacy_local_view_node_budget is not None
    ):
        raise ProfileGateError("performance profiling architecture differs from exact A0")
    expected_parameters: dict[str, object] = {
        "runtime_graph_root": EXACT_A0_RUNTIME_GRAPH_ROOT,
        "performance_pilot_variant": "vnext_a0_ratio_ring_half",
        "prototype_count": 16384,
        "max_unique_conditions_per_batch": 8,
        "prediction_loss_weight": 1.0,
        "condition_consistency_loss_weight": 0.8,
        "masked_node_loss_weight": 0.4,
        "spread_loss_weight": 0.1,
        "systems_optimizations": "all_seven_semantics_preserving_v1",
        "systems_merged_hdf5_reads": True,
        "systems_control_expression_cache": True,
        "systems_background_prefetch": True,
        "systems_pin_memory": True,
        "systems_nonblocking_transfer": True,
        "systems_prefetch_depth": 2,
        "systems_resident_graph_tensors": True,
        "systems_validation_expression_cache": True,
        "systems_buffered_training_logs": True,
        "systems_log_buffer_steps": 64,
        "systems_single_checkpoint_serialization": True,
    }
    observed_parameters = {name: _parameter(config, name) for name in expected_parameters}
    if observed_parameters != expected_parameters:
        differing = sorted(
            name
            for name in expected_parameters
            if observed_parameters[name] != expected_parameters[name]
        )
        raise ProfileGateError("A0 semantic parameters differ: " + ", ".join(differing))
    training_identity = {
        "formal_run_policy": config.training.formal_run_policy,
        "max_epochs": config.training.max_epochs.value,
        "run_seeds": config.training.run_seeds,
        "early_stopping": config.training.early_stopping,
        "train_batch_size": config.training.train_batch_size.value,
        "eval_batch_size": config.training.eval_batch_size.value,
        "optimizer": config.training.optimizer.value,
        "learning_rate": config.training.learning_rate.value,
        "weight_decay": config.training.weight_decay.value,
        "scheduler": config.training.scheduler.value,
        "result_mode": config.artifacts.result_mode,
    }
    expected_training_identity = {
        "formal_run_policy": "fixed_epoch_pilot",
        "max_epochs": 10,
        "run_seeds": [1],
        "early_stopping": False,
        "train_batch_size": 256,
        "eval_batch_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "scheduler": "none",
        "result_mode": "metrics_only",
    }
    if training_identity != expected_training_identity or args.run_seed != 1:
        raise ProfileGateError("A0 training/artifact identity differs from the frozen coordinate")

    layout = DatasetLayout(Path(args.data_root), config.dataset_id, config.data.protocol_id)
    canonical = CanonicalDataManifest.model_validate_json(
        layout.canonical_manifest.read_text(encoding="utf-8")
    )
    split = SplitManifest.model_validate_json(
        (layout.manifests / "split.json").read_text(encoding="utf-8")
    )
    graph_root = Path(args.data_root) / EXACT_A0_RUNTIME_GRAPH_ROOT
    topology, graph_manifest = load_vnext_graph_topology(graph_root)
    expected_hashes = {
        "canonical_data_sha256": args.expected_canonical_data_sha256,
        "split_content_sha256": args.expected_split_content_sha256,
        "source_h5ad_sha256": args.expected_source_h5ad_sha256,
        "source_registry_sha256": args.expected_source_registry_sha256,
        "graph_gene_order_sha256": args.expected_graph_gene_order_sha256,
        "topology_content_sha256": args.expected_topology_content_sha256,
    }
    observed_hashes = {
        "canonical_data_sha256": canonical.canonical_adata_sha256,
        "split_content_sha256": split.split_content_sha256,
        "source_h5ad_sha256": graph_manifest.source_h5ad_sha256,
        "source_registry_sha256": graph_manifest.source_registry_sha256,
        "graph_gene_order_sha256": graph_manifest.graph_gene_order_sha256,
        "topology_content_sha256": graph_manifest.topology_content_sha256,
    }
    if observed_hashes != expected_hashes:
        differing = sorted(
            name for name in expected_hashes if observed_hashes[name] != expected_hashes[name]
        )
        raise ProfileGateError("A0 runtime lineage hashes differ: " + ", ".join(differing))
    if (
        graph_manifest.protocol_id != EXACT_A0_PROTOCOL_ID
        or graph_manifest.canonical_data_sha256 != canonical.canonical_adata_sha256
        or graph_manifest.split_content_sha256 != split.split_content_sha256
        or graph_manifest.requested_hvg_count != 512
        or graph_manifest.graph_gene_count != EXACT_A0_GRAPH_NODE_COUNT
        or len(topology.gene_ids) != EXACT_A0_GRAPH_NODE_COUNT
        or topology.active_sources != ("string", "go")
    ):
        raise ProfileGateError("A0 graph manifest/runtime topology identity differs")
    return architecture, {
        "config_sha256": observed_config_sha,
        "dataset_id": config.dataset_id,
        "protocol_id": config.data.protocol_id,
        "split_policy": config.data.split_policy,
        "semantic_parameters": expected_parameters,
        "training_identity": training_identity,
        "runtime_hashes": observed_hashes,
        "graph_node_count": len(topology.gene_ids),
        "graph_sources_runtime_order": list(topology.active_sources),
    }


def _run_nvidia_query(fields: tuple[str, ...], *, compute_apps: bool = False) -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "returncode": None, "rows": [], "stderr": "not found"}
    query = "--query-compute-apps=" if compute_apps else "--query-gpu="
    completed = subprocess.run(
        [executable, query + ",".join(fields), "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = [
        {field: value.strip() for field, value in zip(fields, values, strict=True)}
        for values in csv.reader(completed.stdout.splitlines())
        if len(values) == len(fields)
    ]
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "query_fields": list(fields),
        "rows": rows,
        "stderr": completed.stderr.strip(),
    }


def _nvidia_snapshot() -> dict[str, object]:
    gpu_fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
        "memory.used",
        "memory.free",
        "memory.total",
    )
    app_fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    return {
        "gpus": _run_nvidia_query(gpu_fields),
        "compute_apps": _run_nvidia_query(app_fields, compute_apps=True),
    }


def _host_available_bytes() -> int | None:
    return host_available_memory_bytes()


def _top_processes() -> dict[str, object]:
    completed = subprocess.run(
        ["ps", "-eo", "pid,comm,pcpu,pmem", "-r"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "rows": completed.stdout.splitlines()[:21],
        "stderr": completed.stderr.strip(),
    }


def _host_snapshot(path: Path) -> dict[str, object]:
    disk = shutil.disk_usage(path.parent)
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ALLOC_CONF",
        "CUBLAS_WORKSPACE_CONFIG",
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "host_available_bytes": _host_available_bytes(),
        "max_rss_raw": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "environment": {name: os.environ.get(name) for name in thread_names},
        "nvidia_smi": _nvidia_snapshot(),
        "top_processes": _top_processes(),
    }


def _safe_host_snapshot(path: Path) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    try:
        return _host_snapshot(path), None
    except BaseException as error:
        return None, _failure_payload(error)


def _selected_physical_gpu(
    snapshot: Mapping[str, object],
    *,
    device_name: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not device_name.startswith("cuda:"):
        raise ProfileGateError("A0 profiling requires an explicit cuda:N device")
    try:
        local_index = int(device_name.split(":", 1)[1])
    except ValueError as error:
        raise ProfileGateError("A0 profiling device must be cuda:N") from error
    nvidia = snapshot.get("nvidia_smi")
    if not isinstance(nvidia, dict):
        raise ProfileGateError("nvidia-smi snapshot is unavailable")
    gpu_query = nvidia.get("gpus")
    app_query = nvidia.get("compute_apps")
    if not isinstance(gpu_query, dict) or not gpu_query.get("available"):
        raise ProfileGateError("nvidia-smi GPU query failed")
    if not isinstance(app_query, dict) or not app_query.get("available"):
        raise ProfileGateError("nvidia-smi compute-process query failed")
    rows = gpu_query.get("rows")
    if not isinstance(rows, list):
        raise ProfileGateError("nvidia-smi GPU rows are malformed")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [value.strip() for value in visible.split(",")] if visible else []
    if selectors:
        if local_index >= len(selectors):
            raise ProfileGateError("CUDA device index is outside CUDA_VISIBLE_DEVICES")
        selector = selectors[local_index]
    else:
        selector = str(local_index)
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            row.get("index") == selector
            or str(row.get("uuid", "")).startswith(selector)
            or selector.startswith(str(row.get("uuid", "")))
        )
    ]
    if len(matches) != 1:
        raise ProfileGateError("logical CUDA device does not resolve to one physical GPU")
    selected = {str(key): str(value) for key, value in matches[0].items()}
    app_rows = app_query.get("rows", [])
    selected_apps = [
        {str(key): str(value) for key, value in row.items()}
        for row in app_rows
        if isinstance(row, dict)
        and row.get("gpu_uuid") == selected["uuid"]
        and row.get("pid") != str(os.getpid())
    ]
    return selected, selected_apps


def _preflight_predicates(
    args: argparse.Namespace,
    snapshot: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, object]]:
    selected, competing_apps = _selected_physical_gpu(snapshot, device_name=args.device)
    predicates = {
        "no_competing_compute_processes": not competing_apps,
        "gpu_utilization_at_most_limit": float(selected["utilization.gpu"])
        <= args.maximum_idle_gpu_utilization_percent,
        "gpu_memory_used_at_most_limit": int(selected["memory.used"])
        <= args.maximum_idle_gpu_memory_mib,
        "disk_free_at_least_limit": int(snapshot["disk_free_bytes"])
        >= args.minimum_disk_free_bytes,
        "host_available_at_least_limit": (
            snapshot.get("host_available_bytes") is not None
            and int(snapshot["host_available_bytes"]) >= args.minimum_host_available_bytes
        ),
    }
    return predicates, {
        "selected_physical_gpu": selected,
        "competing_compute_processes": competing_apps,
    }


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _runtime_capacity_predicates(
    observed: list[dict[str, object]],
    *,
    total_steps: int,
    minimum_gpu_headroom_fraction: float,
    minimum_gpu_free_bytes: int,
    evaluation_state: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, int | float]]:
    retry_or_oom_total = sum(
        int(value) for row in observed for value in row["cuda_memory_stats"].values()
    )
    minimum_observed_free_bytes = min(
        (int(row["gpu_free_bytes_after_step"]) for row in observed),
        default=0,
    )
    minimum_free_fraction = min(
        (int(row["gpu_free_bytes_after_step"]) / int(row["gpu_total_bytes"]) for row in observed),
        default=0.0,
    )
    predicates = {
        "exact_observed_step_count": len(observed) == total_steps,
        "zero_cuda_allocation_retries_or_ooms": retry_or_oom_total == 0,
        "gpu_headroom_fraction_at_least_limit": minimum_free_fraction
        >= minimum_gpu_headroom_fraction,
        "gpu_free_bytes_at_least_absolute_limit": minimum_observed_free_bytes
        >= minimum_gpu_free_bytes,
        "no_validation_callback": evaluation_state["validation_callback_count"] == 0,
        "no_evaluation_truth_access_attempt": not evaluation_state["truth_access_attempts"],
        "no_validation_cache_materialized": not evaluation_state["validation_cache_materialized"],
    }
    return predicates, {
        "cuda_retry_or_oom_counter_total": retry_or_oom_total,
        "minimum_gpu_free_bytes": minimum_observed_free_bytes,
        "minimum_gpu_free_fraction": minimum_free_fraction,
    }


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
        attempts = self._state.setdefault("truth_access_attempts", [])
        if isinstance(attempts, list):
            attempts.append(f"{self._split_name}.{name}")
        raise EvaluationAccessError(
            f"training-only profiler attempted evaluator access: {self._split_name}.{name}"
        )


def _training_only_evaluator_factory(
    state: dict[str, object],
) -> Callable[..., _TrainingOnlyEvaluationGuard]:
    def factory(*_: object, split_name: str, **__: object) -> _TrainingOnlyEvaluationGuard:
        bindings = state.setdefault("guard_bindings", [])
        if isinstance(bindings, list):
            bindings.append(split_name)
        return _TrainingOnlyEvaluationGuard(state, split_name=split_name)

    return factory


def _evaluation_access_summary(state: Mapping[str, object]) -> dict[str, bool]:
    attempts = [str(attempt) for attempt in state["truth_access_attempts"]]
    bindings = [str(binding) for binding in state["guard_bindings"]]
    return {
        "validation_guard_bound": "val" in bindings,
        "test_guard_bound": "test" in bindings,
        "validation_accessed": (
            bool(state["validation_cache_materialized"])
            or int(state["real_canonical_evaluation_constructor_count"]) > 0
            or any(attempt.startswith("val.") for attempt in attempts)
        ),
        "test_truth_accessed": any(attempt.startswith("test.") for attempt in attempts),
    }


def _make_bounded_train_step(
    original_train_step: Callable[..., Any],
    *,
    total_steps: int,
    before_step: Callable[[Any], None],
    after_step: Callable[[Any, Any, int], None],
) -> Callable[..., Any]:
    completed_steps = 0

    def bounded_train_step(engine: Any, batch: Any, *, global_step: int) -> Any:
        nonlocal completed_steps
        if completed_steps >= total_steps:
            raise AssertionError("bounded profiler requested an N+1 training step")
        before_step(engine)
        metrics = original_train_step(engine, batch, global_step=global_step)
        completed_steps += 1
        after_step(engine, metrics, global_step)
        if completed_steps == total_steps:
            raise ProfileComplete("bounded profile completed immediately after its final step")
        return metrics

    return bounded_train_step


def _profile_run(
    args: argparse.Namespace,
    *,
    config: Path,
    run_root: Path,
    teardown_failures: list[dict[str, str]],
) -> dict[str, object]:
    import torch

    import gradpert.execution.native as native_execution
    from gradpert.hashing import sha256_file
    from gradpert.training.step import GraDPertStepEngine

    warmup_steps, measured_steps = {
        "capacity": (3, 3),
        "profile": (2, 3),
        "timing": (2, 10),
    }[args.phase]
    total_steps = warmup_steps + measured_steps
    observed: list[dict[str, object]] = []
    profiler: Any | None = None
    profiler_stopped = False
    evaluation_state: dict[str, object] = {
        "real_canonical_evaluation_constructor_count": 0,
        "guard_bindings": [],
        "validation_cache_requested": False,
        "validation_cache_materialized": False,
        "validation_callback_count": 0,
        "truth_access_attempts": [],
    }
    original_train_step = GraDPertStepEngine.train_step
    original_evaluator = native_execution.CanonicalEvaluationData
    original_validation = native_execution.evaluate_validation_macro_delta

    def stop_profiler() -> None:
        nonlocal profiler_stopped
        if profiler is None or profiler_stopped:
            return
        try:
            profiler.stop()
        except BaseException as error:
            teardown_failures.append({"stage": "profiler.stop", **_failure_payload(error)})
            profiler_stopped = True
            return
        profiler_stopped = True
        evidence_root = run_root / "profile_evidence"
        try:
            evidence_root.mkdir(parents=True, exist_ok=True)
            trace = evidence_root / "torch-profiler-trace.json"
            profiler.export_chrome_trace(str(trace))
            (evidence_root / "torch-profiler-table.txt").write_text(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=200,
                ),
                encoding="utf-8",
            )
        except BaseException as error:
            teardown_failures.append({"stage": "profiler.export", **_failure_payload(error)})

    def before_step(engine: Any) -> None:
        nonlocal profiler
        if observed:
            return
        contract = engine.local_view_contract
        if (
            contract.graph_node_count != EXACT_A0_GRAPH_NODE_COUNT
            or contract.effective_node_budget != EXACT_A0_LOCAL_NODE_BUDGET
            or contract.node_budget_remainder != 1
            or contract.local_view_count != 8
            or contract.effective_mask_view_count != 0
            or contract.derivation_mode != "ratio"
        ):
            raise ProfileGateError("runtime A0 graph or resolved local-view contract differs")
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        if args.phase == "profile":
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            profiler.start()

    def after_step(engine: Any, metrics: Any, global_step: int) -> None:
        torch.cuda.synchronize(torch.device(args.device))
        if profiler is not None:
            profiler.step()
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(args.device))
        observed.append(
            {
                "global_step": global_step,
                "profile_phase": "warmup" if len(observed) < warmup_steps else "measured",
                "metrics": asdict(metrics),
                "view_stats": engine.last_view_stats,
                "peak_allocated_gpu_bytes": int(
                    torch.cuda.max_memory_allocated(torch.device(args.device))
                ),
                "peak_reserved_gpu_bytes": int(
                    torch.cuda.max_memory_reserved(torch.device(args.device))
                ),
                "gpu_free_bytes_after_step": int(free_bytes),
                "gpu_total_bytes": int(total_bytes),
                "cuda_memory_stats": {
                    key: int(value)
                    for key, value in torch.cuda.memory_stats(torch.device(args.device)).items()
                    if "retry" in key or "oom" in key
                },
            }
        )
        if len(observed) == total_steps:
            stop_profiler()

    bounded_train_step = _make_bounded_train_step(
        original_train_step,
        total_steps=total_steps,
        before_step=before_step,
        after_step=after_step,
    )

    def reject_validation(*_: object, **__: object) -> object:
        evaluation_state["validation_callback_count"] = (
            int(evaluation_state["validation_callback_count"]) + 1
        )
        raise EvaluationAccessError("training-only profiler reached validation callback")

    GraDPertStepEngine.train_step = bounded_train_step
    native_execution.CanonicalEvaluationData = _training_only_evaluator_factory(evaluation_state)
    native_execution.evaluate_validation_macro_delta = reject_validation
    primary_failure: BaseException | None = None
    try:
        native_execution.run_native_experiment(
            config_path=config,
            data_root=args.data_root,
            run_root=run_root,
            run_id=args.run_id,
            run_seed=args.run_seed,
            mode="pilot",
            device_name=args.device,
            repository_root=args.repository_root,
            formal=False,
            development_commit=args.development_commit,
        )
        primary_failure = RuntimeError("profile unexpectedly reached validation/test lifecycle")
    except ProfileComplete:
        pass
    except BaseException as error:
        primary_failure = error
    finally:
        stop_profiler()
        GraDPertStepEngine.train_step = original_train_step
        native_execution.CanonicalEvaluationData = original_evaluator
        native_execution.evaluate_validation_macro_delta = original_validation

    measured = observed[warmup_steps:]
    measured_wall = [float(row["metrics"]["step_wall_ms"]) for row in measured]
    runtime_predicates, capacity_summary = _runtime_capacity_predicates(
        observed,
        total_steps=total_steps,
        minimum_gpu_headroom_fraction=args.minimum_gpu_headroom_fraction,
        minimum_gpu_free_bytes=args.minimum_gpu_free_bytes,
        evaluation_state=evaluation_state,
    )
    evidence_root = run_root / "profile_evidence"
    trace = evidence_root / "torch-profiler-trace.json"
    table = evidence_root / "torch-profiler-table.txt"
    access_summary = _evaluation_access_summary(evaluation_state)
    return {
        "primary_failure": None if primary_failure is None else _failure_payload(primary_failure),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "observed_step_count": len(observed),
        "steps": observed,
        "measured_step_wall_ms": measured_wall,
        "measured_step_wall_ms_percentiles": _percentiles(measured_wall),
        "peak_allocated_gpu_bytes": max(
            (int(row["peak_allocated_gpu_bytes"]) for row in observed), default=0
        ),
        "peak_reserved_gpu_bytes": max(
            (int(row["peak_reserved_gpu_bytes"]) for row in observed), default=0
        ),
        **capacity_summary,
        "runtime_predicates": runtime_predicates,
        "evaluation_guard": evaluation_state,
        **access_summary,
        "instrumentation": {
            "phase": args.phase,
            "torch_profiler_enabled": args.phase == "profile",
            "torch_profiler_schedule": (
                {"wait": 1, "warmup": 1, "active": 3, "repeat": 1}
                if args.phase == "profile"
                else None
            ),
            "step_timer": "native_train_step_receipt_cuda_event_synchronized",
            "memory_stats_sampling": "after_each_bounded_optimizer_step",
            "timing_role": (
                "reference_single_arm_raw_timing_not_abba_acceptance"
                if args.phase == "timing"
                else "not_timing_acceptance"
            ),
        },
        "torch_profiler_trace_sha256": sha256_file(trace) if trace.is_file() else None,
        "torch_profiler_table_sha256": sha256_file(table) if table.is_file() else None,
    }


def _threshold_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "minimum_gpu_headroom_fraction": args.minimum_gpu_headroom_fraction,
        "minimum_gpu_free_bytes": args.minimum_gpu_free_bytes,
        "maximum_idle_gpu_utilization_percent": args.maximum_idle_gpu_utilization_percent,
        "maximum_idle_gpu_memory_mib": args.maximum_idle_gpu_memory_mib,
        "minimum_disk_free_bytes": args.minimum_disk_free_bytes,
        "minimum_host_available_bytes": args.minimum_host_available_bytes,
    }


def _validate_thresholds(args: argparse.Namespace) -> None:
    if not 0 < args.minimum_gpu_headroom_fraction < 1:
        raise ProfileGateError("minimum GPU headroom fraction must be in (0,1)")
    if args.maximum_idle_gpu_utilization_percent < 0:
        raise ProfileGateError("maximum idle GPU utilization must be nonnegative")
    if args.maximum_idle_gpu_memory_mib < 0:
        raise ProfileGateError("maximum idle GPU memory must be nonnegative")
    if (
        args.minimum_gpu_free_bytes < 0
        or args.minimum_disk_free_bytes < 0
        or args.minimum_host_available_bytes < 0
    ):
        raise ProfileGateError("GPU/disk/RAM byte thresholds must be nonnegative")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.resolve()
    accepted_root = False
    teardown_failures: list[dict[str, str]] = []
    receipt: dict[str, object] = {
        "schema_version": "native-a0-bounded-profile-v2",
        "phase": args.phase,
        "run_id": args.run_id,
        "run_seed": args.run_seed,
        "device": args.device,
        "source_commit": args.development_commit,
        "thresholds": _threshold_payload(args),
        "expected_hashes": {
            "config_sha256": args.expected_config_sha256,
            "canonical_data_sha256": args.expected_canonical_data_sha256,
            "split_content_sha256": args.expected_split_content_sha256,
            "source_h5ad_sha256": args.expected_source_h5ad_sha256,
            "source_registry_sha256": args.expected_source_registry_sha256,
            "graph_gene_order_sha256": args.expected_graph_gene_order_sha256,
            "topology_content_sha256": args.expected_topology_content_sha256,
        },
    }
    primary_failure: BaseException | None = None
    started_snapshot: dict[str, object] | None = None
    started_snapshot_failure: dict[str, str] | None = None
    try:
        if run_root.exists() and any(run_root.iterdir()):
            raise FileExistsError("profile run root must be new and empty")
        run_root.mkdir(parents=True, exist_ok=True)
        accepted_root = True
        _validate_thresholds(args)
        if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
            raise ProfileGateError("profile requires PYTORCH_ALLOC_CONF=expandable_segments:True")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ProfileGateError("profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        config = args.config.resolve(strict=True)
        args.repository_root = args.repository_root.resolve(strict=True)
        architecture, exact_identity = _require_reference_a0(args, config)
        receipt["config_path"] = str(config)
        receipt["native_architecture"] = architecture.payload()
        receipt["exact_a0_identity"] = exact_identity
        started_snapshot, started_snapshot_failure = _safe_host_snapshot(run_root)
        if started_snapshot is None:
            raise ProfileGateError("failed to capture preflight host/resource snapshot")
        preflight_predicates, gpu_identity = _preflight_predicates(args, started_snapshot)
        receipt["preflight_predicates"] = preflight_predicates
        receipt["physical_gpu_identity"] = gpu_identity
        if not all(preflight_predicates.values()):
            failed = sorted(name for name, passed in preflight_predicates.items() if not passed)
            raise ProfileGateError("preflight resource predicates failed: " + ", ".join(failed))
        receipt.update(
            _profile_run(
                args,
                config=config,
                run_root=run_root,
                teardown_failures=teardown_failures,
            )
        )
    except BaseException as error:
        primary_failure = error
    completed_snapshot, completed_snapshot_failure = _safe_host_snapshot(run_root)
    if receipt.get("primary_failure") is None and primary_failure is not None:
        receipt["primary_failure"] = _failure_payload(primary_failure)
    receipt.setdefault("primary_failure", None)
    receipt["teardown_failures"] = teardown_failures
    receipt["started_host_snapshot"] = started_snapshot
    receipt["started_host_snapshot_failure"] = started_snapshot_failure
    receipt["completed_host_snapshot"] = completed_snapshot
    receipt["completed_host_snapshot_failure"] = completed_snapshot_failure
    completion_predicates = {
        "disk_free_at_least_limit": (
            completed_snapshot is not None
            and int(completed_snapshot["disk_free_bytes"]) >= args.minimum_disk_free_bytes
        ),
        "host_available_at_least_limit": (
            completed_snapshot is not None
            and completed_snapshot.get("host_available_bytes") is not None
            and int(completed_snapshot["host_available_bytes"]) >= args.minimum_host_available_bytes
        ),
    }
    receipt["completion_resource_predicates"] = completion_predicates
    runtime_predicates = receipt.get("runtime_predicates")
    all_runtime = isinstance(runtime_predicates, dict) and all(runtime_predicates.values())
    status_complete = (
        receipt["primary_failure"] is None
        and not teardown_failures
        and all_runtime
        and all(completion_predicates.values())
    )
    receipt["status"] = "complete" if status_complete else "failed"
    if receipt["primary_failure"] is None and not all_runtime:
        receipt["primary_failure"] = {
            "type": "ProfileGateError",
            "message": "one or more runtime predicates failed",
        }
    if receipt["primary_failure"] is None and not all(completion_predicates.values()):
        receipt["primary_failure"] = {
            "type": "ProfileGateError",
            "message": "one or more completion resource predicates failed",
        }

    receipt_path = (
        run_root / "profile_evidence" / "profile-receipt.json"
        if accepted_root
        else run_root.parent / f".{run_root.name}.profile-failure.json"
    )
    write_failure: BaseException | None = None
    reported_receipt_path = receipt_path
    try:
        _atomic_json(receipt_path, receipt)
    except BaseException as error:
        write_failure = error
        fallback_path = receipt_path.with_name("profile-receipt.minimal-failure.json")
        minimal_receipt = {
            "schema_version": "native-a0-bounded-profile-v2-minimal-failure",
            "status": "failed",
            "phase": args.phase,
            "run_id": args.run_id,
            "run_root": str(run_root),
            "primary_failure": receipt.get("primary_failure"),
            "teardown_failures": teardown_failures,
            "full_receipt_write_failure": _failure_payload(error),
        }
        try:
            _atomic_json(fallback_path, minimal_receipt)
        except BaseException:
            pass
        else:
            reported_receipt_path = fallback_path
    if args.as_json:
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    else:
        print(reported_receipt_path)
    if write_failure is not None:
        print(
            f"failed to persist profiler receipt: {type(write_failure).__name__}: {write_failure}",
            file=sys.stderr,
        )
        return 3
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
