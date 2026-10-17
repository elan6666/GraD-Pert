"""Opt-in diagnostic regions and bounded traces; never enabled by formal training."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import resource
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def environment_snapshot(torch: Any) -> dict[str, Any]:
    """Read-only, bounded hardware metadata; no workload or network mutation."""
    result: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
    }
    for name, command in (
        ("cpu", ["lscpu", "-J"]),
        ("memory", ["free", "-b"]),
        ("gpu_topology", ["nvidia-smi", "topo", "-m"]),
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv",
            ],
        ),
    ):
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            result[name] = {"exit_code": run.returncode, "text": (run.stdout + run.stderr)[:20000]}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"unavailable": type(error).__name__}
    return result


def host_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_unix": time.time(),
        "process_user_seconds": usage.ru_utime,
        "process_system_seconds": usage.ru_stime,
        # Linux ru_maxrss uses KiB. GPU probes are Linux-only; tests do not interpret RSS.
        "process_peak_rss_kib": usage.ru_maxrss,
    }


@contextlib.contextmanager
def diagnostic_regions(runtime: Any) -> Iterator[None]:
    """Wrap only during a probe window, restoring exact attributes even on failure.

    No tensor copies, RNG calls, stream changes, or device synchronizations.
    Checkpoint recomputation naturally re-enters the same labeled functions.
    Labels are nested; their times must not be summed as disjoint phases.
    """
    import torch
    import torch.utils.checkpoint as checkpoint_module

    import gradpert.modeling.v2.graph_mhc as graph_mhc
    import gradpert.modeling.v2.model as model_module
    import gradpert.modeling.v2.operators as operators
    import gradpert.training.v2.distributed as distributed
    import gradpert.training.v2.engine as engine
    import gradpert.training.v2.runtime as runtime_module
    from gradpert.modeling.v2.restored_ffn import MatchedGELUFeedForward

    changes: list[tuple[Any, str, Any, bool]] = []

    def wrap(owner: Any, attr: str, label: str) -> None:
        original = getattr(owner, attr)
        own_attribute = attr in vars(owner)

        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function("gradpert/" + label):
                return original(*args, **kwargs)

        changes.append((owner, attr, original, own_attribute))
        setattr(owner, attr, measured)

    try:
        # Diagnostic-only composition, including dynamically imported calls.
        # Do not change RNG preservation, checkpoint policy or saved tensors.
        original_checkpoint = checkpoint_module.checkpoint

        @functools.wraps(original_checkpoint)
        def measured_checkpoint(function: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("use_reentrant") is not False:
                raise ValueError("diagnostic checkpoint labels require non-reentrant checkpoint")
            context_fn = kwargs.get("context_fn", checkpoint_module.noop_context_fn)

            def contexts() -> tuple[Any, Any]:
                forward, recompute = context_fn()

                @contextlib.contextmanager
                def marked(original: Any, phase: str) -> Iterator[None]:
                    with original, torch.profiler.record_function("gradpert/checkpoint/" + phase):
                        yield

                return marked(forward, "forward"), marked(recompute, "recompute")

            return original_checkpoint(function, *args, **dict(kwargs, context_fn=contexts))

        for owner in (checkpoint_module, model_module, operators, graph_mhc):
            changes.append((owner, "checkpoint", owner.checkpoint, True))
            owner.checkpoint = measured_checkpoint
        for name in ("_materialize_cpu_batch", "_to_training_batch"):
            if hasattr(runtime.data, name):
                wrap(runtime.data, name, "data/" + name.removeprefix("_"))
        wrap(runtime_module, "assemble_batch", "data/assemble_views")
        wrap(engine, "gather_rows", "communication/gather_rows")
        wrap(distributed, "average_gradients", "communication/average_gradients")
        wrap(torch.autograd, "backward", "backward")
        wrap(runtime.optimizer, "step", "optimizer")
        wrap(runtime.objective, "commit_statistics", "ema_centers")
        wrap(runtime.objective, "graph_loss", "loss/ssl1")
        wrap(runtime.objective, "cell_loss", "loss/ssl2")
        wrap(runtime.objective, "_cross_entropy_terms", "loss/prototype_ce")
        for role in ("student", "teacher"):
            model = getattr(runtime.objective, role)
            for name in (
                "graph",
                "cell",
                "response",
                "ssl1_cls",
                "ssl1_node",
                "ssl2_cls",
                "ssl2_node",
                "expression",
                "prediction",
            ):
                wrap(getattr(model, name), "forward", f"{role}/{name}")
            for name, module in model.named_modules():
                if (
                    name.startswith(("response.self_layers.", "response.cross_layers."))
                    and name.count(".") == 2
                ):
                    wrap(module, "forward", f"{role}/{name}")
                if isinstance(module, operators.RelayDeltaAttention):
                    # Graph reads use .graph, not .forward; these three regions
                    # cover both neighborhood and sequence/cross execution.
                    for method, part in (
                        ("_project_writes", "project"),
                        ("_write_state", "scan"),
                        ("_read", "read"),
                    ):
                        wrap(module, method, f"component/{role}/{name}/kda_{part}")
                elif isinstance(module, model_module.Projector):
                    wrap(module.features, "forward", f"component/{role}/{name}/projector_features")
                    wrap(module.prototypes, "forward", f"component/{role}/{name}/prototypes")
                elif isinstance(
                    module, (operators.GatedFeedForward, MatchedGELUFeedForward)
                ) or name.endswith(".ffn"):
                    wrap(module, "forward", f"component/{role}/{name}/ffn")
                elif isinstance(
                    module, (operators.LatentAttention, operators.LatentCrossAttention)
                ):
                    wrap(module, "forward", f"component/{role}/{name}/mla")
                elif isinstance(module, model_module.SparseRead):
                    wrap(module, "attention", f"component/{role}/{name}/graph_mla")
                elif name == "graph.adapter":
                    wrap(module, "forward", f"component/{role}/{name}/genept")
        yield
    finally:
        for owner, attr, original, own_attribute in reversed(changes):
            if own_attribute:
                setattr(owner, attr, original)
            else:
                delattr(owner, attr)


def interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Union overlapping GPU intervals, keeping launch count and gap evidence."""
    return summarize_events(lambda: iter(trace.get("traceEvents", [])))


def summarize_events(factory: Any) -> dict[str, Any]:
    """Two streaming passes; retain intervals and bounded memory samples only."""
    windows = [
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in factory()
        if e.get("ph") == "X"
        and e.get("name") == "gradpert/capture_window"
        and e.get("cat") != "gpu_user_annotation"
    ]
    if len(windows) != 1:
        raise ValueError("trace needs exactly one capture window")
    start, end = windows[0]
    device, kernels, copies = [], 0, 0
    memory = []
    memory_count, memory_peak = 0, None
    labels: dict[str, dict[str, float]] = {}
    for event in factory():
        args = event.get("args", {})
        if (
            event.get("name") == "[memory]"
            and args.get("Device Type") == 1
            and start <= float(event.get("ts", -1)) <= end
        ):
            memory_count += 1
            allocated = args.get("Total Allocated")
            if allocated is not None:
                memory_peak = allocated if memory_peak is None else max(memory_peak, allocated)
            memory.append(
                {
                    "time_us": float(event["ts"]) - start,
                    "allocated_bytes": args.get("Total Allocated"),
                    "reserved_bytes": args.get("Total Reserved"),
                }
            )
            # Keep recent samples plus a progressively thinned history, never all
            # allocation events in million-event CUDA traces.
            if len(memory) > 512:
                memory = memory[::2]
        if event.get("ph") != "X" or "dur" not in event:
            continue
        left = max(start, float(event["ts"]))
        right = min(end, float(event["ts"]) + float(event["dur"]))
        if right <= left:
            continue
        category, name = event.get("cat", ""), event.get("name", "")
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            device.append((left, right))
            kernels += category == "kernel"
            copies += category == "gpu_memcpy"
        if name.startswith("gradpert/") and category != "gpu_user_annotation":
            row = labels.setdefault(name, {"calls": 0, "cpu_inclusive_us": 0})
            row["calls"] += 1
            row["cpu_inclusive_us"] += right - left
    merged = interval_union(device)
    busy = sum(right - left for left, right in merged)
    gaps, cursor = [], start
    for left, right in merged:
        if left > cursor:
            gaps.append((cursor - start, left - cursor))
        cursor = right
    if end > cursor:
        gaps.append((cursor - start, end - cursor))
    return {
        "window_us": end - start,
        "cuda_activity_present": bool(device),
        "gpu_busy_union_us": busy if device else None,
        "gpu_busy_fraction": busy / (end - start) if device and end > start else None,
        "kernel_count": kernels,
        "memcpy_count": copies,
        "largest_gpu_gaps_start_and_duration_us": sorted(gaps, key=lambda pair: -pair[1])[:30]
        if device
        else [],
        "regions": labels,
        "cuda_allocation_event_count": memory_count,
        "cuda_allocation_peak_bytes": memory_peak,
        "cuda_memory_timeline_sample": memory[:: max(1, (len(memory) + 255) // 256)],
        "cuda_memory_timeline_sampling": (
            "bounded thinned history; exact count/peak, not a full timeline"
        ),
        "scope": (
            "one rank, nested CPU-inclusive regions; GPU interval union includes "
            "kernel/memcpy/memset; profiler overhead included; "
            "not utilization or throughput evidence"
        ),
    }


class ProfileCapture:
    def __init__(self, runtime: Any, output: Path, rank: int) -> None:
        self.runtime, self.output, self.rank = runtime, output, rank
        self.stack: contextlib.ExitStack | None = None
        self.profiler = None

    def start(self, *, cuda: bool = True, memory: bool = False) -> None:
        import torch

        if self.stack is not None:
            raise RuntimeError("capture already started")
        self.stack = contextlib.ExitStack()
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = self.stack.enter_context(
                torch.profiler.profile(activities=activities, profile_memory=memory)
            )
            self.stack.enter_context(diagnostic_regions(self.runtime))
            self.stack.enter_context(torch.profiler.record_function("gradpert/capture_window"))
        except BaseException:
            self.close(export=False)
            raise

    def close(self, *, export: bool = True, operator_table: bool = False) -> dict[str, Any] | None:
        if self.stack is None:
            return None
        stack, self.stack = self.stack, None
        stack.close()
        if not export or self.profiler is None:
            return None
        path = self.output / f"rank-{self.rank}-trace.json"
        self.profiler.export_chrome_trace(str(path))
        from trace_costs import events

        def factory() -> Iterator[dict[str, Any]]:
            with path.open() as handle:
                yield from events(handle)

        summary = summarize_events(factory)
        (self.output / f"rank-{self.rank}-trace-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        # Kineto key_averages lazily materializes millions of events and can
        # dominate host memory/time. Raw traces preserve the evidence for later
        # analysis; normal diagnostics need only the bounded summary above.
        if operator_table:
            (self.output / f"rank-{self.rank}-operators.txt").write_text(
                self.profiler.key_averages().table(sort_by="self_device_time_total", row_limit=100)
            )
        self.profiler = None
        return {
            "trace": str(path),
            "trace_bytes": path.stat().st_size,
            "summary": summary,
            "operator_table_exported": operator_table,
        }
