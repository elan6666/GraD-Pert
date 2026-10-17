"""Exclusive training-cost attribution from a single-rank Kineto trace.

Forward/autograd linkage uses (forward thread, sequence), CUDA linkage uses
launch correlation. Unlinked work stays unknown. Raw traces remain on server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from trace_costs import events


class Regions:
    """Resolve nested CPU scopes to disjoint half-open segments per thread."""

    def __init__(self, rows):
        points = defaultdict(list)
        for ident, row in enumerate(rows):
            start, end, name, tid = row
            if end <= start:
                continue
            points[tid].append((start, 1, ident, end - start, name))
            points[tid].append((end, 0, ident, end - start, name))
        self.segments = {}
        for tid, boundaries in points.items():
            active, segments = {}, []
            grouped = defaultdict(list)
            for point in boundaries:
                grouped[point[0]].append(point)
            times = sorted(grouped)
            for i, start in enumerate(times[:-1]):
                for _, enter, ident, duration, name in grouped[start]:
                    if enter:
                        active[ident] = (duration, name)
                    else:
                        active.pop(ident, None)
                if active:
                    names = list(active.values())
                    # Recompute is a context, not a replacement for component.
                    phase = (
                        "recompute"
                        if any(n.endswith("checkpoint/recompute") for _, n in names)
                        else (
                            "backward"
                            if any(
                                n.startswith("backward:") or n == "gradpert/backward"
                                for _, n in names
                            )
                            else "forward"
                        )
                    )
                    components = [
                        (d, n) for d, n in names if not n.startswith("gradpert/checkpoint/")
                    ]
                    name = min(components, default=(0, "unknown"))[1]
                    segments.append((start, times[i + 1], phase, name))
            self.segments[tid] = ([s[0] for s in segments], segments)

    def at(self, tid, ts):
        starts, segments = self.segments.get(tid, ([], []))
        i = bisect_right(starts, ts) - 1
        return segments[i][2:] if i >= 0 and ts < segments[i][1] else ("unknown", "unknown")


def category(name):
    name = name.removeprefix("backward:")
    if "/kda_" in name:
        return name.rsplit("/", 1)[1]
    if name.endswith("/prototypes"):
        return "prototypes"
    if name.endswith("/projector_features"):
        return "projector_features"
    if name.endswith("/prototype_ce"):
        return "prototype_ce"
    if name.endswith("/ffn"):
        return "ffn"
    if name.endswith(("/mla", "/graph_mla", "/genept")):
        return name.rsplit("/", 1)[1]
    if "/communication/" in name:
        return "communication"
    if name in ("gradpert/optimizer", "gradpert/ema_centers"):
        return name.removeprefix("gradpert/")
    if name in ("unknown", "gradpert/capture_window", "gradpert/backward"):
        return "unknown"
    return name.removeprefix("gradpert/")


def union_us(ranges):
    end, total = float("-inf"), 0.0
    for left, right in sorted(ranges):
        if right > max(left, end):
            total += right - max(left, end)
        end = max(end, right)
    return total


def attribute_update(factory, *, group_by="component"):
    if group_by not in ("component", "module"):
        raise ValueError("group_by must be component or module")

    def cost_owner(name):
        if group_by == "component" or category(name) == "unknown":
            return category(name)
        return name.removeprefix("backward:").removeprefix("gradpert/")

    scopes, pids, windows = [], set(), []
    flow_starts, flow_ends = {}, {}
    for e in factory():
        if e.get("cat") == "fwdbwd" and e.get("ph") in ("s", "f"):
            target = flow_starts if e["ph"] == "s" else flow_ends
            target[e["id"]] = (e["tid"], e["ts"])
        if e.get("cat") not in ("cpu_op", "user_annotation"):
            continue
        pids.add(e["pid"])
        name = e.get("name", "")
        if e.get("ph") == "X" and name.startswith("gradpert/"):
            scopes.append((e["ts"], e["ts"] + e["dur"], name, e["tid"]))
            if name == "gradpert/capture_window":
                windows.append((e["ts"], e["ts"] + e["dur"]))
    if len(pids) != 1 or len(windows) != 1:
        raise ValueError("requires one CPU process and exactly one capture window")
    start, stop = windows[0]
    regions = Regions(scopes)
    # Kineto tid is an OS thread ID on some platforms, whereas Fwd thread id
    # is autograd's logical ID. Official fwdbwd flows bridge the two identities.
    links = {end: flow_starts[ident] for ident, end in flow_ends.items() if ident in flow_starts}
    thread_ids = defaultdict(set)
    # Retain only differentiable forward operation identities, not every event.
    forward = {}
    for e in factory():
        args = e.get("args", {})
        seq = args.get("Sequence number", -1)
        if e.get("cat") != "cpu_op" or seq < 0:
            continue
        logical = args.get("Fwd thread id", 0)
        linked = links.get((e["tid"], e["ts"]))
        if logical:
            if linked:
                thread_ids[logical].add(linked[0])
            continue
        phase, name = regions.at(e["tid"], e["ts"])
        if category(name) != "unknown":
            key = e["tid"], seq
            # Smallest CPU op identifies the most specific forward owner.
            if key not in forward or e.get("dur", 0) < forward[key][0]:
                forward[key] = (e.get("dur", 0), phase, name)
    backward_rows = []
    for e in factory():
        if e.get("cat") != "cpu_op" or e.get("ph") != "X":
            continue
        args = e.get("args", {})
        tid = args.get("Fwd thread id", 0)
        seq = args.get("Sequence number", -1)
        if tid and seq >= 0:
            linked = links.get((e["tid"], e["ts"]))
            if linked:
                phase, name = regions.at(*linked)
                matched = (0, phase, name) if category(name) != "unknown" else None
            else:
                mapped = thread_ids.get(tid, set())
                # No guessing when a logical thread has ambiguous/no OS mapping.
                matched = forward.get((next(iter(mapped)), seq)) if len(mapped) == 1 else None
            if matched:
                backward_rows.append((e["ts"], e["ts"] + e["dur"], matched[2], e["tid"]))
    backwards = Regions(backward_rows)
    cpu_regions = Regions(scopes + [(a, b, "backward:" + n, t) for a, b, n, t in backward_rows])
    cpu_costs = defaultdict(float)
    for tid, (_, segments) in cpu_regions.segments.items():
        for left, right, phase, owner in segments:
            duration = min(stop, right) - max(start, left)
            if duration > 0:
                cpu_costs[tid, phase, cost_owner(owner)] += duration
    correlations, launch = {}, defaultdict(lambda: [0, 0.0])
    launch_ends = {}
    synchronization = defaultdict(lambda: [0, 0.0])
    for e in factory():
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("ph") != "X":
            continue
        if not start <= e["ts"] < stop:
            continue
        name = e.get("name", "")
        if "Synchronize" in name:
            synchronization[name][0] += 1
            synchronization[name][1] += e["dur"]
        corr = e.get("args", {}).get("correlation")
        if corr is None:
            continue
        phase, owner = regions.at(e["tid"], e["ts"])
        if phase != "recompute":
            _, parent = backwards.at(e["tid"], e["ts"])
            if parent != "unknown":
                phase, owner = "backward", parent
            elif owner == "gradpert/backward":
                phase = "backward"
        key = phase, cost_owner(owner)
        if "LaunchKernel" in name:
            launch[key][0] += 1
            launch[key][1] += e["dur"]
        if corr in correlations and correlations[corr] != key:
            raise ValueError("ambiguous CUDA correlation ownership")
        correlations[corr] = key
        launch_ends[corr] = e["ts"] + e["dur"]
    costs = defaultdict(lambda: {"count": 0, "duration_sum_us": 0.0, "ranges": []})
    all_ranges = []
    memory = defaultdict(lambda: dict(allocated_bytes=0, freed_bytes=0, events=0))
    memory_peak = None
    queue_delay_count, queue_delay_sum, queue_delay_max = 0, 0.0, 0.0
    for e in factory():
        if (
            e.get("name") == "[memory]"
            and e.get("args", {}).get("Device Type") == 1
            and start <= e.get("ts", -1) < stop
        ):
            # Allocation events have no CUDA correlation. Attribute the host
            # allocator call; frees can release an allocation from an older scope.
            phase, owner = regions.at(e["tid"], e["ts"])
            if phase != "recompute":
                _, parent = backwards.at(e["tid"], e["ts"])
                if parent != "unknown":
                    phase, owner = "backward", parent
                elif owner == "gradpert/backward":
                    phase = "backward"
            row = memory[phase, cost_owner(owner)]
            size = e["args"].get("Bytes", 0)
            row["allocated_bytes"] += max(size, 0)
            row["freed_bytes"] += max(-size, 0)
            row["events"] += 1
            allocated = e["args"].get("Total Allocated")
            if allocated is not None and (memory_peak is None or allocated > memory_peak):
                memory_peak = allocated
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or e.get("ph") != "X":
            continue
        left, right = max(start, e["ts"]), min(stop, e["ts"] + e["dur"])
        if right <= left:
            continue
        phase, owner = correlations.get(
            e.get("args", {}).get("correlation"), ("unknown", "unknown")
        )
        launch_end = launch_ends.get(e.get("args", {}).get("correlation"))
        if launch_end is not None and e["cat"] == "kernel":
            delay = max(0, e["ts"] - launch_end)
            queue_delay_count += 1
            queue_delay_sum += delay
            queue_delay_max = max(queue_delay_max, delay)
        key = phase, owner, e["cat"]
        costs[key]["count"] += 1
        costs[key]["duration_sum_us"] += right - left
        costs[key]["ranges"].append((left, right))
        all_ranges.append((left, right))
    rows = []
    for (phase, component, activity), values in sorted(costs.items()):
        count, api_us = launch[phase, component] if activity == "kernel" else (0, 0.0)
        rows.append(
            dict(
                phase=phase,
                component=component,
                activity=activity,
                count=values["count"],
                gpu_duration_sum_us=values["duration_sum_us"],
                gpu_interval_union_us=union_us(values["ranges"]),
                cpu_kernel_launch_count=count,
                cpu_kernel_launch_api_us=api_us,
            )
        )
    busy = union_us(all_ranges)
    return {
        "schema": "gradpert-update-costs-1",
        "group_by": group_by,
        "window_us": stop - start,
        "cuda_activity_present": bool(all_ranges),
        "gpu_activity_union_us": busy if all_ranges else None,
        "gpu_no_activity_us": stop - start - busy if all_ranges else None,
        "forward_sequence_identities": len(forward),
        "linked_backward_cpu_scopes": len(backward_rows),
        "forward_backward_flow_count": len(links),
        "autograd_to_os_thread_ids": {str(k): sorted(v) for k, v in thread_ids.items()},
        "costs": rows,
        "cpu_region_exclusive_us_by_thread": [
            dict(thread_id=tid, phase=phase, component=component, duration_us=duration)
            for (tid, phase, component), duration in sorted(cpu_costs.items())
        ],
        "kernel_start_after_launch_end": dict(
            linked_kernels=queue_delay_count,
            mean_us=queue_delay_sum / queue_delay_count if queue_delay_count else None,
            max_us=queue_delay_max if queue_delay_count else None,
            scope="queued device/stream work and profiler overhead; not pure launch overhead",
        ),
        "synchronization": {
            n: dict(calls=v[0], cpu_api_duration_sum_us=v[1])
            for n, v in sorted(synchronization.items())
        },
        "cuda_allocator_events": {
            "observed_peak_allocated_bytes": memory_peak,
            "by_host_scope": [
                dict(phase=phase, component=component, **values)
                for (phase, component), values in sorted(memory.items())
            ],
            "limits": "Allocation/free byte traffic, not live saved-tensor size. "
            "A free is attributed to its release scope, not its allocation owner. "
            "Window can start/end with live allocations; per-scope net is not a memory budget. "
            "No allocation events means peak is unknown, not zero.",
        },
        "limits": "Each GPU event has one owner; "
        "per-category GPU unions can overlap across streams. "
        "CPU launch and synchronization times are CPU occupancy, not additive GPU costs. "
        "No-GPU-activity intervals are not automatically CPU starvation. Unknown stays unknown. "
        "Profiler overhead included; use independent unprofiled A/B for throughput.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", choices=("component", "module"), default="component")
    args = parser.parse_args()
    before = args.trace.stat()

    def factory():
        with args.trace.open() as handle:
            yield from events(handle)

    result = attribute_update(factory, group_by=args.group_by)
    digest = hashlib.sha256()
    with args.trace.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    after = args.trace.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("trace changed during attribution")
    result.update(
        trace=str(args.trace),
        trace_sha256=digest.hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
