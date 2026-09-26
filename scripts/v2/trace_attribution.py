"""Attribute GPU kernels to selected CPU scopes without loading the whole trace.

Three passes make no assumption about Chrome event ordering. Scopes are inclusive
and can overlap; their totals must not be added or treated as wall-clock savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from trace_costs import events

TARGETS = {
    "aten::logsumexp",
    "aten::linalg_solve_triangular",
    "NativeDropoutBackward0",
    "gradpert/student/graph",
    "gradpert/teacher/graph",
    "gradpert/backward",
}


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def attribute(factory):
    intervals = defaultdict(list)
    cpu_pids = set()
    for e in factory():
        if e.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        cpu_pids.add(e["pid"])
        if e.get("ph") == "X" and e.get("name") in TARGETS:
            key = (e["pid"], e["tid"], e["name"])
            intervals[key].append((e["ts"], e["ts"] + e["dur"]))
    if len(cpu_pids) != 1:
        raise ValueError("attribution requires exactly one CPU process")
    by_thread = defaultdict(list)
    scope_us = defaultdict(float)
    for (pid, tid, name), ranges in intervals.items():
        merged = merge_intervals(ranges)
        by_thread[pid, tid].append((name, [x[0] for x in merged], merged))
        scope_us[name] += sum(b - a for a, b in merged)
    # Only correlations launched inside selected scopes are retained.
    correlations = {}
    runtime_us = defaultdict(float)
    for e in factory():
        if e.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        if "LaunchKernel" not in e.get("name", ""):
            continue
        correlation = e.get("args", {}).get("correlation")
        if correlation is None:
            continue
        names = []
        for name, starts, ranges in by_thread.get((e["pid"], e["tid"]), []):
            pos = bisect_right(starts, e["ts"]) - 1
            if pos >= 0 and e["ts"] < ranges[pos][1]:
                names.append(name)
                runtime_us[name] += e.get("dur", 0)
        if names:
            correlations.setdefault(correlation, set()).update(names)
    totals = defaultdict(lambda: [0, 0.0])
    kernels = 0
    for e in factory():
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        kernels += 1
        for name in correlations.get(e.get("args", {}).get("correlation"), ()):
            totals[name][0] += 1
            totals[name][1] += e["dur"]
    return {
        "total_kernel_count": kernels,
        "selected_launch_correlations": len(correlations),
        "scopes": {
            name: {
                "cpu_scope_union_us_summed_across_threads": scope_us[name],
                "launch_api_duration_sum_us": runtime_us[name],
                "gpu_kernel_count": totals[name][0],
                "gpu_kernel_duration_sum_us": totals[name][1],
            }
            for name in sorted(scope_us)
        },
        "limits": "Inclusive overlapping scopes, launch-time attribution via correlation IDs; "
        "not disjoint critical path, unprofiled timing, or a predicted speedup. "
        "Single-process trace required (correlation IDs are process-local).",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = args.trace.stat()

    def factory():
        with args.trace.open() as handle:
            yield from events(handle)

    result = attribute(factory)
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
