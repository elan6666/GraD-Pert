"""Local synthetic CPU parser benchmark; no training or real trace access."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import runpy
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="02cb70ca05e0b8285a94cae2ef307f36f5261942")
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:scripts/v2/trace_costs.py"], cwd=root, text=True
    )
    baseline = {"__name__": "baseline"}
    exec(compile(baseline_source, "baseline_trace_costs.py", "exec"), baseline)
    current_path = root / "scripts/v2/trace_costs.py"
    current = runpy.run_path(str(current_path))
    source = json.dumps(
        {
            "traceEvents": [
                {"ph": "X", "cat": f"category{i % 4}", "name": f"op{i % 23}", "dur": i % 101}
                for i in range(args.events)
            ]
        },
        separators=(",", ":"),
    )
    timings = {"baseline": [], "current": []}
    reference = baseline["aggregate"](baseline["events"](io.StringIO(source)))
    for repetition in range(args.repeats):
        order = ("baseline", "current") if repetition % 2 == 0 else ("current", "baseline")
        for name in order:
            module = baseline if name == "baseline" else current
            start = time.perf_counter()
            rows = list(module["events"](io.StringIO(source)))
            timings[name].append(time.perf_counter() - start)
            assert module["aggregate"](iter(rows)) == reference
    result = {
        "scope": (
            "Synthetic CPU JSON parser only; list materialization included, "
            "aggregation excluded. No training speed claim."
        ),
        "baseline_commit": args.baseline,
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "event_count": args.events,
        "source_chars": len(source),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "chunk_size": 65536,
        "repeats": args.repeats,
        "alternating_order": True,
        "exact_aggregate_equality": True,
        "seconds": timings,
        "median_seconds": {name: statistics.median(values) for name, values in timings.items()},
        "median_speedup": (
            statistics.median(timings["baseline"]) / statistics.median(timings["current"])
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
