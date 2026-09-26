"""Bounded-memory operator aggregation of an existing Chrome trace (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


def events(handle: TextIO, chunk_size: int = 65536) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    while True:
        match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
        if match:
            buffer = buffer[match.end() :]
            break
        piece = handle.read(chunk_size)
        if not piece or len(buffer) > 1048576:
            raise ValueError("missing bounded traceEvents header")
        buffer += piece
    expect_value = True
    while True:
        buffer = buffer.lstrip()
        if not buffer:
            buffer = handle.read(chunk_size)
            if not buffer:
                raise ValueError("truncated trace array")
            continue
        if buffer.startswith("]"):
            return
        if not expect_value:
            if not buffer.startswith(","):
                raise ValueError("missing event separator")
            buffer = buffer[1:]
            expect_value = True
            continue
        try:
            event, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError as error:
            piece = handle.read(chunk_size)
            if not piece or len(buffer) > 64 * 1024**2:
                raise ValueError("truncated or oversized event") from error
            buffer += piece
            continue
        if not isinstance(event, dict):
            raise ValueError("trace event must be an object")
        yield event
        buffer = buffer[end:]
        expect_value = False


def aggregate(rows: Iterator[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    count = 0
    for event in rows:
        count += 1
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category, name = str(event.get("cat", "")), str(event.get("name", ""))
        duration = float(event["dur"])
        row = totals[category, name]
        row[0] += 1
        row[1] += duration
    categories = sorted({cat for cat, _ in totals})
    return {
        "events": count,
        "categories": {
            cat: {
                "calls": int(sum(row[0] for (c, _), row in totals.items() if c == cat)),
                "duration_sum_us": sum(row[1] for (c, _), row in totals.items() if c == cat),
                "top_duration": sorted(
                    [
                        {"name": name, "calls": int(row[0]), "duration_sum_us": row[1]}
                        for (c, name), row in totals.items()
                        if c == cat
                    ],
                    key=lambda row: -row["duration_sum_us"],
                )[:40],
            }
            for cat in categories
        },
        "scope": "Whole existing trace, summed event durations; CPU events may nest and GPU "
        "streams may overlap. Not disjoint wall time, critical path, or throughput.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = args.trace.stat()
    with args.trace.open() as handle:
        result = aggregate(events(handle))
    digest = hashlib.sha256()
    with args.trace.open("rb") as handle:
        for piece in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(piece)
    after = args.trace.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("trace changed during analysis")
    result.update(
        trace=str(args.trace.resolve()),
        trace_bytes=after.st_size,
        trace_sha256=digest.hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
