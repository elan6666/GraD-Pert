"""Summarize native per-cell scalar diagnostics without pooling covariances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

KEYS = ("feature", "stage", "stream", "condition_id", "context_id", "cell_type", "batch")
MEASURES = (
    "rows",
    "width",
    "rank_ceiling",
    "effective_rank",
    "relative_variance",
    "variance",
    "mean_norm",
    "mean_energy",
    "total_energy",
)
MISSING = (None, "", "NA", "nan", "None")


def aggregate(path, scope):
    """Keep the historical R12 condition/context/feature estimator unchanged."""
    path = Path(path)
    if scope not in ("medium", "panel"):
        raise ValueError("unknown diagnostic scope")
    seal = dict(
        scope=scope,
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
    )
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if scope == "panel" and row["statistical_axis"] != "genes_within_cell":
                continue
            key = tuple(row.get(k, "") for k in KEYS)
            if not row.get("condition_id") or not row.get("context_id"):
                raise ValueError("condition/context identity is required")
            groups[key].append(row)
    rows = []
    for key, records in sorted(groups.items()):
        result = dict(zip(KEYS, key, strict=True))
        result.update(
            scope=scope,
            axis="within_cell_gene_tokens_condition_mean",
            cells=len(records),
            source_sha256=seal["sha256"],
        )
        for name in MEASURES:
            values = [float(r[name]) for r in records if r.get(name) not in MISSING]
            result[name + "_mean"] = statistics.mean(values) if values else None
        values = [
            float(r["effective_rank"]) / float(r["rank_ceiling"])
            for r in records
            if r.get("effective_rank") not in MISSING and float(r["rank_ceiling"]) > 0
        ]
        result["normalized_rank_mean"] = statistics.mean(values) if values else None
        rows.append(result)
    if not rows:
        raise ValueError("no within-cell scalar rows")
    return seal, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("medium", "panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    paths = [args.medium.resolve(), args.panel.resolve(), args.output.resolve()]
    server = Path("/data/yilangliu")
    if any(not p.is_relative_to(server) or p == server for p in paths) or args.output.exists():
        raise ValueError("server-only inputs and fresh output are required")
    source = Path(__file__).resolve().parents[2]

    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command], text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("aggregate producer must be clean")
    parts = [aggregate(args.medium, "medium"), aggregate(args.panel, "panel")]
    result = {
        args.arm: dict(
            sources=[p[0] for p in parts],
            condition_stats=[r for p in parts for r in p[1]],
            aggregation=(
                "Mean of native per-cell scalar statistics within condition/context/"
                "feature/stage/stream. Not recomputed covariance; do not mix medium/panel."
            ),
            producer_commit=git("rev-parse", "HEAD"),
            producer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
