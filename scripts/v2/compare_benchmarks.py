"""Audit one-factor A1/B1/B2/A2 receipts and report descriptive speed ratios."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(receipts) == 4, "requires A1 B1 B2 A2")
    original = receipts[0]
    rows = []
    for role, record in zip(("A1", "B1", "B2", "A2"), receipts, strict=True):
        require(record["status"] == "passed", "incomplete benchmark")
        require(record["kind"] == "benchmark_only", "not an unprofiled benchmark")
        require(record["steps_completed"] == record["steps_requested"], "partial benchmark")
        for flag in ("profile_last_update", "profile_memory", "sync_phase_timing"):
            require(record[flag] is False, "timing altered by profiler or diagnostic sync")
        source = record["source"]
        require(
            source["dirty"] is False
            and source["formal_eligible"] is True
            and source["commit"] == source["published_commit"],
            "source is not clean and published",
        )
        for key in (
            "source",
            "environment",
            "data",
            "gpu",
            "world_size",
            "steps_requested",
            "warmup_steps",
            "batch_schedule_hashes",
            "ordered_batch_schedule_sha256",
            "view_generator_before_sha256",
            "optimizer_routes",
        ):
            require(record[key] == original[key], f"different {key}")
        require(record["world_size"] == 2, "requires both GPUs")
        for key in ("affinity", "allocator", "torch_threads", "torch_interop_threads"):
            require(record["hardware"][key] == original["hardware"][key], f"different {key}")
        count = record["steps_requested"] - record["warmup_steps"]
        require(count >= 2, "insufficient timed updates")
        ranks = sorted(record["rank_measurements"], key=lambda x: x["rank"])
        reference_ranks = sorted(original["rank_measurements"], key=lambda x: x["rank"])
        require([r["rank"] for r in ranks] == [0, 1], "missing or duplicate rank")
        for rank, reference in zip(ranks, reference_ranks, strict=True):
            for key in ("physical_gpu", "view_generator_after_sha256"):
                require(rank[key] == reference[key], f"different rank {key}")
            require(rank["profile"] is None, "profiled rank")
            for key in ("update_seconds", "data_wait_seconds"):
                values = rank[key]
                require(len(values) == count, f"wrong population for {key}")
                require(all(math.isfinite(v) and v >= 0 for v in values), "invalid timing value")
        updates = [max(t) for t in zip(*(r["update_seconds"] for r in ranks), strict=True)]
        totals = [
            max(r["update_seconds"][i] + r["data_wait_seconds"][i] for r in ranks)
            for i in range(count)
        ]
        for key, recomputed in (
            ("measured_update_seconds", updates),
            ("measured_step_seconds_including_data_wait", totals),
        ):
            require(len(record[key]) == count, "aggregate population mismatch")
            require(
                all(
                    math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) and a > 0
                    for a, b in zip(record[key], recomputed, strict=True)
                ),
                "aggregate timing disagrees with rank records",
            )
        rows.append(
            {
                "role": role,
                "update_median_seconds": statistics.median(updates),
                "update_p95_seconds": statistics.quantiles(updates, n=100, method="inclusive")[94],
                "total_mean_seconds": statistics.mean(totals),
                "total_median_seconds": statistics.median(totals),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in ranks),
                "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in ranks),
            }
        )
    a = statistics.mean(row["total_mean_seconds"] for row in (rows[0], rows[3]))
    b = statistics.mean(row["total_mean_seconds"] for row in (rows[1], rows[2]))
    return {
        "status": "comparable",
        "source_sha": original["source"]["commit"],
        "rows": rows,
        "total_time_speed_ratio_A_over_B": a / b,
        "total_time_reduction_fraction": 1 - b / a,
        "pair_ratios": [
            rows[0]["total_mean_seconds"] / rows[1]["total_mean_seconds"],
            rows[3]["total_mean_seconds"] / rows[2]["total_mean_seconds"],
        ],
        "both_B_faster_than_both_A": max(
            rows[1]["total_mean_seconds"], rows[2]["total_mean_seconds"]
        )
        < min(rows[0]["total_mean_seconds"], rows[3]["total_mean_seconds"]),
        "limitations": "Two runs per variant; descriptive ratios, not a significance claim. "
        "Does not prove numerical parity, sustained batch capacity, or model quality.",
    }


def validate_execution_factor(
    payloads: list[dict[str, Any]], receipts: list[dict[str, Any]], factor: str
) -> None:
    """Reject mixed execution factors, including flags outside the config file."""
    require(len(payloads) == 2 and len(receipts) == 4, "requires two configs and four runs")
    if factor == "relay_validate_once":
        for i, payload in enumerate(payloads):
            option = payload["model"]["parameters"].pop("relay_validate_once", {"value": False})
            require(option["value"] is bool(i), "wrong reference/candidate execution setting")
    elif factor not in {"cpu_prefetch", "fused_sinkhorn", "fused_gram"}:
        raise ValueError("unsupported execution factor")
    require(payloads[0] == payloads[1], "more than one configuration factor changed")
    for record, candidate in zip(receipts, (False, True, True, False), strict=True):
        require(
            record.get("sequence_checkpoint_disabled_diagnostic_only", False) is False,
            "checkpoint override confounds benchmark",
        )
        expected = candidate if factor == "cpu_prefetch" else False
        require(
            record.get("cpu_prefetch_diagnostic_only", False) is expected,
            "wrong CPU prefetch execution flag",
        )
        for name in ("fused_sinkhorn", "fused_gram"):
            expected = candidate if factor == name else False
            require(
                record.get(f"{name}_diagnostic_only", False) is expected,
                f"wrong {name} execution flag",
            )
            if expected:
                require(record.get(f"{name}_module_count", 0) > 0, "no fused modules enabled")
                require(
                    record[f"{name}_module_count"] == receipts[1][f"{name}_module_count"],
                    "different fused module count",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs=4, help="A1 B1 B2 A2 in that order")
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-factor",
        choices=("relay_validate_once", "cpu_prefetch", "fused_sinkhorn", "fused_gram"),
        default="relay_validate_once",
    )
    args = parser.parse_args()
    require(len({p.resolve() for p in args.receipts}) == 4, "duplicate receipt paths")
    digests = [sha256_file(p) for p in args.receipts]
    require(len(set(digests)) == 4, "duplicate receipt contents are not independent runs")
    configs = [load_experiment_config(p) for p in (args.reference_config, args.candidate_config)]
    payloads = [c.model_dump(mode="json") for c in configs]
    expected = [sha256_file(p) for p in (args.reference_config, args.candidate_config)]
    receipts = [json.loads(p.read_text()) for p in args.receipts]
    validate_execution_factor(payloads, receipts, args.execution_factor)
    for record, index in zip(receipts, (0, 1, 1, 0), strict=True):
        require(record["config_sha256"] == expected[index], "config checksum mismatch")
    result = summarize(receipts)
    result["execution_factor"] = args.execution_factor
    global_batch = int(configs[0].training.train_batch_size.value)
    result["effective_batch"] = global_batch
    for row, record in zip(result["rows"], receipts, strict=True):
        rate = record["measured_cells_per_second_including_data_wait"]
        expected_rate = global_batch / row["total_mean_seconds"]
        require(math.isclose(rate, expected_rate, rel_tol=1e-8), "batch/throughput mismatch")
        row["cells_per_second_including_data_wait"] = rate
    result["inputs"] = [
        {"receipt": str(p.resolve()), "sha256": d}
        for p, d in zip(args.receipts, digests, strict=True)
    ]
    result["analysis_script_sha256"] = sha256_file(Path(__file__))
    result["config_sha256"] = expected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "total_time_speed_ratio_A_over_B")}))


if __name__ == "__main__":
    main()
