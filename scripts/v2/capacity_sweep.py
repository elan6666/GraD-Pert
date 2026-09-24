"""Run an ordered two-GPU capacity sweep, stopping at the first failed point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from capacity_report import collect

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file


def validate_profiles(configs: list[Path]) -> list[int]:
    """Require increasing physical batches and otherwise identical experiments."""
    batches = []
    baseline = None
    for path in configs:
        load_experiment_config(path)
        raw = yaml.safe_load(path.read_text())
        parameters = raw["model"]["parameters"]
        accumulation = parameters["accumulation"]["value"]
        if parameters["world_size"]["value"] != 2 or accumulation < 1:
            raise ValueError("sweep requires two ranks and positive accumulation")
        micro = parameters.pop("microbatch")["value"]
        if raw["training"].pop("train_batch_size")["value"] != 2 * micro * accumulation:
            raise ValueError("global batch must be world size times microbatch and accumulation")
        if baseline is not None and raw != baseline:
            raise ValueError("capacity profiles may vary only physical/global batch")
        baseline = raw
        batches.append(micro)
    if not batches or batches != sorted(set(batches)):
        raise ValueError("supply unique increasing per-rank batch profiles")
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--integration-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    batches = validate_profiles(args.config)
    root = Path(__file__).resolve().parents[2]
    rows = []
    for config, micro in zip(args.config, batches, strict=True):
        accumulation = yaml.safe_load(config.read_text())["model"]["parameters"]["accumulation"][
            "value"
        ]
        output = args.output / f"micro{micro}"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "scripts/v2/capacity_probe.py"),
            "--config",
            str(config.resolve()),
            "--gpu",
            "0,1",
            "--data-root",
            str(args.data_root),
            "--output",
            str(output),
            "--publication",
            str(args.publication),
            "--publication-sha256",
            args.publication_sha256,
        ]
        if args.integration_only:
            command.append("--integration-only")
        rows.append(
            {
                "microbatch": micro,
                "global_batch": 2 * micro * accumulation,
                "config_sha256": sha256_file(config),
                "command": command,
            }
        )
    result = {
        "kind": "integration_sweep" if args.integration_only else "capacity_sweep",
        "status": "planned",
        "rows": rows,
    }
    if not args.execute:
        print(json.dumps(result, indent=2))
        return
    if not all(
        p.resolve().is_relative_to("/data/yilangliu") for p in (args.output, args.data_root)
    ):
        parser.error("GPU sweep data and outputs must remain on the server")
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    result["source"] = source.payload()
    args.output.mkdir(parents=True, exist_ok=False)
    result["status"] = "running"
    atomic_json(args.output / "sweep.json", result)
    env = {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "1"}
    for row, config in zip(rows, args.config, strict=True):
        with (args.output / f"micro{row['microbatch']}.log").open("x") as log:
            process = subprocess.run(
                row["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        row["exit_code"] = process.returncode
        receipt = args.output / f"micro{row['microbatch']}" / "receipt.json"
        if receipt.exists():
            row["receipt_sha256"] = sha256_file(receipt)
        if process.returncode != 0 or not receipt.exists():
            result["status"] = "stopped_on_failure_not_a_confirmed_memory_boundary"
            atomic_json(args.output / "sweep.json", result)
            raise SystemExit(1)
        payload = json.loads(receipt.read_text())
        if (
            payload.get("status") != "passed"
            or payload.get("config_sha256") != row["config_sha256"]
        ):
            raise RuntimeError("probe receipt status/config mismatch")
        if not args.integration_only:
            row["capacity"] = collect(receipt, config)
        row["status"] = "passed"
        atomic_json(args.output / "sweep.json", result)
    result["status"] = "all_tested_points_passed_maximum_not_established"
    atomic_json(args.output / "sweep.json", result)


if __name__ == "__main__":
    main()
