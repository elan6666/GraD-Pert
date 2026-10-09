"""Evaluate bounded frozen conditions beside a named active fit, without fitting."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_checkpoint_eval import evaluate_worker, resolve_evaluation_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--conditions", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.19)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    launch = read_json(args.launch)
    training = validate_training_stage(launch)
    root = Path(launch["run_root"])
    last = training["last"]
    request = argparse.Namespace(
        config=Path(launch["config"]),
        runtime=args.runtime,
        training_run_root=root,
        checkpoint=root / "fit" / last["file"],
        checkpoint_sha256=last["sha256"],
        checkpoint_role="last",
        output_root=args.output,
        gpu=args.gpu,
        split="test",
    )
    plan = resolve_evaluation_plan(request)
    if len(plan["gpu"]) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("probe requires exactly its selected visible GPU")
    plan.update(
        cpu_training_state=True,
        cuda_memory_fraction=args.memory_fraction,
        diagnostic_only=True,
        diagnostic_condition_count=args.conditions,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    progress_path = args.fit_root / "fit/live_progress.json"
    before = read_json(progress_path)
    started = time.time()
    receipt = {
        "schema": "gradpert-shared-evaluation-probe-1",
        "scientific_complete": False,
        "plan": plan,
        "active_fit_root": str(args.fit_root),
        "active_fit_manifest_sha256": sha256_file(args.fit_root / "run_manifest.json"),
        "fit_before": before,
        "started_unix": started,
    }
    try:
        result = evaluate_worker(plan, 0)
        receipt.update(status="passed", evaluation=result)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(
            finished_unix=time.time(),
            wall_seconds=time.time() - started,
            fit_after=read_json(progress_path),
        )
        atomic_json(output / "PROBE_RECEIPT.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
