"""Post-fit migration queue on one idle GPU; never train or relaunch a row."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--archive-last", type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--row", choices=("ref", "lr_low", "lr_mid"))
    args = parser.parse_args()
    if args.row:
        from gradpert.execution.postfit import evaluate_best_last

        result = evaluate_best_last(
            training_root=args.training_root / args.row / "full",
            output_root=args.output_root / args.row,
            data_root=args.data_root,
            repository_root=args.source,
            publication=args.publication,
            publication_sha256=args.publication_sha,
            device_name="cuda:0",
            archived_last=args.archive_last if args.row == "lr_mid" else None,
        )
        print(json.dumps(result), flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    for row in ("ref", "lr_low", "lr_mid"):
        while not (args.training_root / row / "COMPLETE.json").exists():
            time.sleep(60)
        while True:
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                text=True,
            )
            if args.gpu_uuid not in apps:
                break
            time.sleep(300)
        print(f"TEST_START {row}", flush=True)
        with (args.output_root / f"{row}.log").open("x") as log:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid)
            result = subprocess.run(
                [sys.executable, __file__, *sys.argv[1:], "--row", row],
                cwd=args.source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"TEST_RC {row} {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError("post-fit failed; preserve evidence, no automatic retry")
    with (args.output_root / "QUEUE_COMPLETE.json").open("x") as f:
        json.dump({"status": "complete", "rows": ["ref", "lr_low", "lr_mid"]}, f)


if __name__ == "__main__":
    main()
