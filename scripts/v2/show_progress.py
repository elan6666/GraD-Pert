"""Display the current v2 training/evaluation phase and a live progress bar."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def render(progress: dict[str, object]) -> str:
    phase = str(progress["phase"])
    if phase in {"training", "checkpointing", "epoch_complete", "training_complete"}:
        done = int(progress.get("epoch_step", 0))
        total = int(progress.get("epoch_steps_total", 0))
        detail = f"epoch {progress.get('epoch')}/{progress.get('epochs_total')} step {done}/{total}"
        global_done = int(progress.get("optimizer_steps_completed", done))
        global_total = int(progress.get("optimizer_steps_total", total))
        if global_total:
            detail += f" overall {global_done}/{global_total} ({global_done / global_total:.1%})"
    elif phase == "validation_joint":
        done = int(progress.get("batches_completed", 0))
        total = int(progress.get("batches_total", 0))
        detail = f"epoch {progress.get('epoch')} joint batches {done}/{total}"
    elif phase in {"validation", "validation_prediction", "test"}:
        done = int(progress.get("conditions_completed", 0))
        total = int(progress.get("conditions_total", 0))
        detail = f"{done}/{total} conditions"
        if phase == "test":
            detail = f"{progress.get('checkpoint_role')} {detail}"
    elif phase == "failed":
        done, total = 0, 1
        detail = f"{progress.get('error_type')}: {progress.get('error')}"
    else:
        done, total, detail = 1, 1, phase
    fraction = min(1.0, max(0.0, done / total)) if total else 0.0
    blocks = round(24 * fraction)
    bar = "█" * blocks + "░" * (24 - blocks)
    terms = progress.get("latest_training_terms")
    loss = (
        f" joint_loss={float(terms['joint_loss']):.6f}"
        if isinstance(terms, dict) and "joint_loss" in terms
        else ""
    )
    throughput = progress.get("throughput")
    speed = (
        f" {float(throughput['cells_per_second_this_epoch']):.2f} cells/s"
        if isinstance(throughput, dict) and "cells_per_second_this_epoch" in throughput
        else ""
    )
    if isinstance(throughput, dict) and "optimizer_steps_per_second_this_epoch" in throughput:
        speed += f" {float(throughput['optimizer_steps_per_second_this_epoch']):.3f} step/s"
    joint = (
        f" joint_loss={float(progress['joint_loss_running_mean']):.6f}"
        if phase == "validation_joint" and "joint_loss_running_mean" in progress
        else ""
    )
    return f"{phase:>21} [{bar}] {fraction:5.1%} {detail}{loss}{joint}{speed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    path = args.run_root / "fit/live_progress.json"
    last: str | None = None
    while True:
        if path.is_file():
            progress = json.loads(path.read_text())
            line = render(progress)
            if line != last:
                print(line, flush=True)
                last = line
            if progress.get("phase") in {"complete", "failed"}:
                return
        elif not args.follow:
            parser.error(f"progress has not been written yet: {path}")
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
