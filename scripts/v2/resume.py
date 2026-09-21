"""Resume the exact published v2 lifecycle identified by its saved launch plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.launch.read_text())
    if args.launch.resolve() != Path(plan["run_root"]).resolve() / "launch.json":
        raise ValueError("launch plan must belong to its original run directory")
    if not (Path(plan["run_root"]) / "fit" / "epoch_state.json").is_file():
        raise ValueError("no committed epoch state; inspect the startup failure before retrying")
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    from gradpert.execution.v2 import run_v2

    run_v2(plan, resume=True)


if __name__ == "__main__":
    main()
