"""torchrun entry for a sealed v2 launch plan; one visible GPU per worker."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    devices = plan["gpu"].split(",")
    if world != len(devices) or len(set(devices)) != world or not 0 <= rank < world:
        raise ValueError("torchrun workers differ from the sealed physical GPU list")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from gradpert.execution.v2 import run_v2

    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        run_v2(plan, resume=args.resume)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
