"""Bound native PyTorch allocator use before invoking the existing CLI."""

from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve-mib", type=int, default=2048)
    options, command = parser.parse_known_args()
    if options.reserve_mib < 2048 or not command:
        parser.error("require at least 2048 MiB reserve and a native CLI command")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("required native allocator is absent")
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("bind exactly one physical GPU through CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info(0)
    budget = free - options.reserve_mib * 1024**2
    if budget < 1024**3:
        raise RuntimeError("insufficient free memory for a shared-GPU attempt")
    torch.cuda.set_per_process_memory_fraction(budget / total, device=0)
    print(
        f"SHARED_GPU_FREE_BYTES={free} ALLOCATOR_BUDGET_BYTES={budget} "
        f"RESERVE_MIB={options.reserve_mib}",
        flush=True,
    )
    sys.argv = ["gradpert", *command]
    runpy.run_module("gradpert", run_name="__main__")


if __name__ == "__main__":
    main()
