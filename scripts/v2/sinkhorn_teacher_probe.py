"""Compare saved and no-grad four-stream Sinkhorn on the same CUDA input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch

from gradpert.modeling.v2._sinkhorn_cuda import forward, forward_no_grad
from gradpert.modeling.v2.sinkhorn_fused import fused_sinkhorn


def measure(
    function: Callable[[torch.Tensor], object], logits: torch.Tensor
) -> dict[str, float | int]:
    for _ in range(3):
        function(logits)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    output = function(logits)
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - before
    del output
    durations = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            function(logits)
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) / 20)
    return {"retained_bytes": retained, "seconds_per_call_median": statistics.median(durations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("PYTORCH_ALLOC_CONF must be expandable_segments:True")
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if not clean:
        raise RuntimeError("source checkout must be clean")
    devices = (0, 1)
    counts = (1, 17, 4096, 32768)
    scales = (0.2, 4.0, 20.0)
    protocol = {
        "devices": devices,
        "counts": counts,
        "scales": scales,
        "iterations": 20,
    }
    result: dict[str, object] = {
        "training_git_sha": sha,
        "source_clean": clean,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Isolated Teacher forward, not full-model training throughput or capacity",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": [],
        "passed": True,
    }
    cases = []
    try:
        for device in devices:
            torch.cuda.set_device(device)
            for count in counts:
                for scale in scales:
                    torch.manual_seed(41)
                    logits = torch.randn(count, 4, 4, device="cuda") * scale
                    old, probabilities = forward(logits, 20)
                    new = forward_no_grad(logits, 20)
                    with torch.no_grad():
                        dispatched = fused_sinkhorn(logits)
                    torch.cuda.synchronize()
                    exact = bool(torch.equal(old, new) and torch.equal(new, dispatched))
                    case = {
                        "device": device,
                        "count": count,
                        "scale": scale,
                        "exact_output": exact,
                        "max_absolute": (old - new).abs().max().item(),
                        "expected_saved_probability_bytes": probabilities.numel()
                        * probabilities.element_size(),
                        "saved": measure(lambda x: forward(x, 20), logits),
                        "no_grad": measure(lambda x: forward_no_grad(x, 20), logits),
                    }
                    result["passed"] = bool(result["passed"] and exact)
                    cases.append(case)
    except Exception as exc:
        result["passed"] = False
        result["exception"] = repr(exc)
    result["cases"] = cases
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
