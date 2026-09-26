"""Synthetic two-device mechanism probe; no training or model-default changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from gradpert.modeling.v2.operators import sinkhorn
from gradpert.modeling.v2.sinkhorn_fused import fused_sinkhorn, reference_with_saved


def check(reference, actual):
    try:
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)
        error = None
    except AssertionError as exc:
        error = str(exc)
    return {
        "passed": error is None,
        "max_absolute": (actual - reference).abs().max().item(),
        "different_elements": int((actual != reference).sum().item()),
        "different_after_bfloat16": int(
            (actual.to(torch.bfloat16) != reference.to(torch.bfloat16)).sum().item()
        ),
        "error": error,
    }


def timed(function, x, upstream):
    def step():
        y = function(x)
        torch.autograd.grad(y, x, upstream)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    return {
        "seconds_per_forward_backward": (time.perf_counter() - start) / 20,
        "peak_additional_bytes": torch.cuda.max_memory_allocated() - start_allocated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    assert clean
    protocol = {
        "devices": [0, 1],
        "counts": [17, 256, 4096],
        "scales": [0.2, 4.0, 20.0],
        "iterations": 20,
        "atol": 3e-6,
        "rtol": 3e-5,
        "warmup": 3,
        "repetitions": 20,
    }
    result = {
        "training_git_sha": sha,
        "source_clean": clean,
        "protocol": protocol,
        "config_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "scope": "Synthetic FP32 four-stream mechanism only; not model update or throughput proof",
        "torch": torch.__version__,
        "cases": [],
        "passed": True,
    }
    try:
        for device in protocol["devices"]:
            torch.cuda.set_device(device)
            for count in protocol["counts"]:
                for scale in protocol["scales"]:
                    torch.manual_seed(41)
                    x = (torch.randn(count, 4, 4, device="cuda") * scale).requires_grad_()
                    upstream = torch.randn_like(x)
                    reference = sinkhorn(x)
                    reference_grad = torch.autograd.grad(reference, x, upstream)[0]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = fused_sinkhorn(x)
                    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
                    torch.cuda.synchronize()
                    case = {
                        "device": device,
                        "count": count,
                        "scale": scale,
                        "first_call_seconds": time.perf_counter() - start,
                        "output": check(reference, actual),
                        "gradient": check(reference_grad, actual_grad),
                    }
                    from gradpert.modeling.v2._sinkhorn_cuda import forward

                    _, expected_saved = reference_with_saved(x.detach())
                    _, actual_saved = forward(x.detach(), 20)
                    case["normalization_stages"] = [
                        check(expected_saved[i], actual_saved[:, i]) for i in range(40)
                    ]
                    case["eager"] = timed(sinkhorn, x, upstream)
                    case["fused"] = timed(fused_sinkhorn, x, upstream)
                    result["passed"] &= case["output"]["passed"] and case["gradient"]["passed"]
                    result["cases"].append(case)
    except Exception as exc:
        result["passed"] = False
        result["exception"] = repr(exc)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
