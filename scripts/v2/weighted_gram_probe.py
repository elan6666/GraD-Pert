"""Standalone weighted-Gram numerical/memory probe; run only on idle GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from gradpert.modeling.v2.weighted_gram import fused_weighted_gram, weighted_gram_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
    root = Path(__file__).resolve().parents[2]
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    assert clean
    protocol = {
        "batch": [1, 16, 64],
        "length": [1, 17, 32],
        "heads": 4,
        "channels": 64,
        "seed": 29,
        "atol": 3e-6,
        "rtol": 3e-5,
        "devices": [0, 1],
    }
    result = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_clean": clean,
        "protocol": protocol,
        "config_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "scope": "Isolated forward-fused/eager-backward candidate, not integrated model throughput",
        "passed": True,
        "cases": [],
    }
    try:
        for device in protocol["devices"]:
            torch.cuda.set_device(device)
            free, total = torch.cuda.mem_get_info()
            assert total - free < 512 * 1024**2, "requires idle GPU"
            for batch in protocol["batch"]:
                for length in protocol["length"]:
                    torch.manual_seed(29)
                    keys = (
                        torch.nn.functional.normalize(
                            torch.randn(batch, length, 4, 64, device="cuda"), dim=-1
                        )
                        .transpose(1, 2)
                        .requires_grad_()
                    )
                    gates = (-torch.rand_like(keys).cumsum(-2)).requires_grad_()
                    upstream = torch.randn(batch, 4, length, length, device="cuda")

                    def evaluate(fn, keys=keys, gates=gates, upstream=upstream):
                        output = fn(keys, gates)
                        gradients = torch.autograd.grad(output, (keys, gates), upstream)
                        return (output, *gradients)

                    reference = evaluate(weighted_gram_reference)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    actual = evaluate(fused_weighted_gram)
                    torch.cuda.synchronize()
                    case = {
                        "device": device,
                        "batch": batch,
                        "length": length,
                        "first_call_seconds": time.perf_counter() - start,
                        "comparisons": [],
                    }
                    for expected, value in zip(reference, actual, strict=True):
                        failure = None
                        try:
                            torch.testing.assert_close(value, expected, atol=3e-6, rtol=3e-5)
                        except AssertionError as exc:
                            failure = str(exc)
                        result["passed"] &= failure is None
                        case["comparisons"].append(
                            {
                                "passed": failure is None,
                                "error": failure,
                                "different_elements": int((expected != value).sum().item()),
                                "max_absolute": (expected - value).abs().max().item(),
                            }
                        )
                    for name, fn in (
                        ("eager", weighted_gram_reference),
                        ("fused", fused_weighted_gram),
                    ):
                        for _ in range(3):
                            evaluate(fn)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated()
                        started = time.perf_counter()
                        for _ in range(20):
                            evaluate(fn)
                        torch.cuda.synchronize()
                        case[name] = {
                            "forward_backward_seconds": (time.perf_counter() - started) / 20,
                            "peak_additional_bytes": torch.cuda.max_memory_allocated() - baseline,
                        }
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
