"""CUDA correctness and timing of the opt-in relay chunk backend, without training data."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("CUDA evidence stays on the server")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator missing")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    from gradpert.data._io import atomic_json
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.modeling.v2.operators import chunk_delta_final_state

    root = Path(__file__).resolve().parents[2]
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("kernel probe requires an idle GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "status": "running",
        "kind": "synthetic_kernel_only",
        "source": source.payload(),
        "environment": inspect_environment(root, device_name="cuda:0").payload(),
        "cases": [],
        "autocast_dtype": "bfloat16",
        "atol": 3e-5,
        "rtol": 3e-4,
        "timing": "three warmups then ten synchronized forward+backward times; cold time separate",
        "compiler_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    atomic_json(args.output / "receipt.json", receipt)
    try:
        for bf16 in (False, True):
            for batch, length in ((2, 1), (64, 94), (2, 257)):
                torch.manual_seed(24)
                key = torch.nn.functional.normalize(
                    torch.randn(batch, length, 4, 64, device="cuda"), dim=-1
                )
                value = torch.randn_like(key)
                decay = -torch.rand_like(key) * 2
                beta = torch.rand(batch, length, 4, device="cuda")
                # Include masked no-op writes and a nonzero carried state.
                decay[:, ::7] = 0
                beta[:, ::7] = 0
                state = torch.randn(batch, 4, 64, 64, device="cuda") * 0.1
                inputs = (
                    key.requires_grad_(),
                    value.requires_grad_(),
                    decay.requires_grad_(),
                    beta.requires_grad_(),
                    state.requires_grad_(),
                )
                rng = torch.cuda.get_rng_state()

                def run(
                    compiled: bool,
                    inputs: tuple[
                        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                    ] = inputs,
                    bf16: bool = bf16,
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                        result = chunk_delta_final_state(*inputs, compiled=compiled)
                        loss = result.float().square().mean()
                    grads = torch.autograd.grad(loss, inputs)
                    return result.detach(), tuple(t.detach() for t in grads)

                eager = run(False)
                torch.cuda.synchronize()
                started = time.perf_counter()
                candidate = run(True)
                torch.cuda.synchronize()
                cold_seconds = time.perf_counter() - started
                errors = []
                for a, b in zip((eager[0], *eager[1]), (candidate[0], *candidate[1]), strict=True):
                    errors.append(
                        {
                            "max_absolute": (a - b).abs().max().item(),
                            "rms_relative": (
                                (a - b).square().mean().sqrt()
                                / a.square().mean().sqrt().clamp_min(1e-12)
                            ).item(),
                        }
                    )
                row: dict[str, Any] = {
                    "bf16": bf16,
                    "batch": batch,
                    "length": length,
                    "cold_candidate_seconds": cold_seconds,
                    "errors_output_then_gradients": errors,
                }
                receipt["cases"].append(row)
                atomic_json(args.output / "receipt.json", receipt)
                for a, b in zip((eager[0], *eager[1]), (candidate[0], *candidate[1]), strict=True):
                    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
                assert torch.equal(rng, torch.cuda.get_rng_state()), "kernel changed RNG"
                for compiled in (False, True):
                    for _ in range(3):
                        run(compiled)
                    times = []
                    for _ in range(10):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        run(compiled)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                    row["compiled" if compiled else "eager"] = {
                        "seconds": times,
                        "median_seconds": statistics.median(times),
                    }
                print(json.dumps(row), flush=True)
                atomic_json(args.output / "receipt.json", receipt)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
