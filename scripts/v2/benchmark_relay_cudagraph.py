"""Synthetic CUDA Graph replay candidate; never changes the model's default backend.

Uses the unchanged eager final-state function inside an AOTAutograd CUDA Graph
region. All scientific inputs/checkpoints and live experiments are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def carried_pair(
    function: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    *,
    gradients: bool,
    bf16: bool = False,
) -> tuple[Tensor, ...]:
    """Keep the first output alive while another invocation continues its state."""
    key, value, decay, beta, state = inputs
    with (
        torch.set_grad_enabled(gradients),
        torch.autocast(key.device.type, dtype=torch.bfloat16, enabled=bf16),
    ):
        first = function(key, value, decay, beta, state)
        second = function(key.flip(1), value.flip(1), decay.flip(1), beta.flip(1), first)
        # Both outputs contribute, so overwriting a live graph output is observable.
        loss = first.square().mean() + 0.3 * second.sin().mean()
    outputs: tuple[Tensor, ...] = (first, second)
    if gradients:
        # The real engine also runs backward outside autocast.
        outputs += torch.autograd.grad(loss, inputs)
    # The next marked step may reuse the graph pool. Retain independent evidence.
    return tuple(x.detach().clone() for x in outputs)


def error_summary(expected: Tensor, actual: Tensor) -> dict[str, float]:
    diff = expected.double() - actual.double()
    return {
        "max_absolute": diff.abs().max().item(),
        "rms_relative": (
            diff.square().mean().sqrt() / expected.double().square().mean().sqrt().clamp_min(1e-12)
        ).item(),
    }


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
    # Configure before CUDA initialization; this isolates arithmetic equivalence.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)

    import torch._dynamo.backends.cudagraphs as backend_source
    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as aot_config
    import torch._inductor.config as inductor_config
    from torch._dynamo.utils import counters
    from torch._inductor.cudagraph_trees import get_manager

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
        raise RuntimeError("synthetic probe requires an idle GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "status": "running",
        "kind": "synthetic_cudagraph_only",
        "source": source.payload(),
        "environment": inspect_environment(root, device_name="cuda:0").payload(),
        "backend": "cudagraphs",
        "torch_git_version": torch.version.git_version,
        "backend_source_sha256": hashlib.sha256(
            Path(backend_source.__file__).read_bytes()
        ).hexdigest(),
        "region": "unchanged chunk_delta_final_state, compiled=False",
        "deterministic_diagnostic_only": True,
        "atol": 3e-5,
        "rtol": 3e-4,
        "timing": "paired live-output forward/backward; 4 warmups and 10 timed repeats",
        "cases": [],
    }
    receipt_path = args.output / "receipt.json"
    atomic_json(receipt_path, receipt)
    try:
        # No elementwise fusion is requested. Unsupported replay must fail rather
        # than quietly timing eager execution. The larger recompile bound is only
        # for the finite synthetic dtype/shape/autograd matrix in this process.
        with (
            dynamo_config.patch(recompile_limit=32),
            aot_config.patch(backward_pass_autocast="off"),
            inductor_config.patch({"triton.cudagraph_or_error": True}),
        ):
            candidate = torch.compile(
                chunk_delta_final_state, backend="cudagraphs", fullgraph=True, dynamic=False
            )
            for bf16 in (False, True):
                for batch, length in ((2, 1), (64, 94), (2, 257)):
                    for gradients in (True, False):
                        torch.manual_seed(24)
                        key = torch.nn.functional.normalize(
                            torch.randn(batch, length, 4, 64, device="cuda"), dim=-1
                        )
                        value, decay = torch.randn_like(key), -torch.rand_like(key) * 2
                        beta = torch.rand(batch, length, 4, device="cuda")
                        # Keep length-one (CLS-like) writes active; mask later tokens.
                        decay[:, 1::7], beta[:, 1::7] = 0, 0
                        state = torch.randn(batch, 4, 64, 64, device="cuda") * 0.1
                        inputs = tuple(
                            x.requires_grad_(gradients) for x in (key, value, decay, beta, state)
                        )
                        rng = torch.cuda.get_rng_state().clone()

                        run = partial(carried_pair, inputs=inputs, gradients=gradients, bf16=bf16)

                        expected = run(chunk_delta_final_state)
                        torch.cuda.synchronize()
                        torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
                        start = time.perf_counter()
                        actual = run(candidate)
                        torch.cuda.synchronize()
                        row: dict[str, Any] = {
                            "bf16": bf16,
                            "gradients": gradients,
                            "batch": batch,
                            "length": length,
                            "cold_candidate_seconds": time.perf_counter() - start,
                            "errors": [
                                error_summary(a, b) for a, b in zip(expected, actual, strict=True)
                            ],
                        }
                        receipt["cases"].append(row)
                        atomic_json(receipt_path, receipt)
                        for a, b in zip(expected, actual, strict=True):
                            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
                        assert torch.equal(rng, torch.cuda.get_rng_state()), "RNG changed"
                        del actual
                        for name, fn in (("eager", chunk_delta_final_state), ("replay", candidate)):
                            for _ in range(4):
                                torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
                                output = run(fn)
                                del output
                            manager_before = get_manager(0, create_if_none_exists=False)
                            recorded_before = (
                                int(manager_before.graph_counter.__reduce__()[1][0])
                                if manager_before is not None
                                else 0
                            )
                            timings = []
                            for _ in range(10):
                                torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                output = run(fn)
                                torch.cuda.synchronize()
                                timings.append(time.perf_counter() - start)
                                # Verify actual replay, not only the initial eager warmup.
                                for a, b in zip(expected, output, strict=True):
                                    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
                                del output
                            manager_after = get_manager(0, create_if_none_exists=False)
                            recorded_after = (
                                int(manager_after.graph_counter.__reduce__()[1][0])
                                if manager_after is not None
                                else 0
                            )
                            row[name] = {
                                "seconds": timings,
                                "median_seconds": statistics.median(timings),
                                "graphs_recorded_during_timing": recorded_after - recorded_before,
                            }
                            atomic_json(receipt_path, receipt)
                            if name == "replay":
                                assert recorded_before > 0, "no captured graph after warmup"
                                assert recorded_after == recorded_before, "replay did not stabilize"
                        manager = get_manager(0, create_if_none_exists=False)
                        assert manager is not None, "no CUDA Graph manager created"
                        row["recorded_graphs"] = int(manager.graph_counter.__reduce__()[1][0])
                        row["memory_allocated_bytes"] = torch.cuda.memory_allocated()
                        row["memory_reserved_bytes"] = torch.cuda.memory_reserved()
                        row["skip_count"] = counters["inductor"]["cudagraph_skips"]
                        assert row["recorded_graphs"] > 0 and row["skip_count"] == 0
                        assert torch.equal(rng, torch.cuda.get_rng_state()), "RNG changed"
                        atomic_json(receipt_path, receipt)
                        print(
                            {k: row[k] for k in ("bf16", "gradients", "batch", "length")},
                            flush=True,
                        )
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(receipt_path, receipt)


if __name__ == "__main__":
    main()
