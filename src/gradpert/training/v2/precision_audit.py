"""Bounded preflight observations; no instrumentation is installed during fitting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Literal, Self

import torch
from torch import nn


def enabled(options: Any) -> bool:
    return any(
        getattr(options, key, "autocast") != "autocast"
        for key in ("ssl_head_precision", "prediction_precision", "ffn_compute_precision")
    )


def backend_facts(device: torch.device, *, fp8: bool) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native precision preflight requires CUDA")
    facts = dict(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device),
        compute_capability=list(torch.cuda.get_device_capability(device)),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )
    if fp8:
        from gradpert.modeling.v2.fp8_ffn import SUPPORTED_CUDA_BUILDS

        # _scaled_mm is private: a different release requires a new verified
        # capability decision, rather than relying on a compatible-looking name.
        if torch.__version__.split("+")[0] != "2.13.0":
            raise RuntimeError("FP8-v1 preflight is pinned to torch 2.13.0")
        if torch.version.cuda not in SUPPORTED_CUDA_BUILDS:
            raise RuntimeError("FP8-v1 requires an explicitly supported CUDA build")
        if tuple(facts["compute_capability"]) != (12, 0):
            raise RuntimeError("this FP8 candidate requires the approved SM12.0 devices")
        if not callable(getattr(torch, "_scaled_mm", None)):
            raise RuntimeError("pinned FP8-v1 API is missing")
        facts["scaled_mm_schema"] = str(torch.ops.aten._scaled_mm.default._schema)
        facts["recipe"] = "per-tensor current scaling, E4M3 forward / E5M2 gradient"
    return facts


class PrecisionAudit(AbstractContextManager["PrecisionAudit"]):
    """Collect small dtype/kernel counters over native preflight/fullrestore only."""

    def __init__(self, objective: Any, options: Any, device: torch.device) -> None:
        self.objective, self.options = objective, options
        self.fp8 = options.ffn_compute_precision == "fp8_hybrid"
        self.facts = backend_facts(device, fp8=self.fp8)
        self.head_counts: Counter[tuple[str, str]] = Counter()
        self.gemm_counts: Counter[tuple[str, ...]] = Counter()
        self.handles: list[Any] = []
        self.expected_heads: set[str] = set()
        self.original_mm: Callable[..., torch.Tensor] | None = None
        self.phase = "before_restore"
        self.restored_exactly = False
        self.parameter_names: dict[int, str] = {}
        self.payload: dict[str, Any] = dict(
            schema="gradpert-precision-preflight-1",
            status="observing",
            backend=self.facts,
            timing_scope="instrumented preflight only; not uninstrumented throughput",
        )

    def __enter__(self) -> Self:
        # Validate both owners before installing even the first hook: __exit__
        # is not invoked when __enter__ fails.
        if any(
            p.dtype != torch.float32
            for side in ("student", "teacher")
            for p in getattr(self.objective, side).parameters()
        ):
            raise RuntimeError("precision ablations retain FP32 parameter storage")
        for side in ("student", "teacher"):
            model = getattr(self.objective, side)
            if self.fp8:
                from gradpert.modeling.v2.restored_ffn import MatchedGELUFeedForward

                for path, module in model.named_modules():
                    if isinstance(module, MatchedGELUFeedForward):
                        if module.compute_precision != "fp8_hybrid":
                            raise RuntimeError("partial FP8 FFN policy is not accepted")
                        for projection in ("up", "down"):
                            self.parameter_names[id(getattr(module, projection).weight)] = (
                                f"{side}.{path}.{projection}.weight"
                            )
            roots: tuple[str, ...]
            if self.options.ssl_head_precision == "float32":
                roots = ("ssl1_cls", "ssl2_cls", "ssl2_node")
            else:
                roots = ()
            if side == "student" and self.options.prediction_precision == "float32":
                roots += ("prediction",)
            for root in roots:
                for name, module in getattr(model, root).named_modules():
                    if isinstance(module, nn.Linear):
                        key = f"{side}.{root}.{name}"
                        self.expected_heads.add(key)

                        def check(
                            layer: nn.Module,
                            inputs: tuple[torch.Tensor, ...],
                            output: torch.Tensor,
                            *,
                            key: str = key,
                        ) -> None:
                            if inputs[0].dtype != torch.float32 or output.dtype != torch.float32:
                                raise RuntimeError(f"precision head was narrowed: {key}")
                            self.head_counts[(self.phase, key)] += 1

                        self.handles.append(module.register_forward_hook(check))
        if self.fp8:
            from gradpert.modeling.v2 import fp8_ffn

            self.original_mm = fp8_ffn._gemm
            original_mm = fp8_ffn._gemm

            def observed(
                a: torch.Tensor,
                b: torch.Tensor,
                sa: torch.Tensor,
                sb: torch.Tensor,
                dtype: torch.dtype,
                *,
                operation: str,
                parameter_id: int,
            ) -> torch.Tensor:
                result = original_mm(
                    a, b, sa, sb, dtype, operation=operation, parameter_id=parameter_id
                )
                if parameter_id not in self.parameter_names:
                    raise RuntimeError("unattributed FP8 GEMM is not accepted")
                signature = (
                    self.phase,
                    self.parameter_names[parameter_id],
                    operation,
                    str(a.dtype),
                    str(b.dtype),
                    str(result.dtype),
                )
                if (
                    a.device.type != "cuda"
                    or a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                    or b.dtype != torch.float8_e4m3fn
                    or operation not in ("forward", "dgrad", "wgrad")
                    or a.dtype
                    != (torch.float8_e4m3fn if operation == "forward" else torch.float8_e5m2)
                    or result.dtype != (torch.bfloat16 if operation == "forward" else torch.float32)
                ):
                    raise RuntimeError("FP8 preflight observed an unexpected GEMM recipe")
                self.gemm_counts[signature] += 1
                return result

            fp8_ffn._gemm = observed
        return self

    def mark_restored(self, exact: bool) -> None:
        if exact is not True or self.phase != "before_restore":
            raise RuntimeError("precision audit requires one verified exact full-state restore")
        self.restored_exactly = True
        self.phase = "after_restore"

    def finish(self) -> dict[str, Any]:
        if not self.restored_exactly:
            raise RuntimeError("precision preflight has no exact full-state restore evidence")
        if self.fp8:
            observed = {(k[0], k[1], k[2]) for k, n in self.gemm_counts.items() if n}
            expected = {
                (phase, path, op)
                for phase in ("before_restore", "after_restore")
                for path in self.parameter_names.values()
                for op in (
                    ("forward", "dgrad", "wgrad") if path.startswith("student.") else ("forward",)
                )
            }
            if not expected or expected - observed:
                raise RuntimeError("FP8 preflight lacks module/owner/restore-phase GEMM coverage")
        else:
            if any(
                (phase, key) not in self.head_counts
                for phase in ("before_restore", "after_restore")
                for key in self.expected_heads
            ):
                raise RuntimeError("not every selected FP32 head Linear was observed")
            if not self.expected_heads:
                raise RuntimeError("FP32 prediction path was not observed")
        if any(p.grad is not None for p in self.objective.teacher.parameters()):
            raise RuntimeError("precision preflight found Teacher gradients")
        if any(
            p.grad is not None and p.grad.dtype != torch.float32
            for p in self.objective.student.parameters()
        ):
            raise RuntimeError("precision preflight narrowed parameter-gradient storage")
        self.payload.update(
            status="passed",
            exact_full_state_restore=True,
            head_linear_calls=[
                dict(phase=phase, path=path, calls=n)
                for (phase, path), n in sorted(self.head_counts.items())
            ],
            fp8_gemm_calls=[
                dict(
                    phase=phase,
                    path=path,
                    operation=op,
                    a=a,
                    b=b,
                    output=out,
                    use_fast_accum=False,
                    calls=n,
                )
                for (phase, path, op, a, b, out), n in sorted(self.gemm_counts.items())
            ],
        )
        return self.payload

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        for handle in self.handles:
            handle.remove()
        if self.original_mm is not None:
            from gradpert.modeling.v2 import fp8_ffn

            fp8_ffn._gemm = self.original_mm
        return False
