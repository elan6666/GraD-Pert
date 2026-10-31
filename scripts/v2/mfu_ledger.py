"""Pass-through, partial native arithmetic ledger; never a hardware FLOP counter."""

from __future__ import annotations

import contextlib
import functools
import math
from collections import Counter
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def tensors(value: Any) -> list[torch.Tensor]:
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]


def signature(value: Any) -> tuple[Any, ...]:
    return tuple((tuple(x.shape), str(x.dtype)) for x in tensors(value))


def arithmetic(func: Any, args: tuple[Any, ...], out: Any) -> tuple[int | None, str]:
    """Only declared shape formulas; unknown arithmetic is never silently zero."""
    name = func._schema.name
    output = tensors(out)
    floating = [x for x in tensors((args, out)) if x.is_floating_point() or x.is_complex()]
    if any(x.is_complex() for x in floating):
        return None, "unsupported_complex"
    if not floating:
        return 0, "non_floating"
    if name in ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::_scaled_mm"):
        a, b = args[1:3] if name in ("aten::addmm", "aten::baddbmm") else args[:2]
        if a.shape[-1] != b.shape[-2]:
            raise ValueError("GEMM ledger inner dimension mismatch")
        flops = 2 * math.prod(a.shape[:-2]) * a.shape[-2] * a.shape[-1] * b.shape[-1]
        # Matches the conventional GEMM-only 2mnk definition: bias/scales are
        # disclosed separately rather than claimed as covered arithmetic.
        return int(flops), "padded_gemm_only" if name == "aten::_scaled_mm" else "gemm_only"
    if name in {"aten::add", "aten::sub", "aten::mul", "aten::div", "aten::neg"}:
        # add/sub alpha can add multiplication; leave that case unknown.
        if len(args) > 2 and name in {"aten::add", "aten::sub"} and args[2] != 1:
            return None, "unsupported_alpha"
        return sum(x.numel() for x in output), "elementwise_arithmetic"
    if name == "aten::sum" and output:
        return args[0].numel() - output[0].numel(), "logical_reduction_adds"
    if name in {"aten::scatter", "aten::scatter_"} and "reduce" in func._schema.overload_name:
        return None, "unsupported_scatter_reduction"
    zero = {
        "alias",
        "view",
        "_unsafe_view",
        "reshape",
        "transpose",
        "t",
        "permute",
        "expand",
        "as_strided",
        "slice",
        "select",
        "squeeze",
        "unsqueeze",
        "detach",
        "clone",
        "copy_",
        "contiguous",
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "new_empty",
        "new_zeros",
        "new_ones",
        "zero_",
        "fill_",
        "lift_fresh",
        "cat",
        "stack",
        "split",
        "split_with_sizes",
        "unbind",
        "index",
        "index_select",
        "gather",
        "scatter",
        "scatter_",
        "masked_fill",
        "masked_fill_",
        "where",
        "tril",
        "triu",
        "eye",
        "equal",
        "isfinite",
        "isnan",
        "isinf",
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "all",
        "any",
        "_local_scalar_dense",
        "_assert_async",
        "is_same_size",
        "_reshape_alias",
        "unfold",
        "diagonal",
        "slice_scatter",
        "select_scatter",
        "expand_as",
    }
    if name.removeprefix("aten::") in zero:
        return 0, "known_non_arithmetic"
    # Casts, special functions, RNG, norm, solves, fused/foreach operators,
    # custom kernels, optimizer updates and unknown ops need their own ledger.
    return None, "unsupported_arithmetic_or_conversion"


class ArithmeticLedger(TorchDispatchMode):
    """Execute each operator once without decompose, tensor reads or profiler."""

    def __init__(self) -> None:
        super().__init__()
        self.scope: ContextVar[tuple[str, str, int]] = ContextVar(
            "mfu_scope", default=("forward", "unattributed", 0)
        )
        self.rows: Counter[tuple[Any, ...]] = Counter()
        self.forward_calls: Counter[tuple[Any, ...]] = Counter()
        self.update = 0

    @contextlib.contextmanager
    def region(
        self, *, phase: str | None = None, owner: str | None = None, recompute: bool = False
    ) -> Iterator[None]:
        previous = self.scope.get()
        token = self.scope.set(
            (phase or previous[0], owner or previous[1], previous[2] + int(recompute))
        )
        try:
            yield
        finally:
            self.scope.reset(token)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # The tensor DAG is exactly the caller's. Do not use FlopCounterMode's
        # decomposition path, registered decompositions or alternate kernels.
        out = func(*args, **kwargs)
        flops, coverage = arithmetic(func, args, out)
        if kwargs.get("alpha", 1) != 1 and func._schema.name in {"aten::add", "aten::sub"}:
            flops, coverage = None, "unsupported_alpha"
        phase, owner, depth = self.scope.get()
        self.rows[
            (
                self.update,
                str(func),
                signature((args, kwargs)),
                signature(out),
                phase,
                owner,
                depth > 0,
                flops,
                coverage,
            )
        ] += 1
        return out

    def payload(self) -> dict[str, Any]:
        rows = [
            dict(
                update=k[0],
                operator=k[1],
                inputs=k[2],
                outputs=k[3],
                phase=k[4],
                owner=k[5],
                recompute=k[6],
                flops_per_call=k[7],
                coverage=k[8],
                calls=n,
            )
            for k, n in sorted(self.rows.items(), key=lambda item: repr(item[0]))
        ]
        counted = sum(r["calls"] * (r["flops_per_call"] or 0) for r in rows)
        replay = sum(r["calls"] * (r["flops_per_call"] or 0) for r in rows if r["recompute"])
        model = sum(
            r["calls"] * (r["flops_per_call"] or 0)
            for r in rows
            if not r["recompute"] and r["phase"] not in ("optimizer", "ema_centers")
        )
        return dict(
            schema="gradpert-partial-arithmetic-ledger-1",
            rows=rows,
            forward_calls=[
                dict(update=k[0], owner=k[1], inputs=k[2], calls=n)
                for k, n in sorted(self.forward_calls.items(), key=lambda item: repr(item[0]))
            ],
            covered_executed_flops=counted,
            covered_recompute_flops=replay,
            covered_model_flops=model,
            covered_optimizer_ema_flops=counted - replay - model,
            unsupported_calls=sum(r["calls"] for r in rows if r["flops_per_call"] is None),
            complete=False,
            mfu=None,
            hfu=None,
            limitation="partial shape arithmetic; GEMM bias/scales, FP8 logical padding, "
            "special/solve/conversion/custom/optimizer coverage not established; "
            "no hardware instruction measurement or FLOP coverage percentage",
        )


@contextlib.contextmanager
def counting_regions(runtime: Any, ledger: ArithmeticLedger) -> Iterator[None]:
    """Compose native checkpoint contexts and restore every callable on failure."""
    import torch.utils.checkpoint as checkpoints

    import gradpert.modeling.v2.graph_mhc as graph_mhc
    import gradpert.modeling.v2.model as model
    import gradpert.modeling.v2.operators as operators
    import gradpert.training.v2.objective as objective_module

    changes: list[tuple[Any, str, Any, bool]] = []

    def replace(owner: Any, name: str, value: Any) -> None:
        changes.append((owner, name, getattr(owner, name), name in vars(owner)))
        setattr(owner, name, value)

    def wrap(target: Any, name: str, **scope: Any) -> None:
        original = getattr(target, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            with ledger.region(**scope):
                if name == "forward":
                    ledger.forward_calls[
                        (ledger.update, scope["owner"], signature((args, kwargs)))
                    ] += 1
                return original(*args, **kwargs)

        replace(target, name, observed)

    original = checkpoints.checkpoint

    @functools.wraps(original)
    def checkpoint(function, *args, **kwargs):
        if kwargs.get("use_reentrant") is not False:
            raise ValueError("MFU supports only the native non-reentrant checkpoint")
        context_fn = kwargs.get("context_fn", checkpoints.noop_context_fn)

        def contexts():
            forward, replay = context_fn()

            class Replay:
                def __init__(self):
                    self.active: list[contextlib.ExitStack] = []

                def __enter__(self):
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(replay)
                        stack.enter_context(ledger.region(recompute=True))
                        self.active.append(stack.pop_all())

                def __exit__(self, *exc):
                    return self.active.pop().__exit__(*exc)

            return forward, Replay()

        return original(function, *args, **dict(kwargs, context_fn=contexts))

    try:
        for module in (checkpoints, model, operators, graph_mhc, objective_module):
            replace(module, "checkpoint", checkpoint)
        wrap(torch.autograd, "backward", phase="backward", owner="student")
        wrap(runtime.optimizer, "step", phase="optimizer", owner="optimizer")
        wrap(runtime.objective, "commit_statistics", phase="ema_centers", owner="teacher")
        for role in ("student", "teacher"):
            for name in (
                "graph",
                "cell",
                "response",
                "ssl1_cls",
                "ssl1_node",
                "ssl2_cls",
                "ssl2_node",
                "prediction",
                "expression",
            ):
                wrap(
                    getattr(getattr(runtime.objective, role), name),
                    "forward",
                    owner=f"{role}.{name}",
                )
        yield
    finally:
        for owner, name, value, own in reversed(changes):
            if own:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
