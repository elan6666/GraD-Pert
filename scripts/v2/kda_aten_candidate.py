"""Opt-in diagnostic candidate: execute the existing KDA ATen DAG from C++.

Not imported by native training. GPU kernel count/arithmetic should remain the
same; only host orchestration is moved. Numerical and end-to-end tests decide
whether this is useful. Compiler/cache initialization is a measured cold cost.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_candidate():
    from torch.utils.cpp_extension import load

    load(
        name="gradpert_kda_aten_v1",
        sources=[str(Path(__file__).resolve().parents[1] / "performance/kda_aten.cpp")],
        extra_cflags=["-O2"],
        with_cuda=False,
        is_python_module=False,
    )
    return torch.ops.gradpert_kda_diagnostic.final_state


def final_state(k, v, log_decay, beta, state=None, chunk_size=32, **options):
    if options.keys() - {"compiled", "fused_gram", "cache_constants"} or any(options.values()):
        raise ValueError("ATen candidate isolates eager execution; other candidates must be off")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("ATen candidate is scoped to the current float32 default")
    return load_candidate()(k, v, log_decay, beta, state, chunk_size)


@contextmanager
def candidate_scope():
    """Probe-only replacement, also reached by checkpoint recomputation."""
    import gradpert.modeling.v2.operators as operators

    load_candidate()
    original = operators.chunk_delta_final_state
    operators.chunk_delta_final_state = final_state
    try:
        yield
    finally:
        operators.chunk_delta_final_state = original
