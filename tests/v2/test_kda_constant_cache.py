"""Shape-constant reuse must not alter final-state arithmetic or gradients."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2.operators import _delta_static_tensors, chunk_delta_final_state


def test_kda_constant_cache_configs_change_only_execution() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    for kind in ("profiling_m2_a2", "capacity_m32_a2", "capacity_m64_a2"):
        relative = Path(kind) / "gradpert_v2/nadig_jurkat.yaml"
        baseline = yaml.safe_load((root / "optimized_single_pass_jurkat" / relative).read_text())
        candidate_path = root / "mechanism_kda_constants_jurkat" / relative
        candidate = yaml.safe_load(candidate_path.read_text())
        arch, _ = V2Options.parse_parameters(
            load_experiment_config(candidate_path).model.parameters
        )
        assert arch.cache_kda_constants is True
        candidate["model"]["parameters"].pop("cache_kda_constants")
        assert candidate == baseline
    assert "cache_kda_constants" not in V2Architecture().payload()
    with pytest.raises(ValueError, match="eager relay attention"):
        replace(V2Architecture(), cache_kda_constants=True)


@pytest.mark.parametrize("length", [1, 7, 32, 47])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_constants_keep_exact_outputs_and_gradients(length: int, dtype: torch.dtype) -> None:
    torch.manual_seed(87 + length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (3, length, 4, 64)
    key = torch.nn.functional.normalize(torch.randn(shape, device=device), dim=-1)
    value = torch.randn(shape, device=device, dtype=dtype)
    decay = -torch.rand(shape, device=device) * 0.2
    beta = torch.rand(shape[:-1], device=device) * 0.8
    upstream = torch.randn((3, 4, 64, 64), device=device)
    reference_inputs = tuple(
        x.detach().clone().requires_grad_(True) for x in (key, value, decay, beta)
    )
    candidate_inputs = tuple(
        x.detach().clone().requires_grad_(True) for x in (key, value, decay, beta)
    )
    rng_before = torch.get_rng_state().clone()
    reference = chunk_delta_final_state(*reference_inputs)
    reference_grad = torch.autograd.grad((reference * upstream).sum(), reference_inputs)
    candidate = chunk_delta_final_state(*candidate_inputs, cache_constants=True)
    candidate_grad = torch.autograd.grad((candidate * upstream).sum(), candidate_inputs)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.equal(reference, candidate)
    for expected, actual in zip(reference_grad, candidate_grad, strict=True):
        assert torch.equal(expected, actual)


def test_static_buffers_are_reused_without_gradients() -> None:
    _delta_static_tensors.cache_clear()
    first = _delta_static_tensors(7, torch.device("cpu"), torch.float32)
    again = _delta_static_tensors(7, torch.device("cpu"), torch.float32)
    assert first[0] is again[0] and first[1] is again[1]
    assert all(not tensor.requires_grad for tensor in first)
    assert _delta_static_tensors.cache_info().hits == 1
