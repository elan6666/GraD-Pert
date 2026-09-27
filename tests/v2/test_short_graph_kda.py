"""Short-neighborhood fused recurrence preserves final-state training semantics."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2.operators import chunk_delta_final_state


def test_short_graph_option_is_scoped_and_candidate_configs_change_only_execution() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    for kind in ("profiling_m2_a2", "capacity_m32_a2", "capacity_m64_a2"):
        relative = Path(kind) / "gradpert_v2/nadig_jurkat.yaml"
        baseline = yaml.safe_load((root / "optimized_single_pass_jurkat" / relative).read_text())
        candidate_path = root / "mechanism_short_graph_jurkat" / relative
        candidate = yaml.safe_load(candidate_path.read_text())
        arch, _ = V2Options.parse_parameters(
            load_experiment_config(candidate_path).model.parameters
        )
        assert arch.short_graph_kernel is True
        candidate["model"]["parameters"].pop("short_graph_kernel")
        assert candidate == baseline
    assert "short_graph_kernel" not in V2Architecture().payload()
    with pytest.raises(ValueError, match="single-pass eager relay graph"):
        replace(V2Architecture(), short_graph_kernel=True)


@pytest.mark.parametrize("length", [1, 7, 32, 47])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_graph_cuda_final_state_and_four_input_gradients(
    length: int, dtype: torch.dtype
) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from gradpert.modeling.v2.short_graph_kda import short_graph_final_state

    torch.manual_seed(710 + length)
    shape = (5, length, 4, 64)
    key = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1)
    value = torch.randn(shape, device="cuda", dtype=dtype)
    decay = -torch.rand(shape, device="cuda") * 0.3
    beta = torch.rand(shape[:-1], device="cuda") * 0.9
    inputs = (key, value, decay, beta)
    reference_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    candidate_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    upstream = torch.randn((5, 4, 64, 64), device="cuda")
    before = torch.cuda.get_rng_state()
    reference = chunk_delta_final_state(*reference_inputs)
    reference_grad = torch.autograd.grad((reference * upstream).sum(), reference_inputs)
    candidate = short_graph_final_state(*candidate_inputs)
    candidate_grad = torch.autograd.grad((candidate * upstream).sum(), candidate_inputs)
    assert torch.equal(before, torch.cuda.get_rng_state())
    torch.testing.assert_close(reference, candidate, atol=3e-5, rtol=3e-4)
    for expected, actual in zip(reference_grad, candidate_grad, strict=True):
        torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-4)
