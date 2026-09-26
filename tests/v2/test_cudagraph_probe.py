"""Exercise the live-output/state-carry probe on an independent recurrence."""

import importlib.util
from pathlib import Path

import torch

from gradpert.modeling.v2.operators import chunk_delta_final_state, delta_final_state

SPEC = importlib.util.spec_from_file_location(
    "cudagraph_probe",
    Path(__file__).resolve().parents[2] / "scripts/v2/benchmark_relay_cudagraph.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_carried_pair_preserves_state_and_checks_all_input_gradients():
    torch.manual_seed(23)
    key = torch.nn.functional.normalize(torch.randn(2, 5, 2, 4), dim=-1)
    values = torch.randn_like(key)
    decay = -torch.rand_like(key)
    beta = torch.rand(2, 5, 2)
    state = torch.randn(2, 2, 4, 4) * 0.1
    inputs = tuple(x.requires_grad_() for x in (key, values, decay, beta, state))
    rng = torch.get_rng_state().clone()
    actual = MODULE.carried_pair(chunk_delta_final_state, inputs, gradients=True)
    expected = MODULE.carried_pair(delta_final_state, inputs, gradients=True)
    assert len(actual) == 7  # two live states, all five input gradients
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        assert not a.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.count_nonzero(actual[-1]) > 0  # carried initial state is differentiable
    no_grad = MODULE.carried_pair(chunk_delta_final_state, inputs, gradients=False)
    for a, b in zip(no_grad, actual[:2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_evidence_outputs_are_not_views_into_reused_input_storage():
    inputs = tuple(torch.ones(2, 2) for _ in range(5))
    outputs = MODULE.carried_pair(lambda k, v, d, b, s: s, inputs, gradients=False)
    inputs[-1].zero_()
    assert all(torch.equal(x, torch.ones(2, 2)) for x in outputs)
    error = MODULE.error_summary(torch.zeros(2), torch.ones(2) * 0.01)
    assert error["max_absolute"] > 0.009
