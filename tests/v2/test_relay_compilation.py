"""Compilation is an execution option; scientific state and RNG stay aligned."""

from dataclasses import replace

import pytest
import torch

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import operators
from gradpert.modeling.v2.operators import delta_final_block, delta_final_state


@pytest.mark.parametrize("length", [1, 7, 32])
def test_aot_block_preserves_outputs_gradients_rng_and_extreme_decay(length):
    torch.manual_seed(16)
    key = torch.nn.functional.normalize(torch.randn(2, length, 2, 4), dim=-1).requires_grad_()
    value = torch.randn_like(key, requires_grad=True)
    decay = (-20 * torch.rand_like(key)).requires_grad_()
    beta = torch.rand(2, length, 2, requires_grad=True)
    state = torch.randn(2, 2, 4, 4, requires_grad=True)
    args = (key, value, decay, beta, state)
    before = torch.get_rng_state()
    eager = delta_final_block(*args)
    compiled = torch.compile(delta_final_block, backend="aot_eager", dynamic=True, fullgraph=True)(
        *args
    )
    reference = delta_final_state(*args)
    torch.testing.assert_close(compiled, eager, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(compiled, reference, atol=3e-6, rtol=3e-5)
    for a, b in zip(
        torch.autograd.grad(eager.square().sum(), args, retain_graph=True),
        torch.autograd.grad(compiled.square().sum(), args),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    assert torch.equal(before, torch.get_rng_state())


def test_relay_kernel_option_is_explicit_and_legacy_identity_is_preserved():
    legacy = V2Architecture()
    assert "relay_kernel" not in legacy.payload()
    with pytest.raises(ValueError, match="unknown relay kernel"):
        replace(legacy, relay_kernel="unknown")
    with pytest.raises(ValueError, match="requires the relay"):
        replace(legacy, relay_kernel="inductor")
    relay = replace(
        legacy,
        attention="relay_full",
        graph_read_mode="relay",
        graph_layers=4,
        kda_layers=2,
        relay_kernel="inductor",
    )
    assert relay.payload()["relay_kernel"] == "inductor"


def test_compiled_backward_policy_is_scoped_and_matches_training_engine(monkeypatch):
    import torch._functorch.config as config

    original = config.backward_pass_autocast

    def fake_compile(fn, **kwargs):
        assert kwargs == {
            "fullgraph": True,
            "dynamic": True,
            "options": {"emulate_precision_casts": True},
        }

        def invoke(*args):
            assert config.backward_pass_autocast == "off"
            return args[0]

        return invoke

    monkeypatch.setattr(torch, "compile", fake_compile)
    operators.compiled_delta_block.cache_clear()
    try:
        x = torch.ones(1)
        assert operators.compiled_delta_block()(x) is x
        assert config.backward_pass_autocast == original
    finally:
        operators.compiled_delta_block.cache_clear()
