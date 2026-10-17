"""Native backend routing, backward wiring and historical checkpoint identity."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import weighted_gram
from gradpert.modeling.v2.model import GraDPertV2
from gradpert.modeling.v2.operators import RelayDeltaAttention, chunk_delta_final_state


def architecture():
    return V2Architecture(
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        streams=1,
        projector_hidden=16,
        projector_bottleneck=8,
        prototypes=11,
    )


def test_default_keeps_historical_payload_and_no_parameters_are_added():
    old = architecture()
    assert "weighted_gram_backend" not in old.payload()
    io = replace(old, weighted_gram_backend="triton_io")
    assert io.payload() == {**old.payload(), "weighted_gram_backend": "triton_io"}
    torch.manual_seed(19)
    reference = GraDPertV2(torch.randn(4, 256), old)
    torch.manual_seed(19)
    actual = GraDPertV2(torch.randn(4, 256), io)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], atol=0, rtol=0)
    operators = [m for m in actual.modules() if isinstance(m, RelayDeltaAttention)]
    assert len(operators) == 9  # 3 graph + 2 Cell + 2 Response self + 2 Response cross
    assert all(m.weighted_gram_backend == "triton_io" for m in operators)


@pytest.mark.parametrize(
    "change",
    [
        {"weighted_gram_backend": "unknown"},
        {"relay_kernel": "inductor"},
        {"relay_scan_chunk_size": 64},
        {"relay_sequence_chunk_size": 256},
        {"all_kda_prefix": True},
        {"attention_replacement": "softmax"},
    ],
)
def test_rejects_unsupported_native_execution_instead_of_silent_fallback(change):
    settings = {"weighted_gram_backend": "triton_io", **change}
    with pytest.raises(ValueError, match=r"Gram|weighted"):
        replace(architecture(), **settings)


@pytest.mark.parametrize("length", [1, 17, 33])
def test_native_dispatch_preserves_final_state_and_input_state_gradients(monkeypatch, length):
    calls = []

    def oracle(keys, gates):
        calls.append(keys.shape[-2])
        return weighted_gram.weighted_gram_reference(keys, gates)

    monkeypatch.setattr(weighted_gram, "io_weighted_gram", oracle)
    torch.manual_seed(781)
    k = torch.nn.functional.normalize(torch.randn(2, length, 2, 8), dim=-1).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    g = (-torch.rand_like(k)).requires_grad_()
    beta = torch.rand(2, length, 2, requires_grad=True)
    state = torch.randn(2, 2, 8, 8, requires_grad=True)
    inputs = k, v, g, beta, state
    expected = chunk_delta_final_state(*inputs)
    actual = chunk_delta_final_state(*inputs, gram_backend="triton_io")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(expected)
    for left, right in zip(
        torch.autograd.grad(actual, inputs, upstream),
        torch.autograd.grad(expected, inputs, upstream),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert calls == ([length] if length <= 32 else [32, 1])


def test_custom_backward_wiring_and_cpu_requires_explicit_backend(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gradpert.modeling.v2._weighted_gram_cuda",
        SimpleNamespace(forward=weighted_gram.weighted_gram_reference),
    )
    monkeypatch.setitem(
        sys.modules,
        "gradpert.modeling.v2._weighted_gram_io_cuda",
        SimpleNamespace(backward=weighted_gram.weighted_gram_backward),
    )
    key = torch.randn(2, 4, 17, 64, requires_grad=True)
    gate = (-torch.rand_like(key).cumsum(-2)).requires_grad_()
    expected = weighted_gram.weighted_gram_reference(key, gate)
    actual = weighted_gram._IOWeightedGram.apply(key, gate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, (key, gate), upstream),
        torch.autograd.grad(expected, (key, gate), upstream),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match="CUDA FP32"):
        weighted_gram.io_weighted_gram(key, gate)
