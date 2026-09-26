import pytest
import torch

from gradpert.modeling.v2.weighted_gram import weighted_gram_backward, weighted_gram_reference


@pytest.mark.parametrize("length", [1, 3, 17, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighted_gram_backward_matches_autograd(length, dtype):
    torch.manual_seed(25)
    keys = torch.randn(2, length, 4, 8, dtype=dtype).transpose(1, 2).requires_grad_()
    gates = (-torch.rand_like(keys).cumsum(-2)).requires_grad_()
    result = weighted_gram_reference(keys, gates)
    upstream = torch.randn_like(result)
    expected = torch.autograd.grad(result, (keys, gates), upstream)
    actual = weighted_gram_backward(keys.detach(), gates.detach(), upstream)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert torch.count_nonzero(result.triu()) == 0


def test_weighted_gram_is_original_delta_block_triangle():
    torch.manual_seed(7)
    key = torch.randn(2, 3, 17, 64, requires_grad=True)
    gates = (-torch.rand_like(key).cumsum(-2)).requires_grad_()
    causal = torch.ones(17, 17, dtype=torch.bool).tril()
    decay = (gates.unsqueeze(-2) - gates.unsqueeze(-3)).masked_fill(~causal[..., None], 0).exp()
    original = (key.unsqueeze(-2) * key.unsqueeze(-3) * decay).sum(-1).tril(-1)
    actual = weighted_gram_reference(key, gates)
    torch.testing.assert_close(original, actual, atol=0, rtol=0)
    u = torch.randn_like(actual)
    original_grads = torch.autograd.grad(original, (key, gates), u)
    candidate_grads = weighted_gram_backward(key.detach(), gates.detach(), u)
    for a, b in zip(original_grads, candidate_grads, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_unused_positive_exponents_cannot_overflow():
    keys = torch.ones(1, 1, 4, 2)
    gates = -torch.arange(4).reshape(1, 1, 4, 1).expand_as(keys) * 1000.0
    result = weighted_gram_reference(keys, gates)
    gradients = weighted_gram_backward(keys, gates, torch.ones_like(result))
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("length", [1, 17, 33])
def test_fused_dispatch_preserves_final_state_and_all_gradients(monkeypatch, length):
    from gradpert.modeling.v2 import weighted_gram
    from gradpert.modeling.v2.operators import chunk_delta_final_state

    calls = []

    def reference(keys, gates):
        calls.append(keys.shape[-2])
        return weighted_gram_reference(keys, gates)

    monkeypatch.setattr(weighted_gram, "fused_weighted_gram", reference)
    torch.manual_seed(31)
    k = torch.nn.functional.normalize(torch.randn(2, length, 2, 8), dim=-1).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    g = (-torch.rand_like(k)).requires_grad_()
    beta = torch.rand(2, length, 2, requires_grad=True)
    initial = torch.randn(2, 2, 8, 8, requires_grad=True)
    inputs = (k, v, g, beta, initial)
    expected = chunk_delta_final_state(*inputs)
    actual = chunk_delta_final_state(*inputs, fused_gram=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, inputs, upstream),
        torch.autograd.grad(expected, inputs, upstream),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert calls == ([length] if length <= 32 else [32, 1])
