"""Opt-in precision changes arithmetic only, preserving historical model identity."""

import copy
from dataclasses import replace

import pytest
import torch
from test_components import fixture
from torch import nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.model import Projector, _head_precision_context
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import routes


@pytest.mark.parametrize("name", ["ssl_head_precision", "prediction_precision"])
@pytest.mark.parametrize("value", [True, 1, None, "fp32", "FLOAT32"])
def test_precision_values_are_strict(name, value):
    with pytest.raises(ValueError, match=name):
        V2Architecture(**{name: value})


def test_default_identity_rng_state_and_optimizer_routes():
    old, _ = fixture()
    arch = old.options
    assert "ssl_head_precision" not in arch.payload()
    assert "prediction_precision" not in arch.payload()
    seeds = old.graph.embedding.weight.detach().clone()
    models, rngs = [], []
    for opts in (arch, replace(arch, ssl_head_precision="float32", prediction_precision="float32")):
        torch.manual_seed(42)
        models.append(GraDPertV2(seeds, opts))
        rngs.append(torch.get_rng_state())
        assert V2Architecture.parse(opts.payload()) == opts
    assert torch.equal(*rngs)
    assert routes(models[0]) == routes(models[1])
    assert list(models[0].state_dict()) == list(models[1].state_dict())
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[name])
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    x = torch.randn(3, arch.width)
    head = old.ssl2_cls
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = head.prototypes(F.normalize(head.features(x).float(), dim=-1))
        assert torch.equal(head(x), expected)
        assert head(x).dtype == torch.bfloat16


@pytest.mark.parametrize("with_features", [False, True])
def test_ssl32_intermediates_and_gradients_match_direct_float32(with_features):
    arch = replace(fixture()[0].options, ssl_head_precision="float32")
    head = Projector(arch)
    reference = copy.deepcopy(head)
    x = torch.randn(5, arch.width, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    seen = []
    handles = [
        m.register_forward_hook(lambda m, i, o: seen.append((i[0].dtype, o.dtype)))
        for m in head.modules()
        if isinstance(m, nn.Linear)
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(x, return_features=with_features)
        if with_features:
            features, logits = out
            assert features.dtype == torch.float32
        else:
            logits = out
    for handle in handles:
        handle.remove()
    assert seen == [(torch.float32, torch.float32)] * 3
    expected = F.linear(
        F.normalize(reference.features(xr).float(), dim=-1), reference.prototypes.weight
    )
    assert torch.equal(logits, expected)
    logits.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, xr.grad)
    for actual, ref in zip(head.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual.grad, ref.grad)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert head.forward_with_features(x)[1].dtype == torch.float32


def test_prediction32_decode_gradient_matches_float32():
    old, _ = fixture()
    model = GraDPertV2(
        old.graph.embedding.weight.detach(), replace(old.options, prediction_precision="float32")
    )
    reference = copy.deepcopy(model.prediction)
    x = torch.randn(2, 4, model.options.width, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.decode_score(x)
    expected = reference(xr).squeeze(-1)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, xr.grad)
    for a, b in zip(model.prediction.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)


def test_teacher_ema_restore_and_inactive_ssl1_node():
    old, batch = fixture()
    model = GraDPertV2(
        old.graph.embedding.weight.detach(),
        replace(old.options, ssl_head_precision="float32", prediction_precision="float32"),
    )
    obj = JointObjective(model, ssl1_weights=(0.8, 0.0, 0.1))
    before = {n: p.detach().clone() for n, p in obj.teacher.named_parameters()}
    assert all(not p.requires_grad for p in obj.teacher.parameters())
    assert obj.teacher.options == model.options
    calls = []
    handles = [
        side.ssl1_node.register_forward_hook(lambda *args: calls.append(True))
        for side in (obj.student, obj.teacher)
    ]
    obj(batch)[0].backward()
    for handle in handles:
        handle.remove()
    assert not calls
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    obj.commit_statistics(0.9)
    for name, p in obj.teacher.named_parameters():
        torch.testing.assert_close(p, before[name].lerp(dict(model.named_parameters())[name], 0.1))
        assert p.grad is None
    restored = JointObjective(copy.deepcopy(model), ssl1_weights=(0.8, 0.0, 0.1))
    restored.load_state_dict(obj.state_dict(), strict=True)
    for name, value in obj.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_cuda_tf32_rejected_without_changing_flags(monkeypatch):
    class CudaInput:
        device = torch.device("cuda")

    original = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="TF32 disabled"):
        _head_precision_context(CudaInput(), "float32")
    assert torch.backends.cuda.matmul.allow_tf32 is True
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", original)


def test_nonrelay_response_decode_runs_entire_prediction_head_float32():
    old, batch = fixture()
    model = GraDPertV2(
        old.graph.embedding.weight.detach(), replace(old.options, prediction_precision="float32")
    )
    calls = []
    handles = [
        m.register_forward_hook(lambda m, i, o: calls.append((i[0].dtype, o.dtype)))
        for m in model.prediction.modules()
        if isinstance(m, nn.Linear)
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gene, condition = JointObjective(model)._graph(model, batch.graph, False)
        output = model.encode_response(gene[:4], batch.control, condition)
    for handle in handles:
        handle.remove()
    assert calls == [(torch.float32, torch.float32)] * 2
    assert output["prediction"].dtype == torch.float32


def test_protected_prediction_head_retains_input_gradients_only():
    from test_population_head_gradient import model as protected_model

    model = protected_model()
    model.options = replace(model.options, prediction_precision="float32")
    x = torch.randn(2, 4, 2 * model.options.width, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model.decode_score(x, prediction_head_grad=False)
    assert out.dtype == torch.float32
    out.square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.prediction.parameters())


def test_fp8_option_validation_and_default_identity():
    assert "ffn_compute_precision" not in V2Architecture().payload()
    for value in (True, None, "fp8", 0):
        with pytest.raises(ValueError, match="ffn_compute_precision"):
            V2Architecture(ffn_compute_precision=value)
    with pytest.raises(ValueError, match="restored profile and gelu"):
        V2Architecture(ffn_compute_precision="fp8_hybrid")
