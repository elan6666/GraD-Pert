"""Graph mHC scope, residual parity and complete training-state continuation."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_relay_method import small_architecture
from test_restored_ffn import restored
from torch import nn

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.graph_mhc import GraphManifoldLayer
from gradpert.modeling.v2.model import RelayGraphLayer, SparseRead
from gradpert.modeling.v2.operators import ManifoldResidual
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def graph_inputs(streams=1):
    torch.manual_seed(872)
    memory = torch.randn(7, streams, 8)
    neighbors = torch.tensor([[i, (i + 1) % 7, 6] for i in range(7)])
    valid = torch.tensor([[True, True, False]] * 7)
    sources = torch.zeros(7, 3, 4)
    sources[:, 0, 3] = 1
    sources[:, 1, 0] = 1
    sources[:, 1, 1] = 1
    return memory, neighbors, valid, sources


def core_layer(kind, *, checkpointed=False, dropout=0):
    if kind == "kda":
        layer = RelayGraphLayer(8, 2, dropout, chunk_rows=3, checkpoint_chunks=checkpointed)
        layer.read.write_passes = 1
        return layer
    layer = SparseRead(8, 2, dropout, rank=4, ffn_type="swiglu", source_key_gate=True)
    with torch.no_grad():
        layer.source_bias.copy_(torch.randn_like(layer.source_bias) * 0.3)
        layer.source_key_gate.copy_(torch.randn_like(layer.source_key_gate) * 0.3)
    return layer


@pytest.mark.parametrize("kind", ["kda", "mla"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_single_stream_preserves_original_outputs_and_all_gradients(kind, checkpointed):
    memory, neighbors, valid, sources = graph_inputs()
    base = core_layer(kind, checkpointed=checkpointed).eval()
    wrapped = GraphManifoldLayer(copy.deepcopy(base), 8, 1).eval()
    x = memory.squeeze(1).clone().requires_grad_()
    y = memory.clone().requires_grad_()
    expected = (
        base(x, neighbors, valid, sources)
        if kind == "kda"
        else base(x, x, neighbors, valid, sources)
    )
    actual = wrapped(y, neighbors, valid, sources).squeeze(1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.testing.assert_close(y.grad.squeeze(1), x.grad, atol=0, rtol=0)
    for name, parameter in base.named_parameters():
        candidate = dict(wrapped.core.named_parameters())[name]
        assert parameter.grad is not None and candidate.grad is not None, name
        torch.testing.assert_close(candidate.grad, parameter.grad, atol=0, rtol=0)
    assert isinstance(wrapped.attention_route.norm, nn.Identity)
    assert isinstance(wrapped.ffn_route.norm, nn.Identity)


@pytest.mark.parametrize("kind", ["kda", "mla"])
def test_graph_attention_is_invariant_to_memory_outside_legal_neighbors(kind):
    memory, neighbors, valid, sources = graph_inputs(streams=4)
    layer = GraphManifoldLayer(core_layer(kind), 8, 4).eval()
    expected = layer(memory, neighbors, valid, sources)[0]
    changed = memory.clone()
    # Node 6 is present as masked padding; all changed nodes are outside node 0's edges.
    changed[2:] = torch.randn_like(changed[2:]) * 50
    actual = layer(changed, neighbors, valid, sources)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    input_with_grad = memory.clone().requires_grad_()
    layer(input_with_grad, neighbors, valid, sources)[0].square().sum().backward()
    assert torch.count_nonzero(input_with_grad.grad[2:]) == 0
    assert input_with_grad.grad[:2].abs().sum() > 0


@pytest.mark.parametrize("kind", ["kda", "mla"])
def test_four_streams_have_useful_finite_router_gradients(kind):
    memory, neighbors, valid, sources = graph_inputs(streams=4)
    core = core_layer(kind, checkpointed=True)
    layer = GraphManifoldLayer(core, 8, 4)
    assert layer.core is core
    output = layer(memory, neighbors, valid, sources)
    assert output.shape == memory.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for route in (layer.attention_route, layer.ffn_route):
        for name, parameter in route.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_checkpoint_chunks_preserve_router_and_core_gradients_with_randomness():
    memory, neighbors, valid, sources = graph_inputs(streams=4)
    base = GraphManifoldLayer(core_layer("kda", dropout=0.2), 8, 4)
    checkpointed = copy.deepcopy(base)
    checkpointed.core.checkpoint_chunks = True
    x = memory.clone().requires_grad_()
    y = memory.clone().requires_grad_()
    torch.manual_seed(77)
    expected = base(x, neighbors, valid, sources)
    torch.manual_seed(77)
    actual = checkpointed(y, neighbors, valid, sources)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=0, rtol=0)
    reference = dict(base.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        assert parameter.grad is not None and reference[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, reference[name].grad, atol=0, rtol=0)


def mhc_objective(enabled=True, *, streams=4):
    base, batch = restored("BR")
    architecture = replace(base.student.options, streams=streams, graph_mhc=enabled)
    seeds = base.student.graph.embedding.weight.detach().clone()
    torch.manual_seed(431)
    model = GraDPertV2(seeds, architecture)
    rng_state = torch.get_rng_state().clone()
    return JointObjective(model, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)), batch, rng_state


def test_graph_routes_preserve_common_initialization_rng_and_teacher_homology():
    base, _, base_rng = mhc_objective(False)
    candidate, _, candidate_rng = mhc_objective()
    assert torch.equal(base_rng, candidate_rng)
    shared = base.student.state_dict()
    for name, value in candidate.student.state_dict().items():
        if ".attention_route." in name or ".ffn_route." in name:
            continue
        torch.testing.assert_close(value, shared[name.replace(".core.", ".")], atol=0, rtol=0)
    for model in (candidate.student, candidate.teacher):
        assert len(model.graph.layers) == 4
        assert all(isinstance(layer, GraphManifoldLayer) for layer in model.graph.layers)
        routes = [
            module for module in model.graph.modules() if isinstance(module, ManifoldResidual)
        ]
        assert len(routes) == 8
        assert all(route.streams == 4 and isinstance(route.norm, nn.Identity) for route in routes)
    for name, parameter in candidate.student.named_parameters():
        teacher = dict(candidate.teacher.named_parameters())[name]
        torch.testing.assert_close(parameter, teacher, atol=0, rtol=0)
        assert not teacher.requires_grad


def test_graph_route_private_initialization_does_not_reseed_cuda_generators(monkeypatch):
    # Catch CUDA reseeding even on CPU CI, without allocating a GPU tensor.
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    mhc_objective(False)
    baseline_calls = calls.copy()
    calls.clear()
    mhc_objective()
    assert calls == baseline_calls


def test_full_graph_single_stream_matches_original_output_and_parameter_gradients():
    base, batch, _ = mhc_objective(False, streams=1)
    candidate, _, _ = mhc_objective(streams=1)
    base.eval()
    candidate.eval()
    expected, _ = base._batch_graph(base.student, batch)
    actual, _ = candidate._batch_graph(candidate.student, batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    reference = dict(base.student.graph.named_parameters())
    for name, parameter in candidate.student.graph.named_parameters():
        original = reference[name.replace(".core.", ".")]
        if original.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, original.grad, atol=0, rtol=0)


def test_multistep_optimizer_ema_center_and_exact_checkpoint_continuation(tmp_path):
    objective, batch, _ = mhc_objective()
    optimizer = V2Optimizer(objective.student, 1e-3, 0)
    student_before = copy.deepcopy(objective.student.state_dict())
    teacher_before = copy.deepcopy(objective.teacher.state_dict())
    rng = np.random.default_rng(6)
    for _ in range(2):
        metrics = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(value) for value in metrics.values())
        assert not objective.pending
    for name, parameter in objective.student.named_parameters():
        if ".attention_route.route.weight" in name or ".ffn_route.route.weight" in name:
            assert not torch.equal(parameter, student_before[name]), name
            assert not torch.equal(
                dict(objective.teacher.named_parameters())[name], teacher_before[name]
            )
    assert objective.ssl1_cls_center.abs().sum() > 0 and objective.ssl2_cls_center.abs().sum() > 0
    assert all(parameter.grad is None for parameter in objective.teacher.parameters())
    assert torch.equal(
        objective.student.graph.embedding.weight, student_before["graph.embedding.weight"]
    )
    path = tmp_path / "last.pt"
    identity = {"arm": "GM1"}
    save_checkpoint(
        path, objective, optimizer, identity=identity, progress={"step": 2}, generator=rng
    )
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    reference = copy.deepcopy(objective.state_dict())
    resumed, _, _ = mhc_objective()
    resumed_optimizer = V2Optimizer(resumed.student, 1e-3, 0)
    loaded = load_checkpoint(path, resumed, resumed_optimizer, identity=identity, generator=rng)
    assert loaded["step"] == 2
    actual = optimizer_step(
        resumed, resumed_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, reference[name], atol=0, rtol=0)


def test_legacy_payload_and_unsupported_conditional_graph_combination():
    assert "graph_mhc" not in V2Architecture().payload()
    with pytest.raises(ValueError, match="graph mHC"):
        replace(
            small_architecture(),
            graph_mhc=True,
            control_conditioned_graph=True,
            mlp_profile="unified",
            graph_source_key_gate=True,
        )
