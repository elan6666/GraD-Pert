import copy
import math
from dataclasses import replace

import pytest
import torch
from test_relay_method import assert_tree_close, relay_training_fixture

from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.attention_ablation import ReplacementAttention
from gradpert.modeling.v2.model import RelayGraphLayer, RelayResponseEncoder
from gradpert.modeling.v2.operators import RelayDeltaAttention, TokenEncoder, delta_scan
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def test_position_readout_matches_recurrent_reads_and_restores_gene_identity():
    torch.manual_seed(31)
    layer = RelayDeltaAttention(8, 2).eval()
    layer.write_passes, layer.self_readout, layer.sequence_chunk_size = 1, "position", 2
    x = torch.randn(2, 6, 8, requires_grad=True)
    order = torch.stack((torch.randperm(5), torch.randperm(5)))
    ordered = layer._ordered(x[:, :5], order)
    q = torch.nn.functional.normalize(layer.query(ordered).reshape(2, 5, 2, 4).float(), dim=-1) / 2
    writes = tuple(layer._ordered(t, order) for t in layer._project_writes(x[:, :5]))
    y = delta_scan(q, *writes)
    expected = layer._ordered(
        layer.output(
            layer.head_norm(y).flatten(-2) * torch.nn.functional.silu(layer.output_gate(ordered))
        ),
        order.argsort(-1),
    )
    actual = layer(x, order=order)[:, :5]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, layer.key.weight), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (x, layer.key.weight)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    # A gene's prefix output cannot read the tail CLS.
    changed = x.detach().clone()
    changed[:, -1] += 100
    torch.testing.assert_close(actual, layer(changed, order=order)[:, :5])


@pytest.mark.parametrize("kind", ["softmax", "retention"])
def test_native_replacement_cross_formula_and_gradients(kind):
    torch.manual_seed(17)
    layer = ReplacementAttention(8, 2, 0, kind)
    x, control = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 5, 8, requires_grad=True)
    q = layer.query(x).reshape(2, 3, 2, 4).transpose(1, 2)
    k, v = (
        projection(control).reshape(2, 5, 2, 4).transpose(1, 2)
        for projection in (layer.key, layer.value)
    )
    if kind == "softmax":
        y = (q @ k.transpose(-1, -2) / 2).softmax(-1) @ v
    else:
        # Explicit token sum checks the official ReLU feature map and both
        # 1/sqrt(d_h) factors, rather than calling the native implementation.
        y = sum(
            (q.relu() @ k[:, :, j : j + 1].relu().transpose(-1, -2)) @ v[:, :, j : j + 1] / 4
            for j in range(5)
        )
        y = y / (y.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12))
        y = y * torch.nn.functional.silu(layer.output_gate(x)).reshape(2, 3, 2, 4).transpose(1, 2)
    expected = layer.output(y.transpose(1, 2).flatten(-2))
    actual = layer(x, control)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, control), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (x, control)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("kind", ["softmax", "retention"])
def test_replacement_graph_never_reads_illegal_neighbors(kind):
    torch.manual_seed(26)
    layer = ReplacementAttention(8, 2, 0, kind)
    memory = torch.randn(4, 8, requires_grad=True)
    neighbors = torch.tensor([[0, 1, -1]])
    valid = neighbors >= 0
    sources = torch.zeros(1, 3, 4)
    sources[0, 0, 3], sources[0, 1, 0] = 1, 1
    result = layer.graph(memory[:1], memory, neighbors, valid, sources)
    result.square().sum().backward()
    assert memory.grad[:2].abs().sum() > 0 and torch.equal(
        memory.grad[2:], torch.zeros_like(memory.grad[2:])
    )
    changed = memory.detach().clone()
    changed[2:] += 100
    torch.testing.assert_close(result, layer.graph(changed[:1], changed, neighbors, valid, sources))


@pytest.mark.parametrize("variant", ["K1", "K2", "A1", "A2"])
def test_complete_two_update_checkpoint_parity_teacher_centers_and_optimizer(variant):
    fixture, batch = relay_training_fixture()
    changes = {
        "K1": {"relay_passes": 2},
        "K2": {"self_readout": "position"},
        "A1": {"attention_replacement": "softmax"},
        "A2": {"attention_replacement": "retention"},
    }[variant]
    architecture = replace(
        fixture.student.options,
        relay_passes=1,
        graph_source_key_gate=True,
        **({} if variant == "K1" else changes),
    )
    if variant == "K1":
        architecture = replace(architecture, **changes)
    objective = JointObjective(
        GraDPertV2(fixture.student.graph.embedding.weight.detach().clone(), architecture),
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
        loss_reduction="row_mean",
    )
    plain = copy.deepcopy(objective)
    for module in plain.modules():
        if isinstance(module, (TokenEncoder, RelayResponseEncoder)):
            module.checkpoint_layers = False
        if isinstance(module, RelayGraphLayer):
            module.checkpoint_chunks = False
    if variant.startswith("A"):
        assert not any(isinstance(m, RelayDeltaAttention) for m in objective.modules())
        assert all(
            isinstance(layer.sublayer, ReplacementAttention)
            for layer in objective.student.response.cross_layers
        )
    elif variant == "K2":
        assert all(
            layer.sublayer.self_readout == "final"
            for layer in objective.student.response.cross_layers[:2]
        )
        assert all(
            layer.read.self_readout == "final" for layer in objective.student.graph.layers[:3]
        )
    snapshots = []
    for network in (objective, plain):
        opt = V2Optimizer(network.student, 0.001, 0)
        initial = copy.deepcopy(network.student.state_dict())
        torch.manual_seed(83)
        for _ in range(2):
            terms = optimizer_step(
                network, opt, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
            )
            assert all(math.isfinite(v) for v in terms.values())
            assert not network.pending
            assert all(p.grad is None for p in network.teacher.parameters())
        assert any(
            not torch.equal(initial[k], value) for k, value in network.student.state_dict().items()
        )
        assert network.ssl1_cls_center.abs().sum() > 0 and network.ssl2_cls_center.abs().sum() > 0
        snapshots.append(
            (
                terms,
                copy.deepcopy(network.state_dict()),
                copy.deepcopy(opt.state_dict()),
                torch.get_rng_state(),
            )
        )
    assert_tree_close(*snapshots)
