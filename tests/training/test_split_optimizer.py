from copy import deepcopy

import pytest
import torch
from torch import nn

from gradpert.modeling.encoders import _SparseGraphTransformerLayer
from gradpert.training.optimizers import SplitMatrixAdamW, parameter_routes


def model():
    result = nn.Module()
    result.student_encoder = _SparseGraphTransformerLayer(
        hidden_dim=8, head_count=2, dropout=0, add_local_message_passing=False
    )
    result.student_encoder.gene_embeddings = nn.Parameter(torch.randn(9, 8))
    result.expression_decoder = nn.Module()
    result.expression_decoder.network = nn.Sequential(
        nn.Linear(8, 8), nn.Identity(), nn.Identity(), nn.Identity(), nn.Linear(8, 9)
    )
    result.teacher_encoder = deepcopy(result.student_encoder).requires_grad_(False)
    return result


def gradients(net, index):
    for p in net.parameters():
        if p.requires_grad:
            p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape) / p.numel() + index


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree(x, y)
    else:
        assert a == b


def test_routes_are_complete_and_head_axis_matches_native_projection():
    net = model()
    routes = parameter_routes(net)
    assert {r["name"] for r in routes} == {n for n, p in net.named_parameters() if p.requires_grad}
    assert len(routes) == len({r["name"] for r in routes})
    by_name = {r["name"]: r for r in routes}
    for name in ("query", "key", "value"):
        assert by_name[f"student_encoder.{name}.weight"]["heads"] == 2
    assert by_name["student_encoder.edge.weight"]["heads"] == 1
    assert by_name["student_encoder.gene_embeddings"]["optimizer"] == "adamw"
    assert by_name["expression_decoder.network.4.weight"]["optimizer"] == "adamw"


def test_split_step_matches_independent_torch_heads_and_keeps_gradients():
    net = model()
    opt = SplitMatrixAdamW(net, lr=0.001)
    gradients(net, 1)
    q = net.student_encoder.query.weight
    saved_grad = q.grad.clone()
    expected = [nn.Parameter(t.clone()) for t in q.detach().chunk(2)]
    refs = [
        torch.optim.Muon([p], lr=0.001, weight_decay=0, adjust_lr_fn="match_rms_adamw")
        for p in expected
    ]
    for p, g, ref in zip(expected, saved_grad.chunk(2), refs, strict=True):
        p.grad = g.clone()
        ref.step()
    teacher = deepcopy(net.teacher_encoder.state_dict())
    opt.step()
    assert torch.equal(q, torch.cat(expected))
    assert torch.equal(q.grad, saved_grad)
    assert_tree(teacher, net.teacher_encoder.state_dict())
    assert opt.step_count == 1


def test_mixed_resume_exact_with_lr_change_and_missing_gradient():
    net = model()
    opt = SplitMatrixAdamW(net, lr=0.001)
    gradients(net, 1)
    opt.step()
    restored = model()
    restored.load_state_dict(net.state_dict())
    resumed = SplitMatrixAdamW(restored, lr=0.001)
    resumed.load_state_dict(deepcopy(opt.state_dict()))
    for index, lr in ((2, 0), (3, 0.0002), (4, 0.001)):
        for m, o in ((net, opt), (restored, resumed)):
            gradients(m, index)
            m.student_encoder.query.weight.grad = None
            o.param_groups[0]["lr"] = lr
            o.step()
        assert_tree(net.state_dict(), restored.state_dict())
        assert_tree(opt.state_dict(), resumed.state_dict())


def test_resume_rejects_route_change_before_mutation():
    opt = SplitMatrixAdamW(model(), lr=0.001)
    state = deepcopy(opt.state_dict())
    state["routes"][0]["heads"] = 99
    with pytest.raises(ValueError, match="routing"):
        opt.load_state_dict(state)


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_zero_and_rank_deficient_gradients_are_finite(fill):
    net = model()
    opt = SplitMatrixAdamW(net, lr=0.001)
    before = deepcopy(net.state_dict())
    for p in net.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, fill)
    opt.step()
    assert all(torch.isfinite(p).all() for p in net.parameters())
    if fill == 0:
        assert_tree(before, net.state_dict())


def test_update_health_does_not_change_state_or_rng():
    plain = model()
    observed = model()
    observed.load_state_dict(plain.state_dict())
    a = SplitMatrixAdamW(plain, lr=0.001)
    b = SplitMatrixAdamW(observed, lr=0.001, health_interval=2)
    for step in range(3):
        gradients(plain, step + 1)
        gradients(observed, step + 1)
        rng = torch.get_rng_state().clone()
        a.step()
        b.step()
        assert torch.equal(rng, torch.get_rng_state())
        assert_tree(plain.state_dict(), observed.state_dict())
        assert_tree(a.state_dict(), b.state_dict())
    assert [s["optimizer_step"] for s in b.update_health] == [0, 2]
    assert a.update_health == []
    for sample in b.update_health:
        assert sample["host_dispatch_ms"] >= 0
        for values in sample["groups"].values():
            assert values["weight_l2"] > 0
            assert values["update_l2"] > 0


def test_attention_health_preserves_forward_backward_rng_and_is_bounded():
    layer = model().student_encoder
    layer.dropout = 0.2
    observed = deepcopy(layer)
    observed.capture_attention_health = True
    x = torch.randn(4, 8)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    features = torch.randn(4, 8)
    rng = torch.get_rng_state().clone()
    expected = layer(x, edges, features, edges)
    expected.square().sum().backward()
    state_after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = observed(x, edges, features, edges)
    actual.square().sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(state_after, torch.get_rng_state())
    for a, b in zip(layer.parameters(), observed.parameters(), strict=True):
        assert_tree(a.grad, b.grad)
    assert observed.attention_health["finite"]
    assert observed.attention_health["mean_target_head_entropy"] == 0
    assert observed.capture_attention_health is False
    assert layer.attention_health is None
