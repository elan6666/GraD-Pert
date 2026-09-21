import copy

import pytest
import torch

from gradpert.modeling.v2 import GraDPertV2, V2Architecture
from gradpert.modeling.v2.operators import delta_scan, sinkhorn
from gradpert.training.v2.objective import CellView, GraphView, JointObjective, TrainingBatch


def fixture():
    torch.manual_seed(10)
    options = V2Architecture(
        width=8,
        heads=2,
        latent_rank=4,
        streams=2,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
        dropout=0,
        checkpoint_layers=False,
    )
    model = GraDPertV2(torch.randn(6, 5), options)
    ids = torch.arange(6)
    neighbors = torch.stack((ids, ids.roll(1)), -1)
    sources = torch.zeros(6, 2, 4)
    sources[:, 0, 3] = 1
    sources[:, 1, 0] = 1
    view = GraphView(
        ids,
        neighbors,
        torch.ones_like(neighbors, dtype=torch.bool),
        sources,
        torch.tensor([[0], [1]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([4]),
    )
    control = torch.randn(2, 4)
    mask = torch.zeros_like(control, dtype=torch.bool)
    mask[:, 1] = True
    cell = CellView(torch.arange(4), mask)
    batch = TrainingBatch(
        view,
        torch.arange(4),
        control,
        control + 0.2,
        torch.tensor([0, 1]),
        (view, view, view, view, view, view),
        (cell, cell, cell),
    )
    return model, batch


def test_scan_matches_closed_matrix_transition_and_gradient():
    torch.manual_seed(1)
    q, k, v = [torch.randn(2, 5, 2, 3, requires_grad=True) for _ in range(3)]
    k = torch.nn.functional.normalize(k, dim=-1)
    decay = -torch.rand_like(k)
    beta = torch.rand(2, 5, 2)
    output = delta_scan(q, k, v, decay, beta)
    state = torch.zeros(2, 2, 3, 3)
    rows = []
    for t in range(5):
        diag = torch.diag_embed(decay[:, t].exp())
        key = k[:, t]
        erase = torch.eye(3) - beta[:, t, :, None, None] * key.unsqueeze(-1) * key.unsqueeze(-2)
        write = beta[:, t, :, None, None] * key.unsqueeze(-1) * v[:, t].unsqueeze(-2)
        state = erase @ diag @ state + write
        rows.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2))
    expected = torch.stack(rows, 1)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    a = torch.autograd.grad(output.sum(), q, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), q)[0]
    torch.testing.assert_close(a, b)


def test_sinkhorn_conserves_rows_columns():
    matrix = sinkhorn(torch.randn(3, 4, 4) * 0.2)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(3, 4))
    torch.testing.assert_close(matrix.sum(-2), torch.ones(3, 4))


def test_expression_mask_blocks_hidden_value():
    model, batch = fixture()
    model.eval()
    graph, condition = JointObjective(model)._graph(model, batch.graph, False)
    mask = batch.cell_views[0].mask
    a = model.encode_response(graph[:4], batch.control, condition, mask)
    changed = batch.control.clone()
    changed[mask] += 100
    b = model.encode_response(graph[:4], changed, condition, mask)
    torch.testing.assert_close(a["response_cls"], b["response_cls"])
    torch.testing.assert_close(a["delta"], b["delta"])
    # Raw control residual is deliberately outside the distillation representation.
    assert not torch.equal(a["prediction"], b["prediction"])


@pytest.mark.parametrize("l1,l2", [(0, 0), (1, 0), (0, 0.1), (1, 0.1)])
def test_all_loss_routes_and_teacher_update(l1, l2):
    model, batch = fixture()
    objective = JointObjective(model, l1, l2).train()
    teacher_before = copy.deepcopy(objective.teacher.state_dict())
    loss, _metrics = objective(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert model.graph.embedding.weight.grad.abs().sum() > 0
    for name, tensor in objective.teacher.state_dict().items():
        torch.testing.assert_close(tensor, teacher_before[name])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    objective.commit_statistics(0.9)
    assert not objective.pending
    for name, tensor in objective.teacher.state_dict().items():
        expected = 0.9 * teacher_before[name] + 0.1 * model.state_dict()[name]
        torch.testing.assert_close(tensor, expected)
    restored = copy.deepcopy(objective)
    restored.load_state_dict(objective.state_dict())
    objective.eval()
    restored.eval()
    torch.testing.assert_close(objective(batch)[0], restored(batch)[0])


def test_unknown_config_rejected():
    with pytest.raises(ValueError, match="unknown"):
        V2Architecture.parse({"mysterious_option": True})


def test_real_optimizer_checkpoint_resume(tmp_path):
    import numpy as np

    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.optimizer import V2Optimizer

    model, batch = fixture()
    objective = JointObjective(model, 0, 0)
    optimizer = V2Optimizer(model, 0.001, 0.0)
    rng = np.random.default_rng(9)

    def step():
        optimizer.zero_grad()
        loss, _ = objective(batch)
        loss.backward()
        optimizer.step(0.001)
        objective.commit_statistics(0.99)
        return loss.detach()

    step()
    path = tmp_path / "last.pt"
    identity = {"config_hash": "test", "training_commit": "test"}
    save_checkpoint(
        path, objective, optimizer, identity=identity, progress={"step": 1}, generator=rng
    )
    expected_numpy = np.random.random(3)
    expected = step()
    state = copy.deepcopy(objective.state_dict())
    load_checkpoint(path, objective, optimizer, identity=identity, generator=rng)
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    actual = step()
    torch.testing.assert_close(expected, actual)
    for name, tensor in objective.state_dict().items():
        torch.testing.assert_close(tensor, state[name])
    legacy = torch.load(path, weights_only=False)
    del legacy["numpy_rng"]
    torch.save(legacy, tmp_path / "legacy-v2.pt")
    before = np.random.get_state()
    load_checkpoint(
        tmp_path / "legacy-v2.pt", objective, optimizer, identity=identity, generator=rng
    )
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    torch.save({"model_version": "v1"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="v2 checkpoint"):
        load_checkpoint(tmp_path / "old.pt", objective, optimizer, identity=identity, generator=rng)


@pytest.mark.parametrize("length", [1, 7, 17, 33])
def test_chunk_delta_matches_recurrence_full_gradients(length):
    from gradpert.modeling.v2.operators import chunk_delta_scan

    torch.manual_seed(42)
    tensors = [torch.randn(2, length, 2, 4, requires_grad=True) for _ in range(3)]
    q, k, v = tensors
    k = torch.nn.functional.normalize(k, dim=-1)
    gates = -torch.rand_like(k, requires_grad=True)
    beta = torch.rand(2, length, 2, requires_grad=True)
    inputs = [q, tensors[1], v, gates, beta]
    expected = delta_scan(q, k, v, gates, beta)
    actual = chunk_delta_scan(q, k, v, gates, beta)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    a = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-4)


def test_sparse_graph_read_matches_dense_masked_reference():
    from gradpert.modeling.v2.model import SparseRead

    torch.manual_seed(3)
    layer = SparseRead(8, 2, 0).eval()
    query = torch.randn(3, 8, requires_grad=True)
    memory = torch.randn(5, 8, requires_grad=True)
    neighbors = torch.tensor([[0, 2], [1, 4], [2, 3]])
    valid = torch.ones(3, 2, dtype=torch.bool)
    sources = torch.zeros(3, 2, 4)
    sources[..., 0] = 1
    output = layer(query, memory, neighbors, valid, sources)
    q = layer.query(layer.norm1(query)).reshape(3, 2, 4)
    k = layer.key(memory).reshape(5, 2, 4)
    v = layer.value(memory).reshape(5, 2, 4)
    score = torch.einsum("nhd,mhd->nhm", q, k) / 2
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask.scatter_(1, neighbors, False)
    score = score.masked_fill(mask[:, None], float("-inf"))
    weight = score.softmax(-1)
    read = torch.einsum("nhm,mhd->nhd", weight, v).flatten(-2)
    expected = query + layer.output(read)
    expected = expected + layer.ffn(layer.norm2(expected))
    torch.testing.assert_close(output, expected)
    assert torch.equal(weight.masked_select(mask[:, None].expand_as(weight)), torch.zeros(18))
    a = torch.autograd.grad(output.sum(), memory, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), memory)[0]
    torch.testing.assert_close(a, b)


def test_accumulated_prediction_gradients_match_full_batch(monkeypatch):
    from gradpert.training.v2.engine import optimizer_step
    from gradpert.training.v2.optimizer import V2Optimizer

    model, batch = fixture()
    other = copy.deepcopy(model)
    a, b = JointObjective(model, 0, 0), JointObjective(other, 0, 0)
    oa, ob = V2Optimizer(model, 0.001, 0.0), V2Optimizer(other, 0.001, 0.0)
    gradients = []

    def recording_step(optimizer, module):
        original = optimizer.step

        def step(lr):
            gradients.append(
                {
                    n: p.grad.detach().clone()
                    for n, p in module.named_parameters()
                    if p.grad is not None
                }
            )
            original(lr)

        return step

    monkeypatch.setattr(oa, "step", recording_step(oa, model))
    monkeypatch.setattr(ob, "step", recording_step(ob, other))
    ma = optimizer_step(a, oa, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False)
    mb = optimizer_step(b, ob, batch, microbatch=1, lr=0.001, momentum=0.99, bf16=False)
    assert ma["prediction"] == pytest.approx(mb["prediction"], rel=1e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-7, rtol=1e-4)
    assert oa.steps == ob.steps == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_per_gene_ablation_has_no_cross_gene_expression_path():
    from dataclasses import replace

    base, batch = fixture()
    model = GraDPertV2(torch.randn(6, 5), replace(base.options, attention="per_gene")).eval()
    graph, condition = JointObjective(model)._graph(model, batch.graph, False)
    first = model.encode_response(graph[:4], batch.control, condition)
    changed = batch.control.clone()
    changed[:, 0] += 2
    second = model.encode_response(graph[:4], changed, condition)
    torch.testing.assert_close(
        first["response_tokens"][:, 1:], second["response_tokens"][:, 1:], rtol=0, atol=0
    )
    assert not torch.equal(first["response_cls"], second["response_cls"])


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize(
    "cls_on,node_on,spread_on",
    [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)],
)
def test_component_ablations_skip_disabled_heads_and_centers(stage, cls_on, node_on, spread_on):
    model, batch = fixture()
    weights = (0.8 * cls_on, 0.4 * node_on, 0.1 * spread_on)
    kwargs = {f"ssl{stage}_weights": weights}
    objective = JointObjective(model, lambda1=int(stage == 1), lambda2=0.1 * (stage == 2), **kwargs)

    def reject(*args):
        raise AssertionError("disabled head executed")

    for role in ("student", "teacher"):
        module = getattr(objective, role)
        if not cls_on:
            getattr(module, f"ssl{stage}_cls").register_forward_pre_hook(reject)
        if not node_on:
            getattr(module, f"ssl{stage}_node").register_forward_pre_hook(reject)
    loss, _ = objective(batch)
    loss.backward()
    assert torch.isfinite(loss)
    if not cls_on:
        assert f"ssl{stage}_cls" not in objective.pending
    if not node_on:
        assert f"ssl{stage}_node" not in objective.pending


def test_ssl1_row_reduction_weights_condition_ce_only():
    model, batch = fixture()
    objective = JointObjective(model, ssl1_reduction="row_mean")
    first = objective.graph_loss(batch.graph_views, torch.tensor([0]))
    second = objective.graph_loss(batch.graph_views, torch.tensor([1]))
    weighted = objective.graph_loss(batch.graph_views, torch.tensor([0, 0, 0, 1]))
    torch.testing.assert_close(
        weighted["condition"], (3 * first["condition"] + second["condition"]) / 4
    )
    for name in ("node", "spread"):
        torch.testing.assert_close(weighted[name], first[name])
        torch.testing.assert_close(weighted[name], second[name])
    objective.ssl1_reduction = "condition_mean"
    distinct = objective.graph_loss(batch.graph_views, torch.tensor([0, 0, 0, 1]))
    torch.testing.assert_close(
        distinct["condition"], (first["condition"] + second["condition"]) / 2
    )


@pytest.mark.parametrize("attention", ["hybrid", "full_latent", "delta_full", "full"])
def test_response_cls_edge_intervention_blocks_gene_sensitivity(attention):
    from dataclasses import replace

    original, batch = fixture()
    model = GraDPertV2(torch.randn(6, 5), replace(original.options, attention=attention))
    model.eval()
    graph, condition = JointObjective(model)._graph(model, batch.graph, False)
    with torch.no_grad():
        before = model.encode_response(graph[:4], batch.control, condition)
        blocked_before = model.encode_response(
            graph[:4], batch.control, condition, block_response_cls_to_gene=True
        )
        model.response_cls.add_(torch.randn_like(model.response_cls) * 3)
        after = model.encode_response(graph[:4], batch.control, condition)
        blocked_after = model.encode_response(
            graph[:4], batch.control, condition, block_response_cls_to_gene=True
        )
    torch.testing.assert_close(blocked_before["prediction"], blocked_after["prediction"])
    assert not torch.allclose(before["prediction"], after["prediction"])
    assert not torch.allclose(blocked_before["response_cls"], blocked_after["response_cls"])
    model.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        model.encode_response(graph[:4], batch.control, condition, block_response_cls_to_gene=True)
