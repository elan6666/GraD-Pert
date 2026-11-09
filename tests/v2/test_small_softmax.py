"""Frozen operator contracts and the new prediction-only lifecycle."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_relay_method import relay_training_fixture
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.small_softmax import (
    DenseSwiGLU,
    FeatureRMSNorm,
    SharedKVAttention,
    StreamConnection,
)
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer, routes
from gradpert.training.v2.process_diagnostics import evaluate_process_diagnostics


def architecture(checkpointed=False):
    return V2Architecture(
        architecture_variant="small_softmax_mhc",
        width=256,
        heads=4,
        streams=4,
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        dropout=0,
        learned_genept_projection=True,
        mlp_profile="restored",
        genept_projection_activation="gelu",
        genept_hidden_width=1024,
        genept_middle_width=256,
        ffn_variant="swiglu",
        ffn_clip=True,
        ffn_linear_bias="none",
        prediction_parameterization="softplus",
        random_gene_order=False,
        response_injection_residual=True,
        checkpoint_layers=checkpointed,
        checkpoint_genept_mapping=checkpointed,
    )


def fixture(checkpointed=False):
    _, batch = relay_training_fixture(False)
    batch = replace(batch, graph_views=(), cell_views=())
    model = GraDPertV2(torch.randn(16, 2048), architecture(checkpointed))
    return JointObjective(
        model,
        lambda1=0,
        lambda2=0,
        ssl1_weights=(0, 0, 0),
        ssl2_weights=(0, 0, 0),
        lambda_mmd=0,
        loss_reduction="row_mean",
    ), batch


def test_norm_and_clipped_swiglu_precision():
    norm = FeatureRMSNorm(8)
    x = (torch.randn(2, 8) * 5).to(torch.bfloat16)
    norm.weight.data.uniform_(0.4, 1.4)
    expected = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20) * norm.weight
    ).to(x.dtype)
    assert torch.equal(norm(x), expected)
    ffn = DenseSwiGLU(8)
    ffn.gate.weight.data.mul_(10)
    ffn.up.weight.data.mul_(10)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gate, up = ffn.gate(x), ffn.up(x)
        expected = ffn.output(
            (F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)).to(x.dtype)
        )
        assert torch.equal(ffn(x), expected)
    assert (gate < -10).any() and (up.abs() > 10).any()


def test_mhc_uses_previous_read_and_source_to_target_direction():
    layer = StreamConnection(8)
    state = torch.randn(2, 4, 8, requires_grad=True)
    previous = torch.tensor([[1.0, 0, 0, 0], [0.0, 0.2, 0.3, 0.5]])
    x, following, write, matrix = layer.prepare(state, previous)
    torch.testing.assert_close(x, layer.norm((previous[..., None] * state).sum(-2)))
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2, 4), atol=2e-6, rtol=0)
    # Finite 20-iteration Sinkhorn is approximate in rows; don't silently add iterations.
    flat = state.flatten(-2).float()
    scores = (
        F.linear(flat, layer.route.weight) / (flat.square().mean(-1, keepdim=True) + 1e-20).sqrt()
    )
    reference = (scores[..., 8:] * layer.scale[2] + layer.base[8:]).reshape(2, 4, 4).softmax(
        -1
    ) + 1e-6
    reference = reference / (reference.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        reference = reference / (reference.sum(-1, keepdim=True) + 1e-6)
        reference = reference / (reference.sum(-2, keepdim=True) + 1e-6)
    torch.testing.assert_close(matrix, reference)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 4), atol=1e-3, rtol=0)
    # Asymmetric matrix catches the tempting transpose; unrelated new read catches timing.
    matrix = torch.rand_like(matrix, requires_grad=True)
    update = torch.randn(2, 8, requires_grad=True)
    expected = torch.stack(
        [
            sum(matrix[:, source, target, None] * state[:, source] for source in range(4))
            + write[:, target, None] * update
            for target in range(4)
        ],
        1,
    )
    actual = layer.finish(state, update, write, matrix)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (state, matrix, update), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (state, matrix, update)),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    assert not torch.equal(previous, following)


def test_attention_matches_dense_zero_value_sink_and_group_output_gradients():
    torch.manual_seed(7)
    layer = SharedKVAttention(16)
    query, memory = (
        torch.randn(2, 67, 16, requires_grad=True),
        torch.randn(2, 7, 16, requires_grad=True),
    )
    q = layer.q_up(layer.q_norm(layer.q_down(query))).reshape(2, 67, 4, 4).transpose(1, 2)
    kv = layer.kv_norm(layer.kv(memory))
    keys = kv[:, None].expand(-1, 4, -1, -1)
    logits = q @ keys.transpose(-1, -2) / 2
    sink = layer.sink[None, :, None, None].expand(2, 4, 67, 1)
    value = torch.cat((keys, keys.new_zeros(2, 4, 1, 4)), -2)
    output = (torch.cat((logits, sink), -1).softmax(-1) @ value).transpose(1, 2).reshape(2, 67, 16)
    expected = layer.output(
        torch.cat(
            [layer.group_output[0](output[..., :8]), layer.group_output[1](output[..., 8:])], -1
        )
    )
    actual = layer(query, memory)
    torch.testing.assert_close(actual, expected)
    inputs = (query, memory, layer.sink, *layer.group_output.parameters())
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True),
        torch.autograd.grad(expected.square().sum(), inputs),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("checkpointed,bf16", [(False, False), (True, True)])
def test_absent_distillation_backward_checkpoint_resume_and_diagnostics(
    tmp_path, checkpointed, bf16
):
    torch.set_num_threads(2)
    objective, batch = fixture(checkpointed)
    assert objective.teacher is None and not objective.teacher_updates_enabled
    assert not any(
        "ssl" in key or "center" in key or "teacher" in key for key in objective.state_dict()
    )
    assert not any("DeltaAttention" in type(module).__name__ for module in objective.modules())
    assert not batch.graph_views and not batch.cell_views
    optimizer = V2Optimizer(objective.student, 0.001, 0)
    before = objective.student.graph.adapter[1].weight.detach().clone()
    optimizer_step(objective, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=bf16)
    assert not torch.equal(before, objective.student.graph.adapter[1].weight)
    assert not objective.pending
    assert objective.student.graph.embedding.weight.grad is None
    route = {v["name"]: v for v in routes(objective.student)}
    assert route["cell.layers.0.self_attention.q_up.weight"]["heads"] == 4
    assert route["cell.layers.0.self_attention.group_output.0.weight"]["optimizer"] == "muon"
    assert route["cell.layers.0.self_connection.route.weight"]["optimizer"] == "adamw"
    path = tmp_path / "state.pt"
    rng = np.random.default_rng(5)
    save_checkpoint(
        path, objective, optimizer, identity={"test": 1}, progress={"steps": 1}, generator=rng
    )
    clone, _ = fixture(checkpointed)
    clone_optimizer = V2Optimizer(clone.student, 0.001, 0)
    assert load_checkpoint(path, clone, clone_optimizer, identity={"test": 1}, generator=rng) == {
        "steps": 1
    }
    for version, opt in ((objective, optimizer), (clone, clone_optimizer)):
        optimizer_step(version, opt, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=bf16)
    for key, value in objective.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key]), key
    state = copy.deepcopy(objective.state_dict())
    report = evaluate_process_diagnostics(objective, batch, bf16=bf16, max_samples=8)
    assert report["centers"] == report["distributions"] == {}
    assert "absent" in report["distillation"]
    assert "student.cell.output.genes" in report["features"]
    for key, value in state.items():
        assert torch.equal(value, objective.state_dict()[key])


def test_variant_fail_closed_and_historical_identity():
    assert "architecture_variant" not in V2Architecture().payload()
    with pytest.raises(ValueError, match="fixed architecture"):
        replace(architecture(), dropout=0.1)
    model = GraDPertV2(torch.randn(3, 2048), architecture())
    with pytest.raises(ValueError, match="only the prediction"):
        JointObjective(model)


def test_graph_scores_respect_multisource_relation_and_legal_neighbors():
    from gradpert.modeling.v2.small_softmax import NeighborhoodAttention

    torch.manual_seed(11)
    layer = NeighborhoodAttention(16)
    layer.source_transform.data.normal_(std=0.2)
    x = torch.randn(5, 16, requires_grad=True)
    neighbors = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 1]])
    valid = torch.ones_like(neighbors, dtype=torch.bool)
    sources = torch.zeros(5, 2, 4)
    sources[:, 0, 3] = 1
    sources[:, 1, :2] = 1
    cq, ckv = layer.q_norm(layer.q_down(x)), layer.kv_norm(layer.kv_down(x))
    q = layer.q_up(cq).reshape(5, 4, 3)
    k = layer.k_up(ckv).reshape(5, 4, 3)
    value = layer.v_up(ckv).reshape(5, 4, 4)
    qr = layer.relation_query(cq).reshape(5, 4, 1)
    kr = layer.relation_key(x)
    reference = []
    for i in range(5):
        heads = []
        for h in range(4):
            logits = []
            for t, j in enumerate(neighbors[i]):
                transform = torch.eye(1) + sum(
                    sources[i, t, a] * layer.source_transform[a] for a in range(4)
                )
                query = torch.cat((q[i, h], transform @ qr[i, h]))
                key = torch.cat((k[j, h], transform @ kr[j]))
                logits.append(query @ key / 2)
            heads.append((torch.stack(logits).softmax(0)[:, None] * value[neighbors[i], h]).sum(0))
        reference.append(torch.cat(heads))
    expected = layer.output(torch.stack(reference))
    actual = layer(x, neighbors, valid, sources)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, layer.source_transform), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (x, layer.source_transform)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    # Node4 is not in node0's neighborhood: its state cannot affect that read.
    changed = x.detach().clone()
    changed[4] += 100
    torch.testing.assert_close(actual[0], layer(changed, neighbors, valid, sources)[0])


def test_self_contained_recipe_preserves_the_frozen_training_protocol():
    from pathlib import Path

    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    config = load_experiment_config(
        Path(__file__).resolve().parents[2]
        / "configs/v2/small_softmax_mhc_no_distill/gradpert_v2/nadig_jurkat.yaml"
    )
    arch, opts = V2Options.parse_parameters(config.model.parameters)
    assert arch.architecture_variant == "small_softmax_mhc"
    assert (opts.query_count, opts.microbatch, opts.accumulation, opts.world_size) == (
        1000,
        80,
        4,
        1,
    )
    assert (config.training.train_batch_size.value, config.training.max_epochs.value) == (320, 2)
    assert opts.validation_mode == "disabled" and opts.process_diagnostics
    assert opts.lambda1 == opts.lambda2 == 0
    assert arch.relay_validate_once and arch.cache_kda_constants


def test_prediction_equivariance_masking_and_label_isolation():
    torch.set_num_threads(2)
    objective, batch = fixture(False)
    model = objective.student.eval()
    with torch.no_grad():
        graph, condition = objective._batch_graph(model, batch)
        gene = graph[batch.query_positions]
        baseline = model.encode_response(gene, batch.control, condition)["prediction"]
        permutation = torch.randperm(gene.shape[0])
        permuted = model.encode_response(
            gene[permutation], batch.control[:, permutation], condition
        )["prediction"]
        torch.testing.assert_close(permuted, baseline[:, permutation], atol=3e-6, rtol=3e-5)
        mask = torch.zeros_like(batch.control, dtype=torch.bool)
        mask[:, 1] = True
        changed = batch.control.clone()
        changed[:, 1] += 1000
        one = model.encode_response(gene, batch.control, condition, mask)["prediction"]
        two = model.encode_response(gene, changed, condition, mask)["prediction"]
        assert torch.equal(one, two)
        context = batch.graph.context
        permutation = torch.randperm(len(context.ids))
        inverse = permutation.argsort()
        reordered = replace(
            context,
            ids=context.ids[permutation],
            neighbors=inverse[context.neighbors[permutation]],
            valid=context.valid[permutation],
            sources=context.sources[permutation],
            query_positions=inverse[context.query_positions],
        )
        view = replace(batch.graph, context=reordered)
        g, c = objective._batch_graph(model, replace(batch, graph=view, truth=batch.truth + 1000))
        torch.testing.assert_close(g, graph, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(c, condition, atol=3e-6, rtol=3e-5)


def test_controller_preserves_other_gpu_work_and_rejects_selected_foreign_process():
    from scripts.v2.run_small_softmax import admit_selected_idle

    observed = dict(
        gpus={1: dict(uuid="chosen", free_mib=30000)},
        processes=[dict(gpu_uuid="other", owned=False, pid=123)],
    )
    admit_selected_idle(observed, 1, "chosen")
    observed["processes"][0]["gpu_uuid"] = "chosen"
    with pytest.raises(RuntimeError, match="foreign work"):
        admit_selected_idle(observed, 1, "chosen")
    observed["processes"] = []
    observed["gpus"][1]["free_mib"] = 1024
    with pytest.raises(RuntimeError, match="reserve"):
        admit_selected_idle(observed, 1, "chosen")


def test_evaluation_graph_cache_uses_the_same_complete_graph_and_invalidates_updates():
    from test_relay_method import small_index

    from gradpert.training.v2.evaluation import predict_controls
    from gradpert.training.v2.evaluation_graph import prepare_evaluation_graph

    torch.set_num_threads(2)
    objective, _ = fixture(False)
    model, index = objective.student.eval(), small_index(16)
    memory = prepare_evaluation_graph(model, index, device=torch.device("cpu"))
    assert memory is not None
    controls = np.random.default_rng(9).random((3, 16), dtype=np.float32)
    options = dict(device=torch.device("cpu"), cell_batch=2, query_count=6)
    cached = predict_controls(model, index, controls, (0,), graph_memory=memory, **options)
    uncached = predict_controls(model, index, controls, (0,), **options)
    np.testing.assert_allclose(cached, uncached, atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        model.graph.layers[0].attention.q_up.weight.add_(0.01)
    with pytest.raises(ValueError, match="same frozen"):
        memory.select(model, index, torch.arange(4))
