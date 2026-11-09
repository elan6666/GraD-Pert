"""Training and compatibility contracts for the two small-model mechanism arms."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from test_small_softmax import fixture

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer, routes


def recipe(arm):
    cfg = load_experiment_config(
        Path(__file__).resolve().parents[2] / "configs/v2" / arm / "gradpert_v2/nadig_jurkat.yaml"
    )
    arch, opts = V2Options.parse_parameters(cfg.model.parameters)
    assert not arch.has_distillation and opts.process_diagnostics
    assert (opts.microbatch, opts.accumulation, opts.world_size, opts.query_count) == (
        80,
        4,
        1,
        1000,
    )
    return arch, opts


def test_deep_head_and_layer_budget_preserve_routes():
    torch.set_num_threads(2)
    arch, _ = recipe("small_softmax_mhc_deep_no_distill")
    torch.manual_seed(1)
    model = GraDPertV2(torch.randn(6506, 2048), arch)
    assert (len(model.graph.layers), len(model.cell.layers), len(model.response.layers)) == (
        6,
        6,
        3,
    )
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 13_071_276
    assert sum(p.numel() for p in model.parameters()) == 26_395_564
    route = {r["name"]: r["optimizer"] for r in routes(model)}
    assert route["prediction.0.weight"] == route["prediction.2.weight"] == "muon"
    assert route["prediction.4.weight"] == route["prediction.4.bias"] == "adamw"
    base, _ = recipe("small_softmax_mhc_no_distill")
    assert "small_cell_layers" not in base.payload()
    assert "small_prediction_hidden_layers" not in base.payload()
    assert "small_gene_table" not in base.payload()
    objective, batch = fixture()
    deep = JointObjective(
        model, lambda1=0, lambda2=0, ssl1_weights=(0, 0, 0), ssl2_weights=(0, 0, 0), lambda_mmd=0
    )
    opt = V2Optimizer(deep.student, 0.001, 0)
    assert objective.student.options.small_cell_layers == 3
    terms = optimizer_step(deep, opt, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False)
    assert np.isfinite(terms["prediction_mse"]) and terms["gradient_norm"] > 0


def test_random_gene_table_is_independent_trainable_and_restorable(tmp_path):
    torch.set_num_threads(2)
    arch, opts = recipe("small_softmax_mhc_random_gene_no_distill")
    assert not opts.genept_artifact_path and not opts.genept_sha256
    torch.manual_seed(19)
    first = GraDPertV2(6506, arch, gene_seed=7)
    ending_rng = torch.get_rng_state()
    torch.manual_seed(19)
    second = GraDPertV2(6506, arch, gene_seed=8)
    assert torch.equal(ending_rng, torch.get_rng_state())
    state, other = first.state_dict(), second.state_dict()
    differing = [k for k in state if not torch.equal(state[k], other[k])]
    assert differing == ["graph.embedding.weight"]
    table = first.graph.embedding.weight
    torch.testing.assert_close(
        table, torch.randn(6506, 256, generator=torch.Generator().manual_seed(7))
    )
    assert sum(p.numel() for p in first.parameters() if p.requires_grad) == 9_181_010
    assert not any(k.startswith("graph.adapter.") for k in state)
    _, batch = fixture()
    batch = replace(batch, graph_views=(), cell_views=())

    def objective(model):
        return JointObjective(
            model,
            lambda1=0,
            lambda2=0,
            ssl1_weights=(0, 0, 0),
            ssl2_weights=(0, 0, 0),
            lambda_mmd=0,
        )

    runtime = objective(first)
    optimizer = V2Optimizer(first, 0.001, 0)
    before = table.detach().clone()
    terms = optimizer_step(
        runtime, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
    )
    assert np.isfinite(terms["prediction_mse"]) and not torch.equal(table, before)
    assert torch.equal(table[16:], before[16:])
    rng = np.random.default_rng(1)
    path = tmp_path / "resume.pt"
    identity = {"graph_gene_order_sha256": "synthetic_axis"}
    save_checkpoint(
        path, runtime, optimizer, identity=identity, progress={"steps": 1}, generator=rng
    )
    restored = objective(second)
    restored_optimizer = V2Optimizer(second, 0.001, 0)
    assert load_checkpoint(
        path, restored, restored_optimizer, identity=identity, generator=rng
    ) == {"steps": 1}
    with torch.no_grad():
        for candidate in (runtime, restored):
            candidate.eval()

        def prediction(candidate):
            graph, condition = candidate._batch_graph(candidate.student, batch)
            return candidate.student.encode_response(
                graph[batch.query_positions], batch.control, condition
            )["prediction"]

        torch.testing.assert_close(prediction(runtime), prediction(restored), rtol=0, atol=0)
