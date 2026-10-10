"""First-batch mechanisms through actual optimizer/EMA/center/checkpoint updates."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_relay_method import small_architecture, small_index

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.batch import GraDPertTrainingBatch
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.evaluation import predict_query_set
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.views import assemble_batch

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "configs/v2/cap40_genept_gelu_20261010/gradpert_v2/nadig_jurkat.yaml"
ARMS = ("N0", "U24", "MR1", "P1", "C1", "O1", "VH", "S1-L4", "CG1", "S12-L4")


def fixture(arm="N0", *, checkpointed=True):
    _, options = V2Options.parse_parameters(load_experiment_config(PARENT).model.parameters)
    options = replace(
        options,
        query_count=6,
        microbatch=4,
        accumulation=1,
        world_size=1,
        mask_probability=1,
        graph_mask_ratio=0.5,
        independent_view_rng=True,
        ssl1_local_views=2,
        ssl2_local_views=2,
    )
    architecture = replace(
        small_architecture(),
        learned_genept_projection=True,
        mlp_profile="unified",
        relay_passes=1,
        dropout=0.1,
        streams=2,
        checkpoint_layers=checkpointed,
        graph_source_key_gate=True,
    )
    objective_options = dict(
        loss_reduction="row_mean", ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)
    )
    if arm == "U24":
        architecture = replace(architecture, gene_conditioned_readout=True)
    elif arm == "MR1":
        objective_options.update(masked_response_ratio=0.25, lambda_masked_response=1)
    elif arm == "P1":
        objective_options.update(population_response=True, lambda_mmd=1)
    elif arm == "C1":
        architecture = replace(architecture, perturbation_injection="entry")
    elif arm == "O1":
        architecture = replace(architecture, random_gene_order=False)
    elif arm == "VH":
        options = replace(
            options,
            global_min_ratio=0.8,
            global_max_ratio=1,
            local_min_ratio=0.4,
            local_max_ratio=0.65,
        )
    elif arm == "S1-L4":
        options = replace(options, ssl1_local_views=4, ssl1_local_layout="go_string_random")
    elif arm == "CG1":
        architecture = replace(architecture, control_conditioned_graph=True)
    elif arm == "S12-L4":
        options = replace(
            options, ssl1_local_views=4, ssl2_local_views=4, ssl1_local_layout="go_go_string_string"
        )
    torch.manual_seed(62)
    seeds = torch.randn(16, 24)
    control, truth = torch.rand(4, 16), torch.rand(4, 16)
    conditions = ("g0",) * 4 if arm == "P1" else ("g0", "g1", "g0", "g1")
    raw = GraDPertTrainingBatch(
        control,
        truth,
        conditions,
        {name: (int(name[1:]),) for name in set(conditions)},
        ("t0", "t1", "t2", "t3"),
        ("c0", "c1", "c2", "c3"),
    )
    batch = assemble_batch(
        raw,
        small_index(16),
        options,
        np.random.default_rng(8),
        allowed_expression_ids=np.arange(12),
    )
    model = GraDPertV2(seeds, architecture)
    return JointObjective(model, **objective_options), batch, raw, options


@pytest.mark.parametrize("arm", ARMS)
def test_three_complete_updates_and_exact_checkpoint_continuation(arm, tmp_path):
    objective, batch, _, _ = fixture(arm)
    optimizer = V2Optimizer(objective.student, 1e-3, 0)
    rng = np.random.default_rng(21)
    for _ in range(2):
        terms = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(value) for value in terms.values())
        assert not objective.pending
    assert objective.student.graph.embedding.weight.grad is None
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert objective.ssl1_cls_center.abs().sum() > 0
    assert objective.ssl2_cls_center.abs().sum() > 0
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, objective, optimizer, identity={"arm": arm}, progress={"step": 2}, generator=rng
    )
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    expected_state = copy.deepcopy(objective.state_dict())
    restored, _, _, _ = fixture(arm)
    restored_optimizer = V2Optimizer(restored.student, 1e-3, 0)
    load_checkpoint(path, restored, restored_optimizer, identity={"arm": arm}, generator=rng)
    actual = optimizer_step(
        restored, restored_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)


def test_u24_starts_at_n0_prediction_and_shares_raw_prior_without_duplicate_buffer():
    base, batch, _, _ = fixture()
    combined, _, _, _ = fixture("U24")
    for name, value in base.student.state_dict().items():
        torch.testing.assert_close(value, combined.student.state_dict()[name], atol=0, rtol=0)
    assert "readout_prior" not in combined.student.state_dict()
    base.eval()
    combined.eval()

    def predict(o):
        graph, condition = o._batch_graph(o.student, batch)
        return o.student.encode_response(
            o._select_graph(graph, batch.query_positions),
            batch.control,
            condition,
            **o.response_metadata(batch),
        )["prediction"]

    torch.testing.assert_close(predict(base), predict(combined), atol=0, rtol=0)


def test_mr_hidden_raw_control_has_zero_derivative_and_private_rng():
    objective, batch, _, _ = fixture("MR1", checkpointed=False)
    objective.eval()
    control = batch.control.detach().clone().requires_grad_()
    batch = replace(batch, control=control)
    generator = torch.Generator().manual_seed(objective.auxiliary_seed)
    positions = torch.rand(control.shape, generator=generator).argsort(-1)[:, :1]
    mask = torch.zeros_like(control, dtype=torch.bool).scatter_(1, positions, True)
    graph, condition = objective._batch_graph(objective.student, batch)
    before = torch.get_rng_state().clone()
    loss = objective.masked_response_loss(batch, graph, condition, torch.full((4,), 0.25))
    assert torch.equal(before, torch.get_rng_state())
    gradient = torch.autograd.grad(loss, control)[0]
    assert torch.equal(gradient[mask], torch.zeros_like(gradient[mask]))
    changed = replace(batch, control=control.detach() + mask * 1000)
    other = objective.masked_response_loss(changed, graph, condition, torch.full((4,), 0.25))
    torch.testing.assert_close(loss, other, atol=0, rtol=0)


def test_cg_masked_summary_cannot_read_hidden_control_values():
    objective, batch, _, _ = fixture("CG1", checkpointed=False)
    objective.eval()
    with torch.no_grad():
        objective.student.graph_modulation.projection.weight.normal_(std=0.1)
    view = batch.cell_views[0]
    a, _ = objective._batch_graph(objective.student, batch, view.mask, view.positions)
    changed = batch.control.clone()
    changed[:, view.positions] += view.mask * 1000
    b, _ = objective._batch_graph(
        objective.student, replace(batch, control=changed), view.mask, view.positions
    )
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_local_counts_do_not_change_other_branch_rng_or_expression_holdout():
    _, base, _, _ = fixture()
    _, more, _, _ = fixture("S1-L4")
    assert len(more.graph_views) == 6 and len(more.cell_views) == 4
    for a, b in zip(base.cell_views, more.cell_views, strict=True):
        assert torch.equal(a.positions, b.positions) and torch.equal(a.mask, b.mask)
    assert (more.graph.ids[more.query_positions] < 12).all()


def test_cg_prediction_reuses_prefix_and_matches_control_batch_partition():
    objective, _, raw, _ = fixture("CG1", checkpointed=False)
    model = objective.student.eval()
    kwargs = dict(
        model=model,
        index=small_index(16),
        controls=raw.control_expression.numpy(),
        targets=(0,),
        queries=np.arange(6),
        device=torch.device("cpu"),
    )
    a = predict_query_set(**kwargs, cell_batch=4)
    b = predict_query_set(**kwargs, cell_batch=1)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
