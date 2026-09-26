"""New relay profile: state semantics, graph scope and complete response path."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.graphs import GraphTopology
from gradpert.graphs.pruning import PrunedSourceGraph
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import (
    RelayDeltaAttention,
    chunk_delta_final_state,
    delta_final_state,
)
from gradpert.training.batch import GraDPertTrainingBatch
from gradpert.training.data import random_mixed_epoch_batches
from gradpert.training.v2.evaluation import predict_query_set
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.reductions import nearest_neighbor_terms
from gradpert.training.v2.views import NeighborhoodIndex, assemble_batch


def small_architecture() -> V2Architecture:
    return V2Architecture(
        width=8,
        heads=2,
        latent_rank=4,
        streams=1,
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        dropout=0,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
        checkpoint_layers=False,
    )


def small_index(n: int = 9) -> NeighborhoodIndex:
    genes = tuple(f"g{i}" for i in range(n))
    topology = GraphTopology(
        genes,
        {name: PrunedSourceGraph(name, n, genes, (), 20) for name in ("go", "string")},
    )
    return NeighborhoodIndex(topology, 1, 7, expander_type="hamiltonian", relay=True)


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_chunk_final_state_and_carried_state_match_reference_gradients(chunk_size: int) -> None:
    torch.manual_seed(4)
    k = torch.randn(2, 9, 2, 4, requires_grad=True)
    v = torch.randn(2, 9, 2, 4, requires_grad=True)
    decay = (-torch.rand_like(k)).requires_grad_()
    beta = torch.rand(2, 9, 2, requires_grad=True)
    initial = torch.randn(2, 2, 4, 4, requires_grad=True)
    reference = delta_final_state(k, v, decay, beta, initial)
    chunked = chunk_delta_final_state(k, v, decay, beta, initial, chunk_size=chunk_size)
    torch.testing.assert_close(reference, chunked, atol=2e-6, rtol=2e-5)
    for a, b in zip(
        torch.autograd.grad(
            reference.square().sum(), (k, v, decay, beta, initial), retain_graph=True
        ),
        torch.autograd.grad(chunked.square().sum(), (k, v, decay, beta, initial)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


def test_relay_cls_writes_once_after_twice_scanned_genes() -> None:
    torch.manual_seed(2)
    layer = RelayDeltaAttention(8, 2).eval()
    x = torch.randn(2, 5, 8)
    order = torch.tensor([[3, 1, 0, 2], [2, 3, 1, 0]])
    genes = layer._project_writes(x[:, :4])
    cls = layer._project_writes(x[:, 4:])
    forward = [layer._ordered(t, order) for t in genes]
    state = delta_final_state(*forward)
    state = delta_final_state(*(t.flip(1) for t in forward), state=state)
    state = delta_final_state(*cls, state=state)
    torch.testing.assert_close(layer(x, order=order), layer._read(x, state))


def test_graph_selected_output_matches_full_synchronous_4_layer_graph() -> None:
    torch.manual_seed(9)
    model = GraDPertV2(torch.randn(9, 8), small_architecture()).eval()
    index = small_index()
    selected = np.array([0, 2], dtype=np.int64)
    partial = index.view(
        selected, [(0,)], rng=np.random.default_rng(3), device=torch.device("cpu"), induced=False
    )
    full = index.view(
        np.arange(9),
        [(0,)],
        rng=np.random.default_rng(3),
        device=torch.device("cpu"),
        induced=False,
    )
    assert partial.context is not None and full.context is not None
    a = model.graph(
        partial.ids, partial.neighbors, partial.valid, partial.sources, context=partial.context
    )
    b = model.graph(full.ids, full.neighbors, full.valid, full.sources, context=full.context)
    torch.testing.assert_close(a, b[selected])


def test_relay_local_graph_cannot_read_outside_induced_nodes() -> None:
    torch.manual_seed(3)
    seeds = torch.randn(9, 8)
    index = small_index()
    view = index.view(
        np.array([0, 1, 2], dtype=np.int64),
        [(0,)],
        rng=np.random.default_rng(6),
        device=torch.device("cpu"),
        induced=True,
    )
    assert view.context is not None
    model = GraDPertV2(seeds, small_architecture()).eval()
    original = model.graph(view.ids, view.neighbors, view.valid, view.sources, context=view.context)
    with torch.no_grad():
        model.graph.embedding.weight[3:].add_(100)
    changed = model.graph(view.ids, view.neighbors, view.valid, view.sources, context=view.context)
    torch.testing.assert_close(original, changed)


def test_relay_training_orders_change_but_evaluation_order_is_reproducible() -> None:
    model = GraDPertV2(torch.randn(9, 8), small_architecture())
    gene, control, condition = torch.randn(5, 8), torch.randn(2, 5), torch.randn(2, 8)
    model.train()
    torch.manual_seed(2)
    train_orders = [
        model.encode_response(gene, control, condition)["response_cls"] for _ in range(2)
    ]
    assert not torch.equal(*train_orders)
    model.eval()
    first = model.encode_response(gene, control, condition)["response_cls"]
    second = model.encode_response(gene, control, condition)["response_cls"]
    torch.testing.assert_close(first, second)


def test_relay_response_preserves_expression_mask_and_backpropagates() -> None:
    torch.manual_seed(8)
    model = GraDPertV2(torch.randn(9, 8), small_architecture()).eval()
    gene = torch.randn(3, 8)
    control = torch.randn(2, 3)
    condition = torch.randn(2, 8)
    mask = torch.tensor([[False, True, False], [True, False, False]])
    first = model.encode_response(gene, control, condition, mask)
    changed = control + mask.float() * 100
    second = model.encode_response(gene, changed, condition, mask)
    torch.testing.assert_close(first["response_cls"], second["response_cls"])
    torch.testing.assert_close(first["delta"], second["delta"])
    assert not torch.equal(first["prediction"], second["prediction"])
    first["prediction"].square().mean().backward()
    assert model.response.cross_layers[0].sublayer.query.weight.grad is not None


def test_response_cls_intervention_blocks_gene_route() -> None:
    torch.manual_seed(6)
    model = GraDPertV2(torch.randn(9, 8), small_architecture()).eval()
    gene = torch.randn(3, 8)
    control = torch.randn(2, 3)
    condition = torch.randn(2, 8)
    blocked = model.encode_response(gene, control, condition, block_response_cls_to_gene=True)[
        "delta"
    ]
    gradient = torch.autograd.grad(blocked.sum(), model.response_cls, allow_unused=True)[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0
    normal = model.encode_response(gene, control, condition)["delta"]
    assert not torch.equal(normal, blocked)


def test_condition_excluding_koleo_and_uncapped_mixed_rows() -> None:
    values = torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    ids = torch.tensor([0, 0, 1])
    terms = nearest_neighbor_terms(values, ids)
    assert torch.isfinite(terms).all()
    assert terms[0] < nearest_neighbor_terms(values)[0]
    conditions = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    batches = random_mixed_epoch_batches(
        condition_ids=conditions, run_seed=1, epoch=0, batch_size=12
    )
    assert sorted(row for batch in batches for row in batch) == list(range(60))
    assert all(len({conditions[row] for row in batch}) >= 2 for batch in batches)


def test_new_profile_config_is_distinct_from_historical_method() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/relay_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert (arch.attention, arch.graph_layers, arch.graph_read_mode) == ("relay_full", 4, "relay")
    assert (options.local_views, options.max_conditions) == (2, 0)
    assert (options.global_min_ratio, options.global_max_ratio) == (0.6, 0.9)
    assert (options.local_min_ratio, options.local_max_ratio) == (0.25, 0.5)


def test_relay_joint_losses_and_evaluation_reuse_same_model() -> None:
    config = load_experiment_config(
        Path(__file__).resolve().parents[2]
        / "configs/v2/relay_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    options = replace(options, query_count=3)
    index = small_index()
    raw = GraDPertTrainingBatch(
        torch.randn(4, 9),
        torch.randn(4, 9),
        ("g0", "g1", "g0", "g1"),
        {"g0": (0,), "g1": (1,)},
        ("t0", "t1", "t2", "t3"),
        ("c0", "c1", "c2", "c3"),
    )
    batch = assemble_batch(raw, index, options, np.random.default_rng(4))
    model = GraDPertV2(torch.randn(9, 8), small_architecture())
    objective = JointObjective(
        model, loss_reduction="row_mean", koleo_exclude_same_condition=True
    ).train()
    loss, terms = objective(batch)
    assert torch.isfinite(loss)
    assert {"prediction", "ssl1_condition", "ssl2_dino", "ssl2_koleo"} <= terms.keys()
    loss.backward()
    assert model.graph.embedding.weight.grad is not None
    assert all(parameter.grad is None for parameter in objective.teacher.parameters())
    predicted = predict_query_set(
        model.eval(),
        index,
        raw.control_expression.numpy(),
        (0,),
        np.array([0, 1, 2]),
        device=torch.device("cpu"),
        cell_batch=2,
    )
    assert predicted.shape == (4, 3) and np.isfinite(predicted).all()


def test_relay_checkpointing_preserves_graph_and_response_gradients() -> None:
    torch.manual_seed(11)
    plain = GraDPertV2(torch.randn(9, 8), small_architecture()).train()
    saved = GraDPertV2(
        torch.randn(9, 8), replace(small_architecture(), checkpoint_layers=True)
    ).train()
    saved.load_state_dict(plain.state_dict())
    view = small_index().view(
        np.arange(9),
        [(0,)],
        rng=np.random.default_rng(4),
        device=torch.device("cpu"),
        induced=False,
    )
    assert view.context is not None
    gene = torch.randn(2, 8)
    control = torch.randn(2, 2)
    condition = torch.randn(2, 8)

    def run(model: GraDPertV2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(19)
        graph = model.graph(
            view.ids, view.neighbors, view.valid, view.sources, context=view.context
        )
        output = model.encode_response(gene, control, condition)["response_cls"]
        loss = graph.square().sum() + output.square().sum()
        loss.backward()
        return (
            loss.detach(),
            model.graph.embedding.weight.grad.detach().clone(),
            model.response.cross_layers[0].sublayer.key.weight.grad.detach().clone(),
        )

    for actual, expected in zip(run(saved), run(plain), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def relay_training_fixture(checkpointed: bool = True):
    config = load_experiment_config(
        Path(__file__).resolve().parents[2]
        / "configs/v2/relay_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    options = replace(options, query_count=6, mask_probability=1.0, graph_mask_ratio=0.5)
    raw = GraDPertTrainingBatch(
        torch.randn(4, 16),
        torch.randn(4, 16),
        ("g0", "g1", "g0", "g1"),
        {"g0": (0,), "g1": (1,)},
        ("t0", "t1", "t2", "t3"),
        ("c0", "c1", "c2", "c3"),
    )
    batch = assemble_batch(raw, small_index(16), options, np.random.default_rng(8))
    model = GraDPertV2(
        torch.randn(16, 8),
        replace(
            small_architecture(),
            checkpoint_layers=checkpointed,
            dropout=0.1,
            streams=2,
            ffn_type="swiglu",
            relay_eval_seed=1,
        ),
    )
    return JointObjective(
        model, loss_reduction="row_mean", koleo_exclude_same_condition=True
    ), batch


def assert_tree_close(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_close(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_close(a, b)
    else:
        assert (
            actual == pytest.approx(expected) if isinstance(actual, float) else actual == expected
        )


def test_complete_relay_update_checkpointing_preserves_ema_centers_optimizer_rng():
    from gradpert.modeling.v2.model import RelayGraphLayer, RelayResponseEncoder
    from gradpert.modeling.v2.operators import TokenEncoder
    from gradpert.training.v2.engine import optimizer_step
    from gradpert.training.v2.optimizer import V2Optimizer

    objective, batch = relay_training_fixture()
    plain = copy.deepcopy(objective)
    for module in plain.modules():
        if isinstance(module, (TokenEncoder, RelayResponseEncoder)):
            module.checkpoint_layers = False
        if isinstance(module, RelayGraphLayer):
            module.checkpoint_chunks = False
    snapshots = []
    for version in (objective, plain):
        optimizer = V2Optimizer(version.student, 0.001, 0.0)
        torch.manual_seed(91)
        metrics = optimizer_step(
            version, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
        )
        snapshots.append(
            (
                metrics,
                copy.deepcopy(version.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
                torch.get_rng_state(),
            )
        )
        assert not version.pending
        assert not version.teacher.training and version.teacher.randomize_relay_order
        assert all(p.grad is None for p in version.teacher.parameters())
    assert_tree_close(snapshots[0], snapshots[1])


def test_relay_complete_update_resume_and_best_last_loading(tmp_path):
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import (
        load_checkpoint,
        load_evaluation_checkpoint,
        save_checkpoint,
    )
    from gradpert.training.v2.engine import optimizer_step
    from gradpert.training.v2.optimizer import V2Optimizer

    objective, batch = relay_training_fixture()
    optimizer = V2Optimizer(objective.student, 0.001, 0.0)
    generator = np.random.default_rng(7)
    identity = {"config": "synthetic-relay-full"}

    def step():
        return optimizer_step(
            objective, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
        )

    step()
    for role in ("best", "last"):
        save_checkpoint(
            tmp_path / f"{role}.pt",
            objective,
            optimizer,
            identity=identity,
            progress={"step": 1, "role": role},
            generator=generator,
        )
    expected_metrics = step()
    expected = (
        copy.deepcopy(objective.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state(),
    )
    load_checkpoint(
        tmp_path / "last.pt", objective, optimizer, identity=identity, generator=generator
    )
    assert_tree_close(step(), expected_metrics)
    assert_tree_close(
        (objective.state_dict(), optimizer.state_dict(), torch.get_rng_state()), expected
    )
    for role in ("best", "last"):
        path = tmp_path / f"{role}.pt"
        result = load_evaluation_checkpoint(
            path, objective, training_identity=identity, checkpoint_sha256=sha256_file(path)
        )
        assert result == {"step": 1, "role": role}


def test_relay_prediction_does_not_depend_on_cell_batch_partition():
    model = GraDPertV2(torch.randn(9, 8), small_architecture()).eval()
    controls = np.random.default_rng(5).normal(size=(5, 9)).astype(np.float32)
    results = [
        predict_query_set(
            model,
            small_index(),
            controls,
            (0,),
            np.arange(5),
            device=torch.device("cpu"),
            cell_batch=b,
        )
        for b in (1, 2, 5)
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=2e-6, rtol=2e-5)


def test_multiscale_graph_views_keep_targets_and_crop_induced_neighbors():
    _, batch = relay_training_fixture()
    for i, view in enumerate(batch.graph_views):
        lower, upper = (0.6, 0.9) if i < 2 else (0.25, 0.5)
        assert round(16 * lower) <= len(view.ids) <= round(16 * upper)
        assert {0, 1} <= set(view.ids.tolist())
        assert view.context is not None
        assert set(view.neighbors[view.valid].tolist()) <= set(view.ids.tolist())
    assert len(batch.graph_views) == len(batch.cell_views) == 4


def test_random_mixed_tail_respects_capacity_without_losing_rows():
    conditions = ["a", "b", "c"] * 5 + ["a", "b"]
    batches = random_mixed_epoch_batches(
        condition_ids=conditions, run_seed=5, epoch=0, batch_size=8
    )
    assert all(2 <= len(b) <= 8 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(17))
    assert all(len({conditions[i] for i in b}) >= 2 for b in batches)


def test_legacy_architecture_payload_omits_new_default_seed():
    assert "relay_eval_seed" not in V2Architecture().payload()


def _relay_two_rank_update(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from gradpert.training.v2.engine import optimizer_step, slice_cells
    from gradpert.training.v2.objective import CellView
    from gradpert.training.v2.optimizer import V2Optimizer

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(123)
        objective, batch = relay_training_fixture()
        order = torch.tensor([0, 2, 1, 3])
        batch = replace(
            batch,
            control=batch.control[order],
            truth=batch.truth[order],
            condition_index=batch.condition_index[order],
            cell_views=tuple(CellView(v.positions, v.mask[order]) for v in batch.cell_views),
        )
        local = slice_cells(batch, rank * 2, rank * 2 + 2)
        assert len(torch.unique(local.condition_index)) == 1
        optimizer = V2Optimizer(objective.student, 0.001, 0.0)
        torch.manual_seed(81 + rank)
        metrics = optimizer_step(
            objective,
            optimizer,
            local,
            microbatch=1,
            lr=0.001,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index,
        )
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["ssl2_koleo"] != 0
        # Whole train/EMA/center state must agree after a distributed update,
        # despite independent rank-local views and no local unlike-condition neighbor.
        values = torch.cat([t.detach().reshape(-1) for t in objective.state_dict().values()])
        copies = [torch.empty_like(values) for _ in range(2)]
        dist.all_gather(copies, values)
        torch.testing.assert_close(copies[0], copies[1], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_relay_two_rank_update_can_find_only_remote_unlike_conditions(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _relay_two_rank_update, args=(str(tmp_path / "relay-rendezvous"),), nprocs=2, join=True
    )
