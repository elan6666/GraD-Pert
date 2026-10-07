"""Loss-shape, population, information-access and continuation invariants."""

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.reductions import population_weights


def four_rows():
    model, batch = fixture()
    batch = replace(
        batch,
        control=batch.control[[0, 0, 0, 1]],
        truth=batch.truth[[0, 0, 0, 1]],
        condition_index=torch.tensor([0, 0, 0, 1]),
        cell_views=(),
    )
    return model, batch


@pytest.mark.parametrize(
    "power,reduction,expected",
    [(2, "row_mean", 3), (2, "condition_mean", 5), (4, "row_mean", 21), (4, "condition_mean", 41)],
)
def test_prediction_shape_and_condition_weight_are_independent(power, reduction, expected):
    model, batch = four_rows()
    objective = JointObjective(
        model,
        0,
        0,
        loss_reduction="row_mean",
        prediction_error_power=power,
        prediction_reduction_override=reduction,
    ).eval()
    graph, conditions = objective._graph(model, batch.graph, False)
    prediction = model.encode_response(
        graph[batch.query_positions], batch.control, conditions[batch.condition_index]
    )["prediction"].detach()
    batch = replace(batch, truth=prediction + torch.tensor([1, 1, 1, 3])[:, None])
    loss, terms = objective(batch)
    torch.testing.assert_close(loss, loss.new_tensor(float(expected)))
    torch.testing.assert_close(terms["prediction_mse"], loss.new_tensor(3.0))
    assert objective.ssl1_reduction == objective.loss_reduction == "row_mean"


class CaptureUpdate:
    """Inspect the population gradient before an adaptive optimizer can amplify roundoff."""

    def __init__(self, model):
        self.model = model

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self, lr):
        self.gradients = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)


@pytest.mark.parametrize("power", [2, 4])
def test_condition_average_accumulation_matches_complete_batch_gradient(power):
    model, batch = four_rows()
    whole = JointObjective(
        copy.deepcopy(model),
        0,
        0,
        loss_reduction="row_mean",
        prediction_error_power=power,
        prediction_reduction_override="condition_mean",
    )
    accumulated = copy.deepcopy(whole)
    captures = []
    for objective, size in [(whole, 4), (accumulated, 1)]:
        capture = CaptureUpdate(objective.student)
        optimizer_step(
            objective, capture, batch, microbatch=size, lr=0.001, momentum=0.99, bf16=False
        )
        captures.append(capture.gradients)
    assert captures[0].keys() == captures[1].keys()
    for name in captures[0]:
        torch.testing.assert_close(captures[0][name], captures[1][name], atol=2e-6, rtol=2e-5)
    for a, b in zip(whole.student.parameters(), accumulated.student.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-5)


def masked_objective(cls=1.0):
    model, batch = fixture()
    return JointObjective(
        model,
        0,
        0,
        loss_reduction="row_mean",
        auxiliary_mask_ratio=0.2,
        lambda_gene_mask=1.0,
        lambda_cls_mask=cls,
        auxiliary_seed=77,
    ), batch


def test_hidden_control_values_do_not_reach_auxiliary_outputs():
    objective, batch = masked_objective()
    model = objective.student.eval()
    decoder = model.control_reconstruction
    graph, _ = objective._graph(model, batch.graph, False)
    gene = graph[batch.query_positions]
    mask = batch.cell_views[0].mask
    changed = batch.control.clone()
    changed[mask] += 10000
    a = model.encode_control(gene, batch.control, mask, mask_token=decoder.mask_token)
    b = model.encode_control(gene, changed, mask, mask_token=decoder.mask_token)
    for left, right in zip(decoder(a[0], a[1], gene), decoder(b[0], b[1], gene), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_auxiliary_initialization_and_forward_preserve_main_random_stream():
    model, batch = fixture()
    state = torch.get_rng_state().clone()
    reference = copy.deepcopy(model.state_dict())
    objective = JointObjective(
        model, 0, 0, auxiliary_mask_ratio=0.2, lambda_gene_mask=1.0, auxiliary_seed=77
    )
    assert torch.equal(state, torch.get_rng_state())
    for name, value in reference.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    assert objective.teacher.control_reconstruction is None
    graph, _ = objective._graph(model, batch.graph, False)
    state = torch.get_rng_state().clone()
    objective.control_reconstruction_loss(batch, graph, batch.control.new_full((2,), 0.5))
    assert torch.equal(state, torch.get_rng_state())
    assert objective.auxiliary_rng_counter.item() == 1


def test_m1_unused_cls_head_and_teacher_are_not_auxiliary_training_targets():
    objective, batch = masked_objective(cls=0)
    loss, _ = objective(batch)
    loss.backward()
    decoder = objective.student.control_reconstruction
    assert decoder.gene.weight.grad.abs().sum() > 0
    assert decoder.cls_projection.weight.grad is None
    assert decoder.gene_projection.weight.grad is None
    teacher = copy.deepcopy(objective.teacher.state_dict())
    objective.commit_statistics(0.9)
    for name, value in teacher.items():
        torch.testing.assert_close(objective.teacher.state_dict()[name], value, atol=0, rtol=0)


def test_fixed_training_diagnostic_preserves_state_rng_and_gradients():
    objective, batch = masked_objective()
    objective.auxiliary_rng_counter.add_(7)
    state = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    result = evaluate_loss_diagnostics(objective, batch, bf16=False)
    assert result["split"] == "train" and result["selection_use"] == "none"
    assert result["control_copy_mse"] > 0
    assert result["shared_basal_weighted_gradient_l2"]["gene_mask"] > 0
    assert torch.equal(rng, torch.get_rng_state())
    assert objective.training and not objective.pending
    assert all(p.grad is None for p in objective.parameters())
    for name, value in state.items():
        torch.testing.assert_close(objective.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("power", [2, 4])
@pytest.mark.parametrize("strategy", ["row_mean", "condition_mean"])
def test_diagnostic_extreme_tail_uses_actual_prediction_population_weights(power, strategy):
    model, batch = four_rows()
    objective = JointObjective(
        model,
        0,
        0,
        loss_reduction="row_mean",
        prediction_error_power=power,
        prediction_reduction_override=strategy,
    ).eval()
    with torch.no_grad():
        graph, conditions = objective._graph(model, batch.graph, False)
        prediction = model.encode_response(
            graph[batch.query_positions], batch.control, conditions[batch.condition_index]
        )["prediction"]
    batch = replace(batch, truth=prediction + torch.tensor([1, 1, 1, 3])[:, None])
    result = evaluate_loss_diagnostics(objective, batch, bf16=False)
    # Sixteen gene errors: the top one percent rounds to one position. Under
    # condition_mean the one-row rare condition gets half of the total weight.
    weights = [0.25] * 4 if strategy == "row_mean" else [1 / 6] * 3 + [0.5]
    expected = weights[-1] * 3**power / (4 * (sum(weights[:3]) + weights[-1] * 3**power))
    assert result["top_one_percent_loss_fraction"] == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("auxiliary", ["off", "gene", "gene_cls"])
def test_checkpointed_eval_diagnostic_matches_joint_reference_and_keeps_next_update(auxiliary):
    from test_relay_method import relay_training_fixture

    base, batch = relay_training_fixture(checkpointed=False)
    objective = JointObjective(
        base.student,
        loss_reduction="row_mean",
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
        auxiliary_mask_ratio=0 if auxiliary == "off" else 0.2,
        lambda_gene_mask=0 if auxiliary == "off" else 1,
        lambda_cls_mask=int(auxiliary == "gene_cls"),
    )
    reference = copy.deepcopy(objective).eval()
    parameters = tuple(
        p
        for name, p in reference.student.named_parameters()
        if name.startswith(("cell.", "expression.")) or name == "control_cls"
    )
    _, terms = reference(batch)
    ssl = sum(terms[name] for name in ("ssl1_condition", "ssl1_node", "ssl2_dino", "ssl2_ibot"))
    weighted = {"prediction": terms["prediction"], "ssl": ssl}
    if auxiliary != "off":
        weighted.update(
            gene_mask=terms["gene_mask"], cls_mask=terms["cls_mask"] * int(auxiliary == "gene_cls")
        )
    expected_gradients = {}
    for name, term in weighted.items():
        gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        expected_gradients[name] = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(parameters, gradients, strict=True)
            ]
        )
    graph_only = terms["ssl1_condition"] + terms["ssl1_node"]
    assert all(g is None for g in torch.autograd.grad(graph_only, parameters, allow_unused=True))
    for model in (objective.student, objective.teacher):
        for module in model.modules():
            if hasattr(module, "checkpoint_layers"):
                module.checkpoint_layers = True
            if hasattr(module, "checkpoint_chunks"):
                module.checkpoint_chunks = True
    actual = evaluate_loss_diagnostics(objective, batch, bf16=False)
    for name, value in terms.items():
        assert actual["components"][name] == pytest.approx(
            float(value.detach()), rel=2e-5, abs=2e-6
        )
    for name, gradient in expected_gradients.items():
        assert actual["shared_basal_weighted_gradient_l2"][name] == pytest.approx(
            float(gradient.norm()), rel=2e-5, abs=2e-6
        )
    expected_cosine = torch.nn.functional.cosine_similarity(
        expected_gradients["prediction"], expected_gradients["ssl"], dim=0
    )
    assert actual["prediction_ssl_gradient_cosine"] == pytest.approx(
        float(expected_cosine), abs=2e-6
    )

    # A diagnostic between nonzero-LR updates must not change optimizer/EMA,
    # centers, existing .grad tensors, auxiliary counters, or random streams.
    without = copy.deepcopy(objective)
    snapshots = []
    for version, diagnose in [(without, False), (objective, True)]:
        torch.manual_seed(402)
        optimizer = V2Optimizer(version.student, 0.001, 0)
        metrics = []
        for _ in range(2):
            metrics.append(
                optimizer_step(
                    version, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
                )
            )
            if diagnose:
                before = [None if p.grad is None else p.grad.clone() for p in version.parameters()]
                evaluate_loss_diagnostics(version, batch, bf16=False)
                for p, previous in zip(version.parameters(), before, strict=True):
                    assert (p.grad is None) == (previous is None)
                    if previous is not None:
                        torch.testing.assert_close(p.grad, previous, atol=0, rtol=0)
        snapshots.append(
            (
                metrics,
                copy.deepcopy(version.state_dict()),
                optimizer.state_dict(),
                torch.get_rng_state().clone(),
            )
        )

    def exact(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                exact(a, b)
        else:
            assert left == right

    exact(snapshots[0], snapshots[1])


@pytest.mark.parametrize("cls", [0, 1])
def test_auxiliary_counter_heads_optimizer_and_rng_resume_exactly(tmp_path, cls):
    objective, batch = masked_objective(cls=cls)
    optimizer = V2Optimizer(objective.student, 0.001, 0)
    generator = np.random.default_rng(3)
    kwargs = dict(microbatch=1, lr=0.001, momentum=0.9, bf16=False)
    for _ in range(3):
        optimizer_step(objective, optimizer, batch, **kwargs)
    save_checkpoint(
        tmp_path / "toy.pt",
        objective,
        optimizer,
        identity={"test": "toy"},
        progress={"epoch": 1},
        generator=generator,
    )
    expected_metrics = optimizer_step(objective, optimizer, batch, **kwargs)
    expected = copy.deepcopy(objective.state_dict())
    restored, _ = masked_objective(cls=cls)
    restored_optimizer = V2Optimizer(restored.student, 0.001, 0)
    load_checkpoint(
        tmp_path / "toy.pt",
        restored,
        restored_optimizer,
        identity={"test": "toy"},
        generator=generator,
    )
    actual_metrics = optimizer_step(restored, restored_optimizer, batch, **kwargs)
    assert expected_metrics == actual_metrics
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


def test_global_condition_weights_are_not_microbatch_or_rank_means():
    ids = torch.tensor([0, 0, 0, 1])
    weights = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), "condition_mean")
    torch.testing.assert_close(weights, torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2]))
    errors = torch.tensor([1.0, 1, 1, 9], requires_grad=True)
    complete = (weights * errors).sum()
    fragmented = sum((weights[i : i + 1] * errors[i : i + 1]).sum() for i in range(4))
    torch.testing.assert_close(complete, fragmented)
    torch.testing.assert_close(torch.autograd.grad(fragmented, errors)[0], weights)


def _prediction_rank_worker(rank, rendezvous, reference, power):
    from datetime import timedelta

    import torch.distributed as dist
    from test_distributed import _GradientCapture

    from gradpert.training.v2.engine import slice_cells

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model, batch = four_rows()
        objective = JointObjective(
            model,
            0,
            0,
            loss_reduction="row_mean",
            prediction_error_power=power,
            prediction_reduction_override="condition_mean",
        )
        local = slice_cells(batch, rank * 2, rank * 2 + 2)
        metrics = optimizer_step(
            objective,
            _GradientCapture(model),
            local,
            microbatch=1,
            lr=0.001,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index,
        )
        expected = torch.load(reference, weights_only=True)
        assert abs(metrics["prediction"] - expected["prediction"]) < 2e-6
        for name, parameter in model.named_parameters():
            if expected["gradients"][name] is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected["gradients"][name], atol=3e-6, rtol=1e-4
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("power", [2, 4])
def test_two_rank_prediction_only_condition_weights_match_global_gradient(tmp_path, power):
    from test_distributed import _GradientCapture

    model, batch = four_rows()
    objective = JointObjective(
        model,
        0,
        0,
        loss_reduction="row_mean",
        prediction_error_power=power,
        prediction_reduction_override="condition_mean",
    )
    metrics = optimizer_step(
        objective, _GradientCapture(model), batch, microbatch=4, lr=0.001, momentum=0.99, bf16=False
    )
    reference = str(tmp_path / "toy-reference.pt")
    torch.save(
        {
            "gradients": {n: p.grad for n, p in model.named_parameters()},
            "prediction": metrics["prediction"],
        },
        reference,
    )
    torch.multiprocessing.spawn(
        _prediction_rank_worker,
        args=(str(tmp_path / "rank-init"), reference, power),
        nprocs=2,
        join=True,
    )


def _auxiliary_rank_checkpoint_worker(rank, rendezvous, path):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        objective, _ = masked_objective()
        optimizer = V2Optimizer(objective.student, 0.001, 0)
        generator = np.random.default_rng(rank)
        objective.auxiliary_rng_counter.fill_(2 + rank * 5)
        save_checkpoint(
            Path(path),
            objective,
            optimizer,
            identity={"test": "ranked-toy"},
            progress={"epoch": 0},
            generator=generator,
        )
        objective.auxiliary_rng_counter.zero_()
        load_checkpoint(
            Path(path), objective, optimizer, identity={"test": "ranked-toy"}, generator=generator
        )
        assert objective.auxiliary_rng_counter.item() == 2 + rank * 5
    finally:
        dist.destroy_process_group()


def test_auxiliary_checkpoint_restores_each_ranks_own_counter(tmp_path):
    torch.multiprocessing.spawn(
        _auxiliary_rank_checkpoint_worker,
        args=(str(tmp_path / "aux-init"), str(tmp_path / "toy-rank.pt")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("cls", [0, 1])
def test_auxiliary_recomputation_preserves_main_and_auxiliary_updates(cls):
    from test_relay_method import assert_tree_close, relay_training_fixture

    base, batch = relay_training_fixture(checkpointed=False)
    plain = JointObjective(
        base.student,
        loss_reduction="row_mean",
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
        auxiliary_mask_ratio=0.2,
        lambda_gene_mask=1.0,
        lambda_cls_mask=cls,
        auxiliary_seed=91,
    )
    replayed = copy.deepcopy(plain)
    for model in (replayed.student, replayed.teacher):
        for module in model.modules():
            if hasattr(module, "checkpoint_layers"):
                module.checkpoint_layers = True
            if hasattr(module, "checkpoint_chunks"):
                module.checkpoint_chunks = True
    observed = []
    for objective in (plain, replayed):
        optimizer = V2Optimizer(objective.student, 0.001, 0)
        torch.manual_seed(102)
        metrics = []
        for _ in range(3):
            metrics.append(
                optimizer_step(
                    objective, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
                )
            )
        observed.append(
            (
                metrics,
                copy.deepcopy(objective.state_dict()),
                optimizer.state_dict(),
                torch.get_rng_state().clone(),
            )
        )
    assert_tree_close(observed[0], observed[1])


def test_six_epoch_final_only_toy_lifecycle_records_train_diagnostics(tmp_path):
    import json

    from gradpert.config.step_schedule import EndpointLRWarmupCosine
    from gradpert.training.v2.lifecycle import fit

    objective, batch = masked_objective()
    optimizer = V2Optimizer(objective.student, 0.001, 0)
    journal = fit(
        objective,
        optimizer,
        root=tmp_path,
        identity={"source": "toy"},
        generator=np.random.default_rng(4),
        epochs=6,
        steps_per_epoch=2,
        batches=lambda epoch: [batch, batch],
        validate=None,
        schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
        teacher_start=0.99,
        teacher_end=1,
        microbatch=1,
        bf16=False,
        diagnose=lambda: evaluate_loss_diagnostics(objective, batch, bf16=False),
    )
    assert journal["best"] is None and journal["last"]["epoch"] == 6
    assert len(list(tmp_path.glob("loss-diagnostic-epoch-*.json"))) == 7
    history = json.loads((tmp_path / "history.json").read_text())
    assert len({r["training_diagnostic"]["input_sha256"] for r in history}) == 1
    assert all(r["training_diagnostic"]["split"] == "train" for r in history)
    assert all(r["condition_prediction_weight_audit"] for r in history)
    assert all(r["validation"]["performed"] is False for r in history)


def test_frozen_loss_manifest_rejects_unrelated_protocol_changes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(root / "scripts/v2"))
    spec = importlib.util.spec_from_file_location(
        "run_loss_group", root / "scripts/v2/run_loss_group.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = root / "configs/v2/cap40_loss_ablations_20261007/manifest.json"
    assert len(module.verify_manifest(root, manifest)["rows"]) == 6
    import json

    data = json.loads(manifest.read_text())
    data["global_batch"] = 128
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="authorized six-arm"):
        module.verify_manifest(root, changed)


@pytest.mark.parametrize("cls", [0, 1])
def test_weight1_heads_and_shared_encoder_update_and_evaluate_without_auxiliary_outputs(
    tmp_path, cls
):
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint

    objective, batch = masked_objective(cls=cls)
    optimizer = V2Optimizer(objective.student, 0.001, 0)
    initial = copy.deepcopy(objective.student.state_dict())
    for _ in range(3):
        metrics = optimizer_step(
            objective, optimizer, batch, microbatch=1, lr=0.001, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["gene_mask"] > 0 and metrics["prediction_mse"] > 0
    current = objective.student.state_dict()
    assert not torch.equal(
        initial["control_reconstruction.gene.weight"], current["control_reconstruction.gene.weight"]
    )
    cell_keys = [name for name in initial if name.startswith("cell.")]
    assert cell_keys and any(not torch.equal(initial[name], current[name]) for name in cell_keys)
    for name in [
        "control_reconstruction.cls_projection.weight",
        "control_reconstruction.gene_projection.weight",
    ]:
        assert torch.equal(initial[name], current[name]) is (not bool(cls))
    objective.eval()
    graph, conditions = objective._graph(objective.student, batch.graph, False)
    expected = (
        objective.student.encode_response(
            graph[batch.query_positions], batch.control, conditions[batch.condition_index]
        )["prediction"]
        .detach()
        .clone()
    )
    path = tmp_path / "evaluation-toy.pt"
    save_checkpoint(
        path,
        objective,
        optimizer,
        identity={"test": "weight1"},
        progress={"epoch": 6},
        generator=np.random.default_rng(4),
    )
    restored, _ = masked_objective(cls=cls)
    torch.manual_seed(713)
    rng = torch.get_rng_state().clone()
    assert load_evaluation_checkpoint(
        path, restored, training_identity={"test": "weight1"}, checkpoint_sha256=sha256_file(path)
    ) == {"epoch": 6}
    assert torch.equal(torch.get_rng_state(), rng)
    restored.eval()

    def predict():
        g, c = restored._graph(restored.student, batch.graph, False)
        return restored.student.encode_response(
            g[batch.query_positions], batch.control, c[batch.condition_index]
        )["prediction"]

    torch.testing.assert_close(predict(), expected, atol=0, rtol=0)
    with torch.no_grad():
        for parameter in restored.student.control_reconstruction.parameters():
            parameter.add_(100)
    torch.testing.assert_close(predict(), expected, atol=0, rtol=0)
