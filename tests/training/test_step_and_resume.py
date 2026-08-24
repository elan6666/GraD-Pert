from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.graphs import GraphTopology, prune_incoming_edges  # noqa: E402
from gradpert.modeling import CenterState, GraDPertJointModel  # noqa: E402
from gradpert.training.batch import GraDPertTrainingBatch  # noqa: E402
from gradpert.training.checkpoint import (  # noqa: E402
    CheckpointIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)
from gradpert.training.logging import TrainingReceiptWriter  # noqa: E402
from gradpert.training.step import (  # noqa: E402
    GraDPertStepEngine,
    GraDPertStepMetrics,
    build_native_optimizer,
)


def _topology() -> GraphTopology:
    genes = ("A", "B", "C", "D", "E", "F", "TARGET_ONLY")
    edges = (
        ("B", "A", 1.0),
        ("C", "A", 1.0),
        ("D", "B", 1.0),
        ("E", "C", 1.0),
        ("F", "D", 1.0),
        ("A", "E", 1.0),
    )
    return GraphTopology(
        gene_ids=genes,
        sources={
            name: prune_incoming_edges(
                source_name=name,
                gene_ids=genes,
                weighted_edges=edges,
                top_k=20,
            )
            for name in ("string", "go")
        },
    )


def _batch() -> GraDPertTrainingBatch:
    return GraDPertTrainingBatch(
        control_expression=torch.randn(2, 5),
        target_expression=torch.randn(2, 5),
        condition_ids=("A+ctrl", "B+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
        perturbed_row_ids=("p1", "p2"),
        control_row_ids=("c1", "c2"),
    )


def _components(device: torch.device | None = None):  # type: ignore[no-untyped-def]
    target_device = device or torch.device("cpu")
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    ).to(target_device)
    optimizer = build_native_optimizer(model)
    centers = CenterState.zeros(prototype_count=8192, device=target_device)
    engine = GraDPertStepEngine(
        model=model,
        topology=_topology(),
        optimizer=optimizer,
        centers=centers,
        run_seed=1,
        total_schedule_steps=400,
        heldout_target_ids=(6,),
    )
    return model, optimizer, centers, engine


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        source_commit="a" * 40,
        source_tree_sha256="f" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        canonical_data_sha256="d" * 64,
        split_content_sha256="e" * 64,
    )


def _step_metrics() -> GraDPertStepMetrics:
    return GraDPertStepMetrics(
        total_loss=1.0,
        prediction_loss=0.8,
        condition_consistency_loss=0.1,
        masked_node_loss=0.1,
        spread_loss=0.0,
        spread_available=False,
        teacher_momentum=0.996,
        prediction_graph_gradient_norm=1.0,
        ssl_graph_gradient_norm=1.0,
        weighted_ssl_graph_gradient_norm=0.1,
        prediction_to_weighted_ssl_gradient_ratio=10.0,
        condition_target_entropy=1.0,
        masked_node_target_entropy=1.0,
        condition_prototypes_used=1,
        masked_node_prototypes_used=1,
        condition_center_norm=1.0,
        masked_node_center_norm=1.0,
        unique_condition_count=2,
        masked_node_count=1,
    )


def test_native_step_obeys_gradient_and_update_boundaries() -> None:
    torch.manual_seed(10)
    model, _, centers, engine = _components()
    before_teacher = {
        name: value.detach().clone() for name, value in model.teacher_encoder.named_parameters()
    }
    metrics = engine.train_step(_batch(), global_step=0)
    assert metrics.unique_condition_count == 2
    assert metrics.spread_available
    assert metrics.total_loss > 0
    assert metrics.condition_center_norm > 0
    assert centers.condition.norm().item() > 0
    assert any(
        not torch.equal(before_teacher[name], parameter)
        for name, parameter in model.teacher_encoder.named_parameters()
    )
    assert not any(parameter.grad is not None for parameter in model.teacher_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.basal_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.student_projector.parameters())
    assert model.student_encoder.mask_token.grad is not None


def test_checkpoint_resume_reproduces_the_next_step(tmp_path: Path) -> None:
    torch.manual_seed(123)
    batch = _batch()
    _, optimizer, centers, engine = _components()
    engine.train_step(batch, global_step=0)
    checkpoint = tmp_path / "epoch.pt"
    save_training_checkpoint(
        checkpoint,
        model=engine.model,
        optimizer=optimizer,
        centers=centers,
        progress={"completed_epochs": 1, "global_step": 1},
        identity=_identity(),
    )
    uninterrupted = engine.train_step(batch, global_step=1)
    uninterrupted_state = {
        name: value.detach().clone() for name, value in engine.model.state_dict().items()
    }

    _, resumed_optimizer, resumed_centers, resumed_engine = _components()
    progress = load_training_checkpoint(
        checkpoint,
        model=resumed_engine.model,
        optimizer=resumed_optimizer,
        centers=resumed_centers,
        expected_identity=_identity(),
    )
    assert progress == {"completed_epochs": 1, "global_step": 1}
    resumed = resumed_engine.train_step(batch, global_step=1)
    assert resumed.total_loss == pytest.approx(uninterrupted.total_loss, rel=0, abs=1e-7)
    for name, value in resumed_engine.model.state_dict().items():
        assert torch.equal(value, uninterrupted_state[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint path requires a GPU")
def test_checkpoint_loaded_on_cuda_restores_cpu_and_cuda_rng_states(tmp_path: Path) -> None:
    device = torch.device("cuda:0")
    model, optimizer, centers, _ = _components(device)
    checkpoint = tmp_path / "cuda.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        centers=centers,
        progress={"completed_epochs": 1, "global_step": 1},
        identity=_identity(),
    )

    resumed_model, resumed_optimizer, resumed_centers, _ = _components(device)
    progress = load_training_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        centers=resumed_centers,
        expected_identity=_identity(),
    )

    assert progress == {"completed_epochs": 1, "global_step": 1}
    assert torch.get_rng_state().device.type == "cpu"
    assert all(state.device.type == "cpu" for state in torch.cuda.get_rng_state_all())


def test_training_receipts_reject_duplicate_steps_across_resume(tmp_path: Path) -> None:
    writer = TrainingReceiptWriter(tmp_path)
    writer.write_step(epoch=0, global_step=0, metrics=_step_metrics())

    resumed = TrainingReceiptWriter(tmp_path)
    with pytest.raises(RuntimeError, match="expected 1"):
        resumed.write_step(epoch=0, global_step=0, metrics=_step_metrics())
    resumed.write_step(epoch=0, global_step=1, metrics=_step_metrics())


def test_training_receipts_require_identical_run_metadata_on_resume(tmp_path: Path) -> None:
    writer = TrainingReceiptWriter(tmp_path)
    writer.write_run_meta({"run_id": "same", "mode": "full"})
    TrainingReceiptWriter(tmp_path).write_run_meta({"run_id": "same", "mode": "full"})
    with pytest.raises(ValueError, match="differs"):
        TrainingReceiptWriter(tmp_path).write_run_meta({"run_id": "changed", "mode": "full"})


def test_sealed_test_gate_is_claimed_at_most_once(tmp_path: Path) -> None:
    writer = TrainingReceiptWriter(tmp_path)
    writer.claim_test_once()
    with pytest.raises(RuntimeError, match="already been claimed"):
        TrainingReceiptWriter(tmp_path).claim_test_once()
    writer.complete_test_once()
    with pytest.raises(RuntimeError, match="already been claimed"):
        TrainingReceiptWriter(tmp_path).claim_test_once()
