from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.config.native import NativeArchitectureOptions  # noqa: E402
from gradpert.features import GENEPT_EMB_B_SHA256  # noqa: E402
from gradpert.graphs import (  # noqa: E402
    GraphTopology,
    prune_incoming_edges,
    resolve_local_view_contract,
)
from gradpert.modeling import CenterState, GraDPertJointModel  # noqa: E402
from gradpert.training.batch import GraDPertTrainingBatch  # noqa: E402
from gradpert.training.checkpoint import (  # noqa: E402
    CheckpointIdentity,
    clone_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from gradpert.training.logging import TrainingReceiptWriter  # noqa: E402
from gradpert.training.step import (  # noqa: E402
    GRADPERT_STAGE_PHASE_IDS,
    GraDPertStageEvent,
    GraDPertStageObserver,
    GraDPertStepEngine,
    GraDPertStepMetrics,
    LossWeights,
    build_native_optimizer,
    require_local_view_anchor_capacity,
)
from gradpert.training.trainer import GraDPertTrainer  # noqa: E402


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


def _components(
    device: torch.device | None = None,
    *,
    capture_equivalence_health: bool = False,
    stage_observer: GraDPertStageObserver | None = None,
):  # type: ignore[no-untyped-def]
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
        capture_equivalence_health=capture_equivalence_health,
        stage_observer=stage_observer,
    )
    return model, optimizer, centers, engine


def _vnext_architecture(
    *,
    gene_feature_mode: str = "learned_id",
    decoder_mode: str = "additive",
    graph_encoder_family: str = "multi_source_sparse_transformer",
) -> NativeArchitectureOptions:
    parameters: dict[str, object] = {
        "graph_axis_policy": "canonical_full",
        "graph_hvg_count": 5000,
        "graph_sources": "string_go",
        "graph_encoder_family": graph_encoder_family,
        "graph_encoder_dropout": 0.1,
        "local_view_builder": "fanout",
        "local_view_node_budget_ratio": "1/2",
        "local_anchor_mask_view_ratio": "0/1",
        "gene_feature_mode": gene_feature_mode,
        "decoder_mode": decoder_mode,
    }
    if gene_feature_mode != "learned_id":
        parameters["genept_expected_sha256"] = GENEPT_EMB_B_SHA256
    return NativeArchitectureOptions.from_parameters(parameters)


def _vnext_components(
    *,
    gene_feature_mode: str = "learned_id",
    decoder_mode: str = "additive",
    graph_encoder_family: str = "multi_source_sparse_transformer",
    checkpoint_student_local_activations: bool = False,
    checkpoint_student_local_activation_count: int | None = None,
    capture_equivalence_health: bool = False,
):  # type: ignore[no-untyped-def]
    architecture = _vnext_architecture(
        gene_feature_mode=gene_feature_mode,
        decoder_mode=decoder_mode,
        graph_encoder_family=graph_encoder_family,
    )
    genept = (
        None
        if gene_feature_mode == "learned_id"
        else torch.arange(7 * 1536, dtype=torch.float32).reshape(7, 1536) / 1536
    )
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=architecture,
        genept_matrix=genept,
    )
    optimizer = build_native_optimizer(model)
    centers = CenterState.zeros(prototype_count=8192, device=torch.device("cpu"))
    engine = GraDPertStepEngine(
        model=model,
        topology=_topology(),
        optimizer=optimizer,
        centers=centers,
        run_seed=1,
        total_schedule_steps=400,
        heldout_target_ids=(6,),
        architecture=architecture,
        checkpoint_student_local_activations=checkpoint_student_local_activations,
        checkpoint_student_local_activation_count=checkpoint_student_local_activation_count,
        capture_equivalence_health=capture_equivalence_health,
    )
    return model, optimizer, centers, engine


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_nested_exact(observed, expected) -> None:  # type: ignore[no-untyped-def]
    if isinstance(expected, torch.Tensor):
        assert isinstance(observed, torch.Tensor)
        assert torch.equal(observed, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        assert observed.keys() == expected.keys()
        for key in expected:
            _assert_nested_exact(observed[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert type(observed) is type(expected)
        assert len(observed) == len(expected)
        for observed_item, expected_item in zip(observed, expected, strict=True):
            _assert_nested_exact(observed_item, expected_item)
        return
    assert observed == expected


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
        auxiliary_graph_gradient_norm=0.1,
        prediction_to_auxiliary_gradient_ratio=10.0,
        condition_target_entropy=1.0,
        masked_node_target_entropy=1.0,
        condition_prototypes_used=1,
        masked_node_prototypes_used=1,
        condition_center_norm=1.0,
        masked_node_center_norm=1.0,
        unique_condition_count=2,
        masked_node_count=1,
        batch_cell_count=4,
        data_read_ms=1.0,
        host_to_device_ms=1.0,
        view_build_ms=1.0,
        teacher_forward_ms=1.0,
        student_global_ms=1.0,
        student_local_ms=1.0,
        prediction_ms=1.0,
        backward_update_ms=1.0,
        step_wall_ms=5.0,
        local_view_realization_count=16,
        local_node_count_sum=112,
        local_node_count_min=7,
        local_node_count_max=7,
        local_budget_hit_count=16,
        local_node_counts_sha256="a" * 64,
        masked_local_assignment_count=8,
        masked_local_index_counts_json="[1,1,1,1,1,1,1,1]",
        masked_local_assignments_sha256="b" * 64,
    )


def test_default_engine_preserves_v1_fixed_local_view_semantics() -> None:
    _, _, _, engine = _components()
    assert engine.local_view_contract.derivation_mode == "legacy_fixed"
    assert engine.local_view_contract.effective_node_budget == 7
    assert engine.local_view_contract.effective_mask_view_count == 4


def test_explicit_model_architecture_requires_matching_engine_architecture() -> None:
    architecture = _vnext_architecture()
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=architecture,
    )
    optimizer = build_native_optimizer(model)
    centers = CenterState.zeros(prototype_count=8192, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="explicit model architecture"):
        GraDPertStepEngine(
            model=model,
            topology=_topology(),
            optimizer=optimizer,
            centers=centers,
            run_seed=1,
            total_schedule_steps=400,
            heldout_target_ids=(6,),
        )


def test_engine_rejects_mismatched_pre_resolved_local_view_contract() -> None:
    architecture = _vnext_architecture()
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=architecture,
    )
    optimizer = build_native_optimizer(model)
    centers = CenterState.zeros(prototype_count=8192, device=torch.device("cpu"))
    mismatched = resolve_local_view_contract(
        graph_node_count=7,
        local_view_count=8,
        node_budget_ratio=(1, 1),
        mask_view_ratio=(0, 1),
    )
    with pytest.raises(ValueError, match="pre-resolved and engine"):
        GraDPertStepEngine(
            model=model,
            topology=_topology(),
            optimizer=optimizer,
            centers=centers,
            run_seed=1,
            total_schedule_steps=400,
            heldout_target_ids=(6,),
            architecture=architecture,
            local_view_contract=mismatched,
        )


def test_anchor_capacity_preflight_reports_condition_before_model_construction() -> None:
    contract = resolve_local_view_contract(
        graph_node_count=7,
        local_view_count=8,
        node_budget_ratio=(1, 7),
        mask_view_ratio=(0, 1),
    )
    with pytest.raises(ValueError, match="condition='A\\+B', anchors=2, budget=1"):
        require_local_view_anchor_capacity(contract, {"A+B": (0, 1)})


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


def test_stage_observer_emits_stable_order_and_index_context() -> None:
    events: list[GraDPertStageEvent] = []

    def observer(event: GraDPertStageEvent, engine: GraDPertStepEngine) -> None:
        assert engine.stage_observer is observer
        events.append(event)

    _seed_all(41)
    _, _, _, engine = _components(stage_observer=observer)
    engine.train_step(_batch(), global_step=0)

    expected: list[tuple[str, str, int | None, int | None, int | None, str | None]] = []

    def add_pair(
        phase_id: str,
        *,
        global_view_index: int | None = None,
        local_view_index: int | None = None,
        condition_index: int | None = None,
        condition_id: str | None = None,
    ) -> None:
        for status in ("entered", "completed"):
            expected.append(
                (
                    phase_id,
                    status,
                    global_view_index,
                    local_view_index,
                    condition_index,
                    condition_id,
                )
            )

    add_pair("views")
    add_pair("teacher_global_forward")
    for global_index in range(2):
        add_pair("teacher_global", global_view_index=global_index)
    for global_index in range(2):
        add_pair("teacher_global_projection", global_view_index=global_index)
    add_pair("student_global_forward")
    for global_index in range(2):
        add_pair("student_global", global_view_index=global_index)
    for global_index in range(2):
        add_pair("student_global_projection", global_view_index=global_index)
    for local_index in range(8):
        expected.append(("student_local_index", "entered", None, local_index, None, None))
        for condition_index, condition_id in enumerate(("A+ctrl", "B+ctrl")):
            add_pair(
                "student_local_view",
                local_view_index=local_index,
                condition_index=condition_index,
                condition_id=condition_id,
            )
        expected.append(("student_local_index", "completed", None, local_index, None, None))
    for phase_id in (
        "condition_consistency_loss",
        "masked_node_loss",
        "spread_loss",
        "prediction_forward",
        "auxiliary_grad",
        "prediction_backward",
        "gradient_merge",
        "optimizer",
        "ema",
        "centers",
    ):
        add_pair(phase_id)

    observed = [
        (
            event.phase_id,
            event.status,
            event.global_view_index,
            event.local_view_index,
            event.condition_index,
            event.condition_id,
        )
        for event in events
    ]
    assert observed == expected
    assert {event.phase_id for event in events} == set(GRADPERT_STAGE_PHASE_IDS)
    assert all(event.schema_version == "gradpert-stage-event-v1" for event in events)
    assert all(event.failure_type is None and event.failure_message is None for event in events)
    assert engine.stage_observer_failures == []


def test_stage_observer_preserves_loss_gradients_rng_and_state() -> None:
    _seed_all(71)
    baseline_model, _, baseline_centers, baseline = _components(capture_equivalence_health=True)
    baseline_metrics = baseline.train_step(_batch(), global_step=0)
    baseline_state = {
        name: value.detach().clone() for name, value in baseline_model.state_dict().items()
    }
    baseline_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in baseline_model.named_parameters()
    }
    baseline_condition_center = baseline_centers.condition.detach().clone()
    baseline_masked_center = baseline_centers.masked_node.detach().clone()

    observed_events: list[GraDPertStageEvent] = []
    _seed_all(71)
    observed_model, _, observed_centers, observed = _components(
        capture_equivalence_health=True,
        stage_observer=lambda event, _engine: observed_events.append(event),
    )
    observed_metrics = observed.train_step(_batch(), global_step=0)

    assert observed_metrics.total_loss == pytest.approx(baseline_metrics.total_loss, abs=0, rel=0)
    assert observed.first_step_health == baseline.first_step_health
    assert observed.stage_observer_failures == []
    assert observed_events
    for name, value in observed_model.state_dict().items():
        assert torch.equal(value, baseline_state[name])
    for name, parameter in observed_model.named_parameters():
        expected_gradient = baseline_gradients[name]
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected_gradient)
    assert torch.equal(observed_centers.condition, baseline_condition_center)
    assert torch.equal(observed_centers.masked_node, baseline_masked_center)


@pytest.mark.parametrize(
    "graph_encoder_family",
    ["multi_source_sparse_transformer", "adaptive_relation_gat"],
)
def test_local_activation_checkpoint_preserves_complete_first_step_trajectory(
    graph_encoder_family: str,
) -> None:
    batch = GraDPertTrainingBatch(
        control_expression=torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10,
        target_expression=torch.arange(10, 20, dtype=torch.float32).reshape(2, 5) / 10,
        condition_ids=("A+ctrl", "B+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
        perturbed_row_ids=("p1", "p2"),
        control_row_ids=("c1", "c2"),
        perturbed_row_ids_sha256="1" * 64,
        control_row_ids_sha256="2" * 64,
        pretransfer_control_sha256="3" * 64,
        pretransfer_target_sha256="4" * 64,
    )

    _seed_all(20260830)
    baseline_model, baseline_optimizer, baseline_centers, baseline = _vnext_components(
        graph_encoder_family=graph_encoder_family, capture_equivalence_health=True
    )
    _seed_all(930)
    baseline_metrics = baseline.train_step(batch, global_step=0)
    assert baseline.first_step_health is not None
    assert baseline.first_step_health["schema_version"] == "native-first-step-equivalence-v2"
    for name in (
        "teacher_state_after_sha256",
        "gradient_state_after_sha256",
        "optimizer_state_after_sha256",
        "centers_state_after_sha256",
        "prediction_content_sha256",
    ):
        assert len(str(baseline.first_step_health[name])) == 64
    baseline_rng = torch.get_rng_state().clone()
    baseline_model_state = {
        name: value.detach().clone() for name, value in baseline_model.state_dict().items()
    }
    baseline_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in baseline_model.named_parameters()
    }
    baseline_optimizer_state = baseline_optimizer.state_dict()
    baseline_condition_center = baseline_centers.condition.detach().clone()
    baseline_masked_center = baseline_centers.masked_node.detach().clone()

    _seed_all(20260830)
    observed_model, observed_optimizer, observed_centers, observed = _vnext_components(
        graph_encoder_family=graph_encoder_family,
        checkpoint_student_local_activations=True,
        capture_equivalence_health=True,
    )
    _seed_all(930)
    observed_metrics = observed.train_step(batch, global_step=0)

    for name in baseline_metrics.__dataclass_fields__:
        if not name.endswith("_ms"):
            assert getattr(observed_metrics, name) == getattr(baseline_metrics, name), name
    assert observed.first_step_health == baseline.first_step_health
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for name, value in observed_model.state_dict().items():
        assert torch.equal(value, baseline_model_state[name]), name
    for name, parameter in observed_model.named_parameters():
        expected = baseline_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected), name
    _assert_nested_exact(observed_optimizer.state_dict(), baseline_optimizer_state)
    assert torch.equal(observed_centers.condition, baseline_condition_center)
    assert torch.equal(observed_centers.masked_node, baseline_masked_center)


@pytest.mark.parametrize("checkpoint_count", [0, 1, 2])
def test_selective_local_activation_checkpoint_preserves_complete_first_step_trajectory(
    checkpoint_count: int,
) -> None:
    batch = GraDPertTrainingBatch(
        control_expression=torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10,
        target_expression=torch.arange(10, 20, dtype=torch.float32).reshape(2, 5) / 10,
        condition_ids=("A+ctrl", "B+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
        perturbed_row_ids=("p1", "p2"),
        control_row_ids=("c1", "c2"),
        perturbed_row_ids_sha256="1" * 64,
        control_row_ids_sha256="2" * 64,
        pretransfer_control_sha256="3" * 64,
        pretransfer_target_sha256="4" * 64,
    )

    _seed_all(20260830)
    baseline_model, baseline_optimizer, baseline_centers, baseline = _vnext_components(
        capture_equivalence_health=True
    )
    _seed_all(930)
    baseline_metrics = baseline.train_step(batch, global_step=0)
    baseline_rng = torch.get_rng_state().clone()
    baseline_model_state = {
        name: value.detach().clone() for name, value in baseline_model.state_dict().items()
    }
    baseline_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in baseline_model.named_parameters()
    }
    baseline_optimizer_state = baseline_optimizer.state_dict()
    baseline_condition_center = baseline_centers.condition.detach().clone()
    baseline_masked_center = baseline_centers.masked_node.detach().clone()

    _seed_all(20260830)
    observed_model, observed_optimizer, observed_centers, observed = _vnext_components(
        checkpoint_student_local_activations=True,
        checkpoint_student_local_activation_count=checkpoint_count,
        capture_equivalence_health=True,
    )
    _seed_all(930)
    observed_metrics = observed.train_step(batch, global_step=0)

    for name in baseline_metrics.__dataclass_fields__:
        if not name.endswith("_ms"):
            assert getattr(observed_metrics, name) == getattr(baseline_metrics, name), name
    assert observed.first_step_health == baseline.first_step_health
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for name, value in observed_model.state_dict().items():
        assert torch.equal(value, baseline_model_state[name]), name
    for name, parameter in observed_model.named_parameters():
        expected = baseline_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected), name
    _assert_nested_exact(observed_optimizer.state_dict(), baseline_optimizer_state)
    assert torch.equal(observed_centers.condition, baseline_condition_center)
    assert torch.equal(observed_centers.masked_node, baseline_masked_center)


def test_selective_local_activation_checkpoint_count_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires checkpointing"):
        _vnext_components(checkpoint_student_local_activation_count=1)
    with pytest.raises(ValueError, match="requires checkpointing"):
        _vnext_components(checkpoint_student_local_activation_count=0)
    with pytest.raises(ValueError, match="nonnegative"):
        _vnext_components(
            checkpoint_student_local_activations=True,
            checkpoint_student_local_activation_count=-1,
        )
    with pytest.raises(ValueError, match="exceeds the local-view count"):
        _vnext_components(
            checkpoint_student_local_activations=True,
            checkpoint_student_local_activation_count=9,
        )


def test_cpu_vectorized_sparse_union_preserves_complete_first_step_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = GraDPertTrainingBatch(
        control_expression=torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10,
        target_expression=torch.arange(10, 20, dtype=torch.float32).reshape(2, 5) / 10,
        condition_ids=("A+ctrl", "B+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
        perturbed_row_ids=("p1", "p2"),
        control_row_ids=("c1", "c2"),
        perturbed_row_ids_sha256="1" * 64,
        control_row_ids_sha256="2" * 64,
        pretransfer_control_sha256="3" * 64,
        pretransfer_target_sha256="4" * 64,
    )

    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "reference")
    _seed_all(20260830)
    reference_model, reference_optimizer, reference_centers, reference = _vnext_components(
        checkpoint_student_local_activations=True,
        capture_equivalence_health=True,
    )
    _seed_all(930)
    reference_metrics = reference.train_step(batch, global_step=0)
    reference_rng = torch.get_rng_state().clone()
    reference_model_state = {
        name: value.detach().clone() for name, value in reference_model.state_dict().items()
    }
    reference_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in reference_model.named_parameters()
    }
    reference_optimizer_state = reference_optimizer.state_dict()
    reference_condition_center = reference_centers.condition.detach().clone()
    reference_masked_center = reference_centers.masked_node.detach().clone()

    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "cpu_vectorized")
    _seed_all(20260830)
    optimized_model, optimized_optimizer, optimized_centers, optimized = _vnext_components(
        checkpoint_student_local_activations=True,
        capture_equivalence_health=True,
    )
    _seed_all(930)
    optimized_metrics = optimized.train_step(batch, global_step=0)

    for name in reference_metrics.__dataclass_fields__:
        if not name.endswith("_ms"):
            assert getattr(optimized_metrics, name) == getattr(reference_metrics, name), name
    assert optimized.first_step_health == reference.first_step_health
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for name, value in optimized_model.state_dict().items():
        assert torch.equal(value, reference_model_state[name]), name
    for name, parameter in optimized_model.named_parameters():
        expected = reference_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected), name
    _assert_nested_exact(optimized_optimizer.state_dict(), reference_optimizer_state)
    assert torch.equal(optimized_centers.condition, reference_condition_center)
    assert torch.equal(optimized_centers.masked_node, reference_masked_center)


def test_stage_observer_reports_mid_step_primary_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    events: list[GraDPertStageEvent] = []

    def failing_observer(event: GraDPertStageEvent, _engine: GraDPertStepEngine) -> None:
        events.append(event)
        raise LookupError("secondary observer failure")

    _, _, _, engine = _components(stage_observer=failing_observer)
    original_forward_many = engine.model.student_encoder.forward_many
    call_count = 0

    def fail_first_local(views):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic local forward failure")
        return original_forward_many(views)

    monkeypatch.setattr(engine.model.student_encoder, "forward_many", fail_first_local)
    with pytest.raises(RuntimeError, match="synthetic local forward failure"):
        engine.train_step(_batch(), global_step=0)

    assert [(event.phase_id, event.status, event.local_view_index) for event in events[-2:]] == [
        ("student_local_index", "entered", 0),
        ("student_local_index", "failure", 0),
    ]
    assert events[-1].failure_type == "RuntimeError"
    assert events[-1].failure_message == "synthetic local forward failure"
    assert not any(event.phase_id == "optimizer" for event in events)
    assert engine.stage_observer_failures[-1]["failure_type"] == "LookupError"
    assert engine.stage_observer_failures[-1]["failure_message"] == "secondary observer failure"


def test_stage_observer_failure_is_secondary_to_train_step() -> None:
    def broken_observer(_event: GraDPertStageEvent, _engine: GraDPertStepEngine) -> None:
        raise LookupError("observer unavailable")

    _seed_all(81)
    _, _, _, engine = _components(stage_observer=broken_observer)
    metrics = engine.train_step(_batch(), global_step=0)
    assert metrics.total_loss > 0
    assert engine.stage_observer_failures
    assert {failure["failure_type"] for failure in engine.stage_observer_failures} == {
        "LookupError"
    }
    assert all(
        failure["failure_message"] == "observer unavailable"
        for failure in engine.stage_observer_failures
    )


def test_native_step_applies_explicit_loss_weights() -> None:
    torch.manual_seed(10)
    model, optimizer, centers, _ = _components()
    engine = GraDPertStepEngine(
        model=model,
        topology=_topology(),
        optimizer=optimizer,
        centers=centers,
        run_seed=1,
        total_schedule_steps=400,
        heldout_target_ids=(6,),
        loss_weights=LossWeights(
            prediction=1.0,
            condition_consistency=0.8,
            masked_node=0.4,
            spread=0.1,
        ),
    )
    metrics = engine.train_step(_batch(), global_step=0)
    expected = (
        metrics.prediction_loss
        + 0.8 * metrics.condition_consistency_loss
        + 0.4 * metrics.masked_node_loss
        + 0.1 * metrics.spread_loss
    )
    assert metrics.total_loss == pytest.approx(expected, rel=1e-6)


def test_loss_weights_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        LossWeights(spread=-0.1)
    with pytest.raises(ValueError, match="prediction loss weight must be positive"):
        LossWeights(prediction=0.0)


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


@pytest.mark.parametrize(
    ("gene_feature_mode", "decoder_mode"),
    [
        ("learned_id", "control_condition_transformer"),
        ("frozen_genept_projection", "additive"),
        ("genept_id_residual", "additive"),
        ("genept_initialized", "additive"),
        ("genept_shuffled", "additive"),
    ],
)
def test_vnext_trainable_routes_update_ema_and_restore_checkpoint(
    tmp_path: Path,
    gene_feature_mode: str,
    decoder_mode: str,
) -> None:
    _seed_all(37)
    model, optimizer, centers, engine = _vnext_components(
        gene_feature_mode=gene_feature_mode,
        decoder_mode=decoder_mode,
    )
    before_student = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("student_encoder") or name.startswith("control_condition_fusion"))
    }
    before_teacher = {
        name: parameter.detach().clone()
        for name, parameter in model.teacher_encoder.named_parameters()
    }
    metrics = engine.train_step(_batch(), global_step=0)
    assert metrics.total_loss > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in model.named_parameters()
        if name in before_student
    )
    assert any(
        not torch.equal(before_student[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before_student
    )
    assert any(
        not torch.equal(before_teacher[name], parameter)
        for name, parameter in model.teacher_encoder.named_parameters()
    )
    assert not any(parameter.grad is not None for parameter in model.teacher_encoder.parameters())

    checkpoint = tmp_path / f"{gene_feature_mode}-{decoder_mode}.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        centers=centers,
        progress={"completed_epochs": 1, "global_step": 1},
        identity=_identity(),
    )
    expected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resumed_model, resumed_optimizer, resumed_centers, _ = _vnext_components(
        gene_feature_mode=gene_feature_mode,
        decoder_mode=decoder_mode,
    )
    progress = load_training_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        centers=resumed_centers,
        expected_identity=_identity(),
    )
    assert progress == {"completed_epochs": 1, "global_step": 1}
    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, expected_state[name]), name


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


def test_buffered_training_receipts_flush_exact_rows(tmp_path: Path) -> None:
    writer = TrainingReceiptWriter(tmp_path, buffer_steps=3)
    writer.write_step(epoch=0, global_step=0, metrics=_step_metrics())
    writer.write_step(epoch=0, global_step=1, metrics=_step_metrics())
    assert not (tmp_path / "train_steps.csv").exists()
    writer.flush_steps()
    resumed = TrainingReceiptWriter(tmp_path, buffer_steps=3)
    with pytest.raises(RuntimeError, match="expected 2"):
        resumed.write_step(epoch=0, global_step=1, metrics=_step_metrics())
    resumed.write_step(epoch=0, global_step=2, metrics=_step_metrics())
    resumed.flush_steps()
    assert (tmp_path / "train_steps.csv").read_text().count("\n") == 4


def test_checkpoint_peer_copy_fallback_is_byte_identical(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "last.pt"
    destination = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint-bytes" * 1024)

    def no_reflink(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("reflink unavailable")

    monkeypatch.setattr("gradpert.training.checkpoint.fcntl.ioctl", no_reflink)
    method, digest = clone_training_checkpoint(source, destination)
    assert method == "copy"
    assert destination.read_bytes() == source.read_bytes()
    assert len(digest) == 64


def test_resident_graph_tensors_preserve_first_step_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = GraDPertTrainingBatch(
        control_expression=torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10,
        target_expression=torch.arange(10, 20, dtype=torch.float32).reshape(2, 5) / 10,
        condition_ids=("A+ctrl", "B+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
        perturbed_row_ids=("p1", "p2"),
        control_row_ids=("c1", "c2"),
        perturbed_row_ids_sha256="1" * 64,
        control_row_ids_sha256="2" * 64,
        pretransfer_control_sha256="3" * 64,
        pretransfer_target_sha256="4" * 64,
    )

    monkeypatch.setenv("GRADPERT_RING_INDUCED_IMPL", "reference")
    _seed_all(91)
    baseline_model, baseline_optimizer, baseline_centers, _ = _components()
    baseline = GraDPertStepEngine(
        model=baseline_model,
        topology=_topology(),
        optimizer=baseline_optimizer,
        centers=baseline_centers,
        run_seed=1,
        total_schedule_steps=400,
        heldout_target_ids=(6,),
        capture_equivalence_health=True,
    )
    baseline_metrics = baseline.train_step(batch, global_step=0)

    monkeypatch.setenv("GRADPERT_RING_INDUCED_IMPL", "indexed")
    _seed_all(91)
    resident_model, resident_optimizer, resident_centers, _ = _components()
    resident = GraDPertStepEngine(
        model=resident_model,
        topology=_topology(),
        optimizer=resident_optimizer,
        centers=resident_centers,
        run_seed=1,
        total_schedule_steps=400,
        heldout_target_ids=(6,),
        resident_graph_tensors=True,
        capture_equivalence_health=True,
    )
    resident_metrics = resident.train_step(batch, global_step=0)

    assert resident_metrics.total_loss == pytest.approx(baseline_metrics.total_loss, abs=0, rel=0)
    assert resident.first_step_health == baseline.first_step_health
    assert resident.induced_edges is not None
    assert resident.model.student_encoder.resident_graph_tensor_payload() == {
        "active": True,
        "node_count": 7,
        "go_edge_count": 13,
        "string_edge_count": 13,
    }


def test_ring_induced_implementation_selector_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRADPERT_RING_INDUCED_IMPL", "invented")
    with pytest.raises(ValueError, match="reference or indexed"):
        _components()


def test_trainer_serializes_once_then_materializes_best_peer(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    digest = "a" * 64
    serialized: list[Path] = []
    peers: list[tuple[Path, Path]] = []

    def fake_save(path, **kwargs):  # type: ignore[no-untyped-def]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"one serialization")
        serialized.append(destination)
        return digest

    def fake_peer(source, destination):  # type: ignore[no-untyped-def]
        source_path, destination_path = Path(source), Path(destination)
        destination_path.write_bytes(source_path.read_bytes())
        peers.append((source_path, destination_path))
        return "copy", digest

    monkeypatch.setattr("gradpert.training.trainer.save_training_checkpoint", fake_save)
    monkeypatch.setattr("gradpert.training.trainer.clone_training_checkpoint", fake_peer)

    class FakeEngine:
        total_schedule_steps = 200
        model = object()
        optimizer = object()
        centers = object()

        def train_step(self, batch, *, global_step):  # type: ignore[no-untyped-def]
            assert batch == "batch"
            assert global_step == 0
            return _step_metrics()

    trainer = GraDPertTrainer(
        engine=FakeEngine(),  # type: ignore[arg-type]
        checkpoint_identity=_identity(),
        run_root=tmp_path,
        steps_per_epoch=1,
        max_epochs=200,
        run_meta={"run_id": "systems"},
        log_buffer_steps=64,
        single_checkpoint_serialization=True,
    )
    progress = trainer.fit(
        mode="smoke",
        train_epoch_factory=lambda epoch: ("batch",),  # type: ignore[arg-type]
        validate=lambda model, epoch: 1.0,  # type: ignore[arg-type]
    )
    assert progress.completed_epochs == 1
    assert serialized == [tmp_path / "checkpoints" / "last.pt"]
    assert peers == [
        (
            tmp_path / "checkpoints" / "last.pt",
            tmp_path / "checkpoints" / "best.pt",
        )
    ]
    assert trainer.checkpoint_peer_method == "copy"


def test_fixed_epoch_pilot_runs_exactly_ten_epochs_without_early_stop(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    digest = "b" * 64

    def fake_save(path, **kwargs):  # type: ignore[no-untyped-def]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"pilot-checkpoint")
        return digest

    monkeypatch.setattr("gradpert.training.trainer.save_training_checkpoint", fake_save)

    class FakeEngine:
        total_schedule_steps = 10
        model = object()
        optimizer = object()
        centers = object()

        def train_step(self, batch, *, global_step):  # type: ignore[no-untyped-def]
            assert batch == f"batch-{global_step}"
            return _step_metrics()

    trainer = GraDPertTrainer(
        engine=FakeEngine(),  # type: ignore[arg-type]
        checkpoint_identity=_identity(),
        run_root=tmp_path,
        steps_per_epoch=1,
        max_epochs=10,
        run_meta={"run_id": "fixed-ten-epoch-pilot"},
    )
    progress = trainer.fit(
        mode="pilot",
        train_epoch_factory=lambda epoch: (f"batch-{epoch}",),
        validate=lambda model, epoch: 1.0,
    )
    assert progress.completed_epochs == 10
    assert progress.global_step == 10
    assert progress.early_stopping is not None
    assert progress.early_stopping.consecutive_non_improvements == 9


def test_full_200_epoch_budget_stops_after_ten_non_improving_validations(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    digest = "c" * 64

    def fake_save(path, **kwargs):  # type: ignore[no-untyped-def]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"full-checkpoint")
        return digest

    monkeypatch.setattr("gradpert.training.trainer.save_training_checkpoint", fake_save)

    class FakeEngine:
        total_schedule_steps = 200
        model = object()
        optimizer = object()
        centers = object()

        def train_step(self, batch, *, global_step):  # type: ignore[no-untyped-def]
            assert batch == f"batch-{global_step}"
            return _step_metrics()

    trainer = GraDPertTrainer(
        engine=FakeEngine(),  # type: ignore[arg-type]
        checkpoint_identity=_identity(),
        run_root=tmp_path,
        steps_per_epoch=1,
        max_epochs=200,
        run_meta={"run_id": "full-200-epoch-budget"},
    )
    progress = trainer.fit(
        mode="full",
        train_epoch_factory=lambda epoch: (f"batch-{epoch}",),
        validate=lambda model, epoch: 1.0,
    )
    assert progress.completed_epochs == 11
    assert progress.global_step == 11
    assert progress.early_stopping is not None
    assert progress.early_stopping.consecutive_non_improvements == 10


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
