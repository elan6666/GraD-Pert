from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gradpert.config import NativeArchitectureOptions  # noqa: E402
from gradpert.graphs import (  # noqa: E402
    GraphTopology,
    GraphView,
    build_graph_view_batch,
    prune_incoming_edges,
)
from gradpert.modeling import GraDPertJointModel  # noqa: E402
from gradpert.modeling.modules import (  # noqa: E402
    ConcatControlConditionTransformer,
    ControlConditionMLP,
    ControlConditionTransformer,
)


def _views():  # type: ignore[no-untyped-def]
    genes = ("A", "B", "C", "D", "E", "F", "G")
    string = prune_incoming_edges(
        source_name="string",
        gene_ids=genes,
        weighted_edges=(("B", "A", 1.0), ("C", "A", 1.0), ("D", "B", 1.0)),
        top_k=20,
    )
    go = prune_incoming_edges(
        source_name="go",
        gene_ids=genes,
        weighted_edges=(("C", "A", 1.0), ("E", "C", 1.0), ("F", "B", 1.0)),
        top_k=20,
    )
    topology = GraphTopology(gene_ids=genes, sources={"string": string, "go": go})
    return build_graph_view_batch(
        topology,
        anchors=(0,),
        heldout_target_ids=(),
        run_seed=1,
        global_step=0,
        condition_id="A+ctrl",
        local_node_budget=5,
    )


def test_joint_model_shapes_teacher_identity_and_prediction_determinism() -> None:
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    )
    student = model.student_encoder.state_dict()
    teacher = model.teacher_encoder.state_dict()
    assert student.keys() == teacher.keys()
    assert all(torch.equal(student[key], teacher[key]) for key in student)
    student_projector = model.student_projector.state_dict()
    teacher_projector = model.teacher_projector.state_dict()
    assert student_projector.keys() == teacher_projector.keys()
    assert all(
        torch.equal(student_projector[key], teacher_projector[key]) for key in student_projector
    )
    assert not any(parameter.requires_grad for parameter in model.teacher_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.teacher_projector.parameters())

    views = _views()
    model.eval()
    controls = torch.randn(4, 5)
    first = model.predict_expression(controls, views.prediction, (0,))
    second = model.predict_expression(controls, views.prediction, (0,))
    assert first.shape == (4, 5)
    assert torch.equal(first, second)

    encoded = model.student_encoder(views.prediction)
    assert encoded.node_states.shape == (7, 64)
    assert torch.equal(
        encoded.condition_state((0, 1)),
        encoded.node_states[0] + encoded.node_states[1],
    )


def test_masked_view_routes_gradient_to_shared_mask_token() -> None:
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    )
    views = _views()
    masked = views.globals[views.masked_global_index]
    assert masked.masked_node_ids
    encoded = model.student_encoder(masked)
    loss = encoded.node_states.square().mean()
    loss.backward()
    assert model.student_encoder.mask_token.grad is not None
    assert torch.isfinite(model.student_encoder.mask_token.grad).all()


def test_joint_model_allows_graph_axis_smaller_than_expression_axis() -> None:
    model = GraDPertJointModel(
        graph_gene_count=3,
        expression_gene_count=5,
        prototype_count=8192,
    )
    assert model.graph_gene_count == 3
    assert model.expression_gene_count == 5


def test_disconnected_view_batch_matches_independent_encoding() -> None:
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    )
    views = _views()
    selected = (views.globals[0], views.locals[0], views.globals[1])
    model.eval()

    independent = tuple(model.student_encoder(view) for view in selected)
    batched = model.student_encoder.forward_many(selected)

    assert tuple(item.node_ids for item in batched) == tuple(item.node_ids for item in independent)
    for expected, observed in zip(independent, batched, strict=True):
        torch.testing.assert_close(observed.node_states, expected.node_states)


def _vnext_options(**changes: object) -> NativeArchitectureOptions:
    parameters = {
        "graph_axis_policy": "recomputed_hvg_union_candidate_targets",
        "graph_hvg_count": 512,
        "graph_sources": "string_go",
        "graph_encoder_family": "multi_source_sparse_transformer",
        "string_weight_mode": "selection_only",
        "local_view_builder": "fanout",
        "local_view_count": 8,
        "local_view_node_budget_ratio": "1/2",
        "local_anchor_mask_view_ratio": "0/1",
        "gene_feature_mode": "learned_id",
        "decoder_mode": "additive",
    }
    for key, value in changes.items():
        parameters[key] = value
    return NativeArchitectureOptions.from_parameters(parameters)


def test_vnext_default_joint_model_uses_one_shared_native_architecture() -> None:
    options = _vnext_options()
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    assert model.architecture.payload_sha256 == options.payload_sha256
    assert model.student_encoder.options == model.teacher_encoder.options
    assert model.control_condition_fusion is None
    student = model.student_encoder.state_dict()
    teacher = model.teacher_encoder.state_dict()
    assert student.keys() == teacher.keys()
    assert all(torch.equal(student[key], teacher[key]) for key in student)


def test_vnext_resident_graph_contract_caches_sparse_prediction_union() -> None:
    options = _vnext_options()
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    prediction = _views().prediction
    model.eval()
    before = model.student_encoder(prediction).node_states.detach().clone()
    model.student_encoder.configure_string_weight_contract(prediction)
    model.student_encoder.configure_resident_graph_tensors(prediction)
    after = model.student_encoder(prediction).node_states.detach().clone()

    payload = model.student_encoder.resident_graph_tensor_payload()
    assert payload["active"] is True
    assert payload["sparse_union_active"] is True
    assert payload["sparse_union_implementation"] == "cpu_vectorized"
    assert payload["sparse_union_edge_count"] > 0
    assert payload["sparse_union_channel_names"] == [
        "string",
        "string:reverse",
        "go",
        "go:reverse",
        "self",
        "expander",
    ]
    assert (
        model.student_encoder._batched_sparse_union(
            (prediction,),
            (7,),
            model.student_encoder.backend,
        )
        is model.student_encoder._resident_sparse_union
    )
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_sparse_union_implementation_selector_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "unknown")
    with pytest.raises(ValueError, match="reference or cpu_vectorized"):
        GraDPertJointModel(
            graph_gene_count=7,
            expression_gene_count=5,
            prototype_count=8192,
            architecture=_vnext_options(),
        )


def test_cpu_vectorized_sparse_union_is_exact_for_training_state_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _vnext_options()
    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "reference")
    reference_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "cpu_vectorized")
    optimized_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    optimized_model.load_state_dict(reference_model.state_dict())
    reference_model.train()
    optimized_model.train()
    selected = (_views().globals[0], _views().locals[0], _views().locals[1])

    torch.manual_seed(20260830)
    reference = reference_model.student_encoder.forward_many_checkpointed(selected)
    reference_loss = torch.stack([item.node_states.square().sum() for item in reference]).sum()
    reference_loss.backward()
    reference_rng = torch.get_rng_state().clone()
    reference_state = {
        name: value.detach().clone()
        for name, value in reference_model.student_encoder.state_dict().items()
    }
    reference_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in reference_model.student_encoder.named_parameters()
    }

    torch.manual_seed(20260830)
    optimized = optimized_model.student_encoder.forward_many_checkpointed(selected)
    optimized_loss = torch.stack([item.node_states.square().sum() for item in optimized]).sum()
    optimized_loss.backward()

    assert torch.equal(optimized_loss, reference_loss)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for expected, actual in zip(reference, optimized, strict=True):
        assert expected.node_ids == actual.node_ids
        assert torch.equal(actual.node_states, expected.node_states)
    for name, value in optimized_model.student_encoder.state_dict().items():
        assert torch.equal(value, reference_state[name]), name
    for name, parameter in optimized_model.student_encoder.named_parameters():
        expected = reference_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected), name


def test_vnext_transformer_training_view_is_independent_of_companion_views() -> None:
    options = _vnext_options()
    independent_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    paired_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    paired_model.load_state_dict(independent_model.state_dict())
    independent_model.train()
    paired_model.train()
    views = _views()
    primary = views.globals[0]
    companion = views.locals[0]

    torch.manual_seed(20260828)
    independent = independent_model.student_encoder.forward_many((primary,))[0]
    torch.manual_seed(20260828)
    paired = paired_model.student_encoder.forward_many((primary, companion))[0]

    assert paired.node_ids == independent.node_ids
    torch.testing.assert_close(paired.node_states, independent.node_states, rtol=0, atol=0)


def test_vnext_local_activation_checkpoint_is_exact_for_rng_buffers_and_gradients() -> None:
    options = _vnext_options()
    baseline_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    checkpoint_model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    checkpoint_model.load_state_dict(baseline_model.state_dict())
    baseline_model.train()
    checkpoint_model.train()
    selected = (_views().locals[0], _views().locals[1])

    torch.manual_seed(20260830)
    baseline = baseline_model.student_encoder.forward_many(selected)
    baseline_loss = torch.stack([item.node_states.square().sum() for item in baseline]).sum()
    baseline_loss.backward()
    baseline_rng = torch.get_rng_state().clone()
    baseline_state = {
        name: value.detach().clone()
        for name, value in baseline_model.student_encoder.state_dict().items()
    }
    baseline_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in baseline_model.student_encoder.named_parameters()
    }

    torch.manual_seed(20260830)
    observed = checkpoint_model.student_encoder.forward_many_checkpointed(selected)
    observed_loss = torch.stack([item.node_states.square().sum() for item in observed]).sum()
    observed_loss.backward()

    assert torch.equal(observed_loss, baseline_loss)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for expected, actual in zip(baseline, observed, strict=True):
        assert expected.node_ids == actual.node_ids
        assert torch.equal(actual.node_states, expected.node_states)
    for name, value in checkpoint_model.student_encoder.state_dict().items():
        assert torch.equal(value, baseline_state[name]), name
    for name, parameter in checkpoint_model.student_encoder.named_parameters():
        expected = baseline_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected), name


@pytest.mark.parametrize(
    "weight_mode",
    ["edge_feature", "fixed_prior", "prior_residual", "shuffled_edge_feature"],
)
def test_string_weight_contract_is_frozen_on_full_topology_across_views(
    weight_mode: str,
) -> None:
    options = _vnext_options(
        graph_sources="string",
        graph_encoder_family="single_source_gat",
        graph_encoder_dropout=0.2,
        string_weight_mode=weight_mode,
    )
    model = GraDPertJointModel(
        graph_gene_count=3,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    edge_type = type(_views().prediction.edges_by_source["string"][0])
    full = GraphView(
        view_id="prediction",
        node_ids=(0, 1, 2),
        edges_by_source={
            "string": (
                edge_type(0, 0, 1.0),
                edge_type(0, 1, 200.0),
                edge_type(1, 1, 1.0),
                edge_type(2, 1, 800.0),
                edge_type(2, 2, 1.0),
            )
        },
        masked_node_ids=(),
        masked_anchor_ids=(),
        warnings=(),
    )
    crop = GraphView(
        view_id="local-0",
        node_ids=(0, 1),
        edges_by_source={
            "string": (
                edge_type(0, 0, 1.0),
                edge_type(0, 1, 200.0),
                edge_type(1, 1, 1.0),
            )
        },
        masked_node_ids=(),
        masked_anchor_ids=(),
        warnings=(),
    )
    encoder = model.student_encoder
    encoder.configure_string_weight_contract(full)
    full_source = encoder._source_tensors(full)[0]
    crop_source = encoder._source_tensors(crop)[0]
    assert full_source.edge_weight is not None
    assert crop_source.edge_weight is not None
    full_edge_weights = {
        (edge.source, edge.target): float(weight)
        for edge, weight in zip(
            full.edges_by_source["string"], full_source.edge_weight.tolist(), strict=True
        )
    }
    crop_edge_weights = {
        (edge.source, edge.target): float(weight)
        for edge, weight in zip(
            crop.edges_by_source["string"], crop_source.edge_weight.tolist(), strict=True
        )
    }
    assert crop_edge_weights[(0, 1)] == full_edge_weights[(0, 1)]
    if weight_mode != "shuffled_edge_feature":
        assert crop_edge_weights[(0, 1)] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "feature_mode",
    [
        "frozen_genept_projection",
        "genept_id_residual",
        "genept_initialized",
        "genept_shuffled",
    ],
)
def test_vnext_genept_modes_are_explicit_shape_safe_routes(feature_mode: str) -> None:
    options = _vnext_options(
        gene_feature_mode=feature_mode,
        genept_expected_sha256=("fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862"),
    )
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
        genept_matrix=torch.randn(7, 1536),
    )
    assert model.student_encoder._full_node_inputs().shape == (7, 128)
    assert model.teacher_encoder._full_node_inputs().shape == (7, 128)


def test_control_condition_mlp_is_parameter_matched_within_one_percent() -> None:
    transformer = ControlConditionTransformer()
    mlp = ControlConditionMLP()
    transformer_count = sum(parameter.numel() for parameter in transformer.parameters())
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    assert abs(transformer_count - mlp_count) / transformer_count < 0.01
    basal = torch.randn(4, 64)
    condition = torch.randn(4, 64)
    assert transformer(basal, condition).shape == (4, 64)
    assert mlp(basal, condition).shape == (4, 64)


@pytest.mark.parametrize(
    ("decoder_mode", "perturbation_dim", "decoder_input_dim"),
    [
        ("concat", 64, 128),
        ("concat_transformer", 64, 256),
        ("concat", 256, 320),
        ("concat_transformer", 256, 448),
    ],
)
def test_decoder_factorial_preserves_raw_perturbation_width(
    decoder_mode: str,
    perturbation_dim: int,
    decoder_input_dim: int,
) -> None:
    options = _vnext_options(
        decoder_mode=decoder_mode,
        graph_tower_output_dim=perturbation_dim,
    )
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
    )
    assert model.expression_decoder.network[0].in_features == decoder_input_dim
    assert model.student_projector.mlp[0].in_features == perturbation_dim
    assert model.teacher_projector.mlp[0].in_features == perturbation_dim
    prediction = model.decode_expression(
        torch.randn(4, 5),
        torch.randn(4, perturbation_dim),
    )
    assert prediction.shape == (4, 5)


@pytest.mark.parametrize("condition_dim", [64, 256])
def test_concat_transformer_uses_two_64_wide_tokens_and_concat_readout(
    condition_dim: int,
) -> None:
    transformer = ConcatControlConditionTransformer(condition_dim)
    basal = torch.randn(4, 64)
    condition = torch.randn(4, condition_dim)
    observed = transformer(basal, condition)
    assert observed.shape == (4, 128)
    if condition_dim == 64:
        assert isinstance(transformer.condition_projection, torch.nn.Identity)
    else:
        assert isinstance(transformer.condition_projection, torch.nn.Linear)
        assert transformer.condition_projection.in_features == 256
        assert transformer.condition_projection.out_features == 64
