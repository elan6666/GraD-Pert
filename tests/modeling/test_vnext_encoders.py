from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gradpert.modeling.encoders import (  # noqa: E402
    AdaptiveSourceGATEncoder,
    GraphSourceTensors,
    SingleSourceGATEncoder,
    SparseGraphTransformerEncoder,
    StringWeightMode,
    _prepare_normalized_string_weights,
    build_sparse_union,
    build_sparse_union_from_ordered_pairs,
)

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_FIXTURE = ROOT / "docs/provenance/fixtures/txpert_exphormer_mg_contract.json"


def _source(  # type: ignore[no-untyped-def]
    name: str,
    edges: tuple[tuple[int, int], ...],
    weights: tuple[float, ...] | None = None,
):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    return GraphSourceTensors(name=name, edge_index=edge_index, edge_weight=edge_weight)


def _graphs():  # type: ignore[no-untyped-def]
    string = _source(
        "string",
        ((0, 1), (2, 1), (1, 3), (3, 4)),
        (0.2, 0.8, 0.5, 1.0),
    )
    go = _source("go", ((1, 0), (1, 2), (4, 3)), (1.0, 1.0, 1.0))
    return string, go


def test_sparse_union_has_stable_order_and_source_membership() -> None:
    string, go = _graphs()
    first = build_sparse_union(
        node_count=5,
        sources=(string, go),
        expected_names=("string", "go"),
        add_reverse_edges=True,
        add_self_loops=True,
        expander_degree=0,
    )
    second = build_sparse_union(
        node_count=5,
        sources=(string, go),
        expected_names=("string", "go"),
        add_reverse_edges=True,
        add_self_loops=True,
        expander_degree=0,
    )
    assert first.channel_names == ("string", "string:reverse", "go", "go:reverse", "self")
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_membership, second.edge_membership)

    edges = list(zip(first.edge_index[0].tolist(), first.edge_index[1].tolist(), strict=True))
    overlap = edges.index((0, 1))
    # STRING original and GO reverse both support this directed edge.
    assert first.edge_membership[overlap].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    loop = edges.index((2, 2))
    assert first.edge_membership[loop].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_sparse_union_matches_the_frozen_official_multihot_fixture() -> None:
    string, go = _graphs()
    union = build_sparse_union(
        node_count=5,
        sources=(string, go),
        expected_names=("string", "go"),
        add_reverse_edges=True,
        add_self_loops=True,
        expander_degree=0,
    )
    fixture = json.loads(OFFICIAL_FIXTURE.read_text(encoding="utf-8"))
    golden = fixture["synthetic_union_golden"]
    assert list(union.channel_names) == golden["channel_order"]
    assert [
        {"edge": edge, "membership": membership}
        for edge, membership in zip(
            union.edge_index.t().tolist(), union.edge_membership.tolist(), strict=True
        )
    ] == golden["ordered_by_edge"]


@pytest.mark.parametrize("expander_degree", [0, 3])
def test_cpu_pair_sparse_union_is_bitwise_reference_exact(expander_degree: int) -> None:
    string, go = _graphs()
    reference = build_sparse_union(
        node_count=5,
        sources=(string, go),
        expected_names=("string", "go"),
        add_reverse_edges=True,
        add_self_loops=True,
        expander_degree=expander_degree,
    )
    optimized = build_sparse_union_from_ordered_pairs(
        node_count=5,
        sources=(
            (
                "string",
                tuple(
                    zip(
                        string.edge_index[0].tolist(),
                        string.edge_index[1].tolist(),
                        strict=True,
                    )
                ),
            ),
            (
                "go",
                tuple(
                    zip(
                        go.edge_index[0].tolist(),
                        go.edge_index[1].tolist(),
                        strict=True,
                    )
                ),
            ),
        ),
        expected_names=("string", "go"),
        device=string.edge_index.device,
        add_reverse_edges=True,
        add_self_loops=True,
        expander_degree=expander_degree,
    )
    assert optimized.channel_names == reference.channel_names
    assert torch.equal(optimized.edge_index, reference.edge_index)
    assert torch.equal(optimized.edge_membership, reference.edge_membership)
    assert torch.equal(optimized.local_edge_index, reference.local_edge_index)


@pytest.mark.parametrize(
    "mode",
    [
        StringWeightMode.SELECTION_ONLY,
        StringWeightMode.NORMALIZED_EDGE_FEATURE,
        StringWeightMode.FIXED_NORMALIZED_PRIOR,
        StringWeightMode.PRIOR_LOGIT_RESIDUAL,
        StringWeightMode.SHUFFLED_NORMALIZED_EDGE_FEATURE,
    ],
)
def test_single_string_gat_weight_routes_have_shape_and_gradients(mode: StringWeightMode) -> None:
    string, _ = _graphs()
    model = SingleSourceGATEncoder(
        input_dim=8,
        hidden_dim=4,
        output_dim=6,
        layer_count=2,
        head_count=2,
        dropout=0.0,
        string_weight_mode=mode,
    )
    inputs = torch.randn(5, 8, requires_grad=True)
    output = model(inputs, (string,))
    assert output.shape == (5, 6)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_single_string_gat_is_deterministic_in_eval_mode() -> None:
    string, _ = _graphs()
    torch.manual_seed(17)
    model = SingleSourceGATEncoder(
        input_dim=8,
        hidden_dim=4,
        output_dim=6,
        layer_count=2,
        head_count=2,
        dropout=0.3,
    ).eval()
    inputs = torch.randn(5, 8)
    first = model(inputs, (string,))
    second = model(inputs, (string,))
    assert torch.equal(first, second)


def test_shuffled_string_weights_are_a_deterministic_negative_control() -> None:
    string, _ = _graphs()
    torch.manual_seed(19)
    weighted = SingleSourceGATEncoder(
        input_dim=8,
        hidden_dim=4,
        output_dim=6,
        layer_count=1,
        head_count=2,
        dropout=0.0,
        string_weight_mode=StringWeightMode.NORMALIZED_EDGE_FEATURE,
    ).eval()
    shuffled = SingleSourceGATEncoder(
        input_dim=8,
        hidden_dim=4,
        output_dim=6,
        layer_count=1,
        head_count=2,
        dropout=0.0,
        string_weight_mode=StringWeightMode.SHUFFLED_NORMALIZED_EDGE_FEATURE,
    ).eval()
    shuffled.load_state_dict(weighted.state_dict())
    inputs = torch.randn(5, 8)
    assert not torch.equal(weighted(inputs, (string,)), shuffled(inputs, (string,)))
    assert torch.equal(shuffled(inputs, (string,)), shuffled(inputs, (string,)))


def test_string_weights_match_official_global_normalization_and_shuffle_only_nonself() -> None:
    edge_index = torch.tensor(((0, 1, 2, 3), (1, 1, 3, 3)), dtype=torch.long)
    edge_weight = torch.tensor((200.0, 999.0, 800.0, 777.0))
    normalized = _prepare_normalized_string_weights(
        edge_index,
        edge_weight,
        shuffle_nonself=False,
    )
    shuffled = _prepare_normalized_string_weights(
        edge_index,
        edge_weight,
        shuffle_nonself=True,
    )
    torch.testing.assert_close(normalized, torch.tensor((0.25, 1.24875, 1.0, 0.97125)))
    assert shuffled[1].item() == normalized[1].item()
    assert shuffled[3].item() == normalized[3].item()
    assert sorted(shuffled[[0, 2]].tolist()) == sorted(normalized[[0, 2]].tolist())


@pytest.mark.parametrize("source_names", [("string",), ("string", "go")])
def test_sparse_transformer_single_and_multi_source_are_deterministic_and_differentiable(
    source_names: tuple[str, ...],
) -> None:
    string, go = _graphs()
    sources = (string,) if len(source_names) == 1 else (string, go)
    torch.manual_seed(23)
    model = SparseGraphTransformerEncoder(
        source_names=source_names,
        input_dim=8,
        hidden_dim=8,
        output_dim=5,
        layer_count=2,
        head_count=2,
        dropout=0.0,
        expander_degree=1,
    ).eval()
    inputs = torch.randn(5, 8, requires_grad=True)
    first = model(inputs, sources)
    second = model(inputs, sources)
    assert first.shape == (5, 5)
    assert torch.equal(first, second)
    first.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.edge_source_projection.weight.grad is not None


def test_sparse_transformer_fails_closed_on_numerical_string_weights() -> None:
    with pytest.raises(ValueError, match="only selection_only"):
        SparseGraphTransformerEncoder(
            source_names=("string", "go"),
            input_dim=8,
            hidden_dim=8,
            output_dim=5,
            string_weight_mode=StringWeightMode.NORMALIZED_EDGE_FEATURE,
        )


def test_sparse_transformer_keeps_fixed_channel_semantics_for_undirected_source() -> None:
    undirected = _source("string", ((0, 1), (1, 0)), (1.0, 1.0))
    model = SparseGraphTransformerEncoder(
        source_names=("string",),
        input_dim=4,
        hidden_dim=4,
        output_dim=3,
        layer_count=1,
        head_count=1,
        dropout=0.0,
        expander_degree=0,
    ).eval()
    assert model.edge_channel_names == ("string", "string:reverse", "self")
    output = model(torch.randn(2, 4), (undirected,))
    assert output.shape == (2, 3)


def test_sparse_transformer_singleton_graph_preserves_batchnorm_state() -> None:
    isolated = GraphSourceTensors(
        name="string",
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,), dtype=torch.float32),
    )
    model = SparseGraphTransformerEncoder(
        source_names=("string",),
        input_dim=4,
        hidden_dim=4,
        output_dim=3,
        layer_count=1,
        head_count=1,
        dropout=0.0,
        expander_degree=0,
        add_local_message_passing=False,
    ).train()
    layer = model.layers[0]
    before = {
        "attention_mean": layer.attention_norm.running_mean.detach().clone(),
        "attention_var": layer.attention_norm.running_var.detach().clone(),
        "attention_batches": layer.attention_norm.num_batches_tracked.detach().clone(),
        "feed_forward_mean": layer.feed_forward_norm.running_mean.detach().clone(),
        "feed_forward_var": layer.feed_forward_norm.running_var.detach().clone(),
        "feed_forward_batches": layer.feed_forward_norm.num_batches_tracked.detach().clone(),
    }
    inputs = torch.randn(1, 4, requires_grad=True)

    output = model(inputs, (isolated,))
    output.sum().backward()

    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    torch.testing.assert_close(layer.attention_norm.running_mean, before["attention_mean"])
    torch.testing.assert_close(layer.attention_norm.running_var, before["attention_var"])
    torch.testing.assert_close(
        layer.attention_norm.num_batches_tracked,
        before["attention_batches"],
    )
    torch.testing.assert_close(layer.feed_forward_norm.running_mean, before["feed_forward_mean"])
    torch.testing.assert_close(layer.feed_forward_norm.running_var, before["feed_forward_var"])
    torch.testing.assert_close(
        layer.feed_forward_norm.num_batches_tracked,
        before["feed_forward_batches"],
    )


def test_all_encoders_fail_closed_on_wrong_source_order() -> None:
    string, go = _graphs()
    inputs = torch.randn(5, 8)
    transformer = SparseGraphTransformerEncoder(
        source_names=("string", "go"),
        input_dim=8,
        hidden_dim=8,
        output_dim=5,
        layer_count=1,
        head_count=2,
        dropout=0.0,
        expander_degree=0,
    )
    with pytest.raises(ValueError, match="ordered graph sources"):
        transformer(inputs, (go, string))


def test_adaptive_source_fusion_is_native_shape_safe_and_gradient_owned() -> None:
    string, go = _graphs()
    model = AdaptiveSourceGATEncoder(
        source_names=("string", "go"),
        input_dim=8,
        hidden_dim=4,
        output_dim=6,
        layer_count=2,
        head_count=2,
        dropout=0.0,
    )
    inputs = torch.randn(5, 8, requires_grad=True)
    output = model(inputs, (string, go))
    assert output.shape == (5, 6)
    output.mean().backward()
    assert inputs.grad is not None
    assert all(query.grad is not None for query in model.source_queries.values())


def test_gat_numerical_routes_require_weights_and_apply_official_absolute_normalization() -> None:
    unweighted = _source("string", ((0, 1), (1, 2)))
    model = SingleSourceGATEncoder(
        input_dim=4,
        hidden_dim=4,
        output_dim=3,
        layer_count=1,
        head_count=1,
        dropout=0.0,
        string_weight_mode=StringWeightMode.PRIOR_LOGIT_RESIDUAL,
    )
    with pytest.raises(ValueError, match="requires edge weights"):
        model(torch.randn(3, 4), (unweighted,))

    negative = _source("string", ((0, 1), (1, 2)), (1.0, -0.2))
    output = model(torch.randn(3, 4), (negative,))
    assert torch.isfinite(output).all()
