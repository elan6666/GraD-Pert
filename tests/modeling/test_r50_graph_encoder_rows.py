from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gradpert.modeling.encoders import (  # noqa: E402
    GatMlgEncoder,
    GraphSourceTensors,
    HybridBMPEncoder,
)


def _source(  # type: ignore[no-untyped-def]
    name: str,
    edges: tuple[tuple[int, int], ...],
    weights: tuple[float, ...] | None = None,
):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    return GraphSourceTensors(name=name, edge_index=edge_index, edge_weight=edge_weight)


def _graphs():  # type: ignore[no-untyped-def]
    # (0, 1) deliberately appears in both sources so the binarized union
    # adjacency must aggregate that directed pair exactly once (paper Eq. S2).
    string = _source("string", ((0, 1), (2, 1), (1, 3)), (0.2, 0.8, 0.5))
    go = _source("go", ((0, 1), (3, 4), (4, 5)))
    return string, go


def _bmp_model(**kwargs: object) -> HybridBMPEncoder:
    return HybridBMPEncoder(
        source_names=("string", "go"),
        input_dim=8,
        hidden_dim=6,
        output_dim=4,
        dropout=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_hybrid_bmp_matches_dense_binarized_union_reference() -> None:
    torch.manual_seed(7)
    string, go = _graphs()
    model = _bmp_model().eval()
    inputs = torch.randn(6, 8)

    union = set()
    for source in (string, go):
        for src, dst in zip(
            source.edge_index[0].tolist(), source.edge_index[1].tolist(), strict=True
        ):
            union.add((src, dst))
    adjacency = torch.zeros(6, 6)
    for src, dst in union:
        adjacency[dst, src] = 1.0
    expected_in = adjacency @ model.role_in(inputs)
    expected_out = adjacency.t() @ model.role_out(inputs)
    hidden = expected_in + expected_out
    expected = model.mlp[3](torch.nn.functional.leaky_relu(model.mlp[0](hidden), 0.2))

    output = model(inputs, (string, go))
    assert output.shape == (6, 4)
    assert torch.allclose(output, expected, atol=1e-6)


def test_hybrid_bmp_gradients_are_native_and_fail_closed() -> None:
    torch.manual_seed(11)
    string, go = _graphs()
    model = _bmp_model()
    inputs = torch.randn(6, 8, requires_grad=True)
    output = model(inputs, (string, go))
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.role_in.weight.grad is not None
    assert model.role_out.weight.grad is not None
    assert model.mlp[0].weight.grad is not None

    with pytest.raises(ValueError, match="at least two unique sources"):
        HybridBMPEncoder(source_names=("string",), input_dim=8, hidden_dim=6, output_dim=4)
    single = HybridBMPEncoder(
        source_names=("string", "go"), input_dim=8, hidden_dim=6, output_dim=4
    )
    with pytest.raises(ValueError, match="ordered graph sources"):
        single(torch.randn(6, 8), (go, string))


def test_gat_mlg_supra_construction_shapes_and_gradients() -> None:
    torch.manual_seed(3)
    string, go = _graphs()
    model = GatMlgEncoder(
        source_names=("string", "go"),
        input_dim=8,
        hidden_dim=6,
        output_dim=4,
        layer_count=2,
        head_count=2,
        dropout=0.0,
    )
    edge_index, edge_weight = model._supra_graph((string, go), 6, string.edge_index.device)
    node_count = 2 * 6
    assert int(edge_index.min()) >= 0 and int(edge_index.max()) < node_count
    pairs = {(int(a), int(b)) for a, b in edge_index.t().tolist()}
    # inter-layer identity couplings exist in both directions for every gene
    for gene in range(6):
        assert (gene, gene + 6) in pairs
        assert (gene + 6, gene) in pairs
    # intra-layer edges carry the per-layer offset
    assert (0, 1) in pairs and (0 + 6, 1 + 6) in pairs
    assert edge_weight.shape[0] == edge_index.shape[1]

    inputs = torch.randn(6, 8, requires_grad=True)
    output = model(inputs, (string, go))
    assert output.shape == (6, 4)
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.layer_features[0].weight.grad is not None
    assert model.gate.weight.grad is not None
    assert model.output.weight.grad is not None

    with pytest.raises(ValueError, match="ordered graph sources"):
        model(inputs.detach(), (go, string))
