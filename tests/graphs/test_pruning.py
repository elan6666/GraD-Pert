from __future__ import annotations

from gradpert.graphs import prune_incoming_edges


def test_pruning_is_independent_deterministic_and_uses_weights_only_for_rank() -> None:
    graph = prune_incoming_edges(
        source_name="string",
        gene_ids=("A", "B", "C", "D"),
        weighted_edges=(
            ("D", "A", 0.7),
            ("C", "A", 0.9),
            ("B", "A", 0.9),
            ("B", "A", 0.1),
            ("A", "A", 10.0),
            ("OUTSIDE", "A", 99.0),
        ),
        top_k=2,
    )
    assert [(edge.source, edge.target, edge.weight) for edge in graph.edges] == [
        (1, 0, 0.9),
        (2, 0, 0.9),
    ]
    assert all(edge.source != edge.target for edge in graph.edges)


def test_pruning_rejects_duplicate_gene_axis() -> None:
    try:
        prune_incoming_edges(
            source_name="go",
            gene_ids=("A", "A"),
            weighted_edges=(),
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate gene axis should fail")
