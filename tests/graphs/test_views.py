from __future__ import annotations

from gradpert.graphs import (
    GraphTopology,
    build_graph_view_batch,
    build_ring_induced_view,
    build_training_graph_views,
    clean_graph_view,
    prune_incoming_edges,
)


def _topology() -> GraphTopology:
    genes = ("A", "B", "C", "D", "E", "F", "G")
    string = prune_incoming_edges(
        source_name="string",
        gene_ids=genes,
        weighted_edges=(
            ("B", "A", 1.0),
            ("C", "A", 1.0),
            ("D", "B", 1.0),
            ("E", "C", 1.0),
            ("F", "C", 1.0),
            ("A", "F", 1.0),
        ),
        top_k=20,
    )
    go = prune_incoming_edges(
        source_name="go",
        gene_ids=genes,
        weighted_edges=(
            ("C", "A", 2.0),
            ("D", "B", 2.0),
            ("F", "B", 2.0),
        ),
        top_k=20,
    )
    return GraphTopology(gene_ids=genes, sources={"string": string, "go": go})


def test_ring_view_expands_against_incoming_edges_and_samples_only_boundary() -> None:
    view = build_ring_induced_view(
        _topology(),
        anchors=(0,),
        node_budget=4,
        seed=123,
        view_id="local_0",
        mask_anchors=True,
    )
    assert set(view.node_ids[:3]) == {0, 1, 2}
    assert len(view.node_ids) == 4
    assert set(view.node_ids) <= {0, 1, 2, 3, 4, 5}
    assert view.masked_anchor_ids == (0,)
    for edges in view.edges_by_source.values():
        loops = [edge for edge in edges if edge.source == edge.target]
        assert [edge.source for edge in loops] == list(view.node_ids)
        assert all(edge.source in view.node_ids and edge.target in view.node_ids for edge in edges)


def test_view_batch_has_frozen_counts_masks_and_deterministic_prediction() -> None:
    first = build_graph_view_batch(
        _topology(),
        anchors=(0,),
        heldout_target_ids=(3,),
        run_seed=4,
        global_step=17,
        condition_id="A+ctrl",
        local_node_budget=4,
    )
    second = build_graph_view_batch(
        _topology(),
        anchors=(0,),
        heldout_target_ids=(3,),
        run_seed=4,
        global_step=17,
        condition_id="A+ctrl",
        local_node_budget=4,
    )
    assert first == second
    assert len(first.globals) == 2
    assert len(first.locals) == 8
    assert len(first.masked_local_indices) == 4
    assert sum(bool(view.masked_node_ids) for view in first.globals) == 1
    assert first.globals[first.masked_global_index].masked_node_ids
    assert 0 not in first.globals[first.masked_global_index].masked_node_ids
    assert 3 not in first.globals[first.masked_global_index].masked_node_ids
    for index, view in enumerate(first.locals):
        assert bool(view.masked_anchor_ids) == (index in first.masked_local_indices)

    other_seed = build_graph_view_batch(
        _topology(),
        anchors=(0,),
        heldout_target_ids=(3,),
        run_seed=99,
        global_step=99,
        condition_id="A+ctrl",
        local_node_budget=4,
    )
    assert first.prediction == other_seed.prediction


def test_self_loop_only_anchor_is_retained_warned_and_never_masked() -> None:
    view = build_ring_induced_view(
        _topology(),
        anchors=(6,),
        node_budget=4,
        seed=9,
        view_id="isolated",
        mask_anchors=True,
    )
    assert view.node_ids == (6,)
    assert view.masked_anchor_ids == ()
    assert view.warnings == ("self_loop_only_anchor:G",)


def test_training_views_share_globals_and_build_locals_per_unique_condition() -> None:
    views = build_training_graph_views(
        _topology(),
        anchors_by_condition={"B+ctrl": (1,), "A+ctrl": (0,)},
        heldout_target_ids=(3,),
        run_seed=1,
        global_step=2,
        local_node_budget=4,
    )
    assert tuple(views.anchors_by_condition) == ("A+ctrl", "B+ctrl")
    assert set(views.locals_by_condition) == {"A+ctrl", "B+ctrl"}
    assert all(len(local_views) == 8 for local_views in views.locals_by_condition.values())
    masked_global = views.globals[views.masked_global_index]
    assert not ({0, 1, 3} & set(masked_global.masked_node_ids))
    assert clean_graph_view(masked_global).masked_node_ids == ()
    assert clean_graph_view(masked_global).edges_by_source == masked_global.edges_by_source
