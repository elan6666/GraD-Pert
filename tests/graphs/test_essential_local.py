import hashlib

import pytest

from gradpert.graphs import build_training_graph_views
from gradpert.graphs.essential_local import essential_local_views, load_essential_ids
from tests.graphs.test_views import _topology


def test_annotation_exact_hash_schema_and_gene_order(tmp_path):
    path = tmp_path / "essential.csv"
    path.write_text("gene\nC (3)\nA (1)\nX (9)\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_essential_ids(path, digest, ("A", "B", "C")) == (0, 2)
    with pytest.raises(ValueError, match="hash"):
        load_essential_ids(path, "0" * 64, ("A",))
    path.write_text("wrong\nA\n")
    with pytest.raises(ValueError, match="schema"):
        load_essential_ids(path, hashlib.sha256(path.read_bytes()).hexdigest(), ("A",))


@pytest.mark.parametrize("policy", ["essential_only", "essential_size_ring"])
def test_local_policy_keeps_globals_anchors_count_and_order(policy):
    topology = _topology()
    original = build_training_graph_views(
        topology,
        anchors_by_condition={"A": (0,)},
        heldout_target_ids=(),
        run_seed=1,
        global_step=0,
        local_count=4,
        local_node_budget=2,
        local_anchor_mask_count=0,
    )
    kwargs = dict(ids=(1, 2, 3), policy=policy, run_seed=1, global_step=0)
    actual = essential_local_views(original, topology, **kwargs)
    assert actual == essential_local_views(original, topology, **kwargs)
    assert actual.globals == original.globals
    assert actual.prediction == original.prediction
    assert actual.anchors_by_condition == original.anchors_by_condition
    assert len(actual.locals_by_condition["A"]) == 4
    for view in actual.locals_by_condition["A"]:
        assert len(view.node_ids) == 4 and 0 in view.node_ids
        if policy == "essential_only":
            assert view.node_ids == (0, 1, 2, 3)
        for edges in view.edges_by_source.values():
            assert all(e.source in view.node_ids and e.target in view.node_ids for e in edges)
            assert [e.source for e in edges if e.source == e.target] == list(view.node_ids)


def test_matched_ring_rejects_unreachable_size():
    topology = _topology()
    original = build_training_graph_views(
        topology,
        anchors_by_condition={"G": (6,)},
        heldout_target_ids=(),
        run_seed=1,
        global_step=0,
        local_count=4,
        local_node_budget=2,
        local_anchor_mask_count=0,
    )
    with pytest.raises(ValueError, match="cannot reach"):
        essential_local_views(
            original, topology, ids=(0, 1), policy="essential_size_ring", run_seed=1, global_step=0
        )
