"""Real synthetic forwards keep auxiliary/control-conditional provenance separate."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

from cap40_posthoc import Collector, main_response  # noqa: E402
from test_fgpo_family import new_fixture  # noqa: E402


def records(batch):
    return [
        dict(
            truth_row_id=f"t{i}",
            control_row_id=f"c{i}",
            context_id="T::batch",
            cell_type="T",
            batch="batch",
        )
        for i in range(len(batch.control))
    ]


@pytest.mark.parametrize("arm", ["B0", "CG1", "MR1", "S12-L4"])
def test_observer_labels_actual_forward_paths_and_never_changes_model_or_loss(arm):
    obj, batch = new_fixture(arm)
    obj.eval()
    initial = {k: v.clone() for k, v in obj.state_dict().items()}
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(33)
        expected_loss, expected_terms = obj(batch)
    collector = Collector(batch, records(batch), 42, stats_backend="fast")
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(33)
        collector.install(obj)
        try:
            loss, terms = obj(batch)
        finally:
            collector.close()
    assert collector.module.cross_entropy_terms is collector.original_ce
    assert torch.equal(loss, expected_loss)
    assert terms.keys() == expected_terms.keys()
    for k in terms:
        assert torch.equal(terms[k], expected_terms[k])
    for k, v in obj.state_dict().items():
        assert torch.equal(v, initial[k])
    result = collector.result()
    assert all("ssl1_node" not in pair["student"] for pair in result["pairs"])
    assert ("student.masked_response.response_tokens" in result["features"]) == (arm == "MR1")
    assert all(
        r["view_a"].startswith("view") and r["view_b"].startswith("view")
        for r in result["cross_view"]
    )
    assert set(k for r, k in collector.response_cache if r == "student") == (
        {"main", *[f"view{i}" for i in range(len(batch.cell_views))]}
        | ({"masked_response"} if arm == "MR1" else set())
    )
    if arm == "CG1":
        assert "student.graph.ssl2.view0" in result["features"]
        assert "student.graph.main" in result["features"]


def test_conditional_graph_reuses_only_shared_prefix_and_keeps_each_control_output():
    obj, batch = new_fixture("CG1")
    obj.eval()
    with torch.no_grad():
        prefix = obj.student.graph.relay_prefix(batch.graph.context)
        graph, condition = obj._batch_graph(obj.student, batch)
        expected = obj.student.encode_response(
            obj._select_graph(graph, batch.query_positions),
            batch.control,
            condition,
            **obj.response_metadata(batch),
        )
        cached = main_response(SimpleNamespace(objective=obj), batch, {"prefix": prefix})
    for key in expected:
        torch.testing.assert_close(cached[key], expected[key], rtol=0, atol=0)
