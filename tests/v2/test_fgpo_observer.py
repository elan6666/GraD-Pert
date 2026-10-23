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
            condition_id=f"p{int(batch.condition_index[i])}",
            cell_line="Jurkat",
            control_row_id=f"c{i}",
            context_id="T::batch",
            cell_type="T",
            batch="batch",
        )
        for i in range(len(batch.control))
    ]


def test_ssl_only_observer_does_not_label_first_global_as_prediction_main():
    from test_staged_training import fixture

    obj, batch, _ = fixture()
    obj.eval()
    collector = Collector(batch, records(batch), 42, stats_backend="fast")
    initial = {k: v.clone() for k, v in obj.state_dict().items()}
    with torch.no_grad(), torch.random.fork_rng():
        expected_loss, expected_terms = obj(batch)
    collector.install(obj)
    try:
        with torch.no_grad(), torch.random.fork_rng():
            loss, terms = obj(batch)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        for key in terms:
            torch.testing.assert_close(terms[key], expected_terms[key], rtol=0, atol=0)
        assert ("student", "main") not in collector.response_cache
        assert not hasattr(collector, "main_response_inputs")
        assert {v for role, v in collector.response_cache if role == "student"} == {
            f"view{i}" for i in range(len(batch.cell_views))
        }
    finally:
        collector.close()
    collector.result()
    for k, v in obj.state_dict().items():
        torch.testing.assert_close(v, initial[k], rtol=0, atol=0)


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


def test_disabled_ssl_feature_probes_reuse_main_graph_without_teacher_or_heads():
    from grouped_view_diagnostics import GroupedViewBank

    obj, batch = new_fixture("B0")
    obj.lambda1 = obj.lambda2 = 0
    obj.eval()
    hits = []
    handles = [obj.student.graph.register_forward_hook(lambda *args: hits.append("graph"))]

    def forbidden(*args):
        raise AssertionError("diagnostic-only view invoked teacher or prototype")

    handles += [obj.teacher.register_forward_pre_hook(forbidden)]
    for model in (obj.student, obj.teacher):
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
            handles.append(getattr(model, name).register_forward_pre_hook(forbidden))
    collector = Collector(batch, records(batch), 42, stats_backend="fast")
    collector.install(obj)
    with torch.no_grad():
        obj(batch)
    collector.close()
    assert len(hits) == 1
    rng = torch.get_rng_state().clone()
    result = collector.result()
    assert torch.equal(rng, torch.get_rng_state())
    assert len(hits) == 1 + len(batch.graph_views)
    bank = GroupedViewBank()
    collector.add_to_view_bank(bank)
    grouped = bank.result()
    assert grouped["pairs"] and result["pairs"] == []
    assert all(r["role"] == "student" for r in grouped["pairs"])
    for h in handles:
        h.remove()


@pytest.mark.parametrize("space", ["response_cls", "ssl2_cls_bottleneck"])
def test_population_feature_return_keeps_observer_logits_and_bottleneck_aligned(space):
    from test_cr_integration import fixture

    obj, batch = fixture()
    obj.population_aux_space = space
    obj.eval()
    initial = {k: v.clone() for k, v in obj.state_dict().items()}
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(33)
        expected = obj(batch, deferred_au=[])
    collector = Collector(batch, records(batch), 42, stats_backend="fast")
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(33)
        collector.install(obj)
        try:
            actual = obj(batch, deferred_au=[])
        finally:
            collector.close()
    assert torch.equal(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys()
    for key in actual[1]:
        assert torch.equal(actual[1][key], expected[1][key])
    for key, value in obj.state_dict().items():
        assert torch.equal(value, initial[key])
    result = collector.result()
    for index in (0, 1):
        key = f"student.ssl2_cls.view{index}"
        assert key + ".bottleneck" in result["features"]
        assert key in result["heads"]
    assert result["pairs"]
