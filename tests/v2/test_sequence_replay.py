"""Replay opt-in dispatch must preserve scan semantics and leave graph KDA eager."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import operators


@pytest.mark.parametrize("passes", [1, 2])
def test_sequence_replay_routes_genes_cls_cross_but_not_graph(monkeypatch, passes):
    torch.manual_seed(27)
    layer = operators.RelayDeltaAttention(8, 2).eval()
    layer.write_passes = passes
    x = torch.randn(2, 5, 8, requires_grad=True)
    order = torch.tensor([[1, 3, 0, 2], [2, 1, 0, 3]])
    expected = layer(x, order=order)
    calls = []

    def replay(*args):
        calls.append(args[0].shape[1])
        return operators.chunk_delta_final_state(*args)

    monkeypatch.setattr(operators, "replayed_delta_scan", lambda: replay)
    layer.replay_sequences = True
    actual = layer(x, order=order)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        atol=0,
        rtol=0,
    )
    assert calls == [4 * passes, 1]
    layer.cross(x, x[:, :4], order=order)
    assert calls == [4 * passes, 1, 4 * passes]
    memory = torch.randn(3, 8)
    neighbors = torch.tensor([[0, 1], [1, 2]])
    layer.graph(
        memory[:2], memory, neighbors, torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 2, 4)
    )
    assert calls == [4 * passes, 1, 4 * passes]


def test_replay_profile_is_only_execution_factor():
    root = Path(__file__).resolve().parents[2] / "configs/v2/single_pass_jurkat"
    old = load_experiment_config(root / "profiling_m2_a2/gradpert_v2/nadig_jurkat.yaml")
    new = load_experiment_config(root / "replay_m2_a2/gradpert_v2/nadig_jurkat.yaml")
    a, o = V2Options.parse_parameters(old.model.parameters)
    b, p = V2Options.parse_parameters(new.model.parameters)
    assert replace(b, relay_kernel="eager") == a and o == p
    assert b.payload()["relay_kernel"] == "cudagraphs"


def test_replay_capture_disables_donation_and_restores_global_policy(monkeypatch):
    from types import SimpleNamespace

    import torch._functorch.config as config
    from torch._dynamo.utils import counters

    original = config.donated_buffer, config.backward_pass_autocast
    output = torch.ones(2, requires_grad=True)

    def compile_stub(fn, **kwargs):
        assert kwargs == {"backend": "cudagraphs", "fullgraph": True, "dynamic": False}

        def invoke(*args):
            assert config.donated_buffer is False
            assert config.backward_pass_autocast == "off"
            return output

        return invoke

    monkeypatch.setattr(torch, "compile", compile_stub)
    monkeypatch.setitem(counters["inductor"], "cudagraph_skips", 0)
    operators.replayed_delta_scan.cache_clear()
    try:
        actual = operators.replayed_delta_scan()(
            SimpleNamespace(device=SimpleNamespace(type="cuda"))
        )
        assert actual.data_ptr() != output.data_ptr()
        assert torch.equal(torch.autograd.grad(actual.sum(), output)[0], torch.ones(2))
        assert (config.donated_buffer, config.backward_pass_autocast) == original
    finally:
        operators.replayed_delta_scan.cache_clear()
