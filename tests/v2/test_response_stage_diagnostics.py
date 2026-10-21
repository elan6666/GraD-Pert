"""Fixed-panel Response diagnostics preserve computation and expose contraction."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from response_stage_diagnostics import ResponseStageObserver, spectrum  # noqa: E402

from gradpert.config.v2 import V2Architecture  # noqa: E402
from gradpert.modeling.v2.model import RelayResponseEncoder  # noqa: E402


def records():
    return [
        dict(
            condition_id=p, control_row_id=c, context_id="T::b1", cell_line="Jurkat", cell_type="T"
        )
        for p, c in (("p1", "c1"), ("p1", "c2"), ("p2", "c1"), ("p2", "c3"))
    ]


def hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_real_response_observer_preserves_outputs_state_rng_and_removes_hooks():
    torch.manual_seed(23)
    model = RelayResponseEncoder(
        V2Architecture(
            width=8,
            heads=2,
            latent_rank=4,
            streams=4,
            graph_layers=4,
            graph_read_mode="relay",
            attention="relay_full",
            kda_layers=2,
            dropout=0.1,
            projector_hidden=12,
            projector_bottleneck=4,
            prototypes=11,
            checkpoint_layers=False,
        )
    ).train()
    x, control, p = torch.randn(4, 7, 8), torch.randn(4, 6, 8), torch.randn(4, 8)
    order = (torch.arange(6).expand(4, -1),) * 2
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    hooks = hook_count(model)
    with torch.no_grad():
        expected = model(x, control, p, order)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with (
        torch.no_grad(),
        ResponseStageObserver(
            model, gene_ids=[10, 11, 12, 13, 14, 15], records=records(), max_genes=4
        ) as observer,
    ):
        actual = model(x, control, p, order)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    assert hook_count(model) == hooks
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    result = observer.result()
    assert len(result["sampled_gene_ids"]) == 4
    stages = {r["stage"] for r in result["cell_statistics"]}
    assert stages == {"entry", "final.pre_norm", "final.post_norm"} | {
        f"layer{i}.{stage}"
        for i in (1, 2, 3)
        for stage in ("injection_input", "injection_branch", "injection", "self", "cross", "ffn")
    }
    assert {r["stream"] for r in result["cell_statistics"] if r["stage"] == "layer1.self"} == {
        "stream0",
        "stream1",
        "stream2",
        "stream3",
        "stream_mean",
    }
    assert all(r["rows"] == 4 for r in result["cell_statistics"])
    pair_counts = {
        r["stratum"]: r["pair_count"]
        for r in result["pair_statistics"]
        if r["stage"] == "entry" and r["object"] == "cls"
    }
    assert pair_counts == {
        "same_condition_different_control": 2,
        "different_condition_same_control": 1,
        "different_condition_different_control": 3,
    }


class MinimalResponse(nn.Module):
    def __init__(self, collapse=False, residual=False):
        super().__init__()
        self.residual = residual
        layer = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, :4].copy_(torch.eye(4))
            if collapse:
                layer.weight[1:].zero_()
        self.injections = nn.ModuleList([layer])
        self.self_layers = nn.ModuleList([nn.Identity()])
        self.cross_layers = nn.ModuleList([nn.Identity()])
        self.ffn_layers = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()

    def forward(self, x):
        y = x.unsqueeze(-2).expand(-1, -1, 2, -1)
        branch = self.injections[0](torch.cat((y, torch.zeros_like(y)), -1))
        y = y + branch if self.residual else branch
        y = self.self_layers[0](y)
        y = self.cross_layers[0](y)
        y = self.ffn_layers[0](y)
        return self.norm(y.mean(-2))


def test_identity_copy_and_contraction_are_distinguished_cls_is_excluded():
    x = torch.cat((torch.eye(4), -torch.eye(4), torch.full((1, 4), 1e6)), 0)[None]
    row = records()[:1]
    results = []
    for collapse in (False, True):
        model = MinimalResponse(collapse)
        with (
            torch.no_grad(),
            ResponseStageObserver(model, gene_ids=list(range(8)), records=row) as observer,
        ):
            model(x)
        results.append(observer.result())

    def rank(result, stage):
        return next(
            r["effective_rank"]
            for r in result["cell_statistics"]
            if r["stage"] == stage and r["stream"] in ("single", "stream_mean")
        )

    assert rank(results[0], "entry") == pytest.approx(4)
    assert rank(results[0], "layer1.injection") == pytest.approx(4)
    assert rank(results[1], "entry") == pytest.approx(4)
    assert rank(results[1], "layer1.injection") == pytest.approx(1)
    copied = spectrum(np.ones((8, 4)))
    assert copied["effective_rank"] == 0 and copied["absolute_variance"] == 0
    assert copied["mean_energy"] == 4
    assert copied["relative_variance"] == 0
    assert spectrum(np.ones((1, 4)))["effective_rank"] is None


def test_hash_panel_is_identity_stable_and_failure_removes_hooks():
    ids = list(range(9))
    a = ResponseStageObserver(MinimalResponse(), gene_ids=ids, records=records()[:1], max_genes=4)
    b = ResponseStageObserver(
        MinimalResponse(), gene_ids=ids[::-1], records=records()[:1], max_genes=4
    )
    assert [a.ids[i] for i in a.selected] == [b.ids[i] for i in b.selected]
    model = MinimalResponse()
    x = torch.zeros(1, 10, 4)
    x[0, -1, 0] = float("nan")
    with (
        pytest.raises(ValueError, match="nonfinite"),
        ResponseStageObserver(model, gene_ids=ids, records=records()[:1]) as observer,
    ):
        model(x)
    assert hook_count(model) == 0
    with pytest.raises(ValueError, match="incomplete"):
        observer.result()


def test_bad_identity_and_repeated_forward_rejected():
    model = MinimalResponse()
    with pytest.raises(ValueError, match="unique"):
        ResponseStageObserver(model, gene_ids=[1, 1], records=records()[:1])
    with pytest.raises(ValueError, match="identities"):
        ResponseStageObserver(model, gene_ids=[1, 2], records=[{}])
    observer = ResponseStageObserver(model, gene_ids=[1, 2], records=records()[:1])
    with pytest.raises(ValueError, match="one forward"), observer:
        model(torch.zeros(1, 3, 4))
        model(torch.zeros(1, 3, 4))
    assert hook_count(model) == 0
    with pytest.raises(ValueError, match="single-use"):
        observer.__enter__()


def test_residual_injection_distinguishes_branch_and_actual_state():
    x = torch.cat((torch.eye(4), -torch.eye(4), torch.zeros(1, 4)), 0)[None]
    model = MinimalResponse(collapse=True, residual=True)
    with (
        torch.no_grad(),
        ResponseStageObserver(model, gene_ids=list(range(8)), records=records()[:1]) as observer,
    ):
        output = model(x)
    result = observer.result()
    stats = {r["stage"]: r for r in result["cell_statistics"] if r["stream"] == "stream_mean"}
    assert stats["layer1.injection_input"]["absolute_variance"] == pytest.approx(1)
    assert stats["layer1.injection_branch"]["effective_rank"] == pytest.approx(1)
    assert stats["layer1.injection_branch"]["absolute_variance"] == pytest.approx(0.25)
    assert stats["layer1.injection"]["absolute_variance"] == pytest.approx(1.75)
    assert stats["layer1.injection"]["effective_rank"] > 3
    expected = spectrum(output[0, :-1].numpy())
    for key in ("effective_rank", "relative_variance", "absolute_variance", "mean_energy"):
        assert stats["layer1.injection"][key] == pytest.approx(expected[key])
        assert stats["layer1.self"][key] == pytest.approx(expected[key])
    assert hook_count(model) == 0
