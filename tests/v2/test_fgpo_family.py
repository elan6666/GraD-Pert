"""New anchor semantics and complete updates; preserve historical payloads."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_unified_integration import fixture
from torch import nn

from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import RelayDeltaAttention, delta_scan
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_fgpo_family import ARMS, configs  # noqa: E402


def new_fixture(arm):
    parent, batch, _, _ = fixture("N0" if arm in ("B0", "K2-fixed-prefix") else arm)
    architecture = replace(
        parent.student.options,
        mlp_profile="restored",
        genept_projection_activation="gelu",
        genept_hidden_width=32,
        genept_middle_width=16,
        ffn_clip=False,
        ffn_variant="gelu",
        prediction_parameterization="softplus",
        fixed_gene_order=arm in ("O1", "K2-fixed-prefix"),
        random_gene_order=arm not in ("O1", "K2-fixed-prefix"),
        all_kda_prefix=arm == "K2-fixed-prefix",
    )
    torch.manual_seed(131)
    model = GraDPertV2(parent.student.graph.embedding.weight.detach().clone(), architecture)
    return JointObjective(
        model,
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        loss_reduction="row_mean",
        masked_response_ratio=0.25 if arm == "MR1" else 0,
        lambda_masked_response=1 if arm == "MR1" else 0,
        population_response=arm == "P1",
        lambda_mmd=1,
    ), batch


def test_family_contains_only_authorized_arms_and_shared_baseline():
    values = configs(ROOT)
    assert list(values) == list(ARMS)
    for name, value in values.items():
        p = {k: v["value"] for k, v in value["model"]["parameters"].items()}
        assert p["ffn_variant"] == "gelu" and p["prediction_parameterization"] == "softplus"
        assert (p["ssl1_condition"], p["ssl1_node"], p["ssl1_spread"]) == (1, 0, 0)
        assert (p["ssl2_dino"], p["ssl2_ibot"], p["ssl2_koleo"]) == (1, 1, 0)
        assert p["microbatch"] == 100 and p["world_size"] == p["accumulation"] == 2
        assert value["training"]["max_epochs"]["value"] == 4
        assert p["validation_mode"] == "disabled"
        baseline = copy.deepcopy(values["B0"])
        for key in ARMS[name]:
            baseline["model"]["parameters"][key] = value["model"]["parameters"][key]
        assert baseline == value


@pytest.mark.parametrize("arm", list(ARMS))
def test_complete_nonzero_updates_centers_and_exact_restore(arm, tmp_path):
    objective, batch = new_fixture(arm)
    optimizer = V2Optimizer(objective.student, 1e-3, 0.01)
    initial = objective.student.graph.adapter[0].weight.detach().clone()
    center = objective.ssl1_node_center.clone()
    rng = np.random.default_rng(321)
    for _ in range(2):
        terms = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(x) for x in terms.values())
        assert not objective.pending
    assert not torch.equal(initial, objective.student.graph.adapter[0].weight)
    assert torch.equal(center, objective.ssl1_node_center)
    assert objective.ssl1_cls_center.abs().sum() > 0
    assert objective.ssl2_cls_center.abs().sum() > 0
    assert all(p.grad is None for p in objective.teacher.parameters())
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, objective, optimizer, identity={"arm": arm}, progress={"step": 2}, generator=rng
    )
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    state = copy.deepcopy(objective.state_dict())
    other, _ = new_fixture(arm)
    other_optimizer = V2Optimizer(other.student, 1e-3, 0.01)
    load_checkpoint(path, other, other_optimizer, identity={"arm": arm}, generator=rng)
    actual = optimizer_step(
        other, other_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for name, value in state.items():
        torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)


def test_u24_and_cg_zero_start_keep_backbone_and_positive_prediction():
    baseline, _ = new_fixture("B0")
    for arm in ("U24", "CG1"):
        objective, _ = new_fixture(arm)
        for name, value in baseline.student.state_dict().items():
            torch.testing.assert_close(value, objective.student.state_dict()[name], rtol=0, atol=0)
        if arm == "U24":
            m = objective.student.eval()
            x = torch.randn(2, 6, 16)
            ids = torch.arange(6)
            score = m.decode_score(x, query_gene_ids=ids)
            torch.testing.assert_close(
                score, baseline.student.prediction(x).squeeze(-1), rtol=0, atol=0
            )
            assert (m.decode_delta(x, query_gene_ids=ids, control=torch.zeros(2, 6)) > 0).all()
        else:
            assert any(isinstance(x, nn.GELU) for x in objective.student.control_summary.modules())
            assert torch.count_nonzero(objective.student.graph_modulation.projection.weight) == 0


def test_all_nine_kda_switch_together_and_old_self_only_scope_remains_separate():
    model, _ = new_fixture("K2-fixed-prefix")
    modules = [m for m in model.student.modules() if isinstance(m, RelayDeltaAttention)]
    assert len(modules) == 9 and all(
        m.all_kda_prefix and m.self_readout == "position" for m in modules
    )
    baseline, _ = new_fixture("B0")
    assert "all_kda_prefix" not in baseline.student.options.payload()
    assert "fixed_gene_order" not in baseline.student.options.payload()


def test_prefix_cross_and_graph_match_explicit_recurrent_read_and_gradients():
    torch.manual_seed(95)
    layer = RelayDeltaAttention(8, 2).eval()
    layer.all_kda_prefix, layer.write_passes = True, 1
    query = torch.randn(2, 5, 8, requires_grad=True)
    control = torch.randn(2, 4, 8, requires_grad=True)
    order = torch.arange(4).expand(2, -1)
    actual = layer.cross(query, control, order=order)[:, :4]
    q = (
        torch.nn.functional.normalize(layer.query(query[:, :4]).reshape(2, 4, 2, 4).float(), dim=-1)
        / 2
    )
    y = delta_scan(q, *layer._project_writes(control))
    expected = layer.output(
        layer.head_norm(y).flatten(-2) * torch.nn.functional.silu(layer.output_gate(query[:, :4]))
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), (query, control), retain_graph=True),
        torch.autograd.grad(expected.sum(), (query, control)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    # Target 1 reads only neighbors 0 and1; legal neighbor2 occurs after its self.
    memory = torch.randn(4, 8, requires_grad=True)
    layer.fixed_gene_order = True
    neighbors = torch.tensor([[2, 1, 0, -1]])
    valid = neighbors >= 0
    sources = torch.zeros(1, 4, 4)
    output = layer.graph(
        memory[1:2], memory, neighbors, valid, sources, target_positions=torch.tensor([1])
    )
    output.sum().backward()
    assert memory.grad[:2].abs().sum() > 0
    assert torch.equal(memory.grad[2:], torch.zeros_like(memory.grad[2:]))
