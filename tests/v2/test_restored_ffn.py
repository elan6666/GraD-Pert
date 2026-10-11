"""Restored B0 role conventions and full state updates for the three FFN arms."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_relay_method import small_architecture
from test_unified_integration import fixture
from torch import nn

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import GatedFeedForward, ManifoldResidual
from gradpert.modeling.v2.restored_ffn import UnclippedGate
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_ffn_group import verify_manifest  # noqa: E402
from run_unified_ddp import matrix_contract  # noqa: E402


def restored(arm):
    parent, batch, _, _ = fixture(checkpointed=True)
    seeds = parent.student.graph.embedding.weight.detach().clone()
    arch = replace(
        small_architecture(),
        learned_genept_projection=True,
        genept_projection_activation="gelu",
        mlp_profile="restored",
        ffn_clip=False,
        genept_hidden_width=32,
        genept_middle_width=16,
        ffn_type="swiglu",
        streams=2,
        relay_passes=1,
        checkpoint_layers=True,
        graph_source_key_gate=True,
        ffn_variant="swiglu" if arm == "F1" else "mixed",
        ffn_input_norm="rmsnorm" if arm == "N1" else "historical",
    )
    torch.manual_seed(131)
    return JointObjective(
        GraDPertV2(seeds, arch), ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)
    ), batch


def test_roles_and_biases_are_restored_instead_of_using_unified_profile():
    objective, _ = restored("BR")
    m = objective.student
    assert not m.graph.embedding.weight.requires_grad
    assert isinstance(m.graph.norm, nn.LayerNorm)
    assert isinstance(m.graph.projection_activation, nn.Identity)
    assert [type(x) for x in m.graph.adapter] == [nn.Linear, nn.GELU, nn.Linear, nn.GELU, nn.Linear]
    assert all(x.bias is not None for x in m.graph.adapter if isinstance(x, nn.Linear))
    assert all(isinstance(x.norm2, nn.LayerNorm) for x in m.graph.layers)
    cores = [x for x in m.modules() if isinstance(x, GatedFeedForward)]
    assert len(cores) == 10 and all(not x.clipped for x in cores)
    assert all(x.gate.bias is None and x.up.bias is None and x.down.bias is None for x in cores)
    assert isinstance(m.expression[1], nn.GELU) and m.prediction[0].bias is not None
    assert isinstance(m.ssl1_cls.features[1], nn.GELU)
    assert all(
        isinstance(x.norm, nn.RMSNorm) for x in m.cell.layers if isinstance(x, ManifoldResidual)
    )


def test_norm_arm_changes_only_norm_and_preserves_linear_initialization():
    br, _ = restored("BR")
    n1, _ = restored("N1")
    for x, y in zip(
        [x for x in br.student.modules() if isinstance(x, nn.Linear)],
        [x for x in n1.student.modules() if isinstance(x, nn.Linear)],
        strict=True,
    ):
        torch.testing.assert_close(x.weight, y.weight, rtol=0, atol=0)
        assert (x.bias is None) == (y.bias is None)
        if x.bias is not None:
            torch.testing.assert_close(x.bias, y.bias, rtol=0, atol=0)
    assert all(isinstance(x.norm2, nn.RMSNorm) for x in n1.student.graph.layers)
    assert isinstance(n1.student.expression[0], nn.Linear)
    assert isinstance(n1.student.expression[1], nn.RMSNorm)
    assert n1.student.expression[1].normalized_shape == (8,)
    assert isinstance(n1.student.graph.norm, nn.LayerNorm)


def test_swiglu_retains_role_widths_and_linear_outputs():
    o, _ = restored("F1")
    m = o.student
    assert isinstance(m.graph.adapter[0], UnclippedGate)
    assert isinstance(m.graph.adapter[1], nn.SiLU)
    assert m.graph.adapter[0].gate.out_features == 32
    assert m.graph.adapter[0].down.out_features == 16
    assert isinstance(m.graph.adapter[2], nn.Linear)
    assert m.prediction.down.out_features == 1 and m.prediction.down.bias is not None
    assert m.ssl1_cls.prototypes.bias is None


@pytest.mark.parametrize("arm", ["BR", "F1", "N1"])
def test_three_actual_updates_and_checkpoint_restore(arm, tmp_path):
    o, batch = restored(arm)
    optimizer = V2Optimizer(o.student, 1e-3, 0)
    before = o.student.graph.embedding.weight.detach().clone()
    rng = np.random.default_rng(1)
    for _ in range(2):
        result = optimizer_step(
            o, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(v) for v in result.values())
        assert not o.pending
    assert o.ssl1_cls_center.abs().sum() > 0 and o.ssl2_cls_center.abs().sum() > 0
    assert torch.equal(before, o.student.graph.embedding.weight)
    assert all(p.grad is None for p in o.teacher.parameters())
    path = tmp_path / "state.pt"
    save_checkpoint(path, o, optimizer, identity={"arm": arm}, progress={"step": 2}, generator=rng)
    expected = optimizer_step(o, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)
    reference = copy.deepcopy(o.state_dict())
    resumed, _ = restored(arm)
    resumed_optimizer = V2Optimizer(resumed.student, 1e-3, 0)
    load_checkpoint(path, resumed, resumed_optimizer, identity={"arm": arm}, generator=rng)
    actual = optimizer_step(
        resumed, resumed_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for k, v in resumed.state_dict().items():
        torch.testing.assert_close(v, reference[k], atol=0, rtol=0)


def test_four_epoch_manifest_and_only_single_axis_changes():
    path = ROOT / "configs/v2/ffn_four_20261011/manifest.json"
    manifest = verify_manifest(ROOT, path)
    _, priority, initial = matrix_contract(ROOT, path)
    assert priority == ("BR", "F1", "N1") and initial == ["F1", "BR", "N1"]
    baseline = None
    for row in manifest["rows"]:
        cfg = load_experiment_config(ROOT / row["config"])
        arch, opt = V2Options.parse_parameters(cfg.model.parameters)
        assert cfg.training.max_epochs.value == 4 and cfg.training.monitor == "none"
        assert cfg.training.eval_batch_size.value == 128
        assert (opt.microbatch, opt.world_size, opt.accumulation) == (48, 2, 2)
        value = cfg.model_dump()
        if baseline is None:
            baseline = value
        else:
            for key in row["changes"]:
                value["model"]["parameters"][key] = baseline["model"]["parameters"][key]
            assert value == baseline
        assert not arch.ffn_clip and arch.genept_hidden_width == 4096


def test_legacy_architecture_identity_has_no_new_defaults():
    identity = small_architecture().payload()
    assert not any(
        k in identity for k in ["ffn_clip", "ffn_variant", "ffn_input_norm", "genept_hidden_width"]
    )


def test_four_epoch_train_only_collector_accepts_last_without_fabricated_best(tmp_path):
    from collect_results import collect_run
    from test_collect_results import completed

    completed(tmp_path, epochs=4)

    def update(path, transform):
        value = json.loads(path.read_text())
        transform(value)
        path.write_text(json.dumps(value))

    update(
        tmp_path / "resolved_config.json",
        lambda v: v.update(
            model={"parameters": {"validation_mode": {"value": "disabled"}}},
            training={
                "formal_run_policy": "v2_fixed_4",
                "max_epochs": {"value": 4},
                "monitor": "none",
            },
        ),
    )
    update(
        tmp_path / "fit/epoch_state.json",
        lambda v: v.update(best=None, selection_metric="final_epoch"),
    )
    update(tmp_path / "COMPLETE.json", lambda v: v.update(best=None, test_roles=["last"]))
    history = [
        {"epoch": i, "validation": {"validation_mode": "disabled", "performed": False}}
        for i in range(1, 5)
    ]
    (tmp_path / "fit/history.json").write_text(json.dumps(history))
    result = collect_run(tmp_path)
    assert len(result) == 1 and result[0]["role"] == "last" and result[0]["status"] == "complete"
