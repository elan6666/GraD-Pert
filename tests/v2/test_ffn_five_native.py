"""Five optional arms preserve BR defaults and resumable completed updates."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_relay_method import small_index
from test_restored_ffn import restored
from test_unified_integration import fixture
from torch import nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import GatedFeedForward, ManifoldResidual
from gradpert.modeling.v2.restored_ffn import Float32LayerNorm, MatchedGELUFeedForward
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.evaluation import predict_query_set
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer, routes

ARMS = {
    "BR": {},
    "FB": {"ffn_linear_bias": "none"},
    "FG": {"ffn_variant": "gelu"},
    "FN": {"ffn_input_norm": "layernorm"},
    "DP": {"prototype_weight_norm": True, "prototype_freeze_epochs": 1},
    "PO": {"prediction_parameterization": "softplus"},
}


def arm(name):
    parent, batch = restored("BR")
    torch.manual_seed(131)
    architecture = replace(parent.student.options, **ARMS[name])
    objective = JointObjective(
        GraDPertV2(parent.student.graph.embedding.weight.detach().clone(), architecture),
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
    )
    return objective, batch


def test_disabled_controls_keep_exact_state_initialization_and_identity():
    baseline, _ = restored("BR")
    explicit, _ = arm("BR")
    assert baseline.student.options.payload() == explicit.student.options.payload()
    assert not (set(ARMS["FB"]) | set(ARMS["DP"]) | set(ARMS["PO"])) & set(
        baseline.student.options.payload()
    )
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, explicit.state_dict()[name], rtol=0, atol=0)
    assert routes(baseline.student) == routes(explicit.student)


def test_fb_removes_only_role_linear_bias_keeps_weights_and_mechanism_bias():
    br, _ = arm("BR")
    fb, _ = arm("FB")
    targets = ("graph.adapter.", "expression.", "prediction.", "response.injections.")
    for name, module in fb.student.named_modules():
        if isinstance(module, nn.Linear) and (name.startswith(targets) or ".features." in name):
            assert module.bias is None, name
        if isinstance(module, nn.Linear):
            old = dict(br.student.named_modules())[name]
            torch.testing.assert_close(module.weight, old.weight, rtol=0, atol=0)
    assert fb.student.graph.norm.bias is not None
    for name, value in br.student.named_parameters():
        if "bias" in name and name in dict(fb.student.named_parameters()):
            torch.testing.assert_close(
                value, dict(fb.student.named_parameters())[name], rtol=0, atol=0
            )
    assert any(isinstance(x, ManifoldResidual) and hasattr(x, "bias") for x in fb.student.modules())


def test_fg_parameter_matched_hidden_and_dropout_keep_role_gelu():
    br, _ = arm("BR")
    fg, _ = arm("FG")
    assert not any(isinstance(m, GatedFeedForward) for m in fg.student.modules())
    cores = [m for m in fg.student.modules() if isinstance(m, MatchedGELUFeedForward)]
    originals = [m for m in br.student.modules() if isinstance(m, GatedFeedForward)]
    assert len(cores) == len(originals) == 10
    for a, b in zip(cores, originals, strict=True):
        assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
        assert a.up.out_features == 6 * fg.student.options.width
        assert a.dropout.p == b.dropout.p
        assert a.up.bias is None and a.down.bias is None
    assert isinstance(fg.student.graph.adapter[1], nn.GELU)
    assert isinstance(fg.student.ssl2_cls.features[1], nn.GELU)


def test_fn_fp32_norm_and_only_ffn_pre_norm_positions():
    br, _ = arm("BR")
    fn, _ = arm("FN")
    m = fn.student
    assert isinstance(m.graph.adapter[0], Float32LayerNorm)
    assert isinstance(m.expression[0], nn.Linear) and isinstance(m.expression[1], Float32LayerNorm)
    assert all(isinstance(layer.norm2, Float32LayerNorm) for layer in m.graph.layers)
    assert all(isinstance(layer.norm1, nn.LayerNorm) for layer in m.graph.layers)
    assert isinstance(m.graph.norm, nn.LayerNorm) and not isinstance(m.graph.norm, Float32LayerNorm)
    for layer in m.cell.layers:
        if isinstance(layer.sublayer, GatedFeedForward):
            assert isinstance(layer.norm, Float32LayerNorm)
        else:
            assert isinstance(layer.norm, nn.RMSNorm)
    for head in (m.ssl1_cls, m.ssl1_node, m.ssl2_cls, m.ssl2_node):
        assert isinstance(head.features[0], Float32LayerNorm)
    br_linears = [x for x in br.student.modules() if isinstance(x, nn.Linear)]
    fn_linears = [x for x in m.modules() if isinstance(x, nn.Linear)]
    for a, b in zip(br_linears, fn_linears, strict=True):
        torch.testing.assert_close(a.weight, b.weight, rtol=0, atol=0)
    score = [r for r in routes(m) if r["name"] == "prediction.1.2.weight"]
    assert score[0]["optimizer"] == "adamw"
    x = torch.tensor([[1000.0, 1008.0, 992.0, 1016.0]], dtype=torch.bfloat16, requires_grad=True)
    norm = Float32LayerNorm(4, eps=1e-6)
    actual = norm(x)
    expected = F.layer_norm(x.float(), (4,), norm.weight, norm.bias, 1e-6).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(norm.weight.grad).all()


def test_po_positive_prediction_downregulation_and_inference_path():
    objective, _ = arm("PO")
    model = objective.student.eval()
    score = torch.tensor([[-5.0, 0.0, 5.0]], requires_grad=True)
    control = torch.tensor([[3.0, 2.0, 0.0]])
    prediction, delta = model.prediction_outputs(score, control)
    torch.testing.assert_close(prediction, F.softplus(score), rtol=0, atol=0)
    assert (prediction > 0).all() and (delta[0, :2] < 0).all()
    prediction.sum().backward()
    assert (score.grad > 0).all()
    small = torch.full((1, 1), -30.0, dtype=torch.bfloat16)
    pred, _ = model.prediction_outputs(small, torch.ones_like(small))
    assert pred.item() > 0  # No subtract/add cancellation against control.
    values = predict_query_set(
        model,
        small_index(16),
        np.ones((3, 16), dtype=np.float32),
        (0,),
        np.arange(6),
        device=torch.device("cpu"),
        cell_batch=2,
    )
    assert np.isfinite(values).all() and (values > 0).all()
    with pytest.raises(ValueError, match="aligned control"):
        model.decode_delta(torch.zeros(1, 3, 16))


@pytest.mark.parametrize("name", ["FB", "FG", "FN", "DP", "PO"])
def test_nonzero_lr_multistep_ema_center_and_checkpoint_exact_restore(name, tmp_path):
    objective, batch = arm(name)
    freeze_steps = 2 if name == "DP" else 0
    optimizer = V2Optimizer(objective.student, 1e-3, 0.01, prototype_freeze_steps=freeze_steps)
    prototype_before = {
        k: p.detach().clone() for k, p in objective.student.named_parameters() if "prototypes." in k
    }
    backbone_before = objective.student.graph.adapter[0].weight.detach().clone()
    # FN adapter[0] is a norm, not a matrix; both are trainable and receive signal.
    for _ in range(2):
        result = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(v) for v in result.values())
        assert not objective.pending
    assert not torch.equal(backbone_before, objective.student.graph.adapter[0].weight)
    assert objective.ssl1_cls_center.abs().sum() > 0
    assert objective.ssl2_cls_center.abs().sum() > 0
    assert all(p.grad is None for p in objective.teacher.parameters())
    if name == "DP":
        for k, p in objective.student.named_parameters():
            if k in prototype_before:
                torch.testing.assert_close(p, prototype_before[k], atol=0, rtol=0)
                assert optimizer.adamw.state[p]["step"] == 2
                assert optimizer.adamw.state[p]["exp_avg"].abs().sum() > 0
        assert len(optimizer.adamw.param_groups) == 2
    rng = np.random.default_rng(37)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, objective, optimizer, identity={"arm": name}, progress={"step": 2}, generator=rng
    )
    teacher_before = {k: p.detach().clone() for k, p in objective.teacher.named_parameters()}
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    state = copy.deepcopy(objective.state_dict())
    student_parameters = dict(objective.student.named_parameters())
    for key, value in objective.teacher.named_parameters():
        expected_teacher = teacher_before[key].lerp_(student_parameters[key].detach(), 1 - 0.9)
        torch.testing.assert_close(value, expected_teacher, atol=0, rtol=0)
    if name == "DP":
        assert any(
            not torch.equal(p, prototype_before[k])
            for k, p in objective.student.named_parameters()
            if k in prototype_before
        )
    resumed, _ = arm(name)
    resumed_optimizer = V2Optimizer(
        resumed.student, 1e-3, 0.01, prototype_freeze_steps=freeze_steps
    )
    load_checkpoint(path, resumed, resumed_optimizer, identity={"arm": name}, generator=rng)
    actual = optimizer_step(
        resumed, resumed_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for key, tensor in resumed.state_dict().items():
        torch.testing.assert_close(tensor, state[key], atol=0, rtol=0)


def test_dp_g_initialization_and_freeze_budget_is_fail_closed():
    objective, _ = arm("DP")
    for head in (
        objective.student.ssl1_cls,
        objective.student.ssl1_node,
        objective.student.ssl2_cls,
        objective.student.ssl2_node,
    ):
        torch.testing.assert_close(
            head.prototypes.parametrizations.weight.original0, torch.ones(11, 1)
        )
        assert head.prototypes.bias is None
    with pytest.raises(ValueError, match="resolved positive"):
        V2Optimizer(objective.student, 1e-3, 0)
    a = V2Optimizer(objective.student, 1e-3, 0, prototype_freeze_steps=2)
    b = V2Optimizer(objective.student, 1e-3, 0, prototype_freeze_steps=3)
    with pytest.raises(ValueError, match="freeze budget differs"):
        b.load_state_dict(a.state_dict())


@pytest.mark.parametrize(
    "changes",
    [
        {"ffn_linear_bias": "all"},
        {"prototype_weight_norm": 1},
        {"prototype_freeze_epochs": -1},
        {"prototype_freeze_epochs": True},
        {"prototype_freeze_epochs": 1},
        {"prediction_parameterization": "relu"},
        {"ffn_input_norm": "batchnorm"},
    ],
)
def test_new_controls_reject_invalid_or_ambiguous_settings(changes):
    with pytest.raises(ValueError):
        V2Architecture(**changes)


def test_process_diagnostics_requires_strict_boolean():
    _, _, _, options = fixture()
    assert not options.process_diagnostics
    assert replace(options, process_diagnostics=True).process_diagnostics
    with pytest.raises(ValueError, match="process diagnostics must be boolean"):
        replace(options, process_diagnostics=1)
