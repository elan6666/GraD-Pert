"""Head-only LayerNorm preserves historical models and has resumable EMA state."""

from dataclasses import replace

import pytest
import torch
from test_restored_ffn import restored
from torch import nn
from torch.nn import functional as F

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.restored_ffn import Float32LayerNorm
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def pair():
    obj, _ = restored("BR")
    seeds = obj.student.graph.embedding.weight.detach().clone()
    arch = replace(obj.student.options, ffn_variant="gelu")
    models, states = [], []
    for enabled in (False, True):
        torch.manual_seed(491)
        models.append(GraDPertV2(seeds, replace(arch, ssl2_cls_input_layernorm=enabled)))
        states.append(torch.get_rng_state().clone())
    return models, states


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_input_layernorm_requires_strict_bool(value):
    with pytest.raises(ValueError, match="ssl2_cls_input_layernorm must be boolean"):
        V2Architecture(ssl2_cls_input_layernorm=value)


def test_profile_and_global_norm_combinations_are_explicit():
    with pytest.raises(ValueError, match="requires restored profile"):
        V2Architecture(ssl2_cls_input_layernorm=True)
    obj, _ = restored("BR")
    for norm in ("layernorm", "rmsnorm"):
        with pytest.raises(ValueError, match="historical FFN input norm"):
            replace(obj.student.options, ssl2_cls_input_layernorm=True, ffn_input_norm=norm)
    assert "ssl2_cls_input_layernorm" not in obj.student.options.payload()
    enabled = replace(obj.student.options, ssl2_cls_input_layernorm=True)
    assert enabled.payload()["ssl2_cls_input_layernorm"] is True
    assert V2Architecture.parse(enabled.payload()) == enabled


def test_only_ssl2_cls_features_change_without_changing_rng_or_existing_parameters():
    (old, new), rng = pair()
    assert torch.equal(*rng)
    assert [type(m) for m in old.ssl2_cls.features] == [nn.Linear, nn.GELU, nn.Linear]
    norm = new.ssl2_cls.features[0]
    assert isinstance(norm, Float32LayerNorm)
    assert norm.eps == 1e-6 and norm.elementwise_affine
    assert norm.normalized_shape == (old.options.width,)
    before, after = old.state_dict(), new.state_dict()
    expected_keys = set()
    for key, value in before.items():
        mapped = key.replace("ssl2_cls.features.", "ssl2_cls.features.1.")
        expected_keys.add(mapped)
        torch.testing.assert_close(value, after[mapped], rtol=0, atol=0)
    assert set(after) - expected_keys == {"ssl2_cls.features.0.weight", "ssl2_cls.features.0.bias"}
    # With the option disabled, the original two Linear/GELU/L2/prototype formula is exact.
    x = torch.randn(3, old.options.width)
    expected = F.linear(
        F.normalize(old.ssl2_cls.features[2](F.gelu(old.ssl2_cls.features[0](x))).float(), dim=-1),
        old.ssl2_cls.prototypes.weight,
    )
    torch.testing.assert_close(old.ssl2_cls(x), expected, rtol=0, atol=0)
    expected_new = old.ssl2_cls(norm(x))
    torch.testing.assert_close(new.ssl2_cls(x), expected_new, rtol=0, atol=0)


def test_head_gradient_optimizer_and_teacher_ema_cover_new_norm():
    (_, model), _ = pair()
    obj = JointObjective(model)
    norm = obj.student.ssl2_cls.features[0]
    teacher_norm = obj.teacher.ssl2_cls.features[0]
    assert isinstance(teacher_norm, Float32LayerNorm)
    assert all(not p.requires_grad for p in obj.teacher.parameters())
    x = torch.randn(5, model.options.width, requires_grad=True)
    opt = V2Optimizer(obj.student, 1e-3, 0)
    before = {k: v.detach().clone() for k, v in teacher_norm.named_parameters()}
    out = obj.student.ssl2_cls(x)
    F.cross_entropy(out / 0.1, torch.arange(5) % model.options.prototypes).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for param in norm.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    assert all(p.grad is None for p in obj.teacher.parameters())
    opt.step(1e-3)
    obj.commit_statistics(0.9)
    for key, param in norm.named_parameters():
        assert not torch.equal(before[key], param)
        torch.testing.assert_close(
            dict(teacher_norm.named_parameters())[key], before[key].lerp(param.detach(), 0.1)
        )
    # Strict restore includes affine state and identical Student/Teacher key layouts.
    (_, copy_model), _ = pair()
    restored_obj = JointObjective(copy_model)
    restored_obj.load_state_dict(obj.state_dict(), strict=True)
    for key, value in obj.state_dict().items():
        torch.testing.assert_close(value, restored_obj.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(
        restored_obj.student.ssl2_cls(x.detach()), obj.student.ssl2_cls(x.detach())
    )
