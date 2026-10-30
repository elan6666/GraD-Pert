"""Common evaluation rebuilds cached precision policies after strict restoration."""

from dataclasses import replace

import pytest
import torch
from test_components import fixture
from test_restored_ffn import restored

from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.model import Projector
from gradpert.modeling.v2.precision_evaluation import (
    evaluation_precision,
    evaluation_precision_context,
    evaluation_student,
    precision_policy,
)
from gradpert.modeling.v2.restored_ffn import MatchedGELUFeedForward
from gradpert.training.v2.objective import JointObjective


@pytest.mark.parametrize("value", [None, True, 0, "bf16", "common_ri_bf16", "", []])
def test_unknown_policy_rejected(value):
    with pytest.raises(ValueError, match="evaluation_precision"):
        evaluation_precision(value)


def test_default_native_returns_exact_original_without_rng_or_payload_change():
    model, _ = fixture()
    state = torch.get_rng_state()
    payload = model.options.payload()
    assert evaluation_student(model) is model
    assert evaluation_precision() == "native"
    assert torch.equal(state, torch.get_rng_state())
    assert payload == model.options.payload()
    with pytest.raises(ValueError, match="explicit precision-ablation"):
        evaluation_student(model, "common_ri_native")


def test_common_head_policy_preserves_owner_teacher_weights_modes_and_rng():
    original, _ = fixture()
    model = GraDPertV2(
        original.graph.embedding.weight.detach(),
        replace(original.options, ssl_head_precision="float32", prediction_precision="float32"),
    )
    objective = JointObjective(model)
    objective.eval()
    model.prediction.train()
    before = {name: value.clone() for name, value in objective.state_dict().items()}
    rng = torch.get_rng_state()
    result = evaluation_student(model, "common_ri_native")
    assert torch.equal(rng, torch.get_rng_state())
    assert result is not model
    assert all(value == "autocast" for value in precision_policy(result).values())
    assert model.ssl2_cls.precision == "float32"
    assert all(m.precision == "autocast" for m in result.modules() if isinstance(m, Projector))
    assert list(result.state_dict()) == list(model.state_dict())
    source_modules = dict(model.named_modules())
    for name, module in result.named_modules():
        assert module.training == source_modules[name].training
    for name, parameter in result.named_parameters():
        source = dict(model.named_parameters())[name]
        assert parameter.data_ptr() != source.data_ptr()
        assert parameter.requires_grad == source.requires_grad
        assert torch.equal(parameter, source)
    for name, value in objective.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(not p.requires_grad for p in objective.teacher.parameters())
    x = torch.randn(2, model.options.width)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with evaluation_precision_context("common_ri_native", "cpu"):
            assert not torch.is_autocast_enabled("cpu")
            assert result.ssl2_cls(x).dtype == torch.float32
        assert torch.is_autocast_enabled("cpu")
        with evaluation_precision_context("native", "cpu"):
            assert torch.is_autocast_enabled("cpu")
            assert result.ssl2_cls(x).dtype == torch.bfloat16
        assert model.ssl2_cls(x).dtype == torch.float32


def test_common_fp8_policy_reconstructs_ffn_caches_and_raw_prior():
    parent, _ = restored("BR")
    model = GraDPertV2(
        parent.student.graph.embedding.weight.detach(),
        replace(parent.student.options, ffn_variant="gelu", ffn_compute_precision="fp8_hybrid"),
    )
    result = evaluation_student(model, "common_ri_native")
    source_ffns = [m for m in model.modules() if isinstance(m, MatchedGELUFeedForward)]
    target_ffns = [m for m in result.modules() if isinstance(m, MatchedGELUFeedForward)]
    assert source_ffns and len(source_ffns) == len(target_ffns)
    assert all(m.compute_precision == "fp8_hybrid" for m in source_ffns)
    assert all(m.compute_precision == "autocast" for m in target_ffns)
    result.load_state_dict(model.state_dict(), strict=True)
    for name, value in result.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
        if value.numel():
            assert value.data_ptr() != model.state_dict()[name].data_ptr()
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        evaluation_precision_context("common_ri_native", "cpu"),
    ):
        assert target_ffns[0](torch.randn(2, result.options.width)).dtype == torch.float32


def test_unknown_policy_rejected_before_execution_or_checkpoint_access():
    from gradpert.execution.v2_checkpoint_eval import execute_evaluation_plan

    with pytest.raises(ValueError, match="evaluation_precision"):
        execute_evaluation_plan({"evaluation_precision": "typo"})


def test_default_plan_and_selected_checkpoint_identity_are_unchanged(tmp_path, monkeypatch):
    from test_eval_shards import (
        test_child_parent_best_evaluation_uses_checkpoint_training_identity as original_test,
    )

    from gradpert.execution import v2_checkpoint_eval as execution
    from gradpert.hashing import sha256_file

    original_resolve = execution.resolve_evaluation_plan
    plans = []

    def capture(args):
        digest = sha256_file(args.checkpoint)
        plan = original_resolve(args)
        assert "evaluation_precision" not in plan
        assert plan["checkpoint_sha256"] == digest == sha256_file(args.checkpoint)
        plans.append(plan)
        return plan

    monkeypatch.setattr(execution, "resolve_evaluation_plan", capture)
    original_test(tmp_path, monkeypatch)
    assert len(plans) == 1


def test_common_policy_rejects_test_split_before_checkpoint_access():
    from argparse import Namespace

    from gradpert.execution.v2_checkpoint_eval import resolve_evaluation_plan

    with pytest.raises(ValueError, match="validation research"):
        resolve_evaluation_plan(Namespace(evaluation_precision="common_ri_native", split="test"))
