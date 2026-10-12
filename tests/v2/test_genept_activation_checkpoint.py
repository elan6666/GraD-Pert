"""GenePT-only activation rematerialization keeps the complete update semantics."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_restored_ffn import restored
from test_unified_integration import fixture
from torch import nn

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.hashing import sha256_json
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.restored_ffn import UnclippedGate
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.views import assemble_batch

ROOT = Path(__file__).resolve().parents[2]


def _restored(arm):
    objective, batch = restored("BR" if arm == "GM1" else arm)
    if arm != "GM1":
        return objective, batch
    torch.manual_seed(131)
    student = GraDPertV2(
        objective.student.graph.embedding.weight.detach().clone(),
        replace(objective.student.options, graph_mhc=True),
    )
    return JointObjective(student, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)), batch


def _checkpointed(reference):
    architecture = replace(reference.student.options, checkpoint_genept_mapping=True)
    raw_prior = reference.student.graph.embedding.weight.detach().clone()
    student = GraDPertV2(raw_prior, architecture)
    student.load_state_dict(reference.student.state_dict())
    objective = JointObjective(student, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0))
    objective.load_state_dict(reference.state_dict())
    return objective


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _assert_state_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ["BR", "F1", "N1"])
@pytest.mark.parametrize("selected", [False, True])
def test_mapping_outputs_and_parameter_gradients_are_exact(arm, selected):
    reference, _ = restored(arm)
    checkpointed = _checkpointed(reference)
    ids = torch.tensor([0, 3, 15, 7, 3]) if selected else None
    original = reference.student.graph.identities(ids)
    actual = checkpointed.student.graph.identities(ids)
    torch.testing.assert_close(original, actual, atol=0, rtol=0)
    cotangent = torch.randn_like(original)
    original.backward(cotangent)
    actual.backward(cotangent)
    for (name, left), (other_name, right) in zip(
        reference.student.named_parameters(), checkpointed.student.named_parameters(), strict=True
    ):
        assert name == other_name
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert checkpointed.student.graph.embedding.weight.grad is None
    assert (
        checkpointed.student.graph.adapter[0].gate.weight.grad is not None if arm == "F1" else True
    )


def test_actual_f1_mapping_dimensions_save_unique_retained_activation_bytes(tmp_path):
    """Retained-autograd storage is measured, not claimed as CUDA peak memory."""
    torch.manual_seed(51)
    architecture = V2Architecture(
        width=256,
        heads=4,
        latent_rank=64,
        streams=1,
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
        dropout=0,
        checkpoint_layers=False,
        relay_passes=1,
        learned_genept_projection=True,
        mlp_profile="restored",
        genept_projection_activation="gelu",
        ffn_clip=False,
        ffn_variant="swiglu",
        genept_hidden_width=4096,
        genept_middle_width=1024,
    )
    model = GraDPertV2(torch.randn(128, 2048), architecture)
    assert isinstance(model.graph.adapter[0], UnclippedGate)
    assert model.graph.adapter[0].gate.weight.shape == (4096, 2048)
    assert model.graph.adapter[0].up.weight.shape == (4096, 2048)
    assert model.graph.adapter[0].down.weight.shape == (1024, 4096)
    assert model.graph.adapter[2].weight.shape == (256, 1024)
    persistent = {
        tensor.untyped_storage().data_ptr() for tensor in (*model.parameters(), *model.buffers())
    }

    def measure(enabled):
        model.zero_grad(set_to_none=True)
        model.graph.checkpoint_genept_mapping = enabled
        retained = {}
        logical_bytes = 0

        def pack(tensor):
            nonlocal logical_bytes
            logical_bytes += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in persistent:
                retained[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = model.graph.identities()
        result.backward(torch.ones_like(result))
        return sum(retained.values()), logical_bytes, result.detach()

    baseline_bytes, baseline_logical, baseline_output = measure(False)
    rematerialized_bytes, rematerialized_logical, rematerialized_output = measure(True)
    torch.testing.assert_close(baseline_output, rematerialized_output, atol=0, rtol=0)
    assert baseline_bytes > 8 * 1024**2
    assert rematerialized_bytes < baseline_bytes * 0.1
    assert baseline_bytes - rematerialized_bytes > 8 * 1024**2
    evidence = {
        "scope": "CPU_float32_forward_retained_autograd_storages_not_CUDA_peak",
        "genes": 128,
        "mapping_widths": [2048, 4096, 1024, 256],
        "excludes": "persistent_parameter_and_frozen_prior_storage; aliases_deduplicated",
        "baseline_unique_activation_bytes": baseline_bytes,
        "checkpoint_unique_activation_bytes": rematerialized_bytes,
        "baseline_logical_saved_tensor_bytes": baseline_logical,
        "checkpoint_logical_saved_tensor_bytes": rematerialized_logical,
    }
    (tmp_path / "saved_activations.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


def test_checkpoint_only_runs_for_grad_enabled_training_student(monkeypatch):
    import gradpert.modeling.v2.model as native

    objective, _ = restored("F1")
    objective = _checkpointed(objective)
    original = native.checkpoint
    calls = []

    def observed(function, *args, **kwargs):
        calls.append(kwargs)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(native, "checkpoint", observed)
    objective.train()
    objective.student.graph.identities()
    assert calls == [{"use_reentrant": False, "preserve_rng_state": True}]
    with torch.no_grad():
        objective.student.graph.identities()
    objective.teacher.graph.identities()
    objective.eval()
    objective.student.graph.identities()
    assert len(calls) == 1


def test_checkpoint_preserves_rng_when_mapping_contains_training_dropout():
    reference, _ = restored("F1")
    reference.student.graph.projection_activation = nn.Dropout(0.25)
    checkpointed = _checkpointed(restored("F1")[0])
    checkpointed.student.graph.projection_activation = nn.Dropout(0.25)
    checkpointed.load_state_dict(reference.state_dict())
    rng = torch.get_rng_state().clone()
    expected = reference.student.graph.identities()
    expected.square().sum().backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = checkpointed.student.graph.identities()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    for (_, left), (_, right) in zip(
        reference.student.named_parameters(), checkpointed.student.named_parameters(), strict=True
    ):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_mapping_recomputation_preserves_enclosing_bfloat16_autocast():
    reference, _ = restored("F1")
    checkpointed = _checkpointed(reference)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference.student.graph.identities()
        actual = checkpointed.student.graph.identities()
    assert expected.dtype == actual.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cotangent = torch.randn_like(expected)
    # Backward happens outside autocast, as in the actual update engine.
    expected.backward(cotangent)
    actual.backward(cotangent)
    for (_, left), (_, right) in zip(
        reference.student.named_parameters(), checkpointed.student.named_parameters(), strict=True
    ):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["BR", "F1", "N1", "GM1"])
def test_three_full_nonzero_lr_updates_ema_centers_numpy_and_torch_rng(arm):
    reference, _ = _restored(arm)
    for module in reference.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.1
    checkpointed = _checkpointed(reference)
    for module in checkpointed.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.1
    left_optimizer = V2Optimizer(reference.student, 1e-3, 0)
    right_optimizer = V2Optimizer(checkpointed.student, 1e-3, 0)
    _, _, raw, options = fixture(checkpointed=True)
    from test_relay_method import small_index

    left_generator, right_generator = np.random.default_rng(71), np.random.default_rng(71)
    initial = copy.deepcopy(reference.student.state_dict())
    for _ in range(3):
        left_batch = assemble_batch(
            raw, small_index(16), options, left_generator, allowed_expression_ids=np.arange(12)
        )
        right_batch = assemble_batch(
            raw, small_index(16), options, right_generator, allowed_expression_ids=np.arange(12)
        )
        rng = torch.get_rng_state().clone()
        numpy_before = copy.deepcopy(np.random.get_state())
        expected = optimizer_step(
            reference, left_optimizer, left_batch, microbatch=2, lr=1e-3, momentum=0.9, bf16=False
        )
        next_rng = torch.get_rng_state().clone()
        expected_numpy = copy.deepcopy(np.random.get_state())
        torch.set_rng_state(rng)
        np.random.set_state(numpy_before)
        actual = optimizer_step(
            checkpointed,
            right_optimizer,
            right_batch,
            microbatch=2,
            lr=1e-3,
            momentum=0.9,
            bf16=False,
        )
        assert expected == actual
        torch.testing.assert_close(torch.get_rng_state(), next_rng, atol=0, rtol=0)
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == expected_numpy[0] and actual_numpy[2:] == expected_numpy[2:]
        np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
        assert sha256_json(left_generator.bit_generator.state) == sha256_json(
            right_generator.bit_generator.state
        )
        _assert_state_equal(reference.state_dict(), checkpointed.state_dict())
        _assert_state_equal(left_optimizer.state_dict(), right_optimizer.state_dict())
        assert not reference.pending and not checkpointed.pending
    assert any(
        not torch.equal(value, initial[name])
        for name, value in reference.student.state_dict().items()
    )
    assert reference.ssl1_cls_center.abs().sum() > 0 and reference.ssl2_cls_center.abs().sum() > 0
    assert left_optimizer.steps == right_optimizer.steps == 3
    assert all(parameter.grad is None for parameter in checkpointed.teacher.parameters())


def test_default_false_preserves_historical_payload_and_checkpoint(tmp_path):
    config = load_experiment_config(
        ROOT / "configs/v2/ffn_four_20261011/BR/gradpert_v2/nadig_jurkat.yaml"
    )
    historical, _ = V2Options.parse_parameters(config.model.parameters)
    assert not historical.checkpoint_genept_mapping
    assert "checkpoint_genept_mapping" not in historical.payload()
    assert replace(historical, checkpoint_genept_mapping=False).payload() == historical.payload()
    assert (
        replace(historical, checkpoint_genept_mapping=True).payload()["checkpoint_genept_mapping"]
        is True
    )
    reference, batch = restored("F1")
    explicit_false = replace(reference.student.options, checkpoint_genept_mapping=False)
    identical = GraDPertV2(
        reference.student.graph.embedding.weight.detach().clone(), explicit_false
    )
    assert reference.student.state_dict().keys() == identical.state_dict().keys()
    optimizer = V2Optimizer(reference.student, 1e-3, 0)
    optimizer_step(reference, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)
    generator = np.random.default_rng(10)
    path = tmp_path / "historical.pt"
    save_checkpoint(
        path,
        reference,
        optimizer,
        identity={"scope": "old"},
        progress={"step": 1},
        generator=generator,
    )
    restored_objective = JointObjective(identical, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0))
    restored_optimizer = V2Optimizer(identical, 1e-3, 0)
    progress = load_checkpoint(
        path, restored_objective, restored_optimizer, identity={"scope": "old"}, generator=generator
    )
    assert progress["step"] == restored_optimizer.steps == 1
    _assert_state_equal(reference.state_dict(), restored_objective.state_dict())


def test_same_seed_initialization_and_config_parser_are_unchanged():
    config = load_experiment_config(
        ROOT / "configs/v2/ffn_four_20261011/F1/gradpert_v2/nadig_jurkat.yaml"
    )
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    parameters = dict(config.model.parameters)
    parameters["checkpoint_genept_mapping"] = parameters["learned_genept_projection"].model_copy(
        update={"value": True}
    )
    enabled, enabled_options = V2Options.parse_parameters(parameters)
    assert enabled == replace(architecture, checkpoint_genept_mapping=True)
    assert enabled_options == options
    reference, _ = restored("F1")
    raw = reference.student.graph.embedding.weight.detach().clone()
    torch.manual_seed(89)
    ordinary = GraDPertV2(raw, reference.student.options)
    ordinary_rng = torch.get_rng_state().clone()
    torch.manual_seed(89)
    recomputed = GraDPertV2(raw, replace(reference.student.options, checkpoint_genept_mapping=True))
    torch.testing.assert_close(torch.get_rng_state(), ordinary_rng, atol=0, rtol=0)
    _assert_state_equal(ordinary.state_dict(), recomputed.state_dict())


@pytest.mark.parametrize("value", [1, "true", None])
def test_nonboolean_checkpoint_policy_rejected(value):
    with pytest.raises(ValueError, match="boolean"):
        V2Architecture(checkpoint_genept_mapping=value)


def test_checkpoint_cannot_change_reduced_trainable_gene_table_scope():
    with pytest.raises(ValueError, match="learned raw-prior"):
        V2Architecture(checkpoint_genept_mapping=True)
