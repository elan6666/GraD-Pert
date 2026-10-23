"""Truth-free path dependence, exact input identities and analytic crossed grids."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_representation_panel_integration import native_runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
import response_path_diagnostics as paths
from cap40_posthoc import static_graph
from capacity_probe import validation_state
from representation_panel import assemble_control_only


@pytest.mark.parametrize("axis", ["condition", "control", "interaction"])
def test_decomposition_identifies_orthogonal_effects(axis):
    p = np.array([-1.0, 1.0])[:, None, None]
    c = np.array([-3.0, -1.0, 1.0, 3.0])[None, :, None]
    x = {
        "condition": np.broadcast_to(p, (2, 4, 1)),
        "control": np.broadcast_to(c, (2, 4, 1)),
        "interaction": p * c,
    }[axis]
    result = paths.decomposition(x + 32)
    assert result[axis + "_fraction"] == pytest.approx(1)
    assert sum(
        result[name + "_fraction"] for name in ("condition", "control", "interaction")
    ) == pytest.approx(1)
    for other in {"condition", "control", "interaction"} - {axis}:
        assert result[other + "_energy"] == pytest.approx(0)


def test_zero_energy_and_undefined_cosine_are_null_not_artificial_directions():
    result = paths.decomposition(np.ones((2, 4, 3)))
    assert result["total_energy"] == 0
    assert result["condition_fraction"] is None
    assert result["fraction_unavailable_reason"] == "zero_centered_energy"
    pair = paths.pair_statistics(np.zeros((4, 3)), np.ones((4, 3)))
    assert pair["rms"] == 1
    assert pair["cosine"] is None
    assert pair["valid_vector_pairs"] == 0
    with pytest.raises(ValueError, match="finite complete"):
        paths.decomposition(np.full((2, 4, 3), np.nan))


def softplus_runtime(arm="N0"):
    from gradpert.modeling.v2 import GraDPertV2
    from gradpert.training.v2.objective import JointObjective
    from gradpert.training.v2.optimizer import V2Optimizer

    rt, panel = native_runtime(arm)
    original = rt.objective.student
    architecture = replace(
        original.options,
        mlp_profile="restored",
        prediction_parameterization="softplus",
        genept_projection_activation="gelu",
        genept_hidden_width=16,
        genept_middle_width=8,
        response_injection_residual=True,
    )
    rt.objective = JointObjective(
        GraDPertV2(original.graph.embedding.weight.detach().clone(), architecture),
        loss_reduction="row_mean",
        ssl1_weights=(1, 1, 0),
        ssl2_weights=(1, 1, 0),
    )
    rt.objective.eval()
    rt.optimizer = V2Optimizer(rt.objective.student, 1e-3, 0)
    return rt, panel


@pytest.mark.parametrize(
    "fault",
    [
        "reorder_controls",
        "missing_condition",
        "duplicate_condition",
        "reorder_queries",
        "duplicate_control",
    ],
)
def test_panel_refuses_nonmatching_crossed_keys(fault):
    _, panel = softplus_runtime()
    panel = copy.deepcopy(panel)
    first, second = panel["schedule"]
    if fault == "reorder_controls":
        second["rows"].reverse()
    elif fault == "missing_condition":
        panel["schedule"].pop()
    elif fault == "duplicate_condition":
        panel["schedule"][1] = copy.deepcopy(first)
    elif fault == "reorder_queries":
        second["queries"] = list(reversed(second["queries"]))
    else:
        first["rows"][1] = copy.deepcopy(first["rows"][0])
    panel["panel_sha256"] = paths.digest_json(panel["schedule"])
    with pytest.raises(ValueError):
        paths.PathAuditBank(panel)


@pytest.mark.parametrize("arm", ["N0", "U24"])
def test_native_f_b_d_exact_decoder_only_swap_no_truth_and_state_preservation(arm, monkeypatch):
    torch.set_num_threads(1)
    rt, panel = softplus_runtime(arm)
    bank = paths.PathAuditBank(panel, required_shape=(1, 2, 4))
    before = validation_state(rt)
    flags = [[] for _ in rt.objective.student.response.self_layers]
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs, i=i: flags[i].append(kwargs["block_cls_to_gene"]),
            with_kwargs=True,
        )
        for i, layer in enumerate(rt.objective.student.response.self_layers)
    ]
    calls = []
    native_decode = paths.decode_cached

    def observed_decode(model, tokens, condition, control, metadata):
        calls.append((tokens.clone(), condition.clone(), control.clone()))
        return native_decode(model, tokens, condition, control, metadata)

    monkeypatch.setattr(paths, "decode_cached", observed_decode)
    try:
        with torch.random.fork_rng(), torch.no_grad():
            cached = static_graph(rt)
            group = panel["schedule"][0]
            batch = assemble_control_only(
                rt, group["rows"], group["queries"], group["view_seed"], deep=False
            )
            # Poisoned target values must not affect or be consumed by this path.
            batch = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
            bank.add(rt, batch, cached, group)
            saved_tokens = bank.pending[0]["F"]["response_tokens"].clone()
            saved_condition = bank.pending[0]["condition"].clone()
            group = panel["schedule"][1]
            batch = assemble_control_only(
                rt, group["rows"], group["queries"], group["view_seed"], deep=False
            )
            bank.add(rt, batch, cached, group)
    finally:
        for handle in handles:
            handle.remove()
    assert before == validation_state(rt)
    assert rt.data.read_indices and all(i >= 4 for i in rt.data.read_indices)
    assert all(flag == [False, False, True] * 2 for flag in flags)
    # Last two calls are D. Original F tokens and other condition, identical control.
    assert len(calls) == 6
    torch.testing.assert_close(calls[-2][0], saved_tokens, rtol=0, atol=0)
    torch.testing.assert_close(calls[-1][1], saved_condition, rtol=0, atol=0)
    torch.testing.assert_close(calls[-2][1], calls[2][1], rtol=0, atol=0)
    torch.testing.assert_close(calls[-2][2], calls[0][2], rtol=0, atol=0)
    assert not bank.pending
    result = bank.result()
    assert result["updates"] == 0 and result["contexts"] == 1
    assert result["input_rows"] == 8 and result["forward_row_evaluations"] == 24
    assert {r["mode"] for r in result["interventions"]} == {"B_minus_F", "D_minus_F"}
    assert all(
        v <= 2e-6
        for r in result["receipts"]
        for parity in r["parity"]
        for k, v in parity.items()
        if k.endswith("max_abs")
    )
    assert not any("mse" in k or "truth" in k for r in result["interventions"] for k in r)
    assert result["receipts"][0]["cached_tokens_and_encoder_condition_unchanged"]


def test_fail_closed_on_reordered_input_and_unsupported_native_options():
    rt, panel = softplus_runtime()
    bank = paths.PathAuditBank(panel)
    with pytest.raises(ValueError, match="incomplete"):
        bank.result()
    with pytest.raises(ValueError, match="reordered"):
        bank.add(rt, None, None, panel["schedule"][1])
    with torch.random.fork_rng(), torch.no_grad():
        group = panel["schedule"][0]
        cached = static_graph(rt)
        batch = assemble_control_only(
            rt, group["rows"], group["queries"], group["view_seed"], deep=False
        )
        with pytest.raises(ValueError, match="alignment"):
            paths.collect_group(
                rt, replace(batch, query_positions=batch.query_positions.flip(0)), cached, group
            )
        rt.objective.student.options = replace(
            rt.objective.student.options, direct_target_flag=True
        )
        with pytest.raises(ValueError, match="supports static relay softplus"):
            paths.collect_group(rt, batch, cached, group)


def test_replay_failure_and_zero_condition_swap_denominator(monkeypatch):
    rt, panel = softplus_runtime()
    bank = paths.PathAuditBank(panel)
    native_main = paths.main_response
    count = 0

    def different_replay(*args):
        nonlocal count
        count += 1
        result = native_main(*args)
        if count == 2:
            result["prediction"] = result["prediction"] + 0.001
        return result

    with torch.random.fork_rng(), torch.no_grad():
        cached = static_graph(rt)
        group = panel["schedule"][0]
        batch = assemble_control_only(
            rt, group["rows"], group["queries"], group["view_seed"], deep=False
        )
        monkeypatch.setattr(paths, "main_response", different_replay)
        with pytest.raises(ValueError, match="parity failed"):
            bank.add(rt, batch, cached, group)
        monkeypatch.setattr(paths, "main_response", native_main)
        for parameter in rt.objective.student.prediction.parameters():
            parameter.zero_()
        for group in panel["schedule"]:
            batch = assemble_control_only(
                rt, group["rows"], group["queries"], group["view_seed"], deep=False
            )
            bank.add(rt, batch, cached, group)
    swaps = [r for r in bank.result()["interventions"] if r["mode"] == "D_minus_F"]
    assert len(swaps) == 2
    assert all(r["ratio_to_full_condition_swap"] is None for r in swaps)
    assert all(r["ratio_unavailable_reason"] == "denominator_at_replay_floor" for r in swaps)
