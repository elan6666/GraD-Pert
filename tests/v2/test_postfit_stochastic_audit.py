"""Native Cell/Response factorial and signed whole-repeat energy diagnostics."""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import postfit_stochastic_audit as audit  # noqa: E402
from test_postfit_spectral_audit import fixture_spec, runtime  # noqa: E402


def native_runtime():
    from gradpert.config.v2 import V2Architecture
    from gradpert.modeling.v2 import GraDPertV2

    options = V2Architecture(
        width=8,
        heads=2,
        latent_rank=4,
        streams=1,
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
    rt = runtime()
    rt.objective.student = GraDPertV2(torch.randn(9, 8), options).train()
    rt.objective.teacher = copy.deepcopy(rt.objective.student).eval()
    rt.objective.teacher.set_relay_order_randomization(True)
    rt.optimizer = torch.optim.SGD(rt.objective.parameters(), lr=0.1)
    return rt


def args_for_encoder():
    return torch.randn(5, 8), torch.rand(4, 5), torch.randn(4, 8)


def test_real_native_factorial_order_dropout_replay_and_graph_scope():
    from gradpert.modeling.v2.operators import relay_order

    rt, args = native_runtime(), args_for_encoder()
    original = audit.fingerprint(
        {
            n: (
                m.training,
                getattr(m, "randomize_order", None),
                getattr(m, "randomize_relay_order", None),
            )
            for n, m in rt.objective.named_modules()
        }
    )
    with audit.frozen_observation(rt) as proof:
        model = rt.objective.student
        initial = audit.inventory(model)
        assert any(
            r["scoped"] and r["implementation"] == "nn.Dropout" for r in initial["dropout_owners"]
        )
        assert any(
            r["scoped"] and r["implementation"] == "functional_or_SDPA"
            for r in initial["dropout_owners"]
        )
        with audit.order_observer(model) as observer:
            for factor in audit.PROTOCOL["factors"]:
                modes = audit.set_factor(model, factor)
                assert not any(m.training for m in model.graph.modules())
                assert not rt.objective.teacher.training
                assert all(
                    r["training"] == (factor["dropout"] if r["scoped"] else False)
                    for r in modes["dropout_owners"]
                )
                expected_orders = None
                with torch.random.fork_rng():
                    torch.manual_seed(71)
                    expected_orders = tuple(
                        relay_order(
                            4,
                            5,
                            torch.device("cpu"),
                            factor["random_order"],
                            layer,
                            model.options.relay_eval_seed or 1,
                        )
                        for layer in range(2)
                    )
                outputs = []
                for seed in (71, 71):
                    observer.values.clear()
                    with torch.random.fork_rng():
                        torch.manual_seed(seed)
                        outputs.append(
                            audit.observed_cls(
                                model, partial(model.encode_response, *args, decode=False)
                            )
                        )
                    assert observer.receipt()["cell"] == audit.fingerprint(expected_orders)
                assert torch.equal(*outputs)
                if not factor["dropout"] and not factor["random_order"]:
                    observer.values.clear()
                    with torch.random.fork_rng():
                        torch.manual_seed(999)
                        other = audit.observed_cls(
                            model, partial(model.encode_response, *args, decode=False)
                        )
                    assert torch.equal(other, outputs[0])
            with pytest.raises(ValueError, match="graph forward"):
                model.graph(None)
        assert not model.cell._forward_pre_hooks and not model.response._forward_pre_hooks
        assert not model.graph._forward_pre_hooks
    restored = audit.fingerprint(
        {
            n: (
                m.training,
                getattr(m, "randomize_order", None),
                getattr(m, "randomize_relay_order", None),
            )
            for n, m in rt.objective.named_modules()
        }
    )
    assert proof["unchanged"] and original == restored


@pytest.mark.parametrize("mutated", ["parameter", "buffer", "grad", "optimizer", "pending"])
def test_protected_mutation_still_rejected_and_rng_flags_restored(mutated):
    rt = native_runtime()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="protected keys"), audit.frozen_observation(rt):
        audit.set_factor(rt.objective.student, dict(dropout=True, random_order=True))
        torch.rand(2)
        p = next(rt.objective.student.parameters())
        if mutated == "parameter":
            p.add_(1)
        elif mutated == "buffer":
            rt.objective.ssl2_cls_center.add_(1)
        elif mutated == "grad":
            p.grad = torch.ones_like(p)
        elif mutated == "optimizer":
            rt.optimizer.param_groups[0]["lr"] = 0.9
        else:
            rt.objective.pending.append(torch.ones(1))
    assert torch.equal(rng, torch.get_rng_state())
    assert rt.objective.student.training and rt.objective.student.randomize_relay_order
    assert not rt.objective.teacher.training and rt.objective.teacher.randomize_relay_order


def test_native_reference_encoder_input_cache_and_rng_parity():
    rt, args = native_runtime(), args_for_encoder()
    with audit.frozen_observation(rt):
        cached, receipt = audit.reference_parity(rt.objective.student, args, {})
        assert receipt["used_path_passed"] and receipt["final_rng_equal"]
        assert receipt["graph_max_abs"] == receipt["response_max_abs"] == 0
        assert not receipt["cell_chunk_used"]
        before = audit.fingerprint(cached)
        cached[0][0].add_(1)
        assert audit.fingerprint(cached) != before


def test_whole_population_energy_no_noise_and_zero_norm():
    x = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    out = audit.repeat_statistics([x] * 4, expected_rows=8)
    assert out["Wraw"] == out["Wcenter"] == 0
    assert out["Bcorr"] == out["B"]
    assert out["q_corr"] == pytest.approx(out["repeat_mean"]["q"])
    assert out["repeat_mean"]["rank_ceiling"] == 3
    zero = audit.repeat_statistics([torch.zeros(1, 3)] * 4, expected_rows=1)
    assert (
        zero["q_corr"] is None
        and zero["q_corr_unavailable_reason"] == "nonpositive_corrected_total_energy"
    )
    assert zero["repeat_mean"]["effective_rank"] is None


def test_shared_offset_noise_does_not_get_subtracted_from_between_inputs():
    x = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    out = audit.repeat_statistics(
        [x + scalar for scalar in (-3.0, -1.0, 1.0, 3.0)], expected_rows=8
    )
    assert out["Wraw"] == pytest.approx(20.0)
    assert out["Wcenter"] == pytest.approx(0, abs=1e-20)
    assert out["Bcorr"] == pytest.approx(audit.spectral_statistics(x)["trace"])
    assert out["Ecorr"] == pytest.approx(audit.spectral_statistics(x)["total_energy"] - 5)


def test_negative_noise_corrections_are_saved_not_clipped_and_not_ranked():
    x = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    out = audit.repeat_statistics([x, -x, x, -x], expected_rows=2)
    assert out["Bcorr"] == pytest.approx(-1 / 3)
    assert out["Ecorr"] == pytest.approx(-1 / 3)
    assert out["q_corr"] is None and out["corrected_rank"] is None
    assert out["repeat_mean"]["effective_rank"] == 0
    assert max(abs(v) for v in out["decomposition_residuals"].values()) < 1e-14
    json.dumps(out, allow_nan=False)


def test_mixed_signal_row_dependent_noise_and_rank_axis128():
    generator = torch.Generator().manual_seed(8)
    signal = torch.randn(128, 256, generator=generator, dtype=torch.float64)
    values = [
        signal + 0.2 * torch.randn(128, 256, generator=generator, dtype=torch.float64)
        for _ in range(4)
    ]
    out = audit.repeat_statistics(values, expected_rows=128)
    assert out["rows"] == 128 and out["repeat_mean"]["rank_ceiling"] == 127
    assert all(v["n"] == 128 and v["rank_ceiling"] == 127 for v in out["raw"])
    assert out["Wraw"] > 0 and out["Wcenter"] > 0
    assert out["q_corr"] is not None
    with pytest.raises(ValueError, match="never pool512"):
        audit.repeat_statistics([torch.cat(values)] * 4, expected_rows=128)
    with pytest.raises(ValueError, match="exactly four"):
        audit.repeat_statistics(values[:3], expected_rows=128)


def test_crossed_pairs_signed_stable_distances():
    rows = [
        dict(
            cell_line="J",
            cell_type="T",
            context_id="C",
            batch="B",
            condition_id=p,
            control_row_id=c,
        )
        for p in ("p1", "p2")
        for c in ("c1", "c2")
    ]
    panel = dict(schedule=[dict(rows=rows)])
    x = torch.tensor([[0.0], [1.0], [3.0], [4.0]], dtype=torch.float64)
    result = audit.crossed_statistics([x] * 4, panel)
    assert result["different_p_same_control"]["pairs"] == 2
    assert result["different_p_same_control"]["corrected_squared_distance"] == 9
    assert result["same_p_different_control"]["corrected_squared_distance"] == 1


def test_safe_fingerprint_materializes_zero_stride_singleton_without_source_mutation():
    @dataclasses.dataclass
    class Batch:
        value: torch.Tensor

    x = torch.tensor(3).expand(1)
    original = x.stride()
    assert audit.fingerprint(Batch(x)) == audit.fingerprint(Batch(torch.tensor([3])))
    assert x.stride() == original
    assert audit.fingerprint(Batch(x)) != audit.fingerprint(Batch(torch.tensor([4])))


@pytest.mark.parametrize("revision", list(audit.TRAINING_COMMITS.values()))
def test_actual_retained_native_apis_match_analysis_ast(tmp_path, revision):
    for file in ("model.py", "operators.py"):
        relative = "src/gradpert/modeling/v2/" + file
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        )
    proof = audit.native_compatibility(tmp_path, ROOT)
    assert proof["unchanged"] and len(proof["method_ast_sha256"]) == 13
    path = tmp_path / "src/gradpert/modeling/v2/model.py"
    path.write_text(path.read_text().replace("super().train(mode)", "super().train(False)"))
    with pytest.raises(ValueError, match="native encoder/order methods differ"):
        audit.native_compatibility(tmp_path, ROOT)


def test_schema_original_source_config_and_immutable_inputs(tmp_path):
    spec = fixture_spec(tmp_path)
    spec.update(schema=audit.SCHEMA, protocol=copy.deepcopy(audit.PROTOCOL))
    for endpoint, name in zip(spec["endpoints"], audit.NAMES, strict=True):
        endpoint["name"] = name
        endpoint["source"] = dict(endpoint["source"], commit=audit.TRAINING_COMMITS[name])
    audit.validate_spec(spec, server=False)
    with pytest.raises(ValueError, match="server-only"):
        audit.validate_spec(spec)
    bad = copy.deepcopy(spec)
    bad["protocol"]["repeat_seeds"].reverse()
    with pytest.raises(ValueError, match="protocol"):
        audit.validate_spec(bad, server=False)
    bad = copy.deepcopy(spec)
    bad["output"] = bad["data_root"] + "/audit"
    with pytest.raises(ValueError, match="overlap"):
        audit.validate_spec(bad, server=False)
    Path(spec["panel"]["path"]).write_text("mutated")
    with pytest.raises(ValueError, match="hash differs"):
        audit.validate_spec(spec, server=False)


def test_rng_seeds_are_endpoint_free_distinct_and_preregistered():
    expected = audit.encode_seed(2026102601, 0, "main")
    assert expected == audit.encode_seed(2026102601, 0, "main")
    assert (
        len(
            {
                audit.encode_seed(s, g, v)
                for s in audit.PROTOCOL["repeat_seeds"]
                for g in range(32)
                for v in audit.PROTOCOL["views"]
            }
        )
        == 384
    )


def test_no_training_objective_decoder_or_external_materialization_calls():
    tree = ast.parse((ROOT / "scripts/v2/postfit_stochastic_audit.py").read_text())
    forbidden = {
        "backward",
        "step",
        "zero_grad",
        "update_teacher",
        "cell_loss",
        "prediction_outputs",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden
            if node.func.attr == "encode_response":
                assert any(
                    k.arg == "decode"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is False
                    for k in node.keywords
                )


def test_production_native_cache_assembly_and_actual_control_reads():
    from test_representation_panel_integration import native_runtime as panel_runtime

    rt, panel = panel_runtime("N0")
    admissions = []
    reader, data_keys = rt.data._read_expression_indices, set(vars(rt.data))
    old_step, old_commit = rt.optimizer.step, rt.objective.commit_statistics
    old_backward, old_grad = torch.autograd.backward, torch.autograd.grad
    existing = rt.objective.student.response.register_forward_pre_hook(lambda *_: None)
    before_hooks = dict(rt.objective.student.response._forward_pre_hooks)
    with (
        audit.frozen_observation(rt) as state,
        audit.forbidden_paths(rt) as counts,
        audit.control_only_reads(rt, panel) as reads,
    ):
        inputs, recipes, parities = audit.prepare_inputs(rt, panel, lambda: admissions.append(True))
    assert state["unchanged"] and len(admissions) == len(panel["schedule"]) == 2
    assert reads["calls"] == 4 and reads["rows"] == 16 and reads["forbidden"] == 0
    assert all(i >= 4 for i in rt.data.read_indices)
    assert rt.data._read_expression_indices is reader and set(vars(rt.data)) == data_keys
    assert rt.optimizer.step == old_step and rt.objective.commit_statistics == old_commit
    assert torch.autograd.backward is old_backward and torch.autograd.grad is old_grad
    assert dict(rt.objective.student.response._forward_pre_hooks) == before_hooks
    existing.remove()
    assert counts["native_graph"] == 8 and counts["student_encode"] == 12
    assert all(v == 0 for k, v in counts.items() if k not in ("native_graph", "student_encode"))
    assert len(recipes) == 2 and all(set(row) == {"main", "global0", "global1"} for row in inputs)
    for row in inputs:
        assert len(row["main"][0]) == 3
        assert all(len(row[view][0]) == 4 for view in ("global0", "global1"))
        assert all(
            row[view][0][1].shape == row[view][0][3].shape for view in ("global0", "global1")
        )
    assert all(
        value["native_graph_reference"] and value["response_max_abs"] == 0
        for row in parities
        for value in row.values()
    )


@pytest.mark.parametrize(
    "kind",
    [
        "objective",
        "teacher",
        "projector",
        "decoder",
        "backward",
        "autograd_grad",
        "optimizer_step",
        "statistics_commit",
    ],
)
def test_actual_forbidden_dispatch_rejected_and_hooks_methods_restored(kind):
    from test_representation_panel_integration import native_runtime as panel_runtime

    rt, _ = panel_runtime("N0")
    obj = rt.objective
    old_vars, old_method = set(vars(obj)), obj.commit_statistics
    old_optimizer_vars = set(vars(rt.optimizer))
    old_backward = torch.autograd.backward
    encoder_args = (torch.randn(5, 8), torch.rand(4, 5), torch.randn(4, 8))
    actions = dict(
        objective=lambda: obj(None),
        teacher=lambda: obj.teacher.encode_response(*encoder_args, decode=False),
        projector=lambda: obj.student.ssl2_cls(torch.zeros(4, 8)),
        decoder=lambda: obj.student.prediction(torch.zeros(4, 16)),
        backward=lambda: torch.ones(1, requires_grad=True).sum().backward(),
        autograd_grad=lambda: torch.autograd.grad(torch.ones(1), torch.ones(1)),
        optimizer_step=lambda: rt.optimizer.step(0.1),
        statistics_commit=lambda: obj.commit_statistics(0.99),
    )
    with (
        audit.frozen_observation(rt) as state,
        pytest.raises(ValueError, match="forbidden audit execution path"),
        audit.forbidden_paths(rt) as counts,
    ):
        actions[kind]()
    assert state["unchanged"] and counts[kind] == 1
    assert set(vars(obj)) == old_vars and obj.commit_statistics == old_method
    assert set(vars(rt.optimizer)) == old_optimizer_vars and torch.autograd.backward is old_backward
    assert not obj._forward_pre_hooks and not obj.teacher.expression._forward_pre_hooks


def test_truth_reader_and_resource_failure_are_real_dispatch_guards():
    from test_representation_panel_integration import native_runtime as panel_runtime

    rt, panel = panel_runtime("N0")
    original = rt.data._read_expression_indices
    with (
        pytest.raises(ValueError, match="control-only"),
        audit.control_only_reads(rt, panel) as proof,
    ):
        rt.data._read_expression_indices([0])  # representative truth row, never dispatched
    assert proof["forbidden"] == 1 and rt.data.read_indices == []
    assert rt.data._read_expression_indices is original
    with audit.frozen_observation(rt), pytest.raises(RuntimeError, match="resource admission"):
        audit.prepare_inputs(
            rt, panel, lambda: (_ for _ in ()).throw(RuntimeError("resource admission"))
        )
    assert rt.data.read_indices == []


def test_independent_native_graph_reference_detects_actual_graph_drift():
    from types import SimpleNamespace

    counter = []

    def graph(*_):
        counter.append(True)
        return torch.ones(3, 2) + (0.001 if len(counter) == 2 else 0), torch.ones(4, 2)

    with pytest.raises(ValueError, match="independent native graph"):
        audit.native_graph_reference(
            SimpleNamespace(_batch_graph=graph), None, SimpleNamespace(control=torch.ones(4, 3))
        )
    assert len(counter) == 2


def test_cpu_seed_api_does_not_seed_unobserved_cuda_devices(monkeypatch):
    monkeypatch.setattr(torch, "manual_seed", lambda *_: pytest.fail("global all-device seeding"))
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda *_: pytest.fail("unobserved CUDA RNG touched")
    )
    audit.seed_encode(72, torch.device("cpu"))
    a = torch.rand(3)
    audit.seed_encode(72, torch.device("cpu"))
    assert torch.equal(a, torch.rand(3))


def test_whole_output_zero_pkl_fails_closed(tmp_path):
    assert audit.zero_pkl(tmp_path) == dict(checked=True, count=0)
    nested = tmp_path / "endpoint/raw.pkl"
    nested.parent.mkdir()
    nested.write_bytes(b"synthetic forbidden filename; no data")
    with pytest.raises(ValueError, match="zero-PKL"):
        audit.zero_pkl(tmp_path)
    assert nested.exists()  # preserve evidence, never remove scientific files to pass


def complete_result(name):
    labels = ("d0o0", "d1o0", "d0o1", "d1o1")
    values = [torch.arange(256, dtype=torch.float64).reshape(128, 2)] * 4
    statistics = {
        f: {v: audit.repeat_statistics(values, expected_rows=128) for v in audit.PROTOCOL["views"]}
        for f in labels
    }
    state = dict(unchanged=True, before={"protected": "hash"}, after={"protected": "hash"})
    orders = [
        dict(
            factor=f,
            repeat=r,
            group=g,
            view=v,
            seed=audit.encode_seed(s, g, v),
            cell=audit.digest([g, v]),
            response=audit.digest([g, v]),
        )
        for f in labels
        for r, s in enumerate(audit.PROTOCOL["repeat_seeds"])
        for g in range(32)
        for v in audit.PROTOCOL["views"]
    ]
    cache = [audit.digest([name, g]) for g in range(32)]
    parity = dict(
        used_path_passed=True,
        cell_chunk_used=False,
        final_rng_equal=True,
        native_graph_reference=True,
        graph_max_abs=0.0,
        response_max_abs=0.0,
    )
    return dict(
        endpoint={"name": name},
        protocol=audit.PROTOCOL,
        panel_sha256="panel",
        panel_schedule_sha256="schedule",
        query_sha256="query",
        matched_views_sha256="views",
        training_identity={
            "data": {
                "physical": "same",
                "loss_protocol": {**({"spectral_pair": {"sealed": name}} if name != "RI" else {})},
            }
        },
        cache_sha256=cache,
        cache_after_sha256=cache,
        statistics=statistics,
        state=state,
        factor_states={f: state for f in labels},
        factor_modes={f: {} for f in labels},
        crossed={f: {} for f in labels},
        actual_orders=orders,
        observed_calls=dict(
            student_encode=1728,
            native_graph=128,
            objective=0,
            teacher=0,
            projector=0,
            decoder=0,
            backward=0,
            autograd_grad=0,
            optimizer_step=0,
            statistics_commit=0,
        ),
        control_expression_reads=dict(calls=64, rows=256, forbidden=0),
        zero_pkl=dict(checked=True, count=0),
        updates=0,
        backward_calls=0,
        objective_calls=0,
        parity=[{v: parity for v in audit.PROTOCOL["views"]} for _ in range(32)],
    )


def test_endpoint_validation_does_not_compare_learned_graph_values_across_models():
    results = [complete_result(n) for n in audit.NAMES]
    assert results[0]["cache_sha256"] != results[-1]["cache_sha256"]
    assert audit.combine(results)["scientific_complete"] is False


@pytest.mark.parametrize(
    "bad",
    [
        "duplicate_order",
        "seed",
        "fixed_order",
        "d0noise",
        "d0tensor",
        "factor_state",
        "cache",
        "parity",
        "forbidden",
        "control_read",
        "zero_pkl",
    ],
)
def test_metadata_validator_rejects_incomplete_or_drifted_production_receipts(bad):
    row = complete_result("RI")
    if bad == "duplicate_order":
        row["actual_orders"][-1] = row["actual_orders"][0]
    elif bad == "seed":
        row["actual_orders"][0]["seed"] += 1
    elif bad == "fixed_order":
        row["actual_orders"][0].update(cell="changed", response="changed")
    elif bad == "d0noise":
        row["statistics"]["d0o0"]["main"]["Wraw"] = 1
    elif bad == "d0tensor":
        row["statistics"]["d0o0"]["main"]["repeat_sha256"][0] = "different"
    elif bad == "factor_state":
        row["factor_states"].pop("d1o1")
    elif bad == "cache":
        row["cache_after_sha256"] = ["changed"] * 32
    elif bad == "parity":
        row["parity"][0]["main"]["native_graph_reference"] = False
    elif bad == "forbidden":
        row["observed_calls"]["statistics_commit"] = 1
    elif bad == "control_read":
        row["control_expression_reads"]["forbidden"] = 1
    else:
        row["zero_pkl"]["count"] = 1
    with pytest.raises(ValueError):
        audit.combine([row, complete_result("R13SRAMP"), complete_result("NC")])
