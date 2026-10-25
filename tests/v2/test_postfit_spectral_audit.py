"""Frozen panel statistics, native encoder dispatch and immutable audit gates."""

from __future__ import annotations

import ast
import copy
import json
import random
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import postfit_spectral_audit as audit  # noqa: E402


def test_population_energy_rank_and_scale_invariance():
    x = torch.randn(128, 256, generator=torch.Generator().manual_seed(7), dtype=torch.float64) + 3
    original = audit.spectral_statistics(x)
    scaled = audit.spectral_statistics(x * 1e-6)
    assert original["rank_ceiling"] == 127
    assert original["trace"] + original["mean_energy"] == pytest.approx(original["total_energy"])
    assert scaled["q"] == pytest.approx(original["q"], rel=1e-12)
    assert scaled["effective_rank"] == pytest.approx(original["effective_rank"], rel=1e-12)
    assert scaled["trace"] == pytest.approx(original["trace"] * 1e-12)
    assert original["q"] < 0.2


def test_same_feature_affine_removal_and_row_rms_scale_invariance():
    base = torch.randn(20, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    gamma = torch.tensor([0.1, -2, 4, 0.02, 3, 0.5], dtype=torch.float64)
    first = audit.normalization_statistics(base * gamma, gamma)
    second = audit.normalization_statistics(base * gamma * 5, gamma * 5)
    reference = audit.spectral_statistics(base)
    for key in ("q", "trace", "total_energy", "mean_energy", "effective_rank"):
        assert first["affine_free"][key] == pytest.approx(reference[key])
        assert second["affine_free"][key] == pytest.approx(reference[key])
        assert second["fixed_row_rms"][key] == pytest.approx(first["fixed_row_rms"][key])
    assert first["raw"]["q"] != pytest.approx(reference["q"])
    assert first["raw_activations_saved"] is False


def test_zero_energy_and_zero_gamma_are_explicit_na_without_clamps():
    out = audit.normalization_statistics(torch.zeros(4, 3), torch.tensor([0.0, 1, 2]))
    assert out["raw"]["q"] is None and out["raw"]["effective_rank"] == 0
    assert out["affine_free"] is None and out["zero_gamma_features"] == 1
    assert out["fixed_row_rms"] is None and out["zero_energy_rows"] == 4
    constant = audit.spectral_statistics(torch.ones(4, 3))
    assert constant["q"] == 0 and constant["effective_rank"] == 0
    assert audit.spectral_statistics(torch.ones(1, 3))["effective_rank"] is None
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("x", [torch.empty(0, 3), torch.ones(3), torch.full((3, 2), float("nan"))])
def test_invalid_features_rejected(x):
    with pytest.raises(ValueError, match="matrix"):
        audit.spectral_statistics(x)


class Model(torch.nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        self.response = torch.nn.Module()
        self.response.norm = torch.nn.RMSNorm(3)
        self.options = SimpleNamespace(control_conditioned_graph=conditional)
        self.graph = SimpleNamespace(relay_prefix=lambda context: context)
        self.calls = []

    def encode_response(self, graph, control, condition, *mask, decode=True):
        self.calls.append(dict(mask=mask, control=control.clone(), decode=decode))
        x = torch.stack((control.mean(1), control.sum(1), condition[:, 0]), -1)
        result = self.response.norm(x[:, None, :])
        return {"response_cls": result[:, -1]}


class Objective(torch.nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        self.student, self.teacher = Model(conditional), Model(conditional)
        self.register_buffer("ssl2_cls_center", torch.zeros(3))
        self.pending = []
        self.weights, self.lambda2 = ((1, 1), (1, 1)), 1
        self.auxiliary_rng_counter = 0
        self.graph_calls = []

    def forward(self, *args, **kwargs):
        raise AssertionError("objective/CE must never be invoked")

    def _batch_graph(self, model, batch, expression_mask=None, positions=None, prefix=None):
        self.graph_calls.append((model, expression_mask, positions, prefix))
        return torch.ones(3, 3), torch.ones(len(batch.control), 3)

    def _graph(self, *args, **kwargs):
        raise AssertionError("synthetic cached graph kernel is substituted in this test")

    @staticmethod
    def _select_graph(graph, positions):
        return graph[positions]

    def response_metadata(self, *args):
        return {}


def runtime():
    objective = Objective()
    return SimpleNamespace(
        objective=objective,
        optimizer=torch.optim.SGD(objective.parameters(), lr=0.1),
        generator=np.random.default_rng(123),
        device=torch.device("cpu"),
        index=SimpleNamespace(),
    )


def test_frozen_guard_restores_all_rng_and_modes_without_parameter_updates():
    rt = runtime()
    rt.objective.teacher.eval()
    with audit.frozen_observation(rt) as receipt:
        assert not torch.is_grad_enabled() and not rt.objective.training
        rt.generator.random()
        random.random()
        np.random.random()
        torch.rand(2)
        rt.objective.student.encode_response(None, torch.ones(4, 2), torch.ones(4, 3))
    assert receipt["unchanged"] and receipt["before"] == receipt["after"]
    assert rt.objective.training and not rt.objective.teacher.training
    assert all(p.grad is None for p in rt.objective.parameters())


def test_frozen_guard_detects_buffer_mutation_and_restores_rng_on_exception():
    rt = runtime()
    initial_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="changed"), audit.frozen_observation(rt):
        torch.rand(4)
        rt.objective.ssl2_cls_center.add_(1)
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert torch.equal(rt.objective.ssl2_cls_center, torch.ones(3))  # mutation not concealed


def test_real_relay_train_override_preserves_mixed_training_and_order_flags():
    from gradpert.config.v2 import V2Architecture
    from gradpert.modeling.v2 import GraDPertV2
    from gradpert.modeling.v2.operators import RelayDeltaAttention

    options = V2Architecture(
        width=8,
        heads=2,
        latent_rank=4,
        streams=1,
        graph_layers=4,
        graph_read_mode="relay",
        attention="relay_full",
        kda_layers=2,
        dropout=0,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
        checkpoint_layers=False,
    )
    rt = runtime()
    rt.objective.student = GraDPertV2(torch.randn(9, 8), options).train()
    rt.objective.teacher = GraDPertV2(torch.randn(9, 8), options).eval()
    # Native Teacher may be eval while randomized view ordering remains enabled.
    rt.objective.teacher.set_relay_order_randomization(True)
    attention = next(
        m for m in rt.objective.teacher.modules() if isinstance(m, RelayDeltaAttention)
    )
    attention.randomize_order = None  # exact tri-state restoration, not a blanket .train(mode)
    rt.optimizer = torch.optim.SGD(rt.objective.parameters(), lr=0.1)

    def flags():
        return {
            name: {
                key: getattr(module, key)
                for key in ("training", "randomize_relay_order", "randomize_order")
                if hasattr(module, key)
            }
            for name, module in rt.objective.named_modules()
        }

    original = flags()
    assert rt.objective.student.randomize_relay_order
    assert not rt.objective.teacher.training and rt.objective.teacher.randomize_relay_order
    with audit.frozen_observation(rt) as proof:
        assert not rt.objective.student.randomize_relay_order
        assert not rt.objective.teacher.randomize_relay_order
        assert attention.randomize_order is False
    assert flags() == original
    assert proof["unchanged"] and proof["before"] == proof["after"]
    assert "transient_module_flags" in proof["before"]
    assert proof["changed_protected_keys"] == []


@pytest.mark.parametrize("mutated", ["parameter", "buffer"])
def test_frozen_guard_reports_changed_keys_and_preserves_inner_error(mutated):
    rt = runtime()
    with (
        pytest.raises(ValueError, match="protected keys") as caught,
        audit.frozen_observation(rt) as proof,
    ):
        if mutated == "parameter":
            rt.objective.student.response.norm.weight.add_(1)
        else:
            rt.objective.ssl2_cls_center.add_(1)
        raise RuntimeError("original diagnostic failure")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "original diagnostic failure" in str(caught.value)
    expected_key = "student" if mutated == "parameter" else "centers"
    assert proof["changed_protected_keys"] == [expected_key]
    assert expected_key in str(caught.value)
    if mutated == "parameter":
        assert torch.equal(rt.objective.student.response.norm.weight, torch.full((3,), 2.0))
    else:
        assert torch.equal(rt.objective.ssl2_cls_center, torch.ones(3))


def test_frozen_guard_preserves_original_failure_when_protected_state_unchanged():
    rt = runtime()
    original = RuntimeError("encoder failed before any protected mutation")
    with pytest.raises(RuntimeError) as caught, audit.frozen_observation(rt) as proof:
        raise original
    assert caught.value is original
    assert proof["unchanged"] and proof["changed_protected_keys"] == []


@pytest.mark.parametrize("conditional", [False, True])
def test_native_global_local_inputs_without_objective_backward_or_teacher_locals(
    conditional, monkeypatch
):
    obj = Objective(conditional)
    batch = SimpleNamespace(
        control=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        query_positions=torch.arange(3),
        graph=SimpleNamespace(context=torch.ones(1)),
        cell_views=tuple(
            SimpleNamespace(positions=torch.tensor([0, 2]), mask=torch.tensor([False, True]))
            for _ in range(4)
        ),
    )
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("backward called"))
    with torch.no_grad():
        values = audit.collect_group(SimpleNamespace(objective=obj), batch)
    assert set(values) == {
        "student.global0",
        "student.global1",
        "student.local0",
        "student.local1",
        "teacher.global0",
        "teacher.global1",
    }
    assert len(obj.student.calls) == 4 and len(obj.teacher.calls) == 2
    assert all(len(c["mask"]) == 1 and c["decode"] is False for c in obj.student.calls)
    assert all(not c["mask"] and c["decode"] is False for c in obj.teacher.calls)
    assert all(
        torch.equal(c["control"], batch.control[:, [0, 2]])
        for c in obj.student.calls + obj.teacher.calls
    )
    if conditional:
        assert len(obj.graph_calls) == 8
        assert all(c[1] is None for c in obj.graph_calls if c[0] is obj.teacher)
    assert not obj.pending and all(p.grad is None for p in obj.parameters())
    assert (
        not obj.student.response.norm._forward_hooks
        and not obj.teacher.response.norm._forward_hooks
    )


def test_hook_cleanup_on_bad_return_and_no_grad_gate():
    model = Model()
    with pytest.raises(ValueError, match="without autograd"):
        audit.observed_cls(
            model, lambda: model.encode_response(None, torch.ones(2, 3), torch.ones(2, 3))
        )
    assert not model.response.norm._forward_hooks
    with torch.no_grad(), pytest.raises(ValueError, match="equals"):
        audit.observed_cls(model, lambda: {"response_cls": torch.zeros(2, 3)})
    assert not model.response.norm._forward_hooks


def test_native_source_verification_binds_publication_and_commit(monkeypatch):
    import gradpert.execution.identity as identity

    calls = []
    payload = {"commit": "1" * 40, "dirty": False}
    monkeypatch.setattr(
        identity,
        "inspect_source_identity",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(payload=lambda: payload),
    )
    binding = dict(root="/source", commit="1" * 40, publication="/pub", publication_sha256="2" * 64)
    assert audit.verify_source(binding) == payload
    assert calls[0][1]["formal"] and calls[0][1]["publication_receipt"] == "/pub"
    payload["commit"] = "3" * 40
    with pytest.raises(ValueError, match="source differs"):
        audit.verify_source(binding)


def fixture_spec(tmp_path):
    def artifact(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=audit.sha(path))

    publication = artifact(tmp_path / "publication.json", {})
    source = dict(
        root=str(tmp_path / "source"),
        commit="1" * 40,
        publication=publication["path"],
        publication_sha256=publication["sha256"],
    )
    config = artifact(tmp_path / "source/configs/config.json", {})
    endpoints = []
    for name in audit.NAMES:
        run = tmp_path / f"run-{name}"
        manifest = artifact(run / "run_manifest.json", {})
        endpoints.append(
            dict(
                name=name,
                source=source,
                run_root=str(run),
                run_manifest_sha256=manifest["sha256"],
                config=config["path"],
                config_sha256=config["sha256"],
                checkpoint_sha256="a" * 64,
            )
        )
    return dict(
        schema=audit.SCHEMA,
        analysis_source=source,
        endpoints=endpoints,
        parent=artifact(tmp_path / "parent.json", {}),
        panel=artifact(tmp_path / "panel.json", {}),
        resource_queue=artifact(tmp_path / "queue.json", {}),
        data_root=str(tmp_path / "data"),
        output=str(tmp_path / "audit"),
    )


def test_input_hash_source_output_and_server_gates(tmp_path):
    spec = fixture_spec(tmp_path)
    audit.validate_spec(spec, server=False)
    with pytest.raises(ValueError, match="server-only"):
        audit.validate_spec(spec)
    bad = copy.deepcopy(spec)
    bad["output"] = bad["endpoints"][0]["run_root"] + "/new-audit"
    with pytest.raises(ValueError, match="overlaps"):
        audit.validate_spec(bad, server=False)
    bad = copy.deepcopy(spec)
    bad["endpoints"][0]["config"] = str(tmp_path / "copied.json")
    Path(bad["endpoints"][0]["config"]).write_text("{}")
    with pytest.raises(ValueError, match="original training"):
        audit.validate_spec(bad, server=False)
    Path(spec["panel"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash differs"):
        audit.validate_spec(spec, server=False)


def test_outputs_never_overwritten(tmp_path):
    output = tmp_path / "COMPLETE.json"
    audit.write_new(output, {"a": 1})
    with pytest.raises(FileExistsError):
        audit.write_new(output, {"a": 2})
    assert json.loads(output.read_text()) == {"a": 1}


def result_bank():
    return [
        dict(
            endpoint={"name": name},
            panel_sha256="p",
            panel_schedule_sha256="s",
            query_sha256="q",
            matched_views_sha256="m",
            training_identity={
                "data": {
                    "manifest": "frozen",
                    "loss_protocol": {
                        "base": "unchanged",
                        **({"spectral_pair": {"sealed": name}} if name != "RI" else {}),
                    },
                }
            },
            parity={
                role: dict(
                    used_path_passed=True,
                    cell_chunk_used=False,
                    graph_max_abs=0.0,
                    response_max_abs=1e-7,
                    cell_chunk_passed=False,
                )
                for role in ("student", "teacher")
            },
            updates=0,
            backward_calls=0,
            objective_calls=0,
            state=dict(unchanged=True, before={"rng": "r"}, after={"rng": "r"}),
        )
        for name in audit.NAMES
    ]


@pytest.mark.parametrize(
    "drift", ["matched_views_sha256", "panel_sha256", "data", "updates", "state"]
)
def test_cross_endpoint_actual_input_and_no_update_proofs_are_required(drift):
    results = result_bank()
    assert audit.combine(results)["updates"] == 0
    if drift == "data":
        results[-1]["training_identity"]["data"] = {}
    elif drift == "state":
        results[-1]["state"]["after"] = {}
    elif drift == "updates":
        results[-1][drift] = 1
    else:
        results[-1][drift] = "changed"
    with pytest.raises(ValueError):
        audit.combine(results)


def test_worker_callgraph_has_no_objective_optimizer_or_loss_execution():
    tree = ast.parse((ROOT / "scripts/v2/postfit_spectral_audit.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {
        "backward",
        "step",
        "zero_grad",
        "update_teacher",
        "apply_centers",
        "cell_loss",
        "forward",
    }
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in calls)
    assert not any(
        isinstance(n.func, ast.Name) and n.func.id in {"objective", "obj"} for n in calls
    )


def test_only_declared_spectral_metadata_may_differ():
    results = result_bank()
    before = copy.deepcopy(results)
    audit.combine(results)
    assert results == before  # full original protocols remain in output provenance
    results[-1]["training_identity"]["data"]["loss_protocol"]["alignment_uniformity"] = {}
    with pytest.raises(ValueError, match="data identities differ"):
        audit.combine(results)
    with pytest.raises(ValueError, match="must retain"):
        audit.physical_data(before[0]["training_identity"]["data"], "R12BAL")


@pytest.mark.parametrize("bad", [-1.0, 2.0001e-6, float("nan"), float("inf"), True, None])
def test_used_path_parity_requires_finite_bounded_observations(bad):
    results = result_bank()
    results[-1]["parity"]["teacher"]["response_max_abs"] = bad
    with pytest.raises(ValueError, match="parity"):
        audit.combine(results)


def test_failed_optional_rebatch_is_not_used_or_promoted():
    results = result_bank()
    assert all(not r["parity"]["teacher"]["cell_chunk_passed"] for r in results)
    audit.combine(results)
    results[-1]["parity"]["teacher"]["cell_chunk_used"] = True
    with pytest.raises(ValueError, match="rebatching"):
        audit.combine(results)


def test_worker_assembles_original_runtime_checkpoint_and_full_panel_without_gpu(
    tmp_path, monkeypatch
):
    """Exercise the production worker assembly, substituting only I/O/CUDA/native kernels."""
    import cap40_posthoc as cap40
    import checkpoint_endpoint
    import representation_panel
    import shared_gpu

    import gradpert.training.v2.checkpoint as checkpoint_module
    import gradpert.training.v2.runtime as runtime_module

    spec = fixture_spec(tmp_path)
    endpoint = spec["endpoints"][0]
    output = Path(spec["output"])
    output.mkdir()
    audit.write_new(output / "OWNER.json", {"pid": 123})
    rt = runtime()
    rt.identity = {"loss_protocol": {"base": "unchanged"}, "physical": "frozen"}
    original_source = {"commit": endpoint["source"]["commit"], "dirty": False}
    identity = {"data": rt.identity, "source": original_source}
    parent_path = Path(spec["parent"]["path"])
    parent_path.write_text(json.dumps({"data": rt.identity}))
    spec["parent"]["sha256"] = audit.sha(parent_path)
    queue = Path(spec["resource_queue"]["path"])
    queue.write_text(json.dumps({"shared_gpu_policy": {"max_own_memory_fraction": 0.85}}))
    spec["resource_queue"]["sha256"] = audit.sha(queue)
    cp = Path(endpoint["run_root"]) / "fit/epoch-0002.pt"
    cp.parent.mkdir()
    cp.write_bytes(b"synthetic checkpoint; no weights")
    endpoint["checkpoint_sha256"] = audit.sha(cp)
    selected = {"epoch": 2, "file": cp.name, "sha256": audit.sha(cp)}
    schedule = [
        dict(
            condition_id=f"p{i}",
            rows=[{"control_row_id": f"c{j}"} for j in range(4)],
            queries=[0, 1, 2],
            view_seed=i,
        )
        for i in range(32)
    ]
    panel = dict(schedule=schedule, seed=7, panel_sha256="panel", query_sha256="query")
    batch = SimpleNamespace(
        control=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        query_positions=torch.arange(3),
        graph=SimpleNamespace(ids=torch.arange(3), context=None),
        cell_views=tuple(
            SimpleNamespace(positions=torch.tensor([0, 2]), mask=torch.tensor([False, True]))
            for _ in range(4)
        ),
    )
    calls = []

    @contextmanager
    def prepare(config, **kwargs):
        assert config is sentinel
        assert kwargs == dict(
            data_root=Path(spec["data_root"]),
            run_seed=1,
            device=torch.device("cuda:0"),
            purpose="evaluation",
        )
        calls.append("prepare_original_runtime")
        yield rt

    sentinel = object()

    def runtime_args(args):
        assert vars(args) == dict(
            method_sha=endpoint["source"]["commit"],
            config=Path(endpoint["config"]),
            run_root=Path(endpoint["run_root"]),
        )
        calls.append("runtime_args_original_config_and_sha")
        return torch, sentinel, identity, prepare, Path(endpoint["source"]["root"])

    def load(path, objective, **kwargs):
        assert path == cp and objective is rt.objective
        assert kwargs == dict(training_identity=identity, checkpoint_sha256=audit.sha(cp))
        calls.append("load_original_last")

    class Matched:
        def bind(self, selected_batch, rows):
            return selected_batch, {"matched_cell_views_sha256": "matched"}

    monkeypatch.setattr(audit, "verify_source", lambda binding: original_source)
    monkeypatch.setattr(audit, "validate_spec", lambda *_: None)
    monkeypatch.setattr(audit, "load_panel", lambda *_: panel)
    monkeypatch.setattr(cap40, "runtime_args", runtime_args)
    monkeypatch.setattr(cap40, "static_graph", lambda *_: None)
    monkeypatch.setattr(
        cap40,
        "main_response",
        lambda proxy, b, cache: proxy.objective.student.encode_response(
            None, b.control, torch.ones(4, 3)
        ),
    )
    monkeypatch.setattr(cap40, "verify_cache", lambda *_: result_bank()[0]["parity"]["student"])
    monkeypatch.setattr(
        checkpoint_endpoint,
        "select_checkpoint",
        lambda root, role: (selected, {"training_stage": "joint", "role": "last"}),
    )
    monkeypatch.setattr(
        representation_panel,
        "validate_panel_runtime",
        lambda p, r: calls.append("validate_native_panel"),
    )
    monkeypatch.setattr(representation_panel, "assemble_control_only", lambda *a, **k: batch)
    monkeypatch.setattr(representation_panel, "MatchedCellViews", Matched)
    monkeypatch.setattr(checkpoint_module, "load_evaluation_checkpoint", load)
    monkeypatch.setattr(runtime_module, "_move_batch_tree", lambda b, device: b)
    monkeypatch.setattr(shared_gpu, "snapshot", lambda owner: {"owner": owner})
    monkeypatch.setattr(shared_gpu, "admit", lambda p, o: None)
    for name in ("set_device", "set_per_process_memory_fraction"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: "default")
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *_: "default")
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(rt.optimizer, "step", lambda *a: pytest.fail("optimizer step"))
    audit.worker(spec, endpoint)
    result = audit.read(output / "RI/COMPLETE.json")
    assert calls == [
        "runtime_args_original_config_and_sha",
        "prepare_original_runtime",
        "validate_native_panel",
        "load_original_last",
    ]
    assert result["state"]["unchanged"] and result["updates"] == result["objective_calls"] == 0
    assert len(result["matched_views"]) == 32
    assert all(row["raw"]["n"] == 128 for row in result["statistics"].values())
    assert set(result["statistics"]) == {
        "student.main",
        "teacher.main",
        "student.global0",
        "student.global1",
        "student.local0",
        "student.local1",
        "teacher.global0",
        "teacher.global1",
    }
    assert result["teacher_locals"] is None and "additional" in result["teacher_main"]
