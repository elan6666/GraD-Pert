"""Synthetic postfit hooks/contracts; never touch scientific inputs or checkpoints."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).parents[2] / "scripts/v2/response_anchor_postfit.py"
loader = importlib.util.spec_from_file_location("staged_response_anchor_postfit", SCRIPT)
m = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = m
loader.loader.exec_module(m)


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        return tokens * self.scale


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.response = Response()


class Data:
    def __init__(self, controls=5, width=7):
        self.x = np.arange(controls * width, dtype=np.float64).reshape(controls, width) / 10
        self.controls = tuple(f"control{i}" for i in range(controls))
        self.truth = SimpleNamespace(
            expression=self.x[:3] + np.arange(width) / 7,
            ordered_row_ids=("truth0", "truth1", "truth2"),
        )

    def load_control_rows(self, ids):
        return SimpleNamespace(ordered_row_ids=tuple(ids), expression=self.x)

    def load_truth_rows(self, p):
        return self.truth


def native_fixture(*, width=7, controls=5, query_count=3, cell_batch=2):
    model, data = Model().eval(), Data(controls, width)
    native = SimpleNamespace()

    def predict_query_set(model, index, controls, targets, queries, *, device, cell_batch):
        result = []
        for start in range(0, len(controls), cell_batch):
            chunk = torch.tensor(controls[start : start + cell_batch, queries])
            original = torch.tensor(controls[start : start + cell_batch, 0])
            # Unequal blocks/chunks have explicitly different CLS values.
            cls = original[:, None] + torch.arange(4)[None, :] + float(queries.sum())
            tokens = torch.cat((chunk[:, :, None].expand(-1, -1, 4), cls[:, None, :]), dim=1)
            output = model.response(tokens)
            result.append(output[:, :-1, 0].detach().numpy())
        return np.concatenate(result)

    def predict_controls(model, index, controls, targets, *, device, cell_batch, query_count):
        return np.concatenate(
            [
                native.predict_query_set(
                    model,
                    index,
                    controls,
                    targets,
                    np.arange(i, min(i + query_count, controls.shape[1])),
                    device=device,
                    cell_batch=cell_batch,
                )
                for i in range(0, controls.shape[1], query_count)
            ],
            axis=1,
        )

    native.predict_query_set, native.predict_controls = predict_query_set, predict_controls

    def evaluate():
        results = []
        for p, target in [("p0", (0,)), ("p1", (1,)), ("p2", (2,))]:
            loaded = data.load_control_rows(data.controls)
            results.append(
                native.predict_controls(
                    model,
                    None,
                    loaded.expression,
                    target,
                    device="cpu",
                    cell_batch=cell_batch,
                    query_count=query_count,
                )
            )
            data.load_truth_rows(p)
        return results

    collector = m.NativeValCollector(
        ("p0", "p1", "p2"),
        dict(p0=(0,), p1=(1,), p2=(2,)),
        {p: data.controls for p in ("p0", "p1", "p2")},
        np.arange(width),
        controls=controls,
        expression_genes=width,
        query_count=query_count,
    )
    return model, data, native, collector, evaluate


@pytest.mark.parametrize("cell_batch", [1, 2, 4, 8])
def test_hook_exact_weighted_blocks_chunks_no_extra_forward_or_output_change(cell_batch):
    model, data, native, c, evaluate = native_fixture(cell_batch=cell_batch)
    reference = evaluate()
    initial_calls = model.response.calls
    methods = (native.predict_controls, native.predict_query_set)
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    with c.observe(native, model, data):
        result = evaluate()
    for a, b in zip(reference, result, strict=True):
        np.testing.assert_array_equal(a, b)
    expected_calls = 3 * 3 * ((5 + cell_batch - 1) // cell_batch)
    assert model.response.calls - initial_calls == expected_calls == c.calls["response_forward"]
    expected = np.average([3.0, 12.0, 6.0], weights=[3, 3, 1]) + data.x[:, 0].mean() + np.arange(4)
    np.testing.assert_allclose(
        c.arrays()["cls_means"], np.tile(expected, (3, 1)), rtol=0, atol=1e-14
    )
    assert native.predict_controls is methods[0] and native.predict_query_set is methods[1]
    assert "load_truth_rows" not in vars(data) and "load_control_rows" not in vars(data)
    assert not model.response._forward_hooks and torch.equal(rng, torch.get_rng_state())
    for k, v in state.items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_300_controls_five_blocks_128_128_44_exact_pooling():
    model, data, native, c, evaluate = native_fixture(
        width=5000, controls=300, query_count=1000, cell_batch=128
    )
    with c.observe(native, model, data):
        evaluate()
    expected = sum(np.arange(i, i + 1000).sum() / 5 for i in range(0, 5000, 1000))
    expected = expected + data.x[:, 0].mean() + np.arange(4)
    np.testing.assert_allclose(c.arrays()["cls_means"][0], expected, atol=1e-9, rtol=0)
    assert c.calls["response_forward"] == 3 * 5 * 3


@pytest.mark.parametrize("failure", ["query", "control_order", "unexpected_response", "exception"])
def test_scope_restores_methods_hooks_on_failure(failure):
    model, data, native, c, evaluate = native_fixture()
    old = (native.predict_controls, native.predict_query_set)
    with pytest.raises((ValueError, LookupError)), c.observe(native, model, data):
        if failure == "exception":
            raise LookupError("synthetic")
        if failure == "control_order":
            data.controls = tuple(reversed(data.controls))
            evaluate()
        elif failure == "unexpected_response":
            model.response(torch.zeros(2, 4, 4))
        else:
            data.load_control_rows(data.controls)
            c.begin(data.x, (0,))
            c.begin_block(np.array([1, 2, 3]))
    assert (native.predict_controls, native.predict_query_set) == old
    assert not model.response._forward_hooks
    assert "load_truth_rows" not in vars(data) and "load_control_rows" not in vars(data)


def test_missing_duplicate_or_nonfinite_coverage_rejected():
    _, data, _, c, _ = native_fixture()
    with pytest.raises(ValueError, match="incomplete"):
        c.arrays()
    c.last_control_ids = data.controls
    c.begin(data.x, (0,))
    c.begin_block(np.arange(3))
    c.block["rows"] = 4
    with pytest.raises(ValueError, match="omitted"):
        c.end_block()
    with pytest.raises(ValueError, match="nonfinite"):
        c.response_hook(None, None, torch.full((1, 4, 4), float("nan")))


def synthetic_arrays():
    rng = np.random.default_rng(7)
    n, a, d = 4, 7, 3
    return dict(
        condition_ids=np.array([f"p{i}" for i in range(n)]),
        allowed_gene_indices=np.arange(a, dtype=np.int64),
        own_target_indices=np.arange(n, dtype=np.int64)[:, None],
        prediction_delta=rng.normal(size=(n, a)),
        truth_delta=rng.normal(size=(n, a)),
        cls_means=rng.normal(size=(n, d)),
    )


def test_npz_exact_schema_no_pickle_hash_tamper_and_server_boundary(tmp_path, monkeypatch):
    arrays = synthetic_arrays()
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path / "server")
    with pytest.raises(ValueError, match="server"):
        m.save_arrays(tmp_path / "bad.npz", arrays)
    root = tmp_path / "server"
    root.mkdir()
    b = m.save_arrays(root / "ok.npz", arrays)
    reloaded = m.load_arrays(b)
    for name in arrays:
        np.testing.assert_array_equal(reloaded[name], arrays[name])
    with pytest.raises(ValueError, match="fresh"):
        m.save_arrays(root / "ok.npz", arrays)
    Path(b["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        m.load_arrays(b)


def test_actual_statistics_api_smoke_and_native_point_gates():
    path = os.environ.get("POSTFIT_GEOMETRY_PATH")
    if path:
        spec = importlib.util.spec_from_file_location("synthetic_geometry", path)
        geo = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = geo
        spec.loader.exec_module(geo)
    else:
        from gradpert.evaluation import response_geometry as geo
    arrays = synthetic_arrays()
    result = geo.summarize_endpoint(**arrays)
    assert result["condition_N"] == 4 and len(result["per_condition"]) == 4
    comparison = geo.compare_endpoints(arrays, arrays, draws=10)
    assert comparison["ownmask_MSE_delta"]["candidate_minus_reference"] == 0

    def point(value):
        return dict(
            prediction_loss=1,
            metric_gene_groups={
                "seen_expression": {
                    "metrics": [
                        dict(metric_id=k, macro_mean=value, finite_condition_count=445)
                        for k in (
                            "txpert_macro_pearson_delta_all",
                            "trishift_pearson_delta_all",
                            "systema_pearson_all",
                        )
                    ]
                }
            },
        )

    assert m.native_point_gates(point(0.2), point(0.3))["passed"]
    assert not m.native_point_gates(point(0.2), point(0.1))["passed"]


def spec_fixture(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    def binding(p):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return dict(path=str(p), sha256=m.sha(p))

    endpoints = []
    for name in ("R3-RI", "FixedResponseGroupAnchor"):
        root = tmp_path / name
        source = tmp_path / (name + "-source")
        endpoints.append(
            dict(
                name=name,
                run_root=str(root),
                source=dict(
                    root=str(source),
                    commit="a" * 40,
                    publication=binding(source / "publication.json"),
                ),
                run_manifest=binding(root / "run_manifest.json"),
                journal=binding(root / "fit/epoch_state.json"),
                config=binding(source / "configs/config.json"),
                checkpoint=binding(root / "fit/synthetic.pt"),
            )
        )
    data = tmp_path / "data"
    data.mkdir()
    return dict(
        schema=m.SCHEMA,
        analysis_source=dict(
            root=str(tmp_path / "analysis"),
            commit="b" * 40,
            publication=binding(tmp_path / "analysis/publication.json"),
        ),
        data_root=str(data),
        shared_policy=binding(tmp_path / "policy.json"),
        output=str(tmp_path / "output"),
        validation=dict(
            conditions=[f"p{i}" for i in range(445)],
            controls_per_condition=300,
            expression_genes=5000,
            query_count=1000,
            allowed_gene_indices=list(range(4775)),
            allowed_gene_ids_sha256="1" * 64,
            expression_gene_ids_sha256="2" * 64,
            control_manifest_sha256="3" * 64,
            reference_sha256="4" * 64,
        ),
        endpoints=endpoints,
        bootstrap=dict(draws=10000, seed=20261028),
        medium={},
    )


def test_spec_metadata_only_and_no_test_selection(tmp_path):
    spec = spec_fixture(tmp_path)
    m.validate_spec(spec, server=False)
    spec["validation"]["conditions"][0] = ""
    with pytest.raises(ValueError):
        m.validate_spec(spec, server=False)


@pytest.mark.parametrize("case", ["short", "test", "overlap", "bootstrap", "cp_outside", "extra"])
def test_invalid_or_unsealed_spec_fails(tmp_path, case):
    s = spec_fixture(tmp_path)
    if case == "short":
        s["validation"]["controls_per_condition"] = 299
    elif case == "test":
        s["split"] = "test"
    elif case == "overlap":
        s["output"] = s["data_root"]
    elif case == "bootstrap":
        s["bootstrap"]["draws"] = 199
    elif case == "cp_outside":
        s["endpoints"][0]["checkpoint"] = s["endpoints"][0]["config"]
    elif case == "extra":
        s["endpoints"][0]["warmstart"] = True
    with pytest.raises(ValueError):
        m.validate_spec(s, server=False)


@pytest.mark.parametrize("entry", ["teacher", "optimizer", "commit", "backward", "grad"])
def test_forbidden_update_entry_points_restore_even_on_error(entry):
    teacher = nn.Linear(2, 2).requires_grad_(False)
    optimizer = SimpleNamespace(step=lambda: None)
    objective = SimpleNamespace(teacher=teacher, commit_statistics=lambda x: None)
    runtime = SimpleNamespace(optimizer=optimizer, objective=objective)
    original_backward, original_grad = torch.Tensor.backward, torch.autograd.grad
    original_step = optimizer.step
    with pytest.raises(ValueError, match="forbidden"), m.forbid_updates(runtime) as counts:
        if entry == "teacher":
            teacher(torch.ones(1, 2))
        elif entry == "optimizer":
            optimizer.step()
        elif entry == "commit":
            objective.commit_statistics(0.9)
        elif entry == "backward":
            torch.ones(1, requires_grad=True).sum().backward()
        else:
            torch.autograd.grad(torch.ones(1, requires_grad=True), [])
    assert sum(counts.values()) == 1
    assert torch.Tensor.backward is original_backward and torch.autograd.grad is original_grad
    assert optimizer.step is original_step
    assert not teacher._forward_pre_hooks


def test_native_result_row_identity_and_metric_count_validation():
    model, data, native, c, evaluate = native_fixture()
    with c.observe(native, model, data):
        evaluate()
    names = [
        "txpert_macro_pearson_delta_all",
        "txpert_macro_pearson_delta_deg",
        "trishift_pearson_delta_all",
        "trishift_pearson_delta_deg",
        "systema_pearson_all",
        "systema_pearson_deg",
    ]
    values = [dict(metric_id=k, total_condition_count=3, finite_condition_count=3) for k in names]
    result = dict(
        split="val",
        prediction_loss=0.1,
        metrics=values,
        metric_gene_groups={
            g: dict(metrics=copy.deepcopy(values)) for g in ["seen_expression", "unseen_expression"]
        },
        conditions=[
            dict(
                condition_id=p,
                control_row_ids_sha256=c.rows[p]["control_sha256"],
                truth_row_ids_sha256=c.rows[p]["truth_sha256"],
            )
            for p in c.conditions
        ],
    )
    m.validate_native_result(result, c)
    result["conditions"][0]["truth_row_ids_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        m.validate_native_result(result, c)


def owner_fixture(tmp_path):
    spec = spec_fixture(tmp_path)
    endpoint = spec["endpoints"][1]
    path = tmp_path / "POSTFIT_SPEC.json"
    path.write_text(json.dumps(spec))
    spec_file = dict(path=str(path), sha256=m.sha(path))
    script = str(Path(spec["analysis_source"]["root"]) / "scripts/v2/response_anchor_postfit.py")
    arguments = [
        "/usr/bin/python3",
        script,
        "run",
        "--spec",
        str(path),
        "--spec-sha256",
        spec_file["sha256"],
    ]
    raw = b"\0".join(arg.encode() for arg in arguments) + b"\0"
    process = dict(
        pid=400,
        uid=1006,
        start_ticks="12345",
        command_sha256=m.hashlib.sha256(raw).hexdigest(),
        script_basename="response_anchor_postfit.py",
    )
    live = {400: process}
    receipt = m.make_dispatcher_owner(
        spec,
        endpoint,
        spec_file,
        lookup=live.get,
        pid=400,
        uid=1006,
        command_reader=lambda pid: raw,
    )
    parents = {401: 400, 402: 400, 400: 1, 900: 1}
    return spec, endpoint, spec_file, receipt, live, parents, raw


def native_gpu_api():
    path = os.environ.get("POSTFIT_SHARED_GPU_PATH")
    if path is None:
        path = str(Path(__file__).parents[2] / "scripts/v2/shared_gpu.py")
    # Execute the exact native pure policy source without creating __pycache__ there.
    module = SimpleNamespace(__file__=path)
    exec(compile(Path(path).read_text(), path, "exec"), module.__dict__)
    return module


def policy_fixture(api):
    return dict(
        schema=api.DYNAMIC_SCHEMA,
        devices=[dict(gpu_index=0, gpu_uuid="GPU-a"), dict(gpu_index=1, gpu_uuid="GPU-b")],
        foreign_uid=1003,
        command_basename="run_scbutterfly_archived_predictions.py",
        max_foreign_processes_per_gpu=1,
        max_foreign_memory_mib=1024,
        min_free_memory_mib=2048,
        max_own_memory_fraction=0.85,
        authorization_reference=api.SCBUTTERFLY_AUTHORIZATION,
    )


def test_sibling_already_using_cuda_is_owned_by_sealed_dispatcher_and_callback_rechecks(tmp_path):
    api = native_gpu_api()
    spec, endpoint, specfile, owner, live, parents, raw = owner_fixture(tmp_path)
    policy = policy_fixture(api)
    original_policy = copy.deepcopy(policy)
    seen = []

    def snapshot(pid):
        seen.append(pid)
        # GPU1 sibling initialized before worker0 admission, the original race.
        return dict(
            gpus={
                i: dict(
                    uuid="GPU-a" if i == 0 else "GPU-b",
                    total_mib=32000,
                    used_mib=1000,
                    free_mib=31000,
                )
                for i in (0, 1)
            },
            processes=[
                dict(
                    gpu_uuid="GPU-b",
                    pid=402,
                    memory_mib=1000,
                    owned=api.descendant(402, pid, parents),
                    identity=dict(pid=402, uid=1006, script_basename="response_anchor_postfit.py"),
                )
            ],
        )

    # Old self-owner behavior genuinely fails the unmodified native admission.
    with pytest.raises(RuntimeError, match="foreign"):
        api.admit(policy, snapshot(401))
    guard = m.DispatcherAdmission(
        owner,
        spec,
        endpoint,
        specfile,
        policy,
        lookup=live.get,
        parent_lookup=lambda: parents,
        descendant=api.descendant,
        snapshot=snapshot,
        admit=api.admit,
        worker_pid=401,
        uid=1006,
        worker_index=0,
        command_reader=lambda pid: raw,
    )
    view = guard.check()
    assert view["processes"][0]["owned"]
    guard.on_condition_complete(1, 223, spec["validation"]["conditions"][0])
    assert seen == [401, 400, 400] and guard.conditions_checked == 1 and guard.calls == 2
    assert policy == original_policy


@pytest.mark.parametrize(
    "change", ["start_ticks", "command_sha256", "missing", "uid", "nonancestor", "other_run"]
)
def test_owner_pid_reuse_nonancestor_and_crossrun_rejected_before_snapshot(tmp_path, change):
    api = native_gpu_api()
    spec, endpoint, specfile, owner, live, parents, raw = owner_fixture(tmp_path)
    if change in ("start_ticks", "command_sha256", "uid"):
        live[400] = dict(live[400])
        live[400][change] = {"start_ticks": "98765", "command_sha256": "0" * 64, "uid": 999}[change]
    elif change == "missing":
        live.clear()
    elif change == "nonancestor":
        parents[401] = 900
    else:
        spec = copy.deepcopy(spec)
        spec["output"] += "_other"
    seen = []
    guard = m.DispatcherAdmission(
        owner,
        spec,
        endpoint,
        specfile,
        {},
        lookup=live.get,
        parent_lookup=lambda: parents,
        descendant=api.descendant,
        snapshot=lambda pid: seen.append(pid),
        admit=lambda *args: None,
        worker_pid=401,
        uid=1006,
        worker_index=0,
        command_reader=lambda pid: raw,
    )
    with pytest.raises(ValueError):
        guard.check()
    assert not seen


def test_actual_raw_NUL_hash_not_reconstructed_arguments_and_owner_liveness_each_callback(tmp_path):
    api = native_gpu_api()
    spec, endpoint, specfile, owner, live, parents, raw = owner_fixture(tmp_path)
    # Duplicate terminal NUL leaves decoded args unchanged but must change raw hash.
    with pytest.raises(ValueError, match="raw NUL"):
        m.verified_dispatcher_owner(
            owner,
            spec,
            endpoint,
            specfile,
            lookup=live.get,
            parents=parents,
            descendant=api.descendant,
            worker_pid=401,
            uid=1006,
            command_reader=lambda pid: raw + b"\0",
        )
    guard = m.DispatcherAdmission(
        owner,
        spec,
        endpoint,
        specfile,
        policy_fixture(api),
        lookup=live.get,
        parent_lookup=lambda: parents,
        descendant=api.descendant,
        snapshot=lambda pid: dict(gpus={}, processes=[]),
        admit=lambda *args: None,
        worker_pid=401,
        uid=1006,
        worker_index=0,
        command_reader=lambda pid: raw,
    )
    guard.check()
    live[400] = dict(live[400], start_ticks="999")
    with pytest.raises(ValueError):
        guard.on_condition_complete(1, 223, "p0")
    assert guard.calls == 1


def test_common_owner_does_not_whitelist_foreign_postfit_or_weaken_memory_limits(tmp_path):
    api = native_gpu_api()
    spec, endpoint, specfile, owner, live, parents, raw = owner_fixture(tmp_path)
    policy = policy_fixture(api)
    telemetry = dict(
        gpus={
            i: dict(
                uuid="GPU-a" if i == 0 else "GPU-b", total_mib=32000, used_mib=1000, free_mib=31000
            )
            for i in (0, 1)
        },
        processes=[
            dict(
                gpu_uuid="GPU-b",
                pid=900,
                memory_mib=1000,
                owned=False,
                identity=dict(pid=900, uid=1006, script_basename="response_anchor_postfit.py"),
            )
        ],
    )
    guard = m.DispatcherAdmission(
        owner,
        spec,
        endpoint,
        specfile,
        policy,
        lookup=live.get,
        parent_lookup=lambda: parents,
        descendant=api.descendant,
        snapshot=lambda pid: telemetry,
        admit=api.admit,
        worker_pid=401,
        uid=1006,
        worker_index=0,
        command_reader=lambda pid: raw,
    )
    with pytest.raises(RuntimeError, match="foreign"):
        guard.check()
    telemetry["processes"][0].update(pid=402, memory_mib=30000, owned=True)
    with pytest.raises(RuntimeError, match="allowance"):
        guard.check()
    telemetry["processes"] = []
    telemetry["gpus"][0]["free_mib"] = 1000
    with pytest.raises(RuntimeError, match="reserve"):
        guard.check()


def test_owner_grandparent_is_allowed_but_self_owner_and_wrong_native_callback_rejected(tmp_path):
    api = native_gpu_api()
    spec, endpoint, specfile, owner, live, parents, raw = owner_fixture(tmp_path)
    parents[401] = 399
    parents[399] = 400
    assert (
        m.verified_dispatcher_owner(
            owner,
            spec,
            endpoint,
            specfile,
            lookup=live.get,
            parents=parents,
            descendant=api.descendant,
            worker_pid=401,
            uid=1006,
            command_reader=lambda pid: raw,
        )
        == 400
    )
    with pytest.raises(ValueError):
        m.verified_dispatcher_owner(
            owner,
            spec,
            endpoint,
            specfile,
            lookup=live.get,
            parents=parents,
            descendant=api.descendant,
            worker_pid=400,
            uid=1006,
            command_reader=lambda pid: raw,
        )
    guard = m.DispatcherAdmission(
        owner,
        spec,
        endpoint,
        specfile,
        {},
        lookup=live.get,
        parent_lookup=lambda: parents,
        descendant=api.descendant,
        snapshot=lambda pid: {},
        admit=lambda *args: None,
        worker_pid=401,
        uid=1006,
        worker_index=0,
        command_reader=lambda pid: raw,
    )
    with pytest.raises(ValueError, match="order"):
        guard.on_condition_complete(1, 223, "p2")
