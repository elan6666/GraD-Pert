"""CPU contracts for original-signature dispatch and sealed matched CLS audits."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import postfit_matched_cls_audit as audit  # noqa: E402
from representation_panel import panel_schedule  # noqa: E402


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=audit.sha(path))


@pytest.fixture
def specification(tmp_path, monkeypatch):
    schedules = []
    for i in range(32):
        context = i // 2
        rows = [
            dict(
                condition_id=f"p{i}",
                truth_row_id=f"truth{i}",
                control_row_id=f"c{context}-{j}",
                cell_line="Jurkat",
                cell_type="jurkat",
                context_id=f"ctx{context}",
                batch=str(context),
            )
            for j in range(4)
        ]
        schedules.append(dict(condition_id=f"p{i}", rows=rows, queries=[0, 1, 2]))
    parent = dict(
        analysis_scope="medium_cap40_stratified",
        seed=20261021,
        schedule=schedules,
        data={"physical": "exact"},
    )
    panel = panel_schedule(parent)
    old_recipes = [
        dict(
            condition_id=g["condition_id"],
            context=g["context"],
            query_sha256=panel["query_sha256"],
            controls_sha256=audit.digest([r["control_row_id"] for r in g["rows"]]),
            native_cell_views_sha256="native",
            matched_cell_views_sha256=f"matched-{g['context'][2]}",
        )
        for g in panel["schedule"]
    ]
    parent_binding = save(tmp_path / "parent.json", parent)
    panel_binding = save(tmp_path / "panel.json", panel)
    recipe_binding = save(tmp_path / "recipes.json", old_recipes)
    monkeypatch.setattr(
        audit,
        "INPUT_HASHES",
        dict(
            parent=parent_binding["sha256"],
            panel=panel_binding["sha256"],
            matched_views=recipe_binding["sha256"],
        ),
    )
    endpoints, fixeds = [], copy.deepcopy(audit.ENDPOINTS)
    for name in audit.NAMES:
        source = tmp_path / name / "source"
        config = source / "configs" / "original.json"
        config.parent.mkdir(parents=True)
        config.write_text(name)
        root = tmp_path / name / fixeds[name]["run_id"]
        publication = save(
            tmp_path / name / "pub.json", {"verification_method": "git_ls_remote_live"}
        )
        binding = dict(
            root=str(source),
            commit=fixeds[name]["commit"],
            publication=publication["path"],
            publication_sha256=publication["sha256"],
        )
        identity = dict(
            run_id=fixeds[name]["run_id"],
            source=dict(commit=binding["commit"]),
            data=parent["data"],
        )
        manifest = save(root / "run_manifest.json", identity)
        save(
            root / "resolved_config.json",
            dict(model=dict(parameters={}), training=dict(max_epochs=dict(value=4))),
        )
        cp = root / "fit/epoch-0004.pt"
        cp.parent.mkdir()
        cp.write_bytes(b"small synthetic checkpoint; never torch.load")
        save(
            root / "fit/epoch_state.json",
            dict(epoch=4, last=dict(epoch=4, file=cp.name, sha256=audit.sha(cp))),
        )
        queue = save(
            tmp_path / name / "queue.json",
            dict(
                source_commit=binding["commit"],
                rows=[
                    dict(
                        name=name,
                        plan=dict(
                            run_root=str(root),
                            run_id=identity["run_id"],
                            config=str(config),
                            config_sha256=audit.sha(config),
                        ),
                    )
                ],
                shared_gpu_policy={"synthetic": True},
            ),
        )
        endpoint = dict(
            name=name,
            source=binding,
            run_root=str(root),
            run_manifest_sha256=manifest["sha256"],
            config=str(config),
            config_sha256=audit.sha(config),
            checkpoint_sha256=audit.sha(cp),
            queue=queue,
        )
        fixeds[name].update(
            checkpoint=endpoint["checkpoint_sha256"],
            config=endpoint["config_sha256"],
            manifest=endpoint["run_manifest_sha256"],
            publication=publication["sha256"],
            queue=queue["sha256"],
        )
        endpoints.append(endpoint)
    monkeypatch.setattr(audit, "ENDPOINTS", fixeds)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pub = save(tmp_path / "analysis_pub.json", {"verification_method": "git_ls_remote_live"})
    data = tmp_path / "data"
    data.mkdir()
    spec = dict(
        schema=audit.SCHEMA,
        analysis_source=dict(
            root=str(analysis),
            commit="a" * 40,
            publication=pub["path"],
            publication_sha256=pub["sha256"],
        ),
        parent=parent_binding,
        panel=panel_binding,
        matched_views=recipe_binding,
        resource_queue=endpoints[1]["queue"],
        endpoints=endpoints,
        output=str(tmp_path / "fresh-results"),
        data_root=str(data),
        protocol=audit.PROTOCOL,
    )
    return spec, panel, old_recipes


def test_exact_spec_and_original_endpoint_binding(specification):
    spec, panel, recipes = specification
    audit.validate_spec(spec, server=False)
    assert audit.load_inputs(spec) == (panel, recipes)
    for endpoint in spec["endpoints"]:
        cp, phase = audit.bind_endpoint(
            spec, endpoint, audit.read(Path(endpoint["run_root"]) / "run_manifest.json")
        )
        assert cp["epoch"] == phase["full_training_epochs"] == 4
    with pytest.raises(ValueError, match="server-only"):
        audit.validate_spec(spec)


@pytest.mark.parametrize(
    "change",
    [
        "endpoint",
        "source",
        "config",
        "checkpoint",
        "publication",
        "queue",
        "protocol",
        "output",
        "panel",
    ],
)
def test_spec_rejects_identity_path_and_recipe_drift(specification, change):
    spec, _, _ = specification
    spec = copy.deepcopy(spec)
    if change == "endpoint":
        spec["endpoints"].reverse()
    elif change == "source":
        spec["endpoints"][0]["source"]["commit"] = "b" * 40
    elif change in ("config", "checkpoint"):
        spec["endpoints"][0][change + "_sha256"] = "0" * 64
    elif change == "publication":
        spec["endpoints"][0]["source"]["publication_sha256"] = "0" * 64
    elif change == "queue":
        spec["resource_queue"] = spec["endpoints"][0]["queue"]
    elif change == "protocol":
        spec["protocol"]["views"] = ["main"]
    elif change == "output":
        spec["output"] = spec["analysis_source"]["root"]
    else:
        Path(spec["panel"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        audit.validate_spec(spec, server=False)


@pytest.mark.parametrize("change", ["epoch", "checkpoint_bytes", "queue_duplicate", "data"])
def test_endpoint_rejects_stale_checkpoint_or_queue(specification, change):
    spec, _, _ = specification
    endpoint = spec["endpoints"][0]
    identity = audit.read(Path(endpoint["run_root"]) / "run_manifest.json")
    if change == "epoch":
        f = Path(endpoint["run_root"]) / "fit/epoch_state.json"
        r = audit.read(f)
        r["last"]["epoch"] = 3
        save(f, r)
    elif change == "checkpoint_bytes":
        (Path(endpoint["run_root"]) / "fit/epoch-0004.pt").write_bytes(b"drift")
    elif change == "queue_duplicate":
        r = audit.read(endpoint["queue"]["path"])
        r["rows"] *= 2
        endpoint["queue"] = save(Path(endpoint["queue"]["path"]), r)
    else:
        identity["data"] = {"physical": "drift"}
    with pytest.raises(ValueError):
        audit.bind_endpoint(spec, endpoint, identity)


class Response(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.RMSNorm(3)

    def forward(self, x):
        return self.norm(x)


class LegacyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = torch.nn.Linear(3, 3)
        self.response = Response()
        self.prediction = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.GELU(), torch.nn.Linear(3, 1)
        )
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
            setattr(self, name, torch.nn.Linear(3, 3))
        self.randomize_relay_order = True
        self.options = SimpleNamespace(
            attention="relay_full",
            control_conditioned_graph=False,
            fixed_gene_order=False,
            random_gene_order=True,
        )
        self.calls = []

    def train(self, mode=True):
        super().train(mode)
        self.randomize_relay_order = mode
        return self

    # The real legacy contract intentionally has no decode keyword.
    def encode_response(
        self, gene, control, condition, expression_mask=None, *, query_gene_ids=None
    ):
        self.calls.append((control.shape, expression_mask is not None))
        x = torch.stack((control.mean(1), control.sum(1), condition[:, 0]), -1)
        response = self.response(x[:, None, :])
        return dict(
            response_cls=response[:, -1],
            response_tokens=response,
            prediction=self.prediction(response[:, -1]),
        )


class Objective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student, self.teacher = LegacyModel(), LegacyModel()
        self.register_buffer("ssl2_cls_center", torch.zeros(3))
        self.pending = []
        self._temperature_updates = 0
        self.auxiliary_rng_counter = 0

    def forward(self, *_a, **_k):
        raise AssertionError("native objective is forbidden")

    def commit_statistics(self, momentum):
        self.ssl2_cls_center.add_(momentum)

    def _batch_graph(self, model, batch):
        return model.graph(torch.eye(3)), torch.ones(len(batch.control), 3)

    @staticmethod
    def _select_graph(graph, positions):
        return graph[positions]

    @staticmethod
    def response_metadata(batch, positions=None):
        return dict(
            query_gene_ids=batch.query_positions
            if positions is None
            else batch.query_positions[positions]
        )


class Data:
    def __init__(self, panel):
        ids = {
            r[k]
            for g in panel["schedule"]
            for r in g["rows"]
            for k in ("control_row_id", "truth_row_id")
        }
        self._row_index = {v: i for i, v in enumerate(sorted(ids))}

    def _read_expression_indices(self, indices):
        return np.zeros((len(indices), 3), dtype=np.float32)


def runtime(panel):
    obj = Objective()
    obj.teacher.eval()
    return SimpleNamespace(
        objective=obj,
        optimizer=torch.optim.SGD(obj.parameters(), lr=0.1),
        generator=np.random.default_rng(9),
        device=torch.device("cpu"),
        data=Data(panel),
    )


def batch(deep):
    return SimpleNamespace(
        control=torch.arange(12).float().reshape(4, 3) + 1,
        query_positions=torch.arange(3),
        cell_views=tuple(
            SimpleNamespace(positions=p, mask=torch.zeros(4, len(p), dtype=torch.bool))
            for p in [
                torch.arange(3),
                torch.tensor([0, 2]),
                torch.tensor([1]),
                torch.tensor([0, 1]),
            ]
        )
        if deep
        else (),
    )


def test_actual_legacy_collection_reference_and_decoder_count(specification):
    _, panel, _ = specification
    rt = runtime(panel)
    keys = set(vars(rt.objective)), set(vars(rt.optimizer))
    extra = rt.objective.student.response.register_forward_hook(lambda *_: None)
    original_handles = set(rt.objective.student.response._forward_hooks)
    with audit.protected_observation(rt) as state, audit.observed_paths(rt) as calls:
        main = audit.collect_inputs(rt, panel["schedule"][0], batch(False), first=True)
        deep = audit.collect_inputs(rt, panel["schedule"][0], batch(True), first=True)
    assert set(main) | set(deep) == set(audit.VIEWS)
    assert calls["student_encode"] == calls["decoder"] == 15
    assert calls["native_graph"] == 4 and all(calls[k] == 0 for k in audit.FORBIDDEN)
    assert state["unchanged"] and state["objective_state_before"] == state["objective_state_after"]
    assert (set(vars(rt.objective)), set(vars(rt.optimizer))) == keys
    assert set(rt.objective.student.response._forward_hooks) == original_handles
    assert rt.objective.student.training and not rt.objective.teacher.training
    extra.remove()


@pytest.mark.parametrize(
    "entry", ["objective", "teacher", "projector", "backward", "optimizer", "commit"]
)
def test_real_forbidden_entry_points_fail_before_updates_and_restore(specification, entry):
    _, panel, _ = specification
    rt = runtime(panel)
    original_commit = rt.objective.commit_statistics.__func__
    keys = set(vars(rt.objective))
    with (
        pytest.raises(ValueError, match="forbidden"),
        audit.protected_observation(rt),
        audit.observed_paths(rt),
    ):
        if entry == "objective":
            rt.objective(None)
        elif entry == "teacher":
            rt.objective.teacher.response(torch.ones(1, 1, 3))
        elif entry == "projector":
            rt.objective.student.ssl2_cls(torch.ones(1, 3))
        elif entry == "backward":
            torch.tensor(1.0, requires_grad=True).backward()
        elif entry == "optimizer":
            rt.optimizer.step()
        else:
            rt.objective.commit_statistics(0.99)
    assert set(vars(rt.objective)) == keys
    assert rt.objective.commit_statistics.__func__ is original_commit
    assert not rt.objective.ssl2_cls_center.any()


def test_control_expression_guard_dispatch_and_cleanup(specification):
    _, panel, _ = specification
    rt = runtime(panel)
    keys = set(vars(rt.data))
    control = panel["schedule"][0]["rows"][0]["control_row_id"]
    truth = panel["schedule"][0]["rows"][0]["truth_row_id"]
    with audit.control_only_reads(rt, panel) as proof:
        assert rt.data._read_expression_indices([rt.data._row_index[control]]).shape == (1, 3)
        with pytest.raises(ValueError, match="whitelist"):
            rt.data._read_expression_indices([rt.data._row_index[truth]])
    assert proof["calls"] == proof["rows"] == proof["forbidden"] == 1
    assert set(vars(rt.data)) == keys


@pytest.mark.parametrize("mutation", ["buffer", "parameter", "clock"])
def test_mutation_not_concealed_and_rng_flags_restored_on_failure(specification, mutation):
    _, panel, _ = specification
    rt = runtime(panel)
    before_rng = torch.get_rng_state().clone()
    flags = (
        rt.objective.training,
        rt.objective.teacher.training,
        rt.objective.student.randomize_relay_order,
    )
    with pytest.raises(ValueError, match="changed"), audit.protected_observation(rt):
        random.random()
        np.random.random()
        rt.generator.random()
        torch.rand(2)
        if mutation == "buffer":
            rt.objective.ssl2_cls_center.add_(1)
        elif mutation == "parameter":
            next(rt.objective.student.parameters()).add_(1)
        else:
            rt.objective._temperature_updates += 1
        raise RuntimeError("inner sentinel")
    assert torch.equal(before_rng, torch.get_rng_state())
    assert (
        rt.objective.training,
        rt.objective.teacher.training,
        rt.objective.student.randomize_relay_order,
    ) == flags
    if mutation == "buffer":
        assert rt.objective.ssl2_cls_center.sum() == 3


def test_recipe_mismatch_is_actual_hash_not_seed_or_endpoint_agreement(specification):
    _, _, expected = specification
    actual = copy.deepcopy(expected[0])
    audit.validate_recipe(actual, expected[0])
    actual["matched_cell_views_sha256"] = "same-new-wrong-recipe-both-endpoints"
    with pytest.raises(ValueError, match="matched_cell_views"):
        audit.validate_recipe(actual, expected[0])


@pytest.fixture
def complete_result(specification):
    spec, panel, recipes = specification
    rt = runtime(panel)
    pieces, parity, inputs = [], {}, []
    with audit.protected_observation(rt) as state, audit.observed_paths(rt) as calls:
        for i, group in enumerate(panel["schedule"]):
            values = audit.collect_inputs(rt, group, batch(False), first=i == 0)
            values.update(audit.collect_inputs(rt, group, batch(True), first=i == 0))
            pieces.append({k: v[0] for k, v in values.items()})
            inputs.append(
                dict(
                    group=i,
                    raw={"synthetic": str(i)},
                    cache_sha256={k: v[2] for k, v in values.items()},
                )
            )
            if i == 0:
                parity = {k: v[1] for k, v in values.items()}
    result = dict(
        schema=audit.RESULT_SCHEMA,
        protocol=audit.PROTOCOL,
        endpoint=spec["endpoints"][0],
        analysis_source=spec["analysis_source"],
        source=audit.read(Path(spec["endpoints"][0]["run_root"]) / "run_manifest.json")["source"],
        checkpoint=dict(
            epoch=4, file="epoch-0004.pt", sha256=spec["endpoints"][0]["checkpoint_sha256"]
        ),
        checkpoint_endpoint=dict(
            role="last", epoch=4, training_stage="joint", full_training_epochs=4
        ),
        checkpoint_journal_sha256=audit.sha(
            Path(spec["endpoints"][0]["run_root"]) / "fit/epoch_state.json"
        ),
        training_data={"physical": "exact"},
        updates=0,
        raw_activations_saved=False,
        teacher=None,
        corrected_rank=None,
        scientific_complete=False,
        inputs=inputs,
        state=state,
        observed_calls=calls,
        parity=parity,
        zero_pkl=dict(checked=True, count=0),
        control_expression_reads=dict(forbidden=0, calls=96, rows=384),
        matched_views=copy.deepcopy(recipes),
        expected_views=copy.deepcopy(recipes),
        matched_views_sha256=audit.used_recipe_digest(recipes),
        panel_sha256=spec["panel"]["sha256"],
        panel_schedule_sha256=panel["panel_sha256"],
        query_sha256=panel["query_sha256"],
        **audit.summarize(panel, pieces),
    )
    return spec, panel, result


def test_full_production_collection_counters_and_exact_GG_GL_LL_axes(complete_result):
    _, panel, result = complete_result
    audit.validate_result(result, panel)
    assert result["observed_calls"]["decoder"] == result["observed_calls"]["student_encode"] == 170
    rows = result["relations"]["pairs"]
    assert {r["view_family"] for r in rows} == {"MM", "GG", "GL", "LL"}
    assert not any((r["view_a"] == "main") != (r["view_b"] == "main") for r in rows)
    for va, vb in [
        ("main", "main"),
        ("global0", "global1"),
        ("global0", "local1"),
        ("local0", "local1"),
    ]:
        axis = [r for r in rows if (r["view_a"], r["view_b"]) == (va, vb)]
        assert (
            sum(r["pair_count"] for r in axis if r["stratum"] == "same_condition_different_control")
            == 192
        )
        assert (
            sum(r["pair_count"] for r in axis if r["stratum"] == "different_condition_same_control")
            == 64
        )


@pytest.mark.parametrize(
    "drift", ["counter", "pair", "recipe", "parity", "state", "inputs", "q", "reads"]
)
def test_receipt_rejects_incomplete_or_forged_observations(complete_result, drift):
    _, panel, result = complete_result
    result = copy.deepcopy(result)
    if drift == "counter":
        result["observed_calls"]["decoder"] = 0
    elif drift == "pair":
        result["relations"]["pairs"][0]["pair_count"] += 1
    elif drift == "recipe":
        result["matched_views"][0]["controls_sha256"] = "wrong"
    elif drift == "parity":
        result["parity"]["local0"]["response_max_abs"] = 3e-6
    elif drift == "state":
        result["state"]["unchanged"] = False
    elif drift == "q":
        result["statistics"]["main"]["q"] = 1.2
    elif drift == "reads":
        result["control_expression_reads"]["calls"] = 0
    else:
        result["inputs"].pop()
    with pytest.raises(ValueError):
        audit.validate_result(result, panel)


def test_combine_exact_inputs_and_no_weight_cache_hash_equality_required(complete_result):
    spec, _, first = complete_result
    second = copy.deepcopy(first)
    second["endpoint"] = spec["endpoints"][1]
    second["source"] = audit.read(Path(spec["endpoints"][1]["run_root"]) / "run_manifest.json")[
        "source"
    ]
    second["checkpoint"]["sha256"] = spec["endpoints"][1]["checkpoint_sha256"]
    second["checkpoint_journal_sha256"] = audit.sha(
        Path(spec["endpoints"][1]["run_root"]) / "fit/epoch_state.json"
    )
    second["inputs"][0]["cache_sha256"]["main"] = "legitimate-other-weights"
    assert audit.combine(spec, [first, second])["scientific_complete"] is False
    second["inputs"][0]["raw"]["synthetic"] = "different-input"
    with pytest.raises(ValueError, match="raw control"):
        audit.combine(spec, [first, second])


def test_zero_norm_cosine_and_singleton_stats_remain_NA(specification):
    _, panel, _ = specification
    result = audit.summarize(
        panel,
        [
            {v: torch.zeros(4, 3, dtype=torch.float64) for v in audit.VIEWS}
            for _ in panel["schedule"]
        ],
    )
    assert all(r["q"] is None for r in result["statistics"].values())
    assert all(
        r["cosine_pair_mean"] is None and r["cosine_valid_pairs"] == 0
        for r in result["relations"]["pairs"]
    )
    assert audit.spectral_statistics(torch.ones(1, 3))["effective_rank"] is None


def test_artifact_closure_precedes_success_and_rejects_extra_export(tmp_path):
    save(tmp_path / "ADMISSION.json", {})
    assert audit.artifact_closure(tmp_path, endpoint=True, pending=True)
    (tmp_path / "raw.PKL").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="zero-PKL"):
        audit.artifact_closure(tmp_path, endpoint=True, pending=True)
    assert not (tmp_path / "COMPLETE.json").exists()
    (tmp_path / "raw.PKL").unlink()
    save(tmp_path / "weights.json", {})
    with pytest.raises(ValueError, match="extra"):
        audit.artifact_closure(tmp_path, endpoint=True, pending=True)


def test_real_native_assembly_five_views_read_guard_and_call_contract():
    from representation_panel import MatchedCellViews, assemble_control_only
    from test_representation_panel_integration import native_runtime

    rt, panel = native_runtime("N0")
    matched = MatchedCellViews()
    receipts = []
    old_matcher, expected = MatchedCellViews(), []
    for group in panel["schedule"]:
        deep = assemble_control_only(
            rt, group["rows"], group["queries"], group["view_seed"], deep=True
        )
        _, receipt = old_matcher.bind(deep, group["rows"])
        expected.append(dict(condition_id=group["condition_id"], **receipt))
    before_keys = set(vars(rt.data))
    with (
        audit.protected_observation(rt) as state,
        audit.observed_paths(rt) as counts,
        audit.control_only_reads(rt, panel) as reads,
    ):
        admissions = []
        assert audit.prepare_recipes(rt, panel, expected, lambda: admissions.append(True))
        assert len(admissions) == 2
        assert counts["student_encode"] == counts["native_graph"] == 0
        for i, group in enumerate(panel["schedule"]):
            shallow = assemble_control_only(
                rt, group["rows"], group["queries"], group["view_seed"], deep=False
            )
            deep = assemble_control_only(
                rt, group["rows"], group["queries"], group["view_seed"], deep=True
            )
            assert not shallow.truth.any() and not deep.truth.any()
            deep, receipt = matched.bind(deep, group["rows"])
            receipts.append(receipt)
            values = audit.collect_inputs(rt, group, shallow, first=i == 0)
            values.update(audit.collect_inputs(rt, group, deep, first=i == 0))
            assert set(values) == set(audit.VIEWS)
            assert all(row[0].shape[0] == 4 for row in values.values())
            if i == 0:
                assert all(row[1]["used_path_passed"] for row in values.values())
    assert counts["student_encode"] == counts["decoder"] == 20
    assert counts["native_graph"] == 6
    assert all(counts[k] == 0 for k in audit.FORBIDDEN)
    assert reads["calls"] == 6 and reads["rows"] == 24 and reads["forbidden"] == 0
    assert state["unchanged"] and set(vars(rt.data)) == before_keys
    assert len({r["matched_cell_views_sha256"] for r in receipts}) == 1


def test_late_old_recipe_mismatch_rejected_before_any_native_forward():
    from representation_panel import MatchedCellViews, assemble_control_only
    from test_representation_panel_integration import native_runtime

    rt, panel = native_runtime("N0")
    matcher, expected = MatchedCellViews(), []
    for group in panel["schedule"]:
        deep = assemble_control_only(
            rt, group["rows"], group["queries"], group["view_seed"], deep=True
        )
        _, receipt = matcher.bind(deep, group["rows"])
        expected.append(dict(condition_id=group["condition_id"], **receipt))
    expected[-1]["matched_cell_views_sha256"] = "last-group-old-recipe-mismatch"
    with (
        pytest.raises(ValueError, match="matched_cell_views"),
        audit.protected_observation(rt) as state,
        audit.observed_paths(rt) as calls,
        audit.control_only_reads(rt, panel),
    ):
        audit.prepare_recipes(rt, panel, expected, lambda: None)
    assert calls["student_encode"] == calls["decoder"] == calls["native_graph"] == 0
    assert state["unchanged"]


def two_completed_results(complete_result):
    spec, panel, first = complete_result
    first = copy.deepcopy(first)
    second = copy.deepcopy(first)
    second["endpoint"] = spec["endpoints"][1]
    second["source"] = audit.read(Path(second["endpoint"]["run_root"]) / "run_manifest.json")[
        "source"
    ]
    second["checkpoint"]["sha256"] = second["endpoint"]["checkpoint_sha256"]
    second["checkpoint_journal_sha256"] = audit.sha(
        Path(second["endpoint"]["run_root"]) / "fit/epoch_state.json"
    )
    return spec, panel, [first, second]


def test_cross_endpoint_prebind_difference_is_accepted_but_full_receipts_retained(complete_result):
    spec, panel, results = two_completed_results(complete_result)
    first, second = results
    assert second["matched_views"] is not second["expected_views"]
    assert second["matched_views"][0] is not second["expected_views"][0]
    old = copy.deepcopy(second["expected_views"])
    second["matched_views"][0]["native_cell_views_sha256"] = "different-unused-prebind-realization"
    second["matched_views_sha256"] = audit.used_recipe_digest(second["matched_views"])
    assert second["expected_views"] == old
    assert audit.digest(second["matched_views"]) != audit.digest(first["matched_views"])
    assert second["matched_views_sha256"] == first["matched_views_sha256"]
    audit.validate_result(second, panel)
    assert audit.combine(spec, results)["scientific_complete"] is False
    assert (
        second["matched_views"][0]["native_cell_views_sha256"]
        == "different-unused-prebind-realization"
    )


def test_used_recipe_drift_rejected_even_with_recomputed_used_digest(complete_result):
    spec, _, results = two_completed_results(complete_result)
    old = copy.deepcopy(results[1]["expected_views"])
    results[1]["matched_views"][0]["matched_cell_views_sha256"] = "different-actually-used-mask"
    results[1]["matched_views_sha256"] = audit.used_recipe_digest(results[1]["matched_views"])
    assert results[1]["expected_views"] == old
    assert results[1]["matched_views_sha256"] != results[0]["matched_views_sha256"]
    with pytest.raises(ValueError, match="actual matched expression recipe"):
        audit.combine(spec, results)


def test_result_recomputes_used_projection_not_full_prebind_digest(complete_result):
    _, panel, result = complete_result
    result = copy.deepcopy(result)
    result["matched_views_sha256"] = audit.digest(result["matched_views"])
    with pytest.raises(ValueError, match="recipe receipts"):
        audit.validate_result(result, panel)


def root_output_fixture(spec, results, monkeypatch):
    output = Path(spec["output"])
    save(output / "INPUT.json", spec)
    save(output / "OWNER.json", {"synthetic_metadata_owner": True})
    for result in results:
        name = result["endpoint"]["name"]
        (output / f"{name}.log").write_text("completed synthetic worker metadata\n")
        save(output / name / "ADMISSION.json", {"synthetic_cpu_only": True})
        save(output / name / "COMPLETE.json", result)
    source_calls = []

    def source(binding):
        # Stand in for publication networking while retaining its file hash gate.
        audit.checked(dict(path=binding["publication"], sha256=binding["publication_sha256"]))
        source_calls.append(binding["root"])
        return next(r["source"] for r in results if r["endpoint"]["source"] == binding)

    monkeypatch.setattr(audit, "verify_source", source)
    return output, source_calls


def test_actual_root_closure_restreams_both_checkpoints_and_journals(complete_result, monkeypatch):
    spec, _, results = two_completed_results(complete_result)
    output, source_calls = root_output_fixture(spec, results, monkeypatch)
    result = audit.combine(spec, results)
    audit.write_root_complete(spec, results, result)
    complete = audit.read(output / "COMPLETE.json")
    proofs = complete["final_endpoint_verification"]
    assert set(proofs) == set(audit.NAMES)
    assert source_calls == [e["source"]["root"] for e in spec["endpoints"]]
    for endpoint, receipt in zip(spec["endpoints"], results, strict=True):
        proof = proofs[endpoint["name"]]
        assert proof["streamed_checkpoint_rechecked"] is True
        assert proof["checkpoint_sha256"] == endpoint["checkpoint_sha256"]
        assert proof["checkpoint_journal_sha256"] == receipt["checkpoint_journal_sha256"]
    assert complete["scientific_complete"] is False


@pytest.mark.parametrize(
    "drift", ["checkpoint_bytes", "journal_epoch", "journal_other_field", "publication"]
)
def test_first_worker_checkpoint_or_journal_drift_blocks_actual_root_COMPLETE(
    complete_result, monkeypatch, drift
):
    spec, _, results = two_completed_results(complete_result)
    output, _ = root_output_fixture(spec, results, monkeypatch)
    # Both worker receipts and the old combine/spec gates have already passed.
    result = audit.combine(spec, results)
    first = spec["endpoints"][0]
    if drift == "checkpoint_bytes":
        (Path(first["run_root"]) / "fit/epoch-0004.pt").write_bytes(b"changed after B0 worker")
    elif drift == "publication":
        Path(first["source"]["publication"]).write_text("changed publication")
    else:
        journal = Path(first["run_root"]) / "fit/epoch_state.json"
        data = audit.read(journal)
        if drift == "journal_epoch":
            data["epoch"] = 3  # last metadata itself still claims epoch4
        else:
            data["changed_after_first_worker"] = True
        save(journal, data)
    if drift != "publication":
        audit.validate_spec(spec, server=False)
        audit.combine(spec, results)  # proves the new closure is the rejecting gate
    with pytest.raises(ValueError):
        audit.write_root_complete(spec, results, result)
    assert not (output / "COMPLETE.json").exists()
    assert (output / "B0/COMPLETE.json").is_file()  # preserve prior worker evidence
