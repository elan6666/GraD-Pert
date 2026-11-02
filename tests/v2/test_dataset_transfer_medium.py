"""Synthetic metadata/reader/statistics checks, never a full native medium run."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.execution import dataset_transfer_medium as medium


def records(conditions=3, cells=5, *, missing=False, multi=False):
    rows = []
    for c in range(conditions):
        for i in range(cells):
            rows.append(
                dict(
                    truth_row_id=f"truth{c}-{i}",
                    control_row_id=f"ctrl{i}",
                    condition_id=f"g{c}+g9+ctrl" if multi else f"g{c}+ctrl",
                    context_id="contextA",
                    cell_line=None if missing else "K562",
                    cell_type=None if missing else "typeA",
                    batch="B",
                    control_cell_line=None if missing else "K562",
                    control_cell_type=None if missing else "typeA",
                    control_batch="B",
                )
            )
    return rows


@pytest.mark.parametrize("axis,excluded", [(5000, (1, 7, 12)), (5045, (3, 5, 4000))])
def test_dynamic_population_axis_and_small_support(axis, excluded):
    allowed = tuple(i for i in range(axis) if i not in excluded)
    r = medium.population(
        records(multi=True), expression_genes=axis, allowed=allowed, query_count=20
    )
    assert r["schema"] == medium.SCHEMA and r["status"] == "sealed_metadata_only"
    assert (r["conditions"], r["rows"], r["deep_conditions"], r["gradient_conditions"]) == (
        3,
        12,
        3,
        3,
    )
    assert r["canonical_train_rows"] == 15
    assert r["collection_status"].startswith("not_executed")
    assert r["crossed"]["controls_per_context"] == 4
    for g in r["schedule"]:
        assert "+g9+" in g["condition_id"]
        assert set(g["queries"]) <= set(allowed)
        assert {row["cell_line"] for row in g["rows"]} == {"K562"}
    assert r == medium.population(
        records(multi=True), expression_genes=axis, allowed=allowed, query_count=20
    )


def test_missing_metadata_and_singletons_are_na_without_fake_population():
    r = medium.population(
        records(conditions=1, cells=1, missing=True),
        expression_genes=5000,
        allowed=tuple(range(4998)),
        query_count=10,
    )
    assert r["rows"] == 1 and r["deep_rows"] == 1
    assert r["schedule"][0]["rows"][0]["cell_line"] is None
    assert r["crossed"]["status"] == "not_applicable" and r["crossed"]["rows"] == 0


def test_no_four_distinct_compatible_controls_is_na():
    r = medium.population(
        records(cells=2), expression_genes=5000, allowed=tuple(range(4998)), query_count=10
    )
    assert r["crossed"]["status"] == "not_applicable"
    for g in r["schedule"]:
        assert len(g["rows"]) == 2


@pytest.mark.parametrize("fault", ["duplicate", "context", "axis", "query"])
def test_population_invalid_boundaries(fault):
    rows = records()
    allowed = tuple(range(4998))
    query = 10
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "context":
        rows[0]["context_id"] = None
    elif fault == "axis":
        allowed = (0, False)
    else:
        query = 6000
    with pytest.raises(ValueError):
        medium.population(rows, expression_genes=5000, allowed=allowed, query_count=query)


class GuardedDense:
    def __init__(self):
        self.array = np.arange(40, dtype=np.float32).reshape(5, 8)
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.calls = []

    def __getitem__(self, key):
        mask = np.zeros(self.shape, bool)
        mask[key] = True
        forbidden = np.ones(self.shape, bool)
        forbidden[np.ix_([0, 1, 3], [0, 2, 5])] = False
        assert not (mask & forbidden).any(), "forbidden payload value read"
        self.calls.append(key)
        return self.array[key]


def data_fixture():
    x = GuardedDense()
    data = SimpleNamespace(
        _row_index={"t0": 0, "t1": 1, "c0": 3, "bad": 4},
        train_row_indices=(0, 1),
        control_row_indices=(3,),
        condition_ids=("g0+ctrl", "g1+ctrl", "ctrl", "ctrl", "g9+ctrl"),
        context_ids=("A", "A", "A", "A", "B"),
        pipeline_stats=SimpleNamespace(control_cache_requested=False, yielded_batches=0),
        manifest=SimpleNamespace(n_expression_genes=8),
        _adata=SimpleNamespace(file={"X": x}),
        _read_expression_indices=lambda _: pytest.fail("unrestricted reader called"),
    )
    data.expression_gene_ids = tuple(f"g{i}" for i in range(8))
    data.split = SimpleNamespace(
        test_conditions=tuple(f"g{i}+ctrl" for i in (1, 3, 4, 6, 7)), control_condition_id="ctrl"
    )
    data.configure_training_expression_columns = lambda allowed: setattr(
        data, "_training_expression_columns", allowed
    )
    rows = [
        dict(truth_row_id="t0", control_row_id="c0", condition_id="g0+ctrl", context_id="A"),
        dict(truth_row_id="t1", control_row_id="c0", condition_id="g1+ctrl", context_id="A"),
    ]
    return data, rows, x


def test_strict_reader_restores_duplicates_zeros_and_rejects_nonwhitelisted_rows():
    data, rows, x = data_fixture()
    with medium.strict_train_reads(data, rows, (0, 2, 5)) as receipt:
        result = data._read_expression_indices([3, 0, 3, 1])
        assert result.shape == (4, 8)
        np.testing.assert_array_equal(
            result[:, [0, 2, 5]], x.array[np.ix_([3, 0, 3, 1], [0, 2, 5])]
        )
        assert not result[:, [1, 3, 4, 6, 7]].any()
        assert receipt["calls"] == 1
        with pytest.raises(ValueError):
            data._read_expression_indices([4])
    assert not getattr(data, "_dataset_transfer_strict_guard", False)


def test_controls_only_blocks_truth_values_and_existing_reads_fail_closed():
    data, rows, _ = data_fixture()
    with medium.strict_train_reads(data, rows, (0, 2, 5), controls_only=True):
        data._read_expression_indices([3, 3])
        with pytest.raises(ValueError):
            data._read_expression_indices([0])
    with (
        pytest.raises(ValueError, match="precede"),
        medium.strict_train_reads(data, rows, (0, 2, 5)),
    ):
        pass


def test_assembly_without_guard_fails_closed():
    with pytest.raises(ValueError, match="strict"):
        medium.assemble_diagnostic(SimpleNamespace(data=SimpleNamespace()), {}, deep=True)


def test_metadata_native_sampler_no_x_reads_and_preserves_unknown_cell_line():
    data, _rows, _ = data_fixture()

    class Obs:
        columns = ("cell_type",)

        @property
        def iloc(self):
            return self

        def __getitem__(self, index):
            return {"cell_type": "T"}

    data._adata.obs = Obs()
    data.row_ids = ("t0", "t1", "unused", "c0", "bad")
    data.batch_ids = ("B",) * 5
    data.training_batch_identity_specs = lambda **kw: [
        SimpleNamespace(
            perturbed_row_ids=("t0", "t1"),
            control_row_ids=("c0", "c0"),
            condition_ids=("g0+ctrl", "g1+ctrl"),
        )
    ]
    output = medium.records_from_data(data)
    assert len(output) == 2 and all(r["cell_line"] is None for r in output)
    assert [r["truth_row_id"] for r in output] == ["t0", "t1"]


def test_reused_moments_and_grouped_definitions():
    with medium._scripts() as (cap, _, grouped):
        moments = cap.Moments(2)
        moments.add(np.array([[1.0, 0.0], [-1.0, 0.0]]))
        stats = moments.result()
        assert stats["effective_rank"] == pytest.approx(1)
        assert stats["rank_ceiling"] == 1
        assert stats["relative_variance"] == pytest.approx(1)
        stats2 = grouped.feature_statistics(np.array([[1.0, 0.0]]))
        assert stats2["effective_rank"] is None and "fewer_than_two" in stats2["status"]


def test_strict_reader_cannot_expand_frozen_allowed_axis():
    data, rows, x = data_fixture()
    with (
        pytest.raises(ValueError, match="exclusions"),
        medium.strict_train_reads(data, rows, (0, 1, 2, 5)),
    ):
        pass
    assert not x.calls


@pytest.mark.parametrize("active_ssl", [False, True])
def test_native_synthetic_single_batch_collector_restores_state_and_rng(active_ssl):
    import hashlib

    import torch
    from test_components import fixture

    from gradpert.training.v2.objective import JointObjective

    model, batch = fixture()
    objective = JointObjective(
        model,
        lambda1=float(active_ssl),
        lambda2=float(active_ssl),
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
    )
    rows = [
        dict(
            condition_id=f"p{int(c)}",
            control_row_id=f"ctrl{i}",
            truth_row_id=f"truth{i}",
            context_id="A",
            cell_line="K562",
            cell_type="T",
            batch="B",
            control_treatment="control",
            paired_perturbation=f"p{int(c)}",
        )
        for i, c in enumerate(batch.condition_index)
    ]

    def state():
        digest = hashlib.sha256()
        for name, value in objective.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
        digest.update(torch.get_rng_state().numpy().tobytes())
        return digest.hexdigest()

    before = state()
    result = medium.collect_batch_statistics(
        objective, batch, rows, seed=13, state_probe=state, bf16=False
    )
    assert state() == before
    assert result["observer_loss_term_parity"]
    assert result["status"] == "batch_statistics_only" and not result["full_run_complete"]
    assert result["gradient_statistics"] is None


@pytest.mark.parametrize("steps", [263, 393])
def test_metadata_seal_delegates_exact300_partial_terminal_without_complete(
    tmp_path, monkeypatch, steps
):
    from test_dataset_transfer_validation import terminal as terminal_fixture

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json, read_json
    from gradpert.execution.dataset_transfer_validation import binding
    from gradpert.execution.v2_training_stage import training_stage_evidence
    from gradpert.hashing import sha256_json
    from gradpert.training.expression_policy import expression_policy

    # Reuse the genuine terminal fixture builder; no source/runtime/CUDA is invoked.
    factory = terminal_fixture.__wrapped__(tmp_path, monkeypatch)
    args, plan, identity, _, _ = factory(steps)
    genes = tuple(f"g{i}" for i in range(5000))
    _, policy = expression_policy(genes, ("g7+ctrl",), "ctrl", enabled=True)
    identity["data"].update(
        canonical_sha256="canonical",
        split_sha256="split",
        expression_gene_order_sha256=sha256_json(list(genes)),
        graph_gene_order_sha256=sha256_json(list(genes)),
        training_expression_policy=policy,
        effective_training_expression_ids_sha256=policy["allowed_expression_gene_ids_sha256"],
    )
    run = __import__("pathlib").Path(plan["run_root"])
    atomic_json(run / "run_manifest.json", identity)
    journal = read_json(run / "fit/epoch_state.json")
    journal["identity"] = identity
    atomic_json(run / "fit/epoch_state.json", journal)
    atomic_json(
        run / "TRAIN_COMPLETE.json", training_stage_evidence(run, identity, journal["epoch"])
    )
    config = load_experiment_config(plan["config"])
    data = SimpleNamespace(
        manifest=SimpleNamespace(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            canonical_adata_sha256="canonical",
        ),
        split=SimpleNamespace(
            split_content_sha256="split", test_conditions=("g7+ctrl",), control_condition_id="ctrl"
        ),
        expression_gene_ids=genes,
        graph_gene_ids=genes,
    )
    monkeypatch.setattr(medium, "records_from_data", lambda _: records())
    output = tmp_path / f"medium-{steps}"
    result = medium.seal_metadata(
        training_plan=binding(args["training_plan"]),
        data=data,
        output=output,
        query_count=int(config.model.parameters["query_count"].value),
    )
    assert result["last"]["epoch"] == (300 + steps - 1) // steps
    assert (output / "POPULATION_SEALED.json").is_file()
    assert not (output / "COMPLETE.json").exists()
    assert result["collection_status"] == "not_executed_no_statistics_or_parity_proof"
    with pytest.raises(FileExistsError):
        medium.seal_metadata(
            training_plan=binding(args["training_plan"]),
            data=data,
            output=output,
            query_count=int(config.model.parameters["query_count"].value),
        )


def native_runtime_fixture():
    import torch
    from test_relay_method import small_index
    from test_unified_integration import fixture

    from gradpert.training.v2.optimizer import V2Optimizer

    obj, _, raw, options = fixture(checkpointed=False)
    # Every fixture expression read is guarded; these are synthetic arrays only.
    matrix = np.concatenate([raw.target_expression.numpy(), raw.control_expression.numpy()])

    class Dense:
        shape = matrix.shape
        dtype = matrix.dtype

        def __getitem__(self, key):
            mask = np.zeros(matrix.shape, bool)
            mask[key] = True
            forbidden = np.ones(matrix.shape, bool)
            forbidden[np.ix_(range(8), range(12))] = False
            assert not (mask & forbidden).any()
            return matrix[key]

    ids = tuple(raw.perturbed_row_ids) + tuple(raw.control_row_ids)
    data = SimpleNamespace(
        _row_index={r: i for i, r in enumerate(ids)},
        train_row_indices=tuple(range(4)),
        control_row_indices=tuple(range(4, 8)),
        condition_ids=raw.condition_ids + ("ctrl",) * 4,
        anchors_by_condition={"g0": (0,), "g1": (1,)},
        context_ids=("A",) * 8,
        expression_gene_ids=tuple(f"g{i}" for i in range(16)),
        graph_gene_ids=tuple(f"g{i}" for i in range(16)),
        split=SimpleNamespace(
            test_conditions=tuple(f"g{i}+ctrl" for i in range(12, 16)), control_condition_id="ctrl"
        ),
        manifest=SimpleNamespace(n_expression_genes=16),
        _adata=SimpleNamespace(file={"X": Dense()}),
        pipeline_stats=SimpleNamespace(control_cache_requested=False, yielded_batches=0),
        _read_expression_indices=lambda _: pytest.fail("unrestricted read"),
    )
    data.configure_training_expression_columns = lambda allowed: setattr(
        data, "_training_expression_columns", allowed
    )
    runtime = SimpleNamespace(
        objective=obj,
        optimizer=V2Optimizer(obj.student, lr=0.001, weight_decay=0),
        data=data,
        index=small_index(16),
        generator=np.random.default_rng(91),
        device=torch.device("cpu"),
        options=options,
        allowed_expression_ids=np.arange(12),
    )
    schedule = []
    for c in (0, 1):
        rows = [
            dict(
                truth_row_id=f"t{i}",
                control_row_id=f"c{i}",
                condition_id=f"g{c}",
                context_id="A",
                cell_type="T",
                cell_line="K562",
                batch="B",
                control_batch="B",
                control_cell_type="T",
                control_cell_line="K562",
                control_treatment="control",
                paired_perturbation=f"g{c}",
            )
            for i in range(4)
            if i % 2 == c
        ]
        schedule.append(
            dict(
                index=c,
                condition_id=f"g{c}",
                rows=rows,
                deep_rows=rows,
                gradient_rows=rows,
                queries=list(range(6)),
                view_seed=15 + c,
            )
        )
    manifest = dict(
        seed=15,
        allowed_expression_ids=list(range(12)),
        schedule=schedule,
        conditions=2,
        rows=4,
        deep_conditions=2,
        deep_rows=4,
        gradient_conditions=2,
        gradient_rows=4,
        diagnostic_population_sha256="synthetic-population",
        crossed=dict(
            status="not_applicable", reason="fewerthan4distinctcompatiblecontrols", schedule=[]
        ),
    )
    return runtime, manifest


def test_full_medium_native_cpu_schedule_array_parity_and_gradients():
    runtime, manifest = native_runtime_fixture()
    guards = []
    result = medium._collect_runtime(
        runtime, manifest, phase="medium", resource_check=lambda: guards.append(True), bf16=False
    )
    assert result["status"] == "completed" and result["counts"]["rows"] == 4
    assert result["counts"]["gradient_conditions"] == 2 and len(result["gradients"]) == 2
    assert result["parity"]["used_path_passed"] and result["parity"]["max_abs"] == 0
    assert "response_tokens" in result["parity"]["shapes"]
    assert result["state"]["restored"] and result["state"]["before"] == result["state"]["after"]
    assert result["features"]["student.main.response_cls"]["rows"] == 4
    assert result["grouped_views"]["pairs"] and guards
    assert result["updates"] == 0


def test_failure_transaction_restores_parameters_gradients_rng_and_steps():
    import torch

    runtime, _ = native_runtime_fixture()
    parameter = next(runtime.objective.student.parameters())
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="injected"), medium.state_transaction(runtime) as proof:
        with torch.no_grad():
            parameter.add_(3)
        parameter.grad = torch.ones_like(parameter)
        runtime.optimizer.steps = 12
        torch.rand(5)
        runtime.generator.random()
        raise RuntimeError("injected")
    assert proof["restored"] and proof["before"] == proof["after"]
    assert torch.equal(parameter, before) and parameter.grad is None
    assert runtime.optimizer.steps == 0


def test_matched_crossed_native_full_schedule_and_prototype_decomposition():
    runtime, manifest = native_runtime_fixture()
    # Same four compatible controls for both distinct condition interventions.
    groups = []
    for c in (0, 1):
        representative = manifest["schedule"][c]["rows"][0]
        rows = [{**representative, "control_row_id": f"c{i}"} for i in range(4)]
        groups.append(
            dict(index=c, condition_id=f"g{c}", rows=rows, queries=list(range(6)), view_seed=21)
        )
    manifest["crossed"] = dict(
        status="sealed_metadata_only", schedule=groups, rows=8, controls_per_context=4
    )
    result = medium._collect_runtime(
        runtime, manifest, phase="crossed", resource_check=lambda: None, bf16=False
    )
    assert result["counts"]["rows"] == 8 and result["counts"]["deep_conditions"] == 2
    assert len(result["matched_views"]) == 2
    assert (
        result["matched_views"][0]["matched_cell_views_sha256"]
        == result["matched_views"][1]["matched_cell_views_sha256"]
    )
    assert result["state"]["restored"] and result["parity"]["used_path_passed"]
    pairs = result["prototype_pairs"]["pairs"]
    assert pairs
    for row in pairs:
        if row.get("ce_condition_balanced_mean") is not None:
            assert row["ce_condition_balanced_mean"] == pytest.approx(
                row["teacher_entropy_condition_balanced_mean"] + row["kl_condition_balanced_mean"],
                abs=1e-12,
            )
    assert {row["view_family"] for row in result["grouped_views"]["pairs"]} >= {"GG", "GL", "LL"}
    assert result["gradients"] == [] and not any(
        "prediction_mse" in r for r in result["condition_outputs"]
    )
    assert all(
        "encode_response" not in model.__dict__
        for model in (runtime.objective.student, runtime.objective.teacher)
    )


def test_crossed_na_does_not_assemble_or_claim_parity():
    runtime, manifest = native_runtime_fixture()
    result = medium._collect_runtime(
        runtime,
        manifest,
        phase="crossed",
        resource_check=lambda: pytest.fail("not applicable"),
        bf16=False,
    )
    assert result["status"] == "not_applicable" and "parity" not in result


def test_array_parity_observer_corruption_fails_and_restores(monkeypatch):
    import torch

    runtime, manifest = native_runtime_fixture()
    rows = manifest["schedule"][0]["rows"]
    with medium.strict_train_reads(runtime.data, rows, tuple(manifest["allowed_expression_ids"])):
        group = manifest["schedule"][0]
        batch = medium.assemble_diagnostic(runtime, group, deep=False)
        with medium.state_transaction(runtime):
            runtime.objective.eval()
            with medium._scripts() as (cap, _, _):
                with torch.no_grad():
                    cached = cap.static_graph(runtime)
                original = medium._install_collector

                def changed(collector, obj):
                    original(collector, obj)
                    base = obj.student.encode_response

                    def corrupt(*args, **kw):
                        result = base(*args, **kw)
                        result["prediction"] = result["prediction"] + 1
                        return result

                    obj.student.encode_response = corrupt

                monkeypatch.setattr(medium, "_install_collector", changed)
                with pytest.raises(ValueError, match="array observer parity"):
                    medium._used_array_parity(runtime, batch, rows, cached, 13, False)
    assert all(
        "encode_response" not in model.__dict__
        for model in (runtime.objective.student, runtime.objective.teacher)
    )


def test_batch_contexts_are_isolated_in_existing_statistics_banks():
    original = [
        dict(
            condition_id="g0",
            control_row_id="same",
            context_id="A",
            cell_line="K562",
            cell_type="T",
            batch=batch,
        )
        for batch in ("B1", "B2")
    ]
    transformed = medium._statistical_records(original)
    assert transformed[0]["context_id"] != transformed[1]["context_id"]
    assert all(row["source_context_id"] == "A" for row in transformed)
    with medium._scripts() as (_, _, grouped):
        bank = grouped.GroupedViewBank()
        bank.add(
            "response_cls", "student", "global0", transformed, np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        result = bank.result()
        assert len({row["context_id"] for row in result["features"]}) == 2
    assert original[0]["context_id"] == "A"


def test_incomplete_or_changed_receipt_never_passes_final_closure(tmp_path, monkeypatch):
    from gradpert.data._io import atomic_json
    from gradpert.execution import v2_checkpoint_eval
    from gradpert.execution.dataset_transfer_validation import binding
    from gradpert.hashing import sha256_json

    runtime, frozen = native_runtime_fixture()
    actual = medium._collect_runtime(
        runtime, frozen, phase="medium", resource_check=lambda: None, bf16=False
    )
    monkeypatch.setattr(v2_checkpoint_eval, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(medium, "validate_execution_spec", lambda _: None)
    manifest = tmp_path / "population.json"
    atomic_json(manifest, frozen)
    run = tmp_path / "originalrun"
    run.mkdir()
    train = tmp_path / "plan.json"
    atomic_json(train, dict(run_root=str(run)))
    output = tmp_path / "diagnostics"
    output.mkdir()
    spec = dict(
        population=binding(manifest),
        training_plan=binding(train),
        output=str(output),
        family_root=str(tmp_path),
        checkpoint_sha256="f" * 64,
        config_sha256="e" * 64,
        dataset_id="synthetic",
        training_run_id="synthetic300",
        training_source={"commit": "a" * 40},
        analysis_source={"commit": "b" * 40},
    )
    receipt = dict(
        status="completed",
        updates=0,
        spec_sha256=sha256_json(spec),
        checkpoint_sha256=spec["checkpoint_sha256"],
        config_sha256=spec["config_sha256"],
        dataset_id=spec["dataset_id"],
        training_run_id=spec["training_run_id"],
        training_source=spec["training_source"],
        analysis_source=spec["analysis_source"],
        medium=actual,
        crossed=dict(status="not_applicable", reason="less than four controls"),
    )
    medium.validate_completion(spec, receipt)
    for change in ("count", "parity", "gradient", "source", "pkl"):
        import copy

        broken = copy.deepcopy(receipt)
        if change == "count":
            broken["medium"]["counts"]["rows"] -= 1
        elif change == "parity":
            broken["medium"]["parity"]["used_path_passed"] = False
        elif change == "gradient":
            broken["medium"]["gradients"] = []
        elif change == "source":
            broken["checkpoint_sha256"] = "0" * 64
        else:
            sibling = tmp_path / "sibling-job"
            sibling.mkdir()
            (sibling / "forbidden.pkl").write_bytes(b"synthetic")
        with pytest.raises(ValueError):
            medium.validate_completion(spec, broken)
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("child_success", [True, False])
def test_owned_controller_writes_complete_only_after_verified_child(
    tmp_path, monkeypatch, child_success
):
    import os
    from contextlib import contextmanager

    from gradpert.data._io import atomic_json
    from gradpert.execution import v2_checkpoint_eval
    from gradpert.execution.dataset_transfer_validation import binding

    monkeypatch.setattr(v2_checkpoint_eval, "SERVER_ROOT", tmp_path)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = dict(output=str(tmp_path / "output"), gpu="0", policy={"unchanged": "original"})
    specfile = tmp_path / "spec.json"
    atomic_json(specfile, spec)
    monkeypatch.setattr(medium, "validate_execution_spec", lambda _: None)
    checks = []

    def validate(spec, receipt):
        assert receipt == {"status": "completed", "finite_schedule_verified": True}
        checks.append("terminal")

    monkeypatch.setattr(medium, "validate_completion", validate)
    with medium._scripts():
        import importlib

        shared = importlib.import_module("shared_gpu")
        runner = importlib.import_module("run_dataset_transfer_300")
        leases = importlib.import_module("run_group")

        @contextmanager
        def lock(path):
            assert str(path).endswith("v2-gpu-0.lock")
            checks.append("lock")
            yield 1
            checks.append("unlock")

        monkeypatch.setattr(leases, "lock", lock)
        monkeypatch.setattr(shared, "snapshot", lambda pid: {"owner": pid})
        monkeypatch.setattr(shared, "admit", lambda policy, observed: checks.append("admit"))
        monkeypatch.setattr(
            shared,
            "process_info",
            lambda pid: dict(pid=pid, uid=os.getuid(), start_ticks="1", command_sha256="cmd"),
        )

        def child(command, log, policy):
            assert "--worker-binding" in command
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
            checks.append("child")
            if not child_success:
                raise RuntimeError("synthetic child failure")
            atomic_json(
                tmp_path / "output/WORKER_COMPLETE.json",
                dict(status="completed", finite_schedule_verified=True),
            )

        monkeypatch.setattr(runner, "native_run", child)
        if child_success:
            receipt = medium.execute_medium(spec_binding=binding(specfile))
            assert receipt["status"] == "completed"
            assert (tmp_path / "output/COMPLETE.json").is_file()
        else:
            with pytest.raises(RuntimeError, match="synthetic child failure"):
                medium.execute_medium(spec_binding=binding(specfile))
            assert not (tmp_path / "output/COMPLETE.json").exists()
            assert (tmp_path / "output/FAILURE.json").is_file()
        assert checks[:3] == ["lock", "admit", "child"]


def test_execution_spec_binds_sources_runtime_resource_and_protected_outputs(tmp_path, monkeypatch):
    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.execution import dataset_transfer_validation as validation
    from gradpert.execution import identity, v2_checkpoint_eval
    from gradpert.hashing import sha256_file

    source = tmp_path / "training-source"
    source.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    run = tmp_path / "family" / "runs" / "run"
    run.mkdir(parents=True)
    terminal = dict(
        identity=dict(source=dict(repository_root=str(source), commit="a" * 40)),
        last=dict(sha256="c" * 64),
    )
    monkeypatch.setattr(v2_checkpoint_eval, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(medium, "training_terminal", lambda _: terminal)
    monkeypatch.setattr(
        identity,
        "inspect_source_identity",
        lambda *a, **kw: SimpleNamespace(payload=lambda: dict(commit="b" * 40)),
    )
    compatibility_calls = []
    monkeypatch.setattr(
        validation, "_compatibility", lambda *args: compatibility_calls.append(args)
    )
    with medium._scripts():
        import shared_gpu

        policy = dict(
            schema=shared_gpu.SCHEMA,
            gpu_index=1,
            gpu_uuid="GPU-1",
            foreign_uid=1003,
            command_basename="run_scbutterfly_archived_predictions.py",
            max_foreign_memory_mib=1024,
            min_free_memory_mib=2048,
            max_own_memory_fraction=0.85,
        )
    original = tmp_path / "original-runtime.json"
    atomic_json(original, dict(shared_gpu_policy=policy, runs_root=str(run.parent)))
    runtime = tmp_path / "eval-runtime.json"
    atomic_json(
        runtime,
        dict(data_root=str(data), publication_receipt="receipt", publication_sha256="d" * 64),
    )
    config = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/dataset_transfer_300_20261102/gradpert_v2/replogle_k562_essential.yaml"
    )
    load_experiment_config(config)
    plan = tmp_path / "PLAN.json"
    atomic_json(
        plan,
        dict(
            config=str(config),
            config_sha256=sha256_file(config),
            data_root=str(data),
            run_root=str(run),
            run_id="synthetic300",
            runtime=str(original),
            runtime_sha256=sha256_file(original),
        ),
    )
    frozen = tmp_path / "population.json"
    atomic_json(
        frozen,
        dict(
            schema=medium.SCHEMA,
            training_plan=medium.binding(plan),
            training_identity=terminal["identity"],
            last=terminal["last"],
            collection_status="not_executed_no_statistics_or_parity_proof",
            terminal_inputs=[],
        ),
    )
    kwargs = dict(
        training_plan=medium.binding(plan),
        population_binding=medium.binding(frozen),
        runtime=medium.binding(runtime),
        output=tmp_path / "family" / "new-output",
        family_root=tmp_path / "family",
        gpu="1",
        compatibility={"proof": "synthetic"},
    )
    spec = medium.build_execution_spec(**kwargs)
    assert spec["policy"] == policy and compatibility_calls
    assert spec["training_source"] == terminal["identity"]["source"]
    assert not (tmp_path / "new-output").exists()
    medium.validate_execution_spec(spec)
    for root in (source, data, run):
        with pytest.raises(ValueError):
            medium.build_execution_spec(**{**kwargs, "output": root / "forbidden"})
    with pytest.raises(ValueError, match="resource policy"):
        medium.build_execution_spec(**{**kwargs, "gpu": "0"})
    dynamic = dict(
        schema=shared_gpu.DYNAMIC_SCHEMA,
        foreign_uid=1003,
        command_basename="run_scbutterfly_archived_predictions.py",
        max_foreign_processes_per_gpu=1,
        max_foreign_memory_mib=1024,
        min_free_memory_mib=2048,
        max_own_memory_fraction=0.90,
        authorization_reference=shared_gpu.TOTALVI_MEMORY90_AUTHORIZATION,
        additional_command_basenames=[shared_gpu.TOTALVI_PROGRAM],
        devices=[dict(gpu_index=i, gpu_uuid=f"GPU-{i}") for i in (0, 1)],
    )
    atomic_json(original, dict(shared_gpu_policy=dynamic, runs_root=str(run.parent)))
    train = medium.read_json(plan)
    train["runtime_sha256"] = sha256_file(original)
    atomic_json(plan, train)
    sealed = medium.read_json(frozen)
    sealed["training_plan"] = medium.binding(plan)
    atomic_json(frozen, sealed)
    kwargs.update(training_plan=medium.binding(plan), population_binding=medium.binding(frozen))
    for gpu in ("0", "1"):
        actual = medium.build_execution_spec(**{**kwargs, "gpu": gpu})
        assert actual["policy"] == dynamic
        assert actual["selected_device"] == dynamic["devices"][int(gpu)]
        medium.validate_execution_spec(actual)
        actual["selected_device"]["gpu_uuid"] = "GPU-altered"
        with pytest.raises(ValueError, match="changed"):
            medium.validate_execution_spec(actual)
    with pytest.raises(ValueError):
        medium.build_execution_spec(**{**kwargs, "gpu": "2"})
    with pytest.raises(ValueError, match="sealed runs_root"):
        medium.build_execution_spec(**{**kwargs, "family_root": tmp_path})
    original.write_text("{}")
    with pytest.raises(ValueError, match="runtime changed"):
        medium.build_execution_spec(**kwargs)


@pytest.mark.parametrize("inactive", [False, True])
def test_full_schedule_preserves_native_same_support_loss_pairs(inactive):
    runtime, manifest = native_runtime_fixture()
    # Three active heads, including masked response-token SSL2. No new weighting.
    runtime.objective.weights = ((1, 0, 0), (1, 0 if inactive else 1, 0))
    result = medium._collect_runtime(
        runtime, manifest, phase="medium", resource_check=lambda: None, bf16=False
    )
    pairs = result["loss_pairs"]
    assert pairs
    heads = {
        head
        for head in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
        if any(head in row["student"] for row in pairs)
    }
    assert heads == (
        {"ssl1_cls", "ssl2_cls"} if inactive else {"ssl1_cls", "ssl2_cls", "ssl2_node"}
    )
    assert {row["condition_id"] for row in pairs} == {
        group["condition_id"] for group in manifest["schedule"]
    }
    for row in pairs:
        assert row["cohort"] == "deep" and row["cells"] == 2
        assert row["pairs"] > 0 and row["effective_loss_rows"] > 0
        assert row["CE"] == pytest.approx(row["H_pair"] + row["KL"], abs=2e-6)
        assert row["raw_token_CE"] == pytest.approx(
            row["raw_token_H"] + row["raw_token_KL"], abs=2e-6
        )
        if "ssl2_node" in row["student"]:
            assert row["aggregation"] == "masked_token_mean_then_valid_cell_mean"


@pytest.mark.parametrize("path", ["parity", "deep"])
def test_partial_collector_install_failure_restores_hooks_wrappers_and_ce(monkeypatch, path):
    from gradpert.training.v2 import objective as objective_module

    runtime, manifest = native_runtime_fixture()
    before_ce = objective_module.cross_entropy_terms
    models = (runtime.objective.student, runtime.objective.teacher)
    overrides = [
        (m, "encode_response" in m.__dict__, m.__dict__.get("encode_response")) for m in models
    ]
    hooks = [
        (d, d.copy())
        for m in runtime.objective.modules()
        for name, d in vars(m).items()
        if "hook" in name and isinstance(d, dict)
    ]
    with medium._scripts() as (cap, _, _):
        install = cap.Collector.install

        def broken(self, obj):
            install(self, obj)
            raise RuntimeError("partial installed observer")

        monkeypatch.setattr(cap.Collector, "install", broken)
        if path == "deep":
            monkeypatch.setattr(medium, "_used_array_parity", lambda *a, **k: {})
        with pytest.raises(RuntimeError, match="partial installed"):
            medium._collect_runtime(
                runtime, manifest, phase="medium", resource_check=lambda: None, bf16=False
            )
    assert objective_module.cross_entropy_terms is before_ce
    for model, existed, value in overrides:
        assert ("encode_response" in model.__dict__) == existed
        assert model.__dict__.get("encode_response") is value
    assert all(current == saved for current, saved in hooks)
