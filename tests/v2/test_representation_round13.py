"""R13 sealed coefficients, provenance, physical populations and preflight stages."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round13 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as postfit  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / family.FAMILY
    family.generate(source, output)
    return source, output


def test_exact_two_arms_and_original_ri_configuration(family_root):
    from gradpert.config.v2 import V2Options

    source, output = family_root
    manifest = runner.verify_manifest(source, output / "manifest.json")
    assert manifest["priority"] == list(family.ARMS)
    assert manifest["repeat_baseline"] is False and manifest["test_roles"] == ["last"]
    assert manifest["test_timing"] == "after_research_design_lock"
    assert manifest["completed_optimizer_updates"] == 300
    assert manifest["global_batch"] == 320 and manifest["epochs"] == 2
    assert manifest["eval_batch"] == manifest["eval_capacity_reference_batch"] == 128
    assert family.SCHEMA in runner.RESEARCH_FAMILIES
    assert medium.FAMILY_ARMS[family.SCHEMA] == postfit.FAMILY_ARMS[family.SCHEMA] == family.ARMS
    baseline = first.configs(source)["RS-RI"]
    for arm in family.ARMS:
        candidate = family.configs(source)[arm]
        config = source / next(row["config"] for row in manifest["rows"] if row["name"] == arm)
        _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
        assert (
            runner.expected_epochs(
                dict(
                    repository_root=str(source),
                    config=str(config),
                    resolved_config=load_experiment_config(config).model_dump(mode="json"),
                )
            )
            == 2
        )
        assert options.spectral_amplitude_weight == family.AMPLITUDE_WEIGHT
        assert options.spectral_amplitude_target == 0.0225
        assert options.spectral_cosine_weight == (0.1 if arm == "R13-SRAMP" else 0)
        for key in family.CHANGES[arm]:
            candidate["model"]["parameters"].pop(key)
        assert candidate == baseline
    left, right = (family.configs(source)[arm] for arm in family.ARMS)
    left["model"]["parameters"].pop("spectral_cosine_weight")
    right["model"]["parameters"].pop("spectral_cosine_weight")
    assert left == right


@pytest.mark.parametrize(
    "field,value",
    [
        ("spectral_amplitude_weight", 0.01244),
        ("spectral_amplitude_target", 0.04),
        ("spectral_cosine_weight", 0.2),
        ("spectral_rank_weight", 0.04),
        ("spectral_rank_target", 0.35),
        ("spectral_warmup_steps", 31),
        ("teacher_end", 0.99),
        ("lambda1", 0.05),
    ],
)
def test_rehashed_config_protocol_drift_is_rejected(family_root, field, value):
    source, output = family_root
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    row = manifest["rows"][0]
    config = source / row["config"]
    raw = yaml.safe_load(config.read_text())
    raw["model"]["parameters"][field]["value"] = value
    config.write_text(yaml.safe_dump(raw))
    row["sha256"] = sha256_file(config)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        family.verify_manifest(source, path)


def test_actual_calibration_and_observer_bytes_preserved():
    proof = family.verify_source_compatibility(ROOT, formal_source_commit="a" * 40)
    assert proof["calibration_source_commit"] == family.CALIBRATION_SOURCE
    assert proof["formal_source_commit"] == "a" * 40
    assert proof["byte_identical"]
    assert len(proof["compatible_files_sha256"]) == 13
    assert set(proof["capacity_function_source_sha256"]) == {
        "state_fingerprint",
        "validation_state",
    }


def test_byte_or_observer_drift_invalidates_compatibility(tmp_path):
    import shutil

    for path in family.COMPATIBLE_FILES:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    capacity = tmp_path / "scripts/v2/capacity_probe.py"
    shutil.copyfile(ROOT / "scripts/v2/capacity_probe.py", capacity)
    family.verify_source_compatibility(tmp_path, formal_source_commit="a" * 40)
    capacity.write_text(
        capacity.read_text().replace(
            '"Hash nested optimizer/RNG state exactly',
            '"Changed observer: Hash nested optimizer/RNG state exactly',
        )
    )
    with pytest.raises(ValueError, match="bytes differ"):
        family.verify_source_compatibility(tmp_path, formal_source_commit="a" * 40)


def amplitude_gate(arm, *, active):
    updates = []
    q = 0.0001 if active else 0.1
    raw_amp = max(0.0, math.sqrt(0.0225) - math.sqrt(q + 1e-8))
    for i in range(10):
        warmup = i / 30
        record = dict(
            completed_before=i,
            spectral_amplitude=raw_amp,
            spectral_amplitude_target=0.0225,
            spectral_rank=0.2,
            spectral_cosine=0.1,
            spectral_warmup=warmup,
            spectral_input_count=320,
            spectral_amplitude_weighted=warmup * family.AMPLITUDE_WEIGHT * raw_amp,
            spectral_rank_weighted=warmup * 0.02 * 0.2,
            spectral_cosine_weighted=warmup * family.CHANGES[arm]["spectral_cosine_weight"] * 0.1,
        )
        record["spectral_weighted"] = sum(
            record[k]
            for k in (
                "spectral_amplitude_weighted",
                "spectral_rank_weighted",
                "spectral_cosine_weighted",
            )
        )
        for view in (0, 1):
            record.update(
                {
                    f"spectral_q_g{view}": q,
                    f"spectral_trace_g{view}": 256 * q,
                    f"spectral_total_energy_g{view}": 256.0,
                    f"spectral_mean_energy_g{view}": 256 * (1 - q),
                }
            )
        updates.append(record)
    return dict(
        passed=True,
        checks=dict.fromkeys(
            (
                "ten_native_updates",
                "exact_restore",
                "complete_population",
                "correct_warmup",
                "first_update_zero",
                "finite_losses",
                "amplitude_energy_and_contributions",
            ),
            True,
        ),
        updates=updates,
        resume_checkpoint_sha256="c" * 64,
    )


@pytest.mark.parametrize("arm", family.ARMS)
@pytest.mark.parametrize("active", [False, True])
def test_actual_ten_step_gate_allows_active_and_measured_inactive_floor(arm, active):
    family.validate_amplitude_gate(
        amplitude_gate(arm, active=active), arm=arm, resume_checkpoint_sha256="c" * 64
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("spectral_amplitude", None),
        ("spectral_amplitude_weighted", 0.99),
        ("spectral_q_g0", 0.2),
        ("spectral_total_energy_g1", 128.0),
        ("spectral_input_count", 160),
        ("spectral_warmup", 0.8),
        ("spectral_mean_energy_g0", -1),
    ],
)
def test_amplitude_gate_rejects_partial_or_reweighted_or_wrong_population(field, value):
    gate = amplitude_gate(family.ARMS[0], active=True)
    gate["updates"][1][field] = value
    with pytest.raises(ValueError):
        family.validate_amplitude_gate(gate, arm=family.ARMS[0], resume_checkpoint_sha256="c" * 64)


@pytest.mark.parametrize("arm", family.ARMS)
def test_physical_parent_and_fixed_panel_binding_exact(tmp_path, arm):
    from run_representation_panel import panel_parent_binding
    from test_cr_medium_adapter import cr_case

    parent, identity, queue, args, kwargs = cr_case.__wrapped__(tmp_path, family)
    args.config.write_text(yaml.safe_dump(family.configs(ROOT)[arm]))
    identity["data"] = copy.deepcopy(parent["data"])
    identity["data"]["loss_protocol"]["spectral_pair"] = family.spectral_protocol(arm)
    queue["rows"][0]["name"] = kwargs["arm"] = arm
    queue["rows"][0]["plan"]["config_sha256"] = sha256_file(args.config)
    bound = medium.relative_amplitude_parent_binding(parent, identity, **kwargs)
    assert bound["schedule"] == parent["schedule"]
    assert panel_parent_binding(parent, identity, queue, args) == bound["training_protocol_binding"]
    identity["data"]["canonical_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs beyond"):
        medium.relative_amplitude_parent_binding(parent, identity, **kwargs)


@pytest.mark.parametrize("arm", family.ARMS)
@pytest.mark.parametrize("deep", [False, True])
def test_sampling_keeps_original_options_and_turns_off_all_aux_only_when_shallow(arm, deep):
    from dataclasses import replace

    from diagnostic_view_options import diagnostic_view_options

    from gradpert.config.v2 import V2Options

    config = ROOT / "configs/v2" / family.FAMILY / arm / "gradpert_v2/nadig_jurkat.yaml"
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    original = copy.deepcopy(options)
    actual = diagnostic_view_options(options, deep=deep)
    expected = replace(
        options,
        lambda1=float(deep),
        lambda2=float(deep),
        spectral_rank_weight=options.spectral_rank_weight if deep else 0.0,
        spectral_cosine_weight=options.spectral_cosine_weight if deep else 0.0,
        spectral_amplitude_weight=options.spectral_amplitude_weight if deep else 0.0,
    )
    assert actual == expected and options == original


def test_saved_contributions_do_not_reconstruct_epoch_warmup_or_fill_partial_na():
    config = ROOT / "configs/v2" / family.FAMILY / family.ARMS[0] / "gradpert_v2/nadig_jurkat.yaml"
    history = [
        dict(
            epoch=1,
            training=dict(
                spectral_amplitude=0.1,
                spectral_amplitude_weighted=0.0007,
                spectral_rank=0.2,
                spectral_cosine=0.1,
                spectral_rank_weighted=0.001,
                spectral_cosine_weighted=0.002,
                spectral_weighted=0.0037,
                spectral_warmup=0.5,
                spectral_q_g0=0.01,
                spectral_trace_g0=2.0,
                spectral_total_energy_g0=200.0,
                spectral_mean_energy_g0=198.0,
            ),
        )
    ]
    row = postfit.ssl_loss_contributions(history, config)[0]
    assert row["spectral_amplitude_weighted"] == 0.0007
    assert row["spectral_amplitude_weighted"] != 0.5 * family.AMPLITUDE_WEIGHT * 0.1
    assert row["spectral_q_g0"] == 0.01 and row["spectral_q_g1"] is None


from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


@pytest.fixture
def generated_r13_queue(family_root, new_queue, monkeypatch):
    from gradpert.data._io import atomic_json, read_json

    source, output = family_root
    old, old_output = new_queue
    runtime = Path(old["runtime"])
    atomic_json(runtime, {**read_json(runtime), "repository_root": str(source)})

    def resolve(args):
        plan = copy.deepcopy(old["rows"][0]["plan"])
        arm = args.config.parents[1].name
        plan.update(
            config=str(args.config),
            config_sha256=sha256_file(args.config),
            resolved_config=load_experiment_config(args.config).model_dump(mode="json"),
            runtime_sha256=sha256_file(runtime),
            repository_root=str(source),
            run_id="fresh-" + arm,
            run_root=str(Path(plan["run_root"]).parent / ("fresh-" + arm)),
            test_root=str(Path(plan["test_root"]).parent / ("fresh-" + arm + "-test")),
        )
        return plan

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    monkeypatch.setattr(runner, "__file__", str(source / "scripts/v2/run_br400_group.py"))
    monkeypatch.setattr(
        family,
        "validate_calibration_binding",
        lambda *a, **kw: dict(
            calibration_source_commit=family.CALIBRATION_SOURCE,
            formal_source_commit=kw["formal_source_commit"],
            selected_weight=family.AMPLITUDE_WEIGHT,
        ),
    )
    return runner.prepare(
        source,
        output / "manifest.json",
        runtime,
        old_output.parent / "r13-preflight",
        old["shared_gpu_policy"],
        preflight_only=True,
    )


@pytest.mark.parametrize("gpu", ["0", "1"])
@pytest.mark.parametrize("arm", family.ARMS)
def test_exact_r13_eval128_registration_and_sealed_scope(generated_r13_queue, gpu, arm):
    from eval_batch_capacity import validate_eval_binding
    from test_representation_round11 import r11_eval_args

    queue = copy.deepcopy(generated_r13_queue)
    selected = next(row for row in queue["rows"] if row["name"] == arm)
    args = r11_eval_args({**queue, "rows": [selected]}, gpu)
    args.row_name = arm
    args.output = Path(
        next(
            row["output"]
            for row in queue["eval_bindings"]
            if row["row_name"] == arm and row["gpu"] == gpu
        )
    )
    assert validate_eval_binding(queue, args, queue["source_commit"]) == selected
    queue["only_arms"] = [arm]
    with pytest.raises(ValueError):
        validate_eval_binding(queue, args, queue["source_commit"])


def test_r13_queue_prepare_and_verify_bind_preflight_only(generated_r13_queue):
    queue = generated_r13_queue
    runner.verify(queue)
    assert queue["preflight_only"] is True
    assert (
        queue["relative_amplitude_binding"]["calibration_source_commit"]
        == family.CALIBRATION_SOURCE
    )
    bad = copy.deepcopy(queue)
    bad["preflight_only"] = False
    with pytest.raises(ValueError, match="preflight-only"):
        runner.verify(bad)


@pytest.mark.parametrize(
    "selection", [["R13-SRAMP"], ["R13-SRAMPNC", "R13-SRAMPNC"], list(family.ARMS), []]
)
def test_r13_selection_never_repeats_fitted_arm(family_root, selection):
    source, output = family_root
    matrix = runner.verify_manifest(source, output / "manifest.json")
    original = copy.deepcopy(matrix)
    with pytest.raises(ValueError, match="untrained"):
        runner.selected_matrix(matrix, selection)
    assert matrix == original
    selected = runner.selected_matrix(matrix, ["R13-SRAMPNC"])
    assert selected["rows"] == [original["rows"][1]]
    assert selected["priority"] == selected["preflight_priority"] == ["R13-SRAMPNC"]
    assert matrix == original


@pytest.fixture
def singleton_r13_queue(generated_r13_queue):
    q = generated_r13_queue
    return runner.prepare(
        Path(q["source"]),
        Path(q["manifest"]),
        Path(q["runtime"]),
        Path(q["output"]).parent / "r13-nc-singleton",
        q["shared_gpu_policy"],
        only_arms=["R13-SRAMPNC"],
        preflight_only=True,
    )


@pytest.mark.parametrize("gpu", ["0", "1"])
def test_singleton_eval_scope_config_and_source_remain_strict(singleton_r13_queue, gpu):
    from eval_batch_capacity import validate_eval_binding
    from test_representation_round11 import r11_eval_args

    q = singleton_r13_queue
    runner.verify(q)
    assert len(q["rows"]) == 1 and len(q["eval_bindings"]) == 2
    args = r11_eval_args(q, gpu)
    args.row_name = "R13-SRAMPNC"
    args.output = Path(next(b["output"] for b in q["eval_bindings"] if b["gpu"] == gpu))
    assert validate_eval_binding(q, args, q["source_commit"]) == q["rows"][0]
    for bad_field, value in [("source_commit", "0" * 40), ("resolved_config", {})]:
        bad = copy.deepcopy(q)
        bad["rows"][0]["plan"][bad_field] = value
        with pytest.raises(ValueError, match="manifest configuration"):
            validate_eval_binding(bad, args, q["source_commit"])


def test_singleton_promotion_seals_only_nc_and_never_repeats_sramp(
    singleton_r13_queue, monkeypatch
):
    from gradpert.data._io import atomic_json, read_json

    q = singleton_r13_queue
    directory, calls = mock_preflight_runtime(q, monkeypatch)
    runner.execute(q, directory, preflight_only=True)
    ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
    assert ready["formal_order"] == ["R13-SRAMPNC"]
    assert calls == ["preflight", "eval-capacity-g0", "eval-capacity-g1"]
    atomic_json(directory / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})
    assert runner.validate_fit_promotion(q, directory, read_json(directory / "state.json")) == ready
    with pytest.raises(RuntimeError, match="mock fit reached"):
        runner.execute(q, directory, resume_fit=True)
    assert calls[-1] == "fit" and len(calls) == 4


def test_cli_accepts_only_named_nc_continuation(singleton_r13_queue, monkeypatch, tmp_path):
    from gradpert.data._io import atomic_json

    q = singleton_r13_queue
    policy = tmp_path / "policy.json"
    atomic_json(policy, q["shared_gpu_policy"])
    captured = []
    monkeypatch.setattr(runner, "prepare", lambda *a, **kw: captured.append((a, kw)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--prepare",
            "--preflight-only",
            "--source",
            q["source"],
            "--manifest",
            q["manifest"],
            "--runtime",
            q["runtime"],
            "--output",
            str(tmp_path / "new"),
            "--shared-policy",
            str(policy),
            "--only-arms",
            "R13-SRAMPNC",
        ],
    )
    runner.main()
    assert captured[0][1] == dict(only_arms=["R13-SRAMPNC"], preflight_only=True)


@pytest.fixture
def bound_calibration(tmp_path, monkeypatch):
    import calibrate_relative_amplitude as calibration

    from gradpert.data._io import atomic_json

    root = tmp_path / "calibration"
    path = root / "CALIBRATION_COMPLETE.json"
    value = dict(
        calibration_complete=True,
        engineering_accepted=True,
        updates=0,
        scientific_complete=False,
        source_commit=family.CALIBRATION_SOURCE,
        no_validation_or_test=True,
        selection=dict(selected_weight=family.AMPLITUDE_WEIGHT, eligible_batches=8),
    )
    atomic_json(path, value)
    checksum = sha256_file(path)
    monkeypatch.setattr(family, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(family, "CALIBRATION_RESULT_SHA256", checksum)
    monkeypatch.setattr(calibration, "verify_endpoints", lambda *a, **kw: copy.deepcopy(value))
    runtime = dict(
        relative_amplitude_calibration=dict(
            root=str(root), source_commit=family.CALIBRATION_SOURCE, native_result_sha256=checksum
        )
    )
    return runtime, value, path


def test_binding_preserves_original_calibration_and_actual_formal_sha(bound_calibration):
    runtime, _, _ = bound_calibration
    proof = family.validate_calibration_binding(ROOT, runtime, formal_source_commit="a" * 40)
    assert proof["calibration_source_commit"] == family.CALIBRATION_SOURCE
    assert proof["formal_source_commit"] == "a" * 40
    assert proof["selected_weight"] == family.AMPLITUDE_WEIGHT


@pytest.mark.parametrize("change", ["old_r12", "source", "hash", "weight", "test", "not_accepted"])
def test_binding_rejects_stale_or_relabelled_or_unaccepted_calibration(
    bound_calibration, monkeypatch, change
):
    from gradpert.data._io import atomic_json

    runtime, result, path = bound_calibration
    if change == "old_r12":
        runtime["spectral_pair_calibration"] = {}
    elif change == "source":
        runtime["relative_amplitude_calibration"]["source_commit"] = "a" * 40
    elif change == "hash":
        runtime["relative_amplitude_calibration"]["native_result_sha256"] = "b" * 64
    else:
        if change == "weight":
            result["selection"]["selected_weight"] = 0.1
        elif change == "test":
            result["no_validation_or_test"] = False
        else:
            result["engineering_accepted"] = False
        atomic_json(path, result)
        checksum = sha256_file(path)
        monkeypatch.setattr(family, "CALIBRATION_RESULT_SHA256", checksum)
        runtime["relative_amplitude_calibration"]["native_result_sha256"] = checksum
    with pytest.raises(ValueError):
        family.validate_calibration_binding(ROOT, runtime, formal_source_commit="a" * 40)


def mock_preflight_runtime(queue, monkeypatch):
    import contextlib

    from gradpert.data._io import atomic_json

    directory = Path(queue["output"])
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "lock", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(runner, "snapshot", lambda *a: {"processes": []})
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    monkeypatch.setattr(runner, "validate_training_gate", lambda *a: {})
    monkeypatch.setattr(runner, "validate_eval_gate", lambda *a: {})
    monkeypatch.setattr(runner, "validate_common_gates", lambda *a: {"physical_data": "sealed"})

    def stage(q, row, name, command, gpu, state, guard):
        calls.append(name)
        if name == "fit":
            raise RuntimeError("mock fit reached only after explicit promotion")
        path = (
            directory / f"{row['name']}-preflight/receipt.json"
            if name == "preflight"
            else Path(
                next(
                    b["output"]
                    for b in q["eval_bindings"]
                    if b["row_name"] == row["name"] and b["gpu"] == gpu
                )
            )
            / "receipt.json"
        )
        atomic_json(path, {"actual_mock_step": name})

    monkeypatch.setattr(runner, "run_stage", stage)
    return directory, calls


def test_preflight_only_exits_before_external_acceptance_can_trigger_fit(
    generated_r13_queue, monkeypatch
):
    from gradpert.data._io import atomic_json, read_json

    queue = generated_r13_queue
    directory, calls = mock_preflight_runtime(queue, monkeypatch)
    atomic_json(directory / "MAIN_ACCEPTANCE.json", {"approved": True, "untrusted_existing": True})
    runner.execute(queue, directory, preflight_only=True)
    state = read_json(directory / "state.json")
    assert calls == ["preflight", "eval-capacity-g0", "eval-capacity-g1"] * 2
    assert state["status"] == "preflight_complete" and state["phase"] == "awaiting_main_acceptance"
    assert state["active"] is None and state["formal_fit_authorized"] is False
    assert not (directory / "FIT_PROMOTION.json").exists()


@pytest.fixture
def promotable_queue(generated_r13_queue, monkeypatch):
    from gradpert.data._io import atomic_json, read_json

    queue = generated_r13_queue
    directory, calls = mock_preflight_runtime(queue, monkeypatch)
    runner.execute(queue, directory, preflight_only=True)
    ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
    atomic_json(directory / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})
    return queue, directory, calls


def test_explicit_promotion_reuses_gates_without_rerunning_preflights(promotable_queue):
    from gradpert.data._io import read_json

    queue, directory, calls = promotable_queue
    state = read_json(directory / "state.json")
    assert runner.validate_fit_promotion(queue, directory, state)["preflight_only"]
    before = len(calls)
    with pytest.raises(RuntimeError, match="mock fit reached"):
        runner.execute(queue, directory, resume_fit=True)
    assert calls[before:] == ["fit"]
    assert read_json(directory / "FIT_PROMOTION.json")["preflights_reused"] is True


@pytest.mark.parametrize(
    "change", ["no_accept", "partial", "queue", "source", "gate", "existing_fit", "wrong_mode"]
)
def test_promotion_rejects_missing_or_changed_evidence_or_existing_fit(promotable_queue, change):
    from gradpert.data._io import atomic_json, read_json

    queue, directory, _ = promotable_queue
    state = read_json(directory / "state.json")
    if change == "no_accept":
        (directory / "MAIN_ACCEPTANCE.json").unlink()
    elif change == "partial":
        state["status"] = "running"
    elif change == "queue":
        state["queue_sha256"] = "0" * 64
    elif change == "source":
        state["source_commit"] = "0" * 40
    elif change == "gate":
        ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
        ready["common_identity"] = {"changed": True}
        atomic_json(directory / "PREFLIGHT_COMPLETE.json", ready)
    elif change == "existing_fit":
        Path(queue["rows"][0]["plan"]["run_root"]).mkdir(parents=True)
    else:
        queue["preflight_only"] = False
    with pytest.raises((ValueError, OSError)):
        runner.validate_fit_promotion(queue, directory, state)


def test_actual_r13_report_exports_raw_weighted_and_partial_amplitude_na(tmp_path):
    import csv

    from gradpert.data._io import atomic_json

    root, run = tmp_path / "medium", tmp_path / "run"
    config = ROOT / "configs/v2" / family.FAMILY / family.ARMS[0] / "gradpert_v2/nadig_jurkat.yaml"
    row = dict(
        name=family.ARMS[0],
        plan=dict(run_root=str(run), config=str(config), config_sha256=sha256_file(config)),
    )
    queue = dict(output=str(tmp_path), source_commit="a" * 40, family=family.SCHEMA)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p",
                joint=2.0,
                loss_terms=dict(prediction=1.0, ssl2_dino=1.0),
                diagnostic_objective_scope=dict(
                    unavailable_terms={
                        key: "requires_complete_global_population"
                        for key in ("spectral_rank", "spectral_cosine", "spectral_amplitude")
                    },
                    joint_scope="prediction_and_ssl_excluding_unavailable_representation_terms",
                ),
            )
        ],
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(
        run / "fit/history.json",
        [
            dict(
                epoch=i,
                training=dict(
                    prediction_mse=0.5,
                    ssl1_condition=4.0,
                    ssl2_dino=5.0,
                    ssl2_ibot=6.0,
                    spectral_rank=0.2,
                    spectral_cosine=0.1,
                    spectral_amplitude=0.0,
                    spectral_amplitude_weighted=0.0,
                    spectral_rank_weighted=0.001,
                    spectral_cosine_weighted=0.002,
                    spectral_weighted=0.003,
                    spectral_warmup=0.5,
                    spectral_q_g0=0.1,
                    spectral_trace_g0=25.6,
                    spectral_total_energy_g0=256.0,
                    spectral_mean_energy_g0=230.4,
                    joint_loss=15.503,
                ),
            )
            for i in (1, 2)
        ],
    )
    atomic_json(
        postfit.validation_path(queue, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    postfit.render_medium_report(queue, row, root)
    partial = next(csv.DictReader((root / "report/frozen_losses.csv").open()))
    assert (
        partial["spectral_amplitude"]
        == partial["spectral_rank"]
        == partial["spectral_cosine"]
        == ""
    )
    contribution = next(csv.DictReader((root / "report/training_loss_contributions.csv").open()))
    assert (
        contribution["spectral_amplitude"] == contribution["spectral_amplitude_weighted"] == "0.0"
    )
    assert contribution["spectral_q_g0"] == "0.1" and contribution["spectral_q_g1"] == ""
    assert (root / "report/spectral_losses.png").stat().st_size > 1000
    report = (root / "report/REPORT_CN.md").read_text()
    assert family.CALIBRATION_SOURCE in report and family.CALIBRATION_RESULT_SHA256 in report
    assert "普通medium/panel未采集训练后gamma" in report
    assert "科学验收前需独立冻结0更新补测" in report


@pytest.mark.parametrize("missing", [None, "spectral_amplitude", "exact_restore"])
def test_native_training_gate_requires_actual_r13_amplitude_proof(generated_r13_queue, missing):
    from test_fgpo_group import training_gate

    from gradpert.data._io import atomic_json

    queue = generated_r13_queue
    row = queue["rows"][0]
    receipt = training_gate(queue, row)
    gate = amplitude_gate(row["name"], active=False)
    gate["resume_checkpoint_sha256"] = receipt["resume_checkpoint_sha256"]
    receipt["spectral_amplitude_gate"] = gate
    if missing == "exact_restore":
        gate["checks"]["exact_restore"] = False
    elif missing:
        gate["updates"][4].pop(missing)
    path = Path(queue["output"]) / "r13-gate/receipt.json"
    atomic_json(path, receipt)
    if missing:
        with pytest.raises(ValueError):
            runner.validate_training_gate(queue, row, runner.gate_entry(path))
    else:
        runner.validate_training_gate(queue, row, runner.gate_entry(path))


@pytest.mark.parametrize("competitor_marker", [False, True])
def test_resume_fit_rechecks_disk_state_and_failure_after_waiting_for_locks(
    promotable_queue, monkeypatch, competitor_marker
):
    import contextlib

    from gradpert.data._io import atomic_json, read_json

    queue, directory, calls = promotable_queue
    before = list(calls)

    @contextlib.contextmanager
    def raced_lock(path):
        if path.name == "controller.lock":
            state = read_json(directory / "state.json")
            state.update(status="failed", phase="failed", error="competing controller failure")
            atomic_json(directory / "state.json", state)
            atomic_json(directory / "FAILURE.json", {"error": "competing controller failure"})
            if competitor_marker:
                atomic_json(directory / "FIT_PROMOTION.json", {"owner": "competing controller"})
        yield

    monkeypatch.setattr(runner, "lock", raced_lock)
    with pytest.raises(ValueError, match="terminal state changed"):
        runner.execute(queue, directory, resume_fit=True)
    assert calls == before
    if competitor_marker:
        assert read_json(directory / "FIT_PROMOTION.json") == {"owner": "competing controller"}
    else:
        assert not (directory / "FIT_PROMOTION.json").exists()
    assert read_json(directory / "state.json")["error"] == "competing controller failure"
    assert read_json(directory / "FAILURE.json") == {"error": "competing controller failure"}
    assert list(directory.glob("FIT_PROMOTION_REJECTED-*.json"))


@pytest.fixture
def r13_postfit_binding_case(tmp_path):
    from test_cr_medium_adapter import cr_case

    parent, identity, queue, args, kwargs = cr_case.__wrapped__(tmp_path, family)
    identity["data"] = copy.deepcopy(parent["data"])
    return parent, identity, queue, args, kwargs


def seal_r13_postfit_arm(case, arm):
    parent, identity, queue, args, _ = case
    args.config.write_text(yaml.safe_dump(family.configs(ROOT)[arm]))
    identity["run_id"] = arm + "-run"
    identity["data"]["loss_protocol"]["spectral_pair"] = family.spectral_protocol(arm)
    row = queue["rows"][0]
    row["name"] = arm
    row["plan"].update(run_id=identity["run_id"], config_sha256=sha256_file(args.config))
    return parent, identity, queue, args, row


@pytest.mark.parametrize("arm", family.ARMS)
@pytest.mark.parametrize(
    "drift", [None, "physical", "config"], ids=["valid", "physical_drift", "config_drift"]
)
def test_medium_execute_r13_binding_assembles_exact_native_keywords(
    r13_postfit_binding_case, monkeypatch, arm, drift
):
    from gradpert.data._io import atomic_json, read_json

    parent, identity, queue, args, row = seal_r13_postfit_arm(r13_postfit_binding_case, arm)
    if drift == "physical":
        identity["data"]["canonical_sha256"] = "f" * 64
    elif drift == "config":
        raw = yaml.safe_load(args.config.read_text())
        raw["model"]["parameters"]["spectral_amplitude_weight"]["value"] = 0.1
        args.config.write_text(yaml.safe_dump(raw))
        row["plan"]["config_sha256"] = sha256_file(args.config)
    atomic_json(args.run_root / "run_manifest.json", identity)
    queue_path = Path(queue["output"]) / "queue.json"
    atomic_json(queue_path, queue)
    original_parent = copy.deepcopy(parent)
    original_parent_bytes = args.parent_manifest.read_bytes()
    monkeypatch.setattr(medium, "SERVER_ROOT", args.run_root.parent)
    monkeypatch.setattr(runner, "verify", lambda q: None)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: None)
    monkeypatch.setattr(medium, "validate_analysis_spec", lambda spec: parent)
    monkeypatch.setattr(medium, "select_checkpoint", lambda *a: (dict(sha256="d" * 64), {}))
    original_binding = medium.relative_amplitude_parent_binding
    binding_calls, dispatches = [], []

    def binding(*values, **keywords):
        binding_calls.append(keywords)
        return original_binding(*values, **keywords)

    def native_seal(command, **keywords):
        dispatches.append(command)
        assert "seal-medium" in command and "--reuse-medium-schedule" in command
        assert command[command.index("--method-sha") + 1] == args.method_sha
        assert command[command.index("--config") + 1] == str(args.config)
        assert keywords["env"]["PYTHONPATH"] == str(ROOT / "src")
        bound_path = Path(command[command.index("--parent-manifest") + 1])
        assert sha256_file(bound_path) == command[command.index("--parent-manifest-sha256") + 1]
        bound = read_json(bound_path)
        assert (
            bound["data"] == identity["data"] and bound["schedule"] == original_parent["schedule"]
        )
        assert bound["training_protocol_binding"]["arm"] == arm
        assert bound["training_protocol_binding"]["training_run_id"] == identity["run_id"]
        assert bound["training_protocol_binding"]["training_value"] == family.spectral_protocol(arm)
        raise RuntimeError("R13 native seal reached with exact binding")

    monkeypatch.setattr(medium, "relative_amplitude_parent_binding", binding)
    monkeypatch.setattr(medium.subprocess, "run", native_seal)
    expected_error, message = (
        (ValueError, "sealed|differs beyond")
        if drift
        else (RuntimeError, "R13 native seal reached with exact binding")
    )
    with pytest.raises(expected_error, match=message):
        medium.execute(queue_path, arm)
    assert len(binding_calls) == 1
    assert binding_calls[0]["arm"] == arm and binding_calls[0]["config"] == row["plan"]["config"]
    assert binding_calls[0]["parent_path"] == str(args.parent_manifest)
    assert binding_calls[0]["parent_sha256"] == args.parent_manifest_sha256
    assert bool(dispatches) == (drift is None)
    assert parent == original_parent and args.parent_manifest.read_bytes() == original_parent_bytes


@pytest.mark.parametrize("arm", family.ARMS)
def test_panel_dispatch_passes_exact_r13_binding_keywords(
    r13_postfit_binding_case, monkeypatch, arm
):
    import run_representation_panel as panel

    parent, identity, queue, args, row = seal_r13_postfit_arm(r13_postfit_binding_case, arm)
    original_binding = medium.relative_amplitude_parent_binding
    observed = []

    def binding(*values, **keywords):
        observed.append(keywords)
        return original_binding(*values, **keywords)

    monkeypatch.setattr(medium, "relative_amplitude_parent_binding", binding)
    result = panel.panel_parent_binding(parent, identity, queue, args)
    assert result["arm"] == arm and result["training_run_id"] == identity["run_id"]
    assert observed == [
        dict(
            arm=arm,
            config=args.config,
            parent_path=args.parent_manifest,
            parent_sha256=args.parent_manifest_sha256,
        )
    ]
    before = copy.deepcopy(observed)
    row["plan"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="queue/run/config binding differs"):
        panel.panel_parent_binding(parent, identity, queue, args)
    assert observed == before


@pytest.mark.parametrize("arm", family.ARMS)
def test_r13_repair_delegates_exact_publication_and_selected_arm(tmp_path, monkeypatch, arm):
    from types import SimpleNamespace

    events = []
    queue = dict(family=family.SCHEMA, rows=[dict(name=name) for name in family.ARMS])
    source, publication, output = (
        tmp_path / "source",
        tmp_path / "publication.json",
        tmp_path / "new-analysis",
    )
    module = SimpleNamespace(
        validate_repair_source=lambda q, s: events.append(("source", q, s)),
        validate_repair_publication=lambda q, s, p, h: events.append(("publication", q, s, p, h)),
        validate_output=lambda q, p: events.append(("output", q, p)) or p,
    )
    monkeypatch.setitem(sys.modules, "run_relative_amplitude_analysis_repair", module)
    monkeypatch.setattr(
        postfit, "validate_validation", lambda q, row: events.append(("validation", q, row))
    )
    result = medium.validate_report_repair(
        queue, arm, output, source, publication=publication, publication_sha256="f" * 64
    )
    assert result == output
    assert events == [
        ("source", queue, source),
        ("publication", queue, source, publication, "f" * 64),
        ("validation", queue, next(row for row in queue["rows"] if row["name"] == arm)),
        ("output", queue, output),
    ]


@pytest.mark.parametrize("arm", family.ARMS)
def test_medium_execute_passes_r13_repair_publication_before_any_dispatch(
    r13_postfit_binding_case, tmp_path, monkeypatch, arm
):
    from gradpert.data._io import atomic_json

    _, _, queue, args, _ = seal_r13_postfit_arm(r13_postfit_binding_case, arm)
    queue_path = tmp_path / "queue.json"
    atomic_json(queue_path, queue)
    output, publication = tmp_path / "repair", tmp_path / "publication.json"
    calls = []
    monkeypatch.setattr(runner, "verify", lambda q: None)

    def guard(q, name, target, source, **keywords):
        calls.append((q, name, target, source, keywords))
        raise ValueError("stopped at sealed independent repair guard")

    monkeypatch.setattr(medium, "validate_report_repair", guard)
    with pytest.raises(ValueError, match="sealed independent repair guard"):
        medium.execute(
            queue_path,
            arm,
            repair_output=output,
            repair_publication=publication,
            repair_publication_sha256="e" * 64,
        )
    assert len(calls) == 1 and calls[0][1] == arm
    assert calls[0][2] == output and calls[0][3] == ROOT
    assert calls[0][4] == dict(publication=publication, publication_sha256="e" * 64)
    assert not output.exists() and not args.run_root.exists()
