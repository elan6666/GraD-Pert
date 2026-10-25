"""Only original R12 strengths run; original RI/control/sampling stay immutable."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round12 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as postfit  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_cr_medium_adapter import cr_case  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / family.FAMILY
    family.generate(source, output)
    return source, output


def test_exact_original_ri_only_two_new_arms_and_all_routes(family_root):
    source, output = family_root
    value = runner.verify_manifest(source, output / "manifest.json")
    assert [r["name"] for r in value["rows"]] == ["R12-SRBAL", "R12-SRAGG"]
    assert value["repeat_baseline"] is False and value["main_reference"] == "R3-RI"
    assert value["reused_controls"] == ["R3-RI", "RS-B0", "RS-RI"]
    assert family.SCHEMA in runner.RESEARCH_FAMILIES
    assert medium.FAMILY_ARMS[family.SCHEMA] == postfit.FAMILY_ARMS[family.SCHEMA] == family.ARMS
    assert value["completed_optimizer_updates"] == 300
    for arm in family.ARMS:
        candidate = family.configs(source)[arm]
        base = first.configs(source)["RS-RI"]
        for key in family.CHANGES[arm]:
            candidate["model"]["parameters"].pop(key)
        assert candidate == base
        row = next(r for r in value["rows"] if r["name"] == arm)
        target = source / row["config"]
        plan = dict(
            repository_root=str(source),
            config=str(target),
            resolved_config=load_experiment_config(target).model_dump(mode="json"),
        )
        assert runner.expected_epochs(plan) == 2
    assert value["global_batch"] == 320 and value["preflight_updates"] == 10
    assert value["eval_batch"] == value["eval_capacity_reference_batch"] == 128
    assert value["test_roles"] == ["last"] and value["postfit_validation"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("spectral_rank_target", 0.4),
        ("spectral_rank_weight", 0.3),
        ("spectral_cosine_weight", 0.1),
        ("spectral_softness", 0.3),
        ("spectral_warmup_steps", 20),
        ("cr_weight", 0.00004),
        ("teacher_end", 0.99),
        ("lambda1", 0.05),
    ],
)
def test_rehashed_protocol_drift_rejected(family_root, field, value):
    source, output = family_root
    manifest = json.loads((output / "manifest.json").read_text())
    row = manifest["rows"][1]
    target = source / row["config"]
    raw = yaml.safe_load(target.read_text())
    raw["model"]["parameters"][field]["value"] = value
    target.write_text(yaml.safe_dump(raw))
    row["sha256"] = sha256_file(target)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        family.verify_manifest(source, output / "manifest.json")


def test_actual_losses_preserve_saved_warmup_contributions():
    target = ROOT / "configs/v2" / family.FAMILY / "R12-SRBAL/gradpert_v2/nadig_jurkat.yaml"
    row = postfit.ssl_loss_contributions(
        [
            dict(
                epoch=1,
                training=dict(
                    prediction_mse=0.5,
                    ssl1_condition=4.0,
                    ssl2_dino=5.0,
                    ssl2_ibot=6.0,
                    spectral_rank=0.2,
                    spectral_cosine=0.1,
                    spectral_rank_weighted=0.001,
                    spectral_cosine_weighted=0.002,
                    spectral_weighted=0.003,
                    spectral_warmup=0.5,
                    joint_loss=15.503,
                ),
            )
        ],
        target,
    )[0]
    assert row["prediction"] == 0.5
    assert row["ssl1_contribution"] == 4 and row["ssl2_contribution"] == 11
    assert row["spectral_weighted"] == 0.003
    assert row["spectral_rank_weighted"] == 0.001
    assert row["cr_weighted"] is None


@pytest.mark.parametrize("arm", family.ARMS)
@pytest.mark.parametrize("deep", [False, True])
def test_integrated_medium_panel_sampling_retains_native_recipe_and_training_options(arm, deep):
    from dataclasses import replace

    import numpy as np
    import torch
    from diagnostic_view_options import diagnostic_view_options
    from representation_panel import assemble_control_only
    from test_frozen_readout_medium import runtime

    from gradpert.config.v2 import V2Options
    from gradpert.training.v2.views import assemble_batch

    rt, records, reads = runtime()
    target = ROOT / "configs/v2" / family.FAMILY / arm / "gradpert_v2/nadig_jurkat.yaml"
    _, options = V2Options.parse_parameters(load_experiment_config(target).model.parameters)
    rt.options = replace(
        options,
        query_count=6,
        independent_view_rng=True,
        mask_probability=1.0,
        graph_mask_ratio=0.5,
        graph_edge_dropout=0.2,
    )
    original = copy.deepcopy(rt.options)
    sampling = diagnostic_view_options(rt.options, deep=deep)
    expected_options = replace(
        rt.options,
        lambda1=float(deep),
        lambda2=float(deep),
        spectral_rank_weight=rt.options.spectral_rank_weight if deep else 0.0,
        spectral_cosine_weight=rt.options.spectral_cosine_weight if deep else 0.0,
    )
    assert sampling == expected_options
    assert rt.options == original
    from cap40_posthoc import assemble, make_raw, recipe

    queries, seed = [0, 2, 4, 6, 8, 10], 101
    raw = make_raw(rt, records)
    expected = assemble_batch(
        raw,
        rt.index,
        expected_options,
        np.random.default_rng(seed),
        allowed_expression_ids=np.array(queries),
    )
    reads.clear()
    medium_batch = assemble(rt, records, queries, seed, deep=deep)
    assert reads == [4, 5, 6, 7, 0, 1, 2, 3]
    reads.clear()
    panel_batch = assemble_control_only(rt, records, queries, seed, deep=deep)
    assert reads == [4, 5, 6, 7]
    assert recipe(medium_batch) == recipe(panel_batch) == recipe(expected)
    torch.testing.assert_close(medium_batch.truth, expected.truth, rtol=0, atol=0)
    assert not panel_batch.truth.any()
    assert bool(medium_batch.graph_views) == bool(medium_batch.cell_views) == deep
    assert rt.options == original


def test_shared_sampling_helper_supports_legacy_schema_without_spectral_fields():
    from dataclasses import dataclass, replace

    from diagnostic_view_options import diagnostic_view_options

    @dataclass(frozen=True)
    class LegacyOptions:
        lambda1: float = 1.0
        lambda2: float = 1.0

    original = LegacyOptions()
    assert diagnostic_view_options(original, deep=False) == replace(
        original, lambda1=0.0, lambda2=0.0
    )
    assert diagnostic_view_options(original, deep=True) == original


def test_r11_narrow_repair_source_gate_rejects_integrated_r12_helper(tmp_path, monkeypatch):
    import run_readout_analysis_repair as repair
    from test_readout_analysis_repair import fake_git, queue

    # Even with an allegedly clean allowed-file diff, the old R11-only repair
    # rejects this new helper. Keep its sealed gate and HELPER_SHA unchanged.
    fake_git(monkeypatch)
    assert sha256_file(ROOT / "scripts/v2/diagnostic_view_options.py") != repair.HELPER_SHA
    with pytest.raises(ValueError, match="sampling helper differs"):
        repair.validate_repair_source(queue(tmp_path), ROOT)


@pytest.fixture
def generated_r12_queue(family_root, new_queue, monkeypatch):
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
    return runner.prepare(
        source,
        output / "manifest.json",
        runtime,
        old_output.parent / "r12-eval",
        old["shared_gpu_policy"],
    )


@pytest.mark.parametrize("gpu", ["0", "1"])
@pytest.mark.parametrize("arm", family.ARMS)
def test_both_exact_eval128_leaves_and_reject_mutated_scope(generated_r12_queue, gpu, arm):
    from eval_batch_capacity import PARITY_ATOL, PARITY_RTOL, validate_eval_binding
    from test_representation_round11 import r11_eval_args

    queue = copy.deepcopy(generated_r12_queue)
    selected = next(row for row in queue["rows"] if row["name"] == arm)
    # Existing argument helper takes the first row. Preserve the sealed queue order.
    arguments_queue = {**queue, "rows": [selected]}
    args = r11_eval_args(arguments_queue, gpu)
    args.row_name = arm
    # Select this arm's exact per-GPU output, not another arm's eval leaf.
    binding = next(b for b in queue["eval_bindings"] if b["gpu"] == gpu and b["row_name"] == arm)
    args.output = Path(binding["output"])
    assert validate_eval_binding(queue, args, queue["source_commit"]) == selected
    assert PARITY_ATOL == PARITY_RTOL == 2e-6
    queue["only_arms"] = [arm]
    with pytest.raises(ValueError):
        validate_eval_binding(queue, args, queue["source_commit"])


@pytest.mark.parametrize("arm", family.ARMS)
def test_medium_and_fixed_panel_strict_physical_population_binding(tmp_path, arm):
    from run_representation_panel import panel_parent_binding

    parent, identity, queue, args, kwargs = cr_case.__wrapped__(tmp_path, family)
    args.config.write_text(yaml.safe_dump(family.configs(ROOT)[arm]))
    identity["data"] = copy.deepcopy(parent["data"])
    queue["rows"][0]["name"] = kwargs["arm"] = arm
    queue["rows"][0]["plan"]["config_sha256"] = sha256_file(args.config)
    strength = family.STRENGTHS[arm]
    protocol = dict(
        rank_target=strength["target"],
        rank_weight=strength["rank_weight"],
        cosine_weight=strength["cosine_weight"],
        softness=0.25,
        warmup_completed_optimizer_updates=30,
        representation="student_response_cls_after_final_rmsnorm_before_projector",
        population_scope="complete_effective_batch_all_microbatches_and_ranks_all_contexts",
        deduplication="control_row_id_and_condition_id_before_rank_partition",
        rank_formula="mean_globals_softness_softplus_log_target_minus_log_rho_over_softness",
        rho="exp_spectral_entropy_divided_by_min_D_Nminus1",
        covariance="separate_globals_raw_centered_population_fp64_eigvalsh_no_jitter_no_l2",
        spectrum_roundoff="negative_tolerance_64_D_eps64_max_abs_eigenvalue_then_clamp_zero",
        near_zero_trace_rejection=1e-24,
        cosine="same_input_two_globals_both_student_gradients",
        partial_diagnostics="unavailable_requires_complete_global_population",
    )
    identity["data"]["loss_protocol"]["spectral_pair"] = protocol
    bound = medium.spectral_pair_parent_binding(parent, identity, **kwargs)
    assert bound["schedule"] == parent["schedule"]
    assert panel_parent_binding(parent, identity, queue, args) == bound["training_protocol_binding"]
    identity["data"]["canonical_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs beyond"):
        medium.spectral_pair_parent_binding(parent, identity, **kwargs)


def test_controller_blocks_fit_without_both_real_calibration_endpoints(generated_r12_queue):
    with pytest.raises(ValueError, match="both sealed zero-update"):
        runner.validate_common_gates(generated_r12_queue, {})


@pytest.mark.parametrize(
    "missing", [None, "exact_restore", "complete_population", "first_update_zero"]
)
def test_native_capacity_gate_needs_actual_spectral_proof(generated_r12_queue, missing):
    from test_fgpo_group import training_gate

    from gradpert.data._io import atomic_json

    q = generated_r12_queue
    row = q["rows"][0]
    receipt = training_gate(q, row)
    checks = dict.fromkeys(
        (
            "ten_native_updates",
            "exact_restore",
            "complete_population",
            "correct_warmup",
            "first_update_zero",
            "finite_losses",
        ),
        True,
    )
    receipt["spectral_pair_gate"] = dict(
        passed=True,
        checks=checks,
        updates=[
            dict(
                completed_before=i,
                population=320,
                warmup=min(1.0, i / 30),
                rank=0.2,
                cosine=0.1,
                weighted=i * 0.001,
            )
            for i in range(10)
        ],
        resume_checkpoint_sha256=receipt["resume_checkpoint_sha256"],
    )
    if missing:
        checks[missing] = False
    path = Path(q["output"]) / "r12-preflight/receipt.json"
    atomic_json(path, receipt)
    if missing:
        with pytest.raises(ValueError, match="spectral/warmup/restore"):
            runner.validate_training_gate(q, row, runner.gate_entry(path))
    else:
        runner.validate_training_gate(q, row, runner.gate_entry(path))


def test_actual_report_plot_exports_training_contributions_and_partial_na(tmp_path):
    import csv

    from gradpert.data._io import atomic_json

    root, run = tmp_path / "medium", tmp_path / "run"
    config = ROOT / "configs/v2" / family.FAMILY / "R12-SRBAL/gradpert_v2/nadig_jurkat.yaml"
    row = dict(
        name="R12-SRBAL",
        plan=dict(run_root=str(run), config=str(config), config_sha256=sha256_file(config)),
    )
    queue = dict(output=str(tmp_path), source_commit="a" * 40, family=family.SCHEMA)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p",
                joint=2,
                loss_terms=dict(prediction=1, ssl2_dino=1),
                diagnostic_objective_scope=dict(
                    unavailable_terms={
                        key: "requires_complete_global_population"
                        for key in ("spectral_rank", "spectral_cosine")
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
                    spectral_rank_weighted=0.001,
                    spectral_cosine_weighted=0.002,
                    spectral_weighted=0.003,
                    spectral_warmup=0.5,
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
    assert partial["spectral_rank"] == partial["spectral_cosine"] == ""
    contributions = next(csv.DictReader((root / "report/training_loss_contributions.csv").open()))
    assert contributions["spectral_weighted"] == "0.003"
    assert contributions["ssl1_contribution"] == "4.0"
    assert (root / "report/spectral_losses.png").stat().st_size > 1000
    report = (root / "report/REPORT_CN.md").read_text()
    assert "历史R3-RI" in report and "跨版本单seed" in report
    assert "target=.35/rank=.04/cos=.20" in report
