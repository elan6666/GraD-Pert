"""Seal two amplitude-floor arms and bind their accepted train-only calibration."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import representation_family as sealed
from generate_representation_round1 import configs as first_configs
from generate_representation_round12 import metadata as previous_metadata

from gradpert.hashing import sha256_file

SCHEMA = "representation-round13-relative-amplitude-medium-1"
FAMILY = "representation_round13_20261026"
REFERENCE = "docs/design/RELATIVE_AMPLITUDE_RESEARCH.md"
ARMS = ("R13-SRAMP", "R13-SRAMPNC")
CALIBRATION_SOURCE = "933dfb5cf546adbf5561a4041fbd4737643f2020"
CALIBRATION_RESULT_SHA256 = "6a98d66d761f69e65d2f41d35665d3965702602d6ba11ec300b2bb94d721d191"
AMPLITUDE_WEIGHT = 0.012439571837341717
AMPLITUDE_TARGET = 0.0225
SERVER_ROOT = Path("/data/yilangliu")
CHANGES = {
    arm: dict(
        spectral_rank_target=0.20,
        spectral_rank_weight=0.02,
        spectral_cosine_weight=0.10 if arm == "R13-SRAMP" else 0.0,
        spectral_softness=0.25,
        spectral_warmup_steps=30,
        spectral_amplitude_target=AMPLITUDE_TARGET,
        spectral_amplitude_weight=AMPLITUDE_WEIGHT,
        au_alignment_weight=0.0,
        au_uniformity_weight=0.0,
        cr_weight=0.0,
    )
    for arm in ARMS
}

# Frozen full-file bytes from the actual calibration source, never a current-HEAD substitution.
COMPATIBLE_FILES = {
    "src/gradpert/training/v2/spectral_pair.py": (
        "e99bd77d39d31b10a9c8efca9366c1f6b063504121b3c545e8509ffa5efe7eaf"
    ),
    "src/gradpert/training/v2/objective.py": (
        "d4ed2ed88e3f8993d276726de6cbd84db4056ce11cc5994b2150ba0687179070"
    ),
    "src/gradpert/training/v2/engine.py": (
        "333b7bb01fbdd90dcafda30083583525554a2dc766ffcd62731300195afac3ba"
    ),
    "src/gradpert/training/v2/runtime.py": (
        "40f00de5f497e0972bdda76af0f1c573eac571100fd0bc9d9a2dd2aae59677f9"
    ),
    "src/gradpert/training/v2/diagnostic_objective.py": (
        "2dc92d2183a6374f43e37964254e3d81f50f0b281fc4849ff157f0691b4e679f"
    ),
    "src/gradpert/training/v2/loss_diagnostics.py": (
        "051cb2763377a545976ff5120a200761d510c8b84fbb48b7cc3a30f820cbbe1a"
    ),
    "src/gradpert/config/v2.py": "ca340d8fbdfbde650009c1743be381ba5e290ead5eef342c919a31f8bd97ffe4",
    "src/gradpert/training/v2/checkpoint.py": (
        "aa9a7e86ebd707b12fddc04cbf3bb30f2f73c5cdcd4ee41d8358ce6510b61373"
    ),
    "src/gradpert/training/v2/optimizer.py": (
        "7abd45ec494456d2f2871e233ee96a0c33f1a40d98d860de117fad90ca12de57"
    ),
    "src/gradpert/training/v2/reductions.py": (
        "523dadb212989699cead47d60d9c35b8cf72ab166ad450837705cd0c3aa53265"
    ),
    "src/gradpert/modeling/v2/model.py": (
        "78e43d452b6e5a77f973eae9a08383836173adc809a0ad8d05cab2d458a2f4e9"
    ),
    "scripts/v2/calibrate_alignment_uniformity.py": (
        "f6a2d6487a8ea5bad5d727e6822ac602bedf4d7119c6fb7ac1bf5f80b962e9f5"
    ),
    "scripts/v2/calibrate_relative_amplitude.py": (
        "3ae1d5b3c8080cc09ece9ac94aa60cb220df16c88d141a5e8e5484025bf0ee3f"
    ),
}
CAPACITY_FUNCTIONS = {
    "state_fingerprint": "eba876e18be21f91f8cfce0b745d5fa12f120a4dbe3e5ee6dad5b7df0d088e0d",
    "validation_state": "7d032711c9fc8d80f7e324cfd9b0bd0f77ee70446e6b8c244350b29ec1249108",
}


def configs(source: Path) -> dict:
    anchor = first_configs(source)["RS-RI"]
    values = {}
    for arm, changes in CHANGES.items():
        value = copy.deepcopy(anchor)
        for key, setting in changes.items():
            value["model"]["parameters"][key] = dict(
                value=setting, source="project_preregistered", reference=REFERENCE
            )
        values[arm] = value
    return values


def metadata() -> dict:
    value = previous_metadata()
    value.update(schema=SCHEMA, family=FAMILY, priority=list(ARMS), preflight_priority=list(ARMS))
    value["research_iteration"].update(
        round=13,
        stage_review_after_round=13,
        hypothesis="relative amplitude floor closes rank-only scale loophole; isolate extra cosine",
    )
    value["spectral_pair"]["strengths"] = CHANGES
    value["spectral_pair"]["amplitude"] = dict(
        target_q=AMPLITUDE_TARGET,
        weight=AMPLITUDE_WEIGHT,
        formula="mean_globals_relu_sqrt_target_minus_sqrt_q_plus_epsilon",
        epsilon_q=1e-8,
        q="centered_population_covariance_trace_over_mean_raw_squared_norm",
        denominator_gradients=True,
    )
    value["calibration_required"] = dict(
        source_commit=CALIBRATION_SOURCE,
        native_result_sha256=CALIBRATION_RESULT_SHA256,
        endpoints=["init", "ri_last", "r12bal_last"],
        training_populations_per_endpoint=8,
        effective_batch=320,
        updates=0,
        fitting_endpoint="r12bal_last",
        fitting_group="response",
        selected_weight=AMPLITUDE_WEIGHT,
        median_weighted_gradient_ratio=0.10,
        no_validation_or_test=True,
        source_compatibility=(
            "full math/calibration file bytes and capacity observer source segments"
        ),
        compatible_files_sha256=COMPATIBLE_FILES,
        capacity_function_source_sha256=CAPACITY_FUNCTIONS,
    )
    value["comparison_limits"] = [
        "historical R3-RI main reference; RS-B0/RS-RI context only; no baseline rerun",
        "single seed, cross-version trajectories; no test-driven selection",
        (
            "both arms use the same accepted training-only amplitude coefficient; "
            "only extra cosine differs"
        ),
        (
            "geometry/teacher discrimination and validation utility jointly required; "
            "calibration is engineering only"
        ),
    ]
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    return sealed.verify(source, path, metadata(), configs(source), CHANGES)


def generate(source: Path, output: Path) -> dict:
    return sealed.generate(source, output, metadata(), configs(source), CHANGES)


def capacity_function_hashes(path):
    text = Path(path).read_text()
    nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    return {
        name: hashlib.sha256(ast.get_source_segment(text, nodes[name]).encode()).hexdigest()
        for name in CAPACITY_FUNCTIONS
    }


def verify_source_compatibility(source, *, formal_source_commit):
    source = Path(source)
    if any(
        (source / name).is_symlink()
        or not (source / name).resolve().is_relative_to(source.resolve())
        for name in COMPATIBLE_FILES
    ):
        raise ValueError(
            "R13 compatible source files must remain native source-contained regular files"
        )
    observed = {name: sha256_file(source / name) for name in COMPATIBLE_FILES}
    capacity = capacity_function_hashes(source / "scripts/v2/capacity_probe.py")
    if observed != COMPATIBLE_FILES or capacity != CAPACITY_FUNCTIONS:
        raise ValueError("R13 calibration math/observer source bytes differ from sealed933")
    if (
        not isinstance(formal_source_commit, str)
        or len(formal_source_commit) != 40
        or any(c not in "0123456789abcdef" for c in formal_source_commit)
    ):
        raise ValueError("R13 actual formal source SHA required separately from calibration")
    return dict(
        calibration_source_commit=CALIBRATION_SOURCE,
        formal_source_commit=formal_source_commit,
        compatible_files_sha256=observed,
        capacity_function_source_sha256=capacity,
        byte_identical=True,
    )


def validate_calibration_binding(source, runtime, *, formal_source_commit):
    """Re-read accepted original receipts, without model calls or source relabeling."""
    from calibrate_relative_amplitude import verify_endpoints

    if "spectral_pair_calibration" in runtime:
        raise ValueError("R13 must not retain an operative R12 calibration binding")
    spec = runtime.get("relative_amplitude_calibration", {})
    if not isinstance(spec, dict) or set(spec) != {"root", "source_commit", "native_result_sha256"}:
        raise ValueError("R13 needs the exact accepted three-endpoint calibration binding")
    root = Path(spec["root"]).resolve()
    if (
        not root.is_relative_to(SERVER_ROOT)
        or spec["source_commit"] != CALIBRATION_SOURCE
        or spec["native_result_sha256"] != CALIBRATION_RESULT_SHA256
        or sha256_file(root / "CALIBRATION_COMPLETE.json") != CALIBRATION_RESULT_SHA256
    ):
        raise ValueError("R13 accepted calibration root/source/result hash differs")
    proof = verify_source_compatibility(source, formal_source_commit=formal_source_commit)
    stored = json.loads((root / "CALIBRATION_COMPLETE.json").read_text())
    result = verify_endpoints(
        root / "calibration/init",
        root / "calibration/ri_last",
        root / "calibration/r12bal_last",
        source_commit=CALIBRATION_SOURCE,
    )
    if (
        result != stored
        or result.get("calibration_complete") is not True
        or result.get("engineering_accepted") is not True
        or result.get("updates") != 0
        or result.get("scientific_complete") is not False
        or result.get("source_commit") != CALIBRATION_SOURCE
        or result.get("no_validation_or_test") is not True
        or result.get("selection", {}).get("selected_weight") != AMPLITUDE_WEIGHT
        or result.get("selection", {}).get("eligible_batches") != 8
    ):
        raise ValueError("R13 native accepted coefficient/result verification differs")
    return dict(
        root=str(root),
        native_result_sha256=CALIBRATION_RESULT_SHA256,
        selected_weight=AMPLITUDE_WEIGHT,
        calibration_source_commit=CALIBRATION_SOURCE,
        formal_source_commit=formal_source_commit,
        source_compatibility=proof,
        calibration_complete=True,
        engineering_accepted=True,
        scientific_complete=False,
    )


def spectral_protocol(arm):
    if arm not in ARMS:
        raise ValueError("R13 exact arm required")
    return dict(
        rank_target=0.20,
        rank_weight=0.02,
        cosine_weight=CHANGES[arm]["spectral_cosine_weight"],
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
        amplitude=dict(
            target_q=AMPLITUDE_TARGET,
            weight=AMPLITUDE_WEIGHT,
            formula="mean_globals_relu_sqrt_target_minus_sqrt_q_plus_epsilon",
            q="centered_population_covariance_trace_over_mean_raw_squared_norm",
            epsilon_q=1e-8,
            denominator_gradients=True,
            statistics_dtype="float64",
            near_zero_total_energy_rejection=1e-24,
            q_roundoff="64_D_eps64_tolerance_then_clamp_zero_one",
        ),
    )


AMPLITUDE_STEP_FIELDS = (
    "spectral_amplitude",
    "spectral_amplitude_weighted",
    "spectral_amplitude_target",
    "spectral_rank",
    "spectral_cosine",
    "spectral_rank_weighted",
    "spectral_cosine_weighted",
    "spectral_weighted",
    "spectral_warmup",
    "spectral_input_count",
    "spectral_trace_g0",
    "spectral_trace_g1",
    "spectral_q_g0",
    "spectral_q_g1",
    "spectral_total_energy_g0",
    "spectral_total_energy_g1",
    "spectral_mean_energy_g0",
    "spectral_mean_energy_g1",
)


def validate_amplitude_gate(gate, *, arm, resume_checkpoint_sha256):
    """Require actual ten-step energy/contribution/restore evidence for this arm."""
    import math

    checks = {
        "ten_native_updates",
        "exact_restore",
        "complete_population",
        "correct_warmup",
        "first_update_zero",
        "finite_losses",
        "amplitude_energy_and_contributions",
    }
    updates = gate.get("updates", [])
    if (
        arm not in ARMS
        or gate.get("passed") is not True
        or set(gate.get("checks", {})) != checks
        or not all(v is True for v in gate["checks"].values())
        or gate.get("resume_checkpoint_sha256") != resume_checkpoint_sha256
        or len(updates) != 10
    ):
        raise ValueError("R13 requires actual exact10 amplitude/warmup/restore proof")
    for i, row in enumerate(updates):
        if row.get("completed_before") != i or any(
            type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
            for k in AMPLITUDE_STEP_FIELDS
        ):
            raise ValueError("R13 complete finite amplitude step evidence missing")
        warmup = min(1.0, i / 30)
        if (
            row["spectral_input_count"] != 320
            or row["spectral_warmup"] != warmup
            or row["spectral_amplitude_target"] != AMPLITUDE_TARGET
        ):
            raise ValueError("R13 amplitude population/warmup/target differs")
        for raw, weighted, coefficient in (
            ("spectral_amplitude", "spectral_amplitude_weighted", AMPLITUDE_WEIGHT),
            ("spectral_rank", "spectral_rank_weighted", 0.02),
            ("spectral_cosine", "spectral_cosine_weighted", CHANGES[arm]["spectral_cosine_weight"]),
        ):
            if not math.isclose(
                row[weighted], warmup * coefficient * row[raw], rel_tol=1e-7, abs_tol=1e-12
            ):
                raise ValueError("R13 saved actual weighted contribution differs")
        expected_amp = 0.0
        for view in (0, 1):
            q, trace, total, mean = (
                row[f"spectral_{name}_g{view}"]
                for name in ("q", "trace", "total_energy", "mean_energy")
            )
            if (
                not 0 <= q <= 1
                or trace <= 1e-24
                or total <= 1e-24
                or mean < 0
                or not math.isclose(q, trace / total, rel_tol=1e-7, abs_tol=1e-12)
                or not math.isclose(total, trace + mean, rel_tol=1e-7, abs_tol=1e-12)
            ):
                raise ValueError("R13 raw amplitude energy decomposition differs")
            expected_amp += max(0.0, math.sqrt(AMPLITUDE_TARGET) - math.sqrt(q + 1e-8)) / 2
        if not math.isclose(
            row["spectral_amplitude"], expected_amp, abs_tol=1e-10
        ) or not math.isclose(
            row["spectral_weighted"],
            sum(
                row[k]
                for k in (
                    "spectral_rank_weighted",
                    "spectral_cosine_weighted",
                    "spectral_amplitude_weighted",
                )
            ),
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise ValueError("R13 raw floor or total actual contribution differs")
        if i == 0 and any(
            row[k] != 0
            for k in (
                "spectral_rank_weighted",
                "spectral_cosine_weighted",
                "spectral_amplitude_weighted",
                "spectral_weighted",
            )
        ):
            raise ValueError("R13 first successful-update auxiliary scale must be zero")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2") / FAMILY)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(source, source / args.output), indent=2))
