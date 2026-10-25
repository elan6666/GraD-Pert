"""One completed new-family arm: exact medium rows, parity, diagnostics and report."""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from cap40_posthoc import population_identity, read, sha, write
from checkpoint_endpoint import select_checkpoint
from generate_dino_pair import ARMS as DINO_ARMS
from generate_dino_pair import REMAINING_SCHEMA as DP2_SCHEMA
from generate_dino_pair import SCHEMA as DINO_SCHEMA
from generate_fgpo_family import ARMS, SCHEMA
from generate_no_ssl import ARMS as NO_SSL_ARMS
from generate_no_ssl import SCHEMA as NO_SSL_SCHEMA
from generate_pt1 import ARMS as PT1_ARMS
from generate_pt1 import SCHEMA as PT1_SCHEMA
from generate_pt2 import ARMS as PT2_ARMS
from generate_pt2 import SCHEMA as PT2_SCHEMA
from generate_representation_round1 import ARMS as RS_ARMS
from generate_representation_round1 import SCHEMA as RS_SCHEMA
from generate_representation_round2 import ARMS as RS2_ARMS
from generate_representation_round2 import SCHEMA as RS2_SCHEMA
from generate_representation_round3 import ARMS as RS3_ARMS
from generate_representation_round3 import SCHEMA as RS3_SCHEMA
from generate_representation_round4 import ARMS as RS4_ARMS
from generate_representation_round4 import SCHEMA as RS4_SCHEMA
from generate_representation_round5 import ARMS as RS5_ARMS
from generate_representation_round5 import SCHEMA as RS5_SCHEMA
from generate_representation_round6 import ARMS as RS6_ARMS
from generate_representation_round6 import SCHEMA as RS6_SCHEMA
from generate_representation_round7 import ARMS as RS7_ARMS
from generate_representation_round7 import SCHEMA as RS7_SCHEMA
from generate_representation_round8 import ARMS as RS8_ARMS
from generate_representation_round8 import SCHEMA as RS8_SCHEMA
from generate_representation_round9 import ARMS as RS9_ARMS
from generate_representation_round9 import SCHEMA as RS9_SCHEMA
from generate_representation_round11 import ARMS as RS11_ARMS
from generate_representation_round11 import SCHEMA as RS11_SCHEMA

FAMILY_ARMS = {
    RS9_SCHEMA: RS9_ARMS,
    RS11_SCHEMA: RS11_ARMS,
    RS_SCHEMA: RS_ARMS,
    RS2_SCHEMA: RS2_ARMS,
    RS3_SCHEMA: RS3_ARMS,
    RS4_SCHEMA: RS4_ARMS,
    RS5_SCHEMA: RS5_ARMS,
    RS6_SCHEMA: RS6_ARMS,
    RS7_SCHEMA: RS7_ARMS,
    RS8_SCHEMA: RS8_ARMS,
    SCHEMA: ARMS,
    DINO_SCHEMA: DINO_ARMS,
    DP2_SCHEMA: {"DP2": DINO_ARMS["DP2"]},
    NO_SSL_SCHEMA: NO_SSL_ARMS,
    PT1_SCHEMA: PT1_ARMS,
    PT2_SCHEMA: PT2_ARMS,
}
AUTHORIZED_ARMS = tuple(dict.fromkeys(name for arms in FAMILY_ARMS.values() for name in arms))

SERVER_ROOT = Path("/data/yilangliu")
COUNTS = dict(
    conditions=1335,
    rows=5317,
    deep_conditions=128,
    deep_rows=506,
    gradient_conditions=32,
    gradient_rows=125,
)


def validate_analysis_spec(spec):
    fields = {
        "parent_manifest",
        "parent_manifest_sha256",
        "performance_receipt",
        "performance_receipt_sha256",
    }
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError("medium analysis needs exact sampling and performance receipts")
    for name in ("parent_manifest", "performance_receipt"):
        path = Path(spec[name]).resolve()
        if not path.is_relative_to(SERVER_ROOT) or sha(path) != spec[name + "_sha256"]:
            raise ValueError("medium sampling/performance receipt changed or left server")
    parent = read(spec["parent_manifest"])
    if parent.get("analysis_scope") != "medium_cap40_stratified" or any(
        parent.get(k) != v for k, v in COUNTS.items()
    ):
        raise ValueError("only the original comparable medium population is authorized")
    from cap40_posthoc import reuse_medium_schedule

    reuse_medium_schedule(parent)
    performance = read(spec["performance_receipt"])
    parity = performance.get("full_parity", {})
    fast, reference = (
        performance.get(k, float("nan")) for k in ("full_fast_seconds", "full_reference_seconds")
    )
    if (
        performance.get("status") != "completed"
        or performance.get("same_sample_scope") is not True
        or performance.get("deterministic_algorithms") is not True
        or performance.get("model_updates") != 0
        or parity.get("passed") is not True
        or parity.get("atol") != 1e-8
        or not 0 <= parity.get("max_abs", float("inf")) <= 1e-8
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (fast, reference))
        or not 0 < fast < reference
    ):
        raise ValueError("full-medium correctness/stability and end-to-end gain are required")
    return parent


def medium_command(queue, row):
    return [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/run_fgpo_medium.py"),
        "--queue",
        str(Path(queue["output"]) / "queue.json"),
        "--arm",
        row["name"],
    ]


def validate_medium_complete(queue, row, *, root=None, checkpoint_role=None):
    if queue.get("family") == RS9_SCHEMA and checkpoint_role is None:
        from run_staged_joint_medium import validate_complete

        return validate_complete(queue, row, root=root)
    if queue.get("family") == PT2_SCHEMA and checkpoint_role is None:
        from run_pt2_medium import validate_complete

        return validate_complete(queue, row, root=root)
    root = Path(root) if root is not None else Path(queue["output"]) / f"{row['name']}-medium"
    path = root / "MEDIUM_COMPLETE.json"
    result = read(path)
    plan = row["plan"]
    journal = read(Path(plan["run_root"]) / "fit/epoch_state.json")
    role = checkpoint_role or "last"
    checkpoint = (
        select_checkpoint(Path(plan["run_root"]), role)[0]
        if queue.get("family") in (PT2_SCHEMA, RS9_SCHEMA)
        else journal[role]
    )
    expected = {
        "status": "completed",
        "arm": row["name"],
        "source_commit": queue["source_commit"],
        "queue_sha256": sha(Path(queue["output"]) / "queue.json"),
        "checkpoint_sha256": checkpoint["sha256"],
        "updates": 0,
        **COUNTS,
    }
    parent = validate_analysis_spec(queue["medium_analysis"])
    if queue.get("family") in (PT2_SCHEMA, RS9_SCHEMA):
        _, endpoint = select_checkpoint(Path(plan["run_root"]), role)
        expected["checkpoint_endpoint"] = endpoint
    expected["diagnostic_population_sha256"] = population_identity(parent["schedule"])
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("completed medium arm identity, population or counts differ")
    summary = read(root / "analysis/summary.json")
    complete = read(root / "analysis/COMPLETE.json")
    if (
        complete.get("state_unchanged") is not True
        or summary.get("updates") != 0
        or summary.get("state_before") != summary.get("state_after")
        or summary.get("method_sha") != queue["source_commit"]
        or summary.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or (
            queue.get("family") in (PT2_SCHEMA, RS9_SCHEMA)
            and (
                summary.get("checkpoint_role") != endpoint["role"]
                or summary.get("checkpoint_epoch") != endpoint["epoch"]
                or summary.get("training_stage") != endpoint["training_stage"]
            )
        )
        or summary.get("config_sha256") != plan["config_sha256"]
        or summary.get("diagnostic_population_sha256") != expected["diagnostic_population_sha256"]
        or any(summary.get(k) != v for k, v in COUNTS.items())
        or result.get("parity", {}).get("passed") is not True
        or result["parity"].get("atol") != 1e-8
        or not 0 <= result["parity"].get("max_abs", float("inf")) <= 1e-8
    ):
        raise ValueError("medium analysis state, method, population or parity changed")
    for name, entry in result["outputs"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target) != entry["sha256"]:
            raise ValueError("sealed medium artifact changed")
    return {"receipt": str(path), "sha256": sha(path), **COUNTS}


def validate_report_repair(q, arm, output, report_source):
    """Postfit report repair; freeze numerical code and the parent queue."""
    if (arm, q.get("family")) not in (
        ("D0", NO_SSL_SCHEMA),
        ("PT1", PT1_SCHEMA),
        ("PT2", PT2_SCHEMA),
        ("CG1", SCHEMA),
        ("U24", SCHEMA),
        ("R3-RI", RS3_SCHEMA),
        ("R3-AUB", RS3_SCHEMA),
        ("R3-AUS", RS3_SCHEMA),
    ):
        raise ValueError(
            "analysis-only report repair is scoped to D0, PT1, PT2, CG1, U24 or R3 RI/AU"
        )
    output = Path(output).resolve()
    protected = [Path(q["output"]).resolve()] + [
        Path(row["plan"]["run_root"]).resolve() for row in q["rows"]
    ]
    if not output.is_relative_to(SERVER_ROOT) or any(
        output.is_relative_to(root) for root in protected
    ):
        raise ValueError("repair output must be a new independent server root")
    paths = [
        "src",
        "scripts/v2/cap40_posthoc*.py",
        "scripts/v2/profile_medium_observer.py",
        "scripts/v2/shared_gpu.py",
        "scripts/v2/run_br400_group.py",
        "scripts/v2/generate_no_ssl.py",
        "scripts/v2/generate_pt1.py",
        "scripts/v2/generate_pt2.py",
        "scripts/v2/generate_fgpo_family.py",
    ]
    dirty = subprocess.check_output(
        ["git", "-C", str(report_source), "status", "--porcelain"], text=True
    ).strip()
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(report_source),
            "diff",
            "--name-only",
            q["source_commit"],
            "HEAD",
            "--",
            *paths,
        ],
        text=True,
    ).splitlines()
    if arm in ("R3-AUB", "R3-AUS") and "scripts/v2/cap40_posthoc.py" in changed:
        original = subprocess.check_output(
            [
                "git",
                "-C",
                str(report_source),
                "show",
                q["source_commit"] + ":scripts/v2/cap40_posthoc.py",
            ],
            text=True,
        )
        before = "else dataclasses.replace(runtime.options, lambda1=0.0, lambda2=0.0)"
        after = (
            "else dataclasses.replace(\n"
            "            runtime.options,\n"
            "            lambda1=0.0,\n"
            "            lambda2=0.0,\n"
            "            **(\n"
            '                {"au_alignment_weight": 0.0, "au_uniformity_weight": 0.0}\n'
            '                if getattr(runtime.options, "au_alignment_weight", 0.0)\n'
            '                or getattr(runtime.options, "au_uniformity_weight", 0.0)\n'
            "                else {}\n"
            "            ),\n"
            '            **({"cr_weight": 0.0} '
            'if getattr(runtime.options, "cr_weight", 0.0) else {}),\n'
            "        )"
        )
        if original.count(before) != 1 or (
            report_source / "scripts/v2/cap40_posthoc.py"
        ).read_text() != original.replace(before, after):
            raise ValueError("only the declared shallow AU assembly correction is allowed")
        changed.remove("scripts/v2/cap40_posthoc.py")
    if arm == "PT2" and "scripts/v2/cap40_posthoc.py" in changed:
        original = subprocess.check_output(
            [
                "git",
                "-C",
                str(report_source),
                "show",
                q["source_commit"] + ":scripts/v2/cap40_posthoc.py",
            ],
            text=True,
        )
        before = '\n                if role == "student"\n                else []'
        after = (
            '\n                if role == "student" and obj.prediction_enabled'
            "\n                else []"
        )
        if original.count(before) != 1 or (
            report_source / "scripts/v2/cap40_posthoc.py"
        ).read_text() != original.replace(before, after):
            raise ValueError("only the declared SSL-only observer layout correction is allowed")
        changed.remove("scripts/v2/cap40_posthoc.py")
    if dirty or changed:
        raise ValueError("repair source is dirty or numerical method/analysis code changed")
    return output


def cg1_parent_binding(parent, training_identity, *, parent_path, parent_sha256):
    """Bind CG1's recorded SSL1 scope without changing the diagnostic population."""
    trained = training_identity["data"]
    comparable = copy.deepcopy(trained)
    scope = comparable["loss_protocol"].pop("conditional_ssl1_scope", None)
    if (
        scope != "local_control_rows_global_condition_weights_masked_node_mean"
        or comparable != parent["data"]
    ):
        raise ValueError("CG1 medium parent may differ only by declared conditional_ssl1_scope")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(trained),
        "training_protocol_binding": {
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "changed_field": "data.loss_protocol.conditional_ssl1_scope",
            "previous_value": None,
            "training_value": scope,
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def response_parent_binding(parent, training_identity, *, arm, parent_path, parent_sha256):
    """Bind only preregistered U24/MR1/P1 metadata; preserve the entire population."""
    trained = training_identity["data"]
    expected = copy.deepcopy(parent["data"])
    if arm == "U24":
        if expected["unseen_gene_mechanisms"]["gene_conditioned_readout"] is not False:
            raise ValueError("U24 needs the original unconditioned readout parent")
        expected["unseen_gene_mechanisms"]["gene_conditioned_readout"] = True
    elif arm in ("MR1", "P1"):
        protocol = expected["loss_protocol"]
        if "new_response_supervision" in protocol or "population_scope" in protocol:
            raise ValueError("response binding requires the original joint parent")
        protocol["new_response_supervision"] = {
            "population_response": arm == "P1",
            "lambda_mmd": 1.0,
            "mmd_bandwidth": "detached_training_truth_median_offdiagonal",
            "masked_response_ratio": 0.25 if arm == "MR1" else 0.0,
            "lambda_masked_response": 1.0 if arm == "MR1" else 0.0,
            "masked_raw_residual": "blocked",
            "epoch_budget": "each_selected_training_truth_once",
        }
        if arm == "P1":
            if protocol["batch_order"] != "random_mixed_seeded_v2":
                raise ValueError("P1 needs the original mixed batch parent")
            protocol["batch_order"] = "same_condition_population_train_rows_once"
            protocol["population_scope"] = "complete_effective_batch_all_microbatches_and_ranks"
    else:
        raise ValueError("response parent binding is scoped to U24, MR1 and P1")
    if trained != expected:
        raise ValueError("medium parent differs beyond declared " + arm + " metadata")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(trained),
        "training_protocol_binding": {
            "arm": arm,
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def au_parent_binding(parent, training_identity, *, arm, config, parent_path, parent_sha256):
    """Bind the frozen R3 AU protocol without changing any diagnostic sampling."""
    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    weights = {"R3-AUB": (0.05, 0.005), "R3-AUS": (0.1, 0.01)}
    if arm not in weights:
        raise ValueError("AU parent binding is scoped to R3-AUB and R3-AUS")
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    if (
        (options.au_alignment_weight, options.au_uniformity_weight) != weights[arm]
        or options.au_temperature != 2.0
        or options.au_warmup_steps != 30
    ):
        raise ValueError("AU parent binding requires the frozen arm coefficients and schedule")
    protocol = {
        "alignment_weight": options.au_alignment_weight,
        "uniformity_weight": options.au_uniformity_weight,
        "temperature": options.au_temperature,
        "warmup_completed_optimizer_updates": options.au_warmup_steps,
        "representation": "student_response_cls_after_final_rmsnorm_before_projector",
        "population_scope": "complete_effective_batch_all_microbatches_and_ranks",
        "deduplication": "control_row_id_and_condition_id_before_rank_partition",
        "alignment": "same_control_and_condition_two_global_views_both_student_gradients",
        "uniformity_pairs": "unordered_same_cell_type_batch_context_different_condition",
        "reduction": "log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
        "partial_diagnostics": "unavailable_requires_complete_global_population",
    }
    expected = copy.deepcopy(parent["data"])
    if "alignment_uniformity" in expected["loss_protocol"]:
        raise ValueError("AU binding requires the original parent without AU metadata")
    expected["loss_protocol"]["alignment_uniformity"] = protocol
    if training_identity["data"] != expected:
        raise ValueError("AU medium parent differs beyond the frozen alignment_uniformity protocol")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(training_identity["data"]),
        "training_protocol_binding": {
            "arm": arm,
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "changed_field": "data.loss_protocol.alignment_uniformity",
            "previous_value": None,
            "training_value": protocol,
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def cr_parent_binding(parent, training_identity, *, arm, config, parent_path, parent_sha256):
    """Bind the frozen population auxiliary protocol and its exact EMA variant."""
    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    if arm not in ("R4-CR", "R5-PROJ", "R6-EMA99", "R7-S1D05", "R8-CR0"):
        raise ValueError("CR parent binding is scoped to frozen R4/R5/R6/R7/R8 arms")
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    if (
        (options.au_alignment_weight, options.au_uniformity_weight) != (0.05, 0.0)
        or options.au_temperature != 2.0
        or options.au_warmup_steps != 30
        or options.cr_weight != (0.0 if arm == "R8-CR0" else 0.00004)
        or options.cr_epsilon_squared != 0.05
        or options.cr_warmup_steps != 30
    ):
        raise ValueError("CR parent binding requires the frozen arm coefficients and schedule")
    if arm in ("R6-EMA99", "R7-S1D05", "R8-CR0") and (
        options.teacher_start,
        options.teacher_end,
    ) != (
        0.99,
        0.99,
    ):
        raise ValueError("R6/R7/R8 parent binding requires constant .99 Teacher EMA")
    if arm == "R7-S1D05" and (options.lambda1, options.lambda2) != (0.05, 1.0):
        raise ValueError("R7 parent binding requires lambda1=.05 and locked lambda2=1")
    if arm == "R8-CR0" and (options.lambda1, options.lambda2) != (1.0, 1.0):
        raise ValueError("R8 parent binding requires lambda1=1 and locked lambda2=1")
    expected_space = (
        "ssl2_cls_bottleneck"
        if arm in ("R5-PROJ", "R6-EMA99", "R7-S1D05", "R8-CR0")
        else "response_cls"
    )
    if options.population_aux_space != expected_space:
        raise ValueError("CR parent binding auxiliary representation space differs")
    representation = (
        "student_ssl2_cls_features_before_l2_and_prototypes"
        if arm in ("R5-PROJ", "R6-EMA99", "R7-S1D05", "R8-CR0")
        else "student_response_cls_after_final_rmsnorm_before_projector"
    )
    protocols = {
        "alignment_uniformity": {
            "alignment_weight": options.au_alignment_weight,
            "uniformity_weight": options.au_uniformity_weight,
            "temperature": options.au_temperature,
            "warmup_completed_optimizer_updates": options.au_warmup_steps,
            "representation": representation,
            "population_scope": "complete_effective_batch_all_microbatches_and_ranks",
            "deduplication": "control_row_id_and_condition_id_before_rank_partition",
            "alignment": "same_control_and_condition_two_global_views_both_student_gradients",
            "uniformity_pairs": "unordered_same_cell_type_batch_context_different_condition",
            "reduction": "log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
            "partial_diagnostics": "unavailable_requires_complete_global_population",
        },
        "coding_rate": {
            "weight": options.cr_weight,
            "epsilon_squared": options.cr_epsilon_squared,
            "warmup_completed_optimizer_updates": options.cr_warmup_steps,
            "representation": representation,
            "population_scope": "complete_effective_batch_all_microbatches_and_ranks_all_contexts",
            "deduplication": "control_row_id_and_condition_id_before_rank_partition",
            "formula": "mean_two_globals_negative_half_logdet_I_plus_d_over_N_epsilon_squared_UtU",
            "normalization": "separate_global_unit_rows_uncentered_fp64_cholesky",
            "partial_diagnostics": "unavailable_requires_complete_global_population",
        },
    }
    if arm == "R8-CR0":
        del protocols["coding_rate"]  # Native disabled CR has no runtime protocol entry.
    expected = copy.deepcopy(parent["data"])
    if set(protocols).intersection(expected["loss_protocol"]):
        raise ValueError("CR binding requires the original parent without AU or CR metadata")
    expected["loss_protocol"].update(protocols)
    if training_identity["data"] != expected:
        raise ValueError("CR medium parent differs beyond the frozen AU and coding_rate protocols")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(training_identity["data"]),
        "training_protocol_binding": {
            "arm": arm,
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "changed_fields": ["data.loss_protocol." + key for key in protocols],
            "previous_values": dict.fromkeys(protocols),
            "training_values": protocols,
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def staged_joint_parent_binding(
    parent, training_identity, *, arm, config, parent_path, parent_sha256
):
    """Add only sealed R9 phase metadata to the original R6 population binding."""
    import yaml
    from generate_representation_round9 import configs

    if (
        arm != "R9-SSL1J2"
        or yaml.safe_load(Path(config).read_text())
        != configs(Path(__file__).resolve().parents[2])[arm]
    ):
        raise ValueError("staged-joint binding requires the exact R9 configuration")
    comparable = copy.deepcopy(training_identity)
    staged = comparable["data"]["loss_protocol"].pop("staged_training", None)
    expected = dict(
        mode="ssl_then_joint",
        pretrain_epochs=1,
        pretrain_steps=150,
        joint_epochs=2,
        joint_steps=300,
        total_steps=450,
        pretrain_truth="not_loaded_nan_sentinel",
        transition="student_teacher_centers_optimizer_moments_retained",
        joint_teacher_centers="continuously_updated",
        learning_rate_clocks="separate_prefix_and_joint_endpoint_cosines",
        auxiliary_warmup_clock="global_successful_updates",
        sampler_epochs="prefix_local_then_joint_local",
        view_rng="continuous_checkpointed_not_baseline_replay",
    )
    if staged != expected:
        raise ValueError("R9 staged protocol or budget changed")
    bound = cr_parent_binding(
        parent,
        comparable,
        arm="R6-EMA99",
        config=config,
        parent_path=parent_path,
        parent_sha256=parent_sha256,
    )
    bound["data"] = copy.deepcopy(training_identity["data"])
    binding = bound["training_protocol_binding"]
    binding["arm"] = arm
    binding["changed_fields"].append("data.loss_protocol.staged_training")
    binding["previous_values"]["staged_training"] = None
    binding["training_values"]["staged_training"] = expected
    return bound


def pt1_parent_binding(parent, training_identity, *, parent_path, parent_sha256):
    """Adapt only PT1's declared phase metadata, retaining every sampling field.

    Native sealing still compares the complete live runtime identity with the
    checkpoint identity. This does not relax that check or change numerical code.
    """
    trained = training_identity["data"]
    comparable = copy.deepcopy(trained)
    protocol = comparable["loss_protocol"]
    staged = protocol.pop("staged_training", None)
    if (
        not isinstance(staged, dict)
        or staged.get("mode") != "ssl_then_mse"
        or comparable != parent["data"]
    ):
        raise ValueError("PT1 medium parent may differ only by declared staged_training metadata")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(trained),
        "training_protocol_binding": {
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "changed_field": "data.loss_protocol.staged_training",
            "previous_value": None,
            "training_value": copy.deepcopy(staged),
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def frozen_readout_parent_binding(
    parent, training_identity, *, arm, config, parent_path, parent_sha256
):
    """Preserve every physical data/population field; declare inactive child losses."""
    import yaml
    from generate_representation_round11 import configs

    if (
        arm != "R11-READOUT"
        or yaml.safe_load(Path(config).read_text())
        != configs(Path(__file__).resolve().parents[2])[arm]
    ):
        raise ValueError("R11 requires its sealed frozen readout config")
    trained = training_identity["data"]
    if {k: v for k, v in parent["data"].items() if k != "loss_protocol"} != {
        k: v for k, v in trained.items() if k != "loss_protocol"
    }:
        raise ValueError("R11 physical medium data/population changed")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(trained),
        "training_protocol_binding": dict(
            arm=arm,
            original_parent_manifest=str(parent_path),
            original_parent_sha256=parent_sha256,
            changed_fields=["data.loss_protocol"],
            previous_values=parent["data"]["loss_protocol"],
            training_values=trained["loss_protocol"],
            training_run_id=training_identity["run_id"],
            diagnostic_population_sha256=population_identity(parent["schedule"]),
            sampling_changed=False,
            teacher_ssl_status=(
                "inactive; unchanged R9 reference only with frozen-state digest proof"
            ),
        ),
    }


def execute(
    queue_path,
    arm,
    *,
    repair_output=None,
    checkpoint_role="last",
    endpoint_root=None,
    shared_manifest=None,
):
    from profile_medium_observer import compare
    from run_br400_group import (
        RESEARCH_FAMILIES,
        validate_complete,
        validate_training_contract,
        verify,
    )

    q = read(queue_path)
    if q.get("family") == RS9_SCHEMA and endpoint_root is None:
        if checkpoint_role != "last" or shared_manifest is not None or repair_output is not None:
            raise ValueError("R9 requires both sealed endpoints")
        from run_staged_joint_medium import execute as execute_staged_joint

        return execute_staged_joint(queue_path, arm)
    if q.get("family") == PT2_SCHEMA and endpoint_root is None:
        if checkpoint_role != "last" or shared_manifest is not None:
            raise ValueError("PT2 dual-endpoint scope cannot be replaced by a single endpoint")
        from run_pt2_medium import execute as execute_pt2

        return execute_pt2(queue_path, arm, repair_output=repair_output)
    if (
        checkpoint_role != "last" or endpoint_root is not None or shared_manifest is not None
    ) and q.get("family") not in (PT2_SCHEMA, RS9_SCHEMA):
        raise ValueError("additional diagnostic endpoints are scoped to PT2")
    if arm not in FAMILY_ARMS.get(q.get("family"), {}):
        raise ValueError("medium postfit work is limited to the authorized new family")
    verify(q)
    source = Path(q["source"])
    report_source = Path(__file__).resolve().parents[2]
    if repair_output is None and source.resolve() != report_source:
        raise ValueError("medium worker must use the exact published queue source")
    if repair_output is not None:
        validate_report_repair(q, arm, repair_output, report_source)
    parent = validate_analysis_spec(q["medium_analysis"])
    row = next(r for r in q["rows"] if r["name"] == arm)
    research = q.get("family") in RESEARCH_FAMILIES
    if research:
        validate_training_contract(row["plan"])
    else:
        validate_complete(row["plan"])
    root = (
        Path(endpoint_root)
        if endpoint_root is not None
        else (
            validate_report_repair(q, arm, repair_output, report_source)
            if repair_output is not None
            else Path(q["output"]) / f"{arm}-medium"
        )
    )
    if root.exists() or not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server-only medium run required")
    root.mkdir()
    start = time.perf_counter()
    plan = row["plan"]
    checkpoint, endpoint = select_checkpoint(Path(plan["run_root"]), checkpoint_role)
    env = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    stages = {}

    def call(label, command, gpu):
        begin = time.perf_counter()
        with (root / (label + ".log")).open("x") as log:
            subprocess.run(
                command,
                cwd=source,
                env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        stages[label] = time.perf_counter() - begin

    # A repair changes endpoint metadata only. Native imports stay on the frozen
    # training source via PYTHONPATH; the analysis checkout is separately recorded.
    analysis_source = report_source if repair_output is not None else source
    base = [sys.executable, str(analysis_source / "scripts/v2/cap40_posthoc.py")]
    common = [
        "--run-root",
        plan["run_root"],
        "--config",
        plan["config"],
        "--data-root",
        plan["data_root"],
        "--method-sha",
        q["source_commit"],
        "--checkpoint-role",
        checkpoint_role,
    ]
    try:
        if not research:
            call(
                "saved-training-report",
                [
                    sys.executable,
                    str(report_source / "scripts/v2/analyze_completed_ablation.py"),
                    "--run-root",
                    plan["run_root"],
                    "--output",
                    str(root / "saved"),
                    "--name",
                    arm,
                    "--checkpoint-role",
                    checkpoint_role,
                ],
                "",
            )
        parent_path = q["medium_analysis"]["parent_manifest"]
        parent_sha256 = q["medium_analysis"]["parent_manifest_sha256"]
        if arm in (
            "PT1",
            "PT2",
            "CG1",
            "U24",
            "MR1",
            "P1",
            "R3-AUB",
            "R3-AUS",
            "R4-CR",
            "R5-PROJ",
            "R6-EMA99",
            "R7-S1D05",
            "R8-CR0",
            "R9-SSL1J2",
            "R11-READOUT",
        ):
            bind_parent = (
                frozen_readout_parent_binding
                if arm == "R11-READOUT"
                else staged_joint_parent_binding
                if arm == "R9-SSL1J2"
                else cr_parent_binding
                if arm in ("R4-CR", "R5-PROJ", "R6-EMA99", "R7-S1D05", "R8-CR0")
                else au_parent_binding
                if arm in ("R3-AUB", "R3-AUS")
                else (
                    response_parent_binding
                    if arm in ("U24", "MR1", "P1")
                    else (cg1_parent_binding if arm == "CG1" else pt1_parent_binding)
                )
            )
            bound_parent = bind_parent(
                parent,
                read(Path(plan["run_root"]) / "run_manifest.json"),
                parent_path=parent_path,
                parent_sha256=parent_sha256,
                **(
                    {"arm": arm, "config": plan["config"]}
                    if arm
                    in (
                        "R3-AUB",
                        "R3-AUS",
                        "R4-CR",
                        "R5-PROJ",
                        "R6-EMA99",
                        "R7-S1D05",
                        "R8-CR0",
                        "R9-SSL1J2",
                        "R11-READOUT",
                    )
                    else ({"arm": arm} if arm in ("U24", "MR1", "P1") else {})
                ),
            )
            parent_path = root / f"{arm.lower()}-medium-parent.json"
            write(parent_path, bound_parent)
            parent_sha256 = sha(parent_path)
        if shared_manifest is None:
            call(
                "seal",
                [
                    *base,
                    "seal-medium",
                    *common,
                    "--reuse-medium-schedule",
                    "--parent-manifest",
                    str(parent_path),
                    "--parent-manifest-sha256",
                    parent_sha256,
                    "--output",
                    str(root / "manifest"),
                ],
                "",
            )
        manifest = (
            Path(shared_manifest)
            if shared_manifest is not None
            else root / "manifest/manifest.json"
        )
        derived = read(manifest)
        if (
            population_identity(derived["schedule"]) != population_identity(parent["schedule"])
            or derived["method_sha"] != q["source_commit"]
            or derived["view_recipe_config_sha256"] != plan["config_sha256"]
            or not manifest.resolve().is_relative_to(
                root.parent.resolve() if endpoint_root is not None else root.resolve()
            )
        ):
            raise ValueError("resealing changed the common diagnostic population")
        command = [
            *base,
            "run",
            *common,
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            sha(manifest),
            "--resource-queue",
            str(queue_path),
            "--arm",
            arm,
            "--deterministic",
            "--checkpoint-conditions",
            "100",
        ]
        for label, backend in (("reference-probe", "reference"), ("fast-probe", "fast")):
            call(
                label,
                [
                    *command,
                    "--output",
                    str(root / label),
                    "--stats-backend",
                    backend,
                    "--limit",
                    "3",
                    "--gate-indices",
                    "0,1,15",
                ],
                "0",
            )
        parity = compare(root / "reference-probe", root / "fast-probe")
        call(
            "medium", [*command, "--output", str(root / "analysis"), "--stats-backend", "fast"], "0"
        )
        summary = read(root / "analysis/summary.json")
        if any(summary.get(k) != v for k, v in COUNTS.items()):
            raise ValueError("medium output coverage incomplete")
        if research:
            from run_representation_postfit import render_medium_report

            render_medium_report(q, row, root)
        else:
            call(
                "report",
                [
                    sys.executable,
                    str(source / "scripts/v2/cap40_posthoc_report.py"),
                    "--input",
                    str(root / "analysis"),
                    "--original",
                    str(root / "saved"),
                    "--output",
                    str(root / "report"),
                ],
                "",
            )
        write(
            root / "MEDIUM_COMPLETE.json",
            {
                "status": "completed",
                "arm": arm,
                "source_commit": q["source_commit"],
                "report_controller_commit": subprocess.check_output(
                    ["git", "-C", str(report_source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "parent_failed_output_preserved": repair_output is not None,
                "queue_sha256": sha(queue_path),
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_endpoint": endpoint,
                "diagnostic_population_sha256": population_identity(parent["schedule"]),
                "manifest_sha256": sha(manifest),
                "manifest": str(manifest),
                "updates": 0,
                **COUNTS,
                "parity": parity,
                "end_to_end_seconds": time.perf_counter() - start,
                "stage_seconds": stages,
                "outputs": {
                    str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                    and p.suffix in (".json", ".csv", ".png", ".md")
                    and "recovery" not in p.parts
                },
            },
        )
        validate_medium_complete(q, row, root=root, checkpoint_role=checkpoint_role)
    except BaseException as exc:
        write(
            root / "FAILURE.json",
            {
                "status": "failed",
                "error": repr(exc),
                "queue_sha256": sha(queue_path),
                "stage_seconds": stages,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--arm", choices=AUTHORIZED_ARMS, required=True)
    parser.add_argument("--repair-output", type=Path)
    arguments = parser.parse_args()
    execute(arguments.queue, arguments.arm, repair_output=arguments.repair_output)
