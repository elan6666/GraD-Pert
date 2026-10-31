"""All gates first, then sequential dual-GPU fit and native best/last tests."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from collect_results import collect_run
from generate_br400_group import (
    REMAINING_SCHEMA,
)
from generate_br400_group import (
    SCHEMA as FAMILY_SCHEMA,
)
from generate_br400_group import (
    verify_manifest as verify_br_manifest,
)
from generate_dino_pair import REMAINING_SCHEMA as DP2_SCHEMA
from generate_dino_pair import SCHEMA as DINO_SCHEMA
from generate_dino_pair import verify_manifest as verify_dino_manifest
from generate_ffn_five import SCHEMA as FFN_SCHEMA
from generate_ffn_five import verify_manifest as verify_ffn_manifest
from generate_fgpo_family import SCHEMA as FGPO_SCHEMA
from generate_fgpo_family import verify_manifest as verify_fgpo_manifest
from generate_no_ssl import SCHEMA as NO_SSL_SCHEMA
from generate_no_ssl import verify_manifest as verify_no_ssl_manifest
from generate_precision_family import CHANGES as PRECISION_ARMS
from generate_precision_family import SCHEMA as PRECISION_SCHEMA
from generate_precision_family import verify_manifest as verify_precision_manifest
from generate_pt1 import SCHEMA as PT1_SCHEMA
from generate_pt1 import verify_manifest as verify_pt1_manifest
from generate_pt2 import SCHEMA as PT2_SCHEMA
from generate_pt2 import verify_manifest as verify_pt2_manifest
from generate_representation_round1 import SCHEMA as RS_SCHEMA
from generate_representation_round1 import verify_manifest as verify_rs_manifest
from generate_representation_round2 import SCHEMA as RS2_SCHEMA
from generate_representation_round2 import verify_manifest as verify_rs2_manifest
from generate_representation_round3 import SCHEMA as RS3_SCHEMA
from generate_representation_round3 import verify_manifest as verify_rs3_manifest
from generate_representation_round4 import SCHEMA as RS4_SCHEMA
from generate_representation_round4 import verify_manifest as verify_rs4_manifest
from generate_representation_round5 import SCHEMA as RS5_SCHEMA
from generate_representation_round5 import verify_manifest as verify_rs5_manifest
from generate_representation_round6 import SCHEMA as RS6_SCHEMA
from generate_representation_round6 import verify_manifest as verify_rs6_manifest
from generate_representation_round7 import SCHEMA as RS7_SCHEMA
from generate_representation_round7 import verify_manifest as verify_rs7_manifest
from generate_representation_round8 import SCHEMA as RS8_SCHEMA
from generate_representation_round8 import verify_manifest as verify_rs8_manifest
from generate_representation_round9 import SCHEMA as RS9_SCHEMA
from generate_representation_round9 import verify_manifest as verify_rs9_manifest
from generate_representation_round11 import SCHEMA as RS11_SCHEMA
from generate_representation_round11 import verify_manifest as verify_rs11_manifest
from generate_representation_round12 import SCHEMA as RS12_SCHEMA
from generate_representation_round12 import verify_manifest as verify_rs12_manifest
from generate_representation_round13 import SCHEMA as RS13_SCHEMA
from generate_representation_round13 import verify_manifest as verify_rs13_manifest
from run_group import lock, validate_preflight
from run_unified_ddp import _capture_process_identity, probe
from run_unified_group import ALLOCATOR, LEASE_ROOT, METRICS, child_environment
from shared_gpu import (
    DYNAMIC_SCHEMA,
    admit,
    descendant,
    parent_map,
    process_info,
    snapshot,
    terminate_tree,
    validate_policy,
)

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.execution.v2_deferred_postfit import finalize_deferred_run
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file, sha256_json

RESEARCH_FAMILIES = (
    PRECISION_SCHEMA,
    RS_SCHEMA,
    RS2_SCHEMA,
    RS3_SCHEMA,
    RS4_SCHEMA,
    RS5_SCHEMA,
    RS6_SCHEMA,
    RS7_SCHEMA,
    RS8_SCHEMA,
    RS9_SCHEMA,
    RS11_SCHEMA,
    RS12_SCHEMA,
    RS13_SCHEMA,
)
SERVER_ROOT = Path("/data/yilangliu")
POLL_SECONDS = 5
FGPO_FAMILIES = (
    FGPO_SCHEMA,
    DINO_SCHEMA,
    DP2_SCHEMA,
    NO_SSL_SCHEMA,
    PT1_SCHEMA,
    PT2_SCHEMA,
    *RESEARCH_FAMILIES,
)
LAST_ONLY_FAMILIES = (REMAINING_SCHEMA, FFN_SCHEMA, *FGPO_FAMILIES)
PREFLIGHT_STAGE_FAMILIES = (RS13_SCHEMA, PRECISION_SCHEMA)
PRECISION_POSTFIT = "main_owned_precision_postfit"
PRECISION_BUDGET = {"epochs": 2, "updates_per_epoch": 150, "completed_updates": 300}
DEFERRED_SCDFM_POLICY = "scdfm_fit_gears_terminal_evaluation_deferred_v1"
PENDING_FGPO_ARMS = ("S12-L4", "CG1", "U24", "MR1", "P1", "C1", "O1", "K2-fixed-prefix")
R3_CONTINUATION_ARMS = ("R3-AUB", "R3-AUS")


def selected_matrix(matrix: dict, only_arms: list[str] | None) -> dict:
    """Select an ordered pending scope without changing the sealed family configs."""
    if only_arms is None:
        return matrix
    if matrix["schema"] == RS3_SCHEMA:
        if only_arms not in (list(R3_CONTINUATION_ARMS), ["R3-AUS"]):
            raise ValueError("R3 continuation requires ordered R3-AUB, R3-AUS or singleton R3-AUS")
    elif matrix["schema"] == RS13_SCHEMA:
        if only_arms != ["R13-SRAMPNC"]:
            raise ValueError("R13 continuation permits only the untrained SRAMPNC singleton")
    elif (
        matrix["schema"] != FGPO_SCHEMA
        or not only_arms
        or only_arms != [name for name in PENDING_FGPO_ARMS if name in only_arms]
    ):
        raise ValueError("selection requires ordered, unique, pending FGPO arms")
    selected = copy.deepcopy(matrix)
    selected["rows"] = [row for row in matrix["rows"] if row["name"] in only_arms]
    selected["priority"] = list(only_arms)
    selected["preflight_priority"] = [
        name for name in matrix["preflight_priority"] if name in only_arms
    ]
    if [row["name"] for row in selected["rows"]] != only_arms:
        raise ValueError("selected arms are missing from the sealed family")
    return selected


def deferred_official_dependency_ready(spec: dict, queue: dict) -> bool:
    """Honor an explicit user deferral without inventing a scDFM test result."""
    from benchmarks.expression_holdout.config import load_config, validate_terminal

    if spec.get("authorization_reference") != "user:2026-10-14-native-five-scdfm-eval-deferred":
        raise ValueError("deferred scDFM evaluation needs its explicit authorization")
    rows = {row["name"]: row for row in queue["rows"]}
    if set(rows) != {"scdfm", "gears"}:
        raise ValueError("deferred dependency still requires scDFM and GEARS evidence")
    for dependency in rows.values():
        for key in ("config", "runtime"):
            if sha256_file(Path(dependency[key])) != dependency[key + "_sha256"]:
                raise ValueError("preserved official inputs changed")
    row = rows["scdfm"]
    root = Path(row["run_root"])
    if not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("preserved scDFM checkpoint must stay on server")
    proof = spec["scdfm_fit"]
    if proof.get("evaluation_status") != "deferred_by_user":
        raise ValueError("scDFM evaluation must be explicitly marked deferred")
    for filename, key in (
        ("RUN.json", "run_receipt_sha256"),
        ("fit-receipt.json", "fit_receipt_sha256"),
    ):
        if sha256_file(root / filename) != proof[key]:
            raise ValueError("preserved scDFM training receipts changed")
    identity, fit = read_json(root / "RUN.json"), read_json(root / "fit-receipt.json")
    config = load_config(Path(row["config"]))
    expected = {
        "source_commit": row.get("scientific_source_commit", queue["source_commit"]),
        "source_clean": True,
        "config_sha256": row["config_sha256"],
        "runtime_sha256": row["runtime_sha256"],
        "official_commit": config["source_code"]["commit"],
        "checkpoint_role": "last",
        **config["evaluation"]["expected_hashes"],
    }
    if any(record.get(k) != v for record in (identity, fit) for k, v in expected.items()):
        raise ValueError("preserved scDFM source/config/evaluation identity differs")
    if (
        fit.get("training_complete") is not True
        or fit.get("epochs_completed") != config["training"]["epochs"]
        or fit.get("completed_updates") != fit.get("total_optimizer_updates")
        or not isinstance(fit.get("completed_updates"), int)
        or fit["completed_updates"] <= 0
        or Path(fit["checkpoint"]).resolve() != (root / "fit/last.pt").resolve()
        or sha256_file(root / "fit/last.pt") != proof["checkpoint_sha256"]
    ):
        raise ValueError("preserved scDFM training/checkpoint is incomplete or changed")
    gears = rows["gears"]
    gears_root = Path(gears["run_root"])
    if not gears_root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("GEARS dependency must stay on server")
    if not (gears_root / "COMPLETE.json").exists():
        return False
    terminal = validate_terminal(
        gears_root, gears, gears.get("scientific_source_commit", queue["source_commit"])
    )
    if (
        sha256_file(gears_root / "receipt.json") != spec["gears_receipt_sha256"]
        or terminal["last_test_sha256"] != spec["gears_last_test_sha256"]
    ):
        raise ValueError("GEARS terminal dependency changed")
    return True


def verify_manifest(source: Path, manifest: Path) -> dict:
    schema = read_json(manifest).get("schema")
    verifier = {
        PRECISION_SCHEMA: verify_precision_manifest,
        RS6_SCHEMA: verify_rs6_manifest,
        RS7_SCHEMA: verify_rs7_manifest,
        RS8_SCHEMA: verify_rs8_manifest,
        RS9_SCHEMA: verify_rs9_manifest,
        RS11_SCHEMA: verify_rs11_manifest,
        RS12_SCHEMA: verify_rs12_manifest,
        RS13_SCHEMA: verify_rs13_manifest,
        RS5_SCHEMA: verify_rs5_manifest,
        RS4_SCHEMA: verify_rs4_manifest,
        RS3_SCHEMA: verify_rs3_manifest,
        RS2_SCHEMA: verify_rs2_manifest,
        RS_SCHEMA: verify_rs_manifest,
        PT2_SCHEMA: verify_pt2_manifest,
        PT1_SCHEMA: verify_pt1_manifest,
        NO_SSL_SCHEMA: verify_no_ssl_manifest,
        DINO_SCHEMA: verify_dino_manifest,
        DP2_SCHEMA: verify_dino_manifest,
        FGPO_SCHEMA: verify_fgpo_manifest,
        FFN_SCHEMA: verify_ffn_manifest,
    }.get(schema, verify_br_manifest)
    return verifier(source, manifest)


def official_dependency_ready(spec: dict) -> bool:
    """Verify both actual official terminal evaluations before native dispatch."""
    from benchmarks.expression_holdout.config import validate_terminal

    root = Path(spec["root"])
    queue_path = root / "queue.json"
    if not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("official dependency must stay on server")
    if sha256_file(queue_path) != spec["queue_sha256"]:
        raise ValueError("official dependency queue changed")
    queue = read_json(queue_path)
    if queue["source_commit"] != spec["source_commit"]:
        raise ValueError("official dependency source changed")
    policy = spec.get("completion_policy", "all_official_evaluations")
    if policy == DEFERRED_SCDFM_POLICY:
        return deferred_official_dependency_ready(spec, queue)
    if policy != "all_official_evaluations":
        raise ValueError("unknown official dependency completion policy")
    if (root / "FAILURE.json").exists():
        raise RuntimeError("official dependency failed; native arms remain unstarted")
    if not (root / "COMPLETE.json").exists():
        return False
    complete = read_json(root / "COMPLETE.json")
    if not complete.get("scientific_complete") or complete.get("status") != "complete":
        raise ValueError("official process exit is not scientific completion")
    rows = {row["name"]: row for row in queue["rows"]}
    for name in ("scdfm", "gears"):
        if name not in rows or name not in complete["completed"]:
            raise ValueError("scDFM and GEARS both required before FFN arms")
        row = rows[name]
        terminal = validate_terminal(
            Path(row["run_root"]), row, row.get("scientific_source_commit", queue["source_commit"])
        )
        expected = {
            "root": row["run_root"],
            "receipt_sha256": sha256_file(Path(row["run_root"]) / "receipt.json"),
            "last_test_sha256": terminal["last_test_sha256"],
        }
        if complete["completed"][name] != expected:
            raise ValueError("official terminal identity differs")
    return True


def prepare(
    source: Path,
    manifest: Path,
    runtime: Path,
    output: Path,
    policy: dict,
    *,
    only_arms: list[str] | None = None,
    preflight_only: bool = False,
) -> dict:
    source, manifest, runtime, output = (
        path.resolve() for path in (source, manifest, runtime, output)
    )
    if output.exists() or not output.is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server queue root required")
    matrix = selected_matrix(verify_manifest(source, manifest), only_arms)
    if (matrix["schema"] in PREFLIGHT_STAGE_FAMILIES) != preflight_only:
        raise ValueError("staged family requires explicit sealed preflight-only mode")
    validate_policy(policy)
    if policy["schema"] != DYNAMIC_SCHEMA:
        raise ValueError("BR400 requires exact approved dual-GPU schema3 policy")
    rows = []
    for spec in matrix["rows"]:
        plan = resolve_plan(
            argparse.Namespace(
                config=source / spec["config"],
                runtime=runtime,
                data_root=None,
                gpu="0,1",
                seed=1,
                defer_test=True,
            )
        )
        if any(Path(plan[key]).exists() for key in ("run_root", "test_root")):
            raise FileExistsError("fresh BR400 IDs cannot overwrite prior artifacts")
        plan["cuda_memory_fraction"] = policy["max_own_memory_fraction"]
        rows.append({"name": spec["name"], "plan": plan})
    queue = {
        "schema": "unified-ddp-queue-1",
        "mode": matrix["schema"],
        "family": matrix["schema"],
        "source": str(source),
        "source_commit": rows[0]["plan"]["source_commit"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "output": str(output),
        "rows": rows,
        "steps": 10,
        "shared_gpu_policy": copy.deepcopy(policy),
        "policy_sha256": sha256_json(policy),
        "eval_bindings": [],
    }
    if matrix["schema"] == PRECISION_SCHEMA:
        queue["postfit_mode"] = PRECISION_POSTFIT
        queue["precision_training_budget"] = dict(PRECISION_BUDGET)
    if matrix["schema"] in PREFLIGHT_STAGE_FAMILIES:
        queue["preflight_only"] = True
    if matrix["schema"] == RS13_SCHEMA:
        from generate_representation_round13 import validate_calibration_binding

        queue["preflight_only"] = True
        queue["relative_amplitude_binding"] = validate_calibration_binding(
            source, read_json(runtime), formal_source_commit=queue["source_commit"]
        )
    if only_arms is not None:
        queue["only_arms"] = list(only_arms)
    if matrix["schema"] == FFN_SCHEMA:
        queue["official_dependency"] = read_json(runtime)["official_dependency"]
    elif matrix["schema"] in FGPO_FAMILIES:
        queue["medium_analysis"] = read_json(runtime)["medium_analysis"]
    for row in rows:
        for gpu in ("0", "1"):
            queue["eval_bindings"].append(
                {
                    "row_name": row["name"],
                    "gpu": gpu,
                    "cell_batch": matrix["eval_batch"],
                    "reference_batch": 128,
                    "output": str(output / f"{row['name']}-eval-b{matrix['eval_batch']}-g{gpu}"),
                }
            )
    verify(queue)
    output.mkdir()
    for row in rows:
        atomic_json(output / f"{row['name']}.launch-plan.json", row["plan"])
    atomic_json(output / "queue.json", queue)
    return queue


def verify(queue: dict) -> None:
    if (
        queue.get("schema") != "unified-ddp-queue-1"
        or queue.get("mode") not in (FAMILY_SCHEMA, *LAST_ONLY_FAMILIES)
        or queue.get("family") != queue.get("mode")
        or queue.get("steps") != 10
    ):
        raise ValueError("unknown BR400 queue/protocol")
    source, output = Path(queue["source"]), Path(queue["output"])
    if (
        not output.is_absolute()
        or not output.resolve().is_relative_to(SERVER_ROOT)
        or output.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("BR400 queue must preserve published source")
    if (queue["family"] in PREFLIGHT_STAGE_FAMILIES) != (queue.get("preflight_only") is True):
        raise ValueError("staged queue must seal preflight-only mode")
    if queue["family"] == PRECISION_SCHEMA and (
        queue.get("postfit_mode") != PRECISION_POSTFIT
        or queue.get("precision_training_budget") != PRECISION_BUDGET
    ):
        raise ValueError("precision queue must seal main-owned postfit and exact 300-update budget")
    validate_policy(queue["shared_gpu_policy"])
    if queue["shared_gpu_policy"]["schema"] != DYNAMIC_SCHEMA or queue[
        "policy_sha256"
    ] != sha256_json(queue["shared_gpu_policy"]):
        raise ValueError("shared policy seal changed")
    for key in ("runtime", "manifest"):
        if sha256_file(Path(queue[key])) != queue[key + "_sha256"]:
            raise ValueError("sealed BR400 input changed")
    matrix = selected_matrix(
        verify_manifest(source, Path(queue["manifest"])), queue.get("only_arms")
    )
    if (
        matrix["schema"] in LAST_ONLY_FAMILIES
        and queue["shared_gpu_policy"]["max_own_memory_fraction"] != 0.90
    ):
        raise ValueError("remaining last-only requires the exact approved memory90 policy")
    runtime = read_json(Path(queue["runtime"]))
    if matrix["schema"] in FGPO_FAMILIES:
        from run_fgpo_medium import validate_analysis_spec

        if queue.get("medium_analysis") != runtime.get("medium_analysis"):
            raise ValueError("new B0 medium analysis must be explicitly bound")
        validate_analysis_spec(queue["medium_analysis"])
        if "official_dependency" in queue or "official_dependency" in runtime:
            raise ValueError("new family must not dispatch deferred official baselines")
    if matrix["schema"] == FFN_SCHEMA and (
        not isinstance(runtime.get("official_dependency"), dict)
        or queue.get("official_dependency") != runtime["official_dependency"]
    ):
        raise ValueError("FFN official dependency must be explicitly sealed")
    if matrix["schema"] == RS13_SCHEMA:
        from generate_representation_round13 import validate_calibration_binding

        if queue.get("relative_amplitude_binding") != validate_calibration_binding(
            source, runtime, formal_source_commit=queue["source_commit"]
        ):
            raise ValueError("R13 calibration/source compatibility binding changed")
    if Path(runtime["repository_root"]).resolve() != source.resolve():
        raise ValueError("runtime source binding changed")
    if (
        queue["family"] != matrix["schema"]
        or [row["name"] for row in queue["rows"]] != matrix["priority"]
    ):
        raise ValueError("BR400 arm scope/order changed")
    bindings = []
    for row, spec in zip(queue["rows"], matrix["rows"], strict=True):
        plan = row["plan"]
        if queue["family"] == PRECISION_SCHEMA:
            validate_precision_plan(plan)
        if (
            plan["config"] != str(source / spec["config"])
            or plan["config_sha256"] != spec["sha256"]
            or plan["source_commit"] != queue["source_commit"]
            or Path(plan["repository_root"]).resolve() != source.resolve()
            or plan["resolved_config"]
            != load_experiment_config(plan["config"]).model_dump(mode="json")
            or plan["gpu"] != "0,1"
            or plan["seed"] != 1
            or plan["postfit_policy"] != "deferred"
            or plan["cuda_memory_fraction"] != queue["shared_gpu_policy"]["max_own_memory_fraction"]
            or Path(plan["runtime"]).resolve() != Path(queue["runtime"]).resolve()
            or plan["runtime_sha256"] != queue["runtime_sha256"]
            or Path(plan["data_root"]).resolve() != Path(runtime["data_root"]).resolve()
            or Path(plan["publication"]).resolve() != Path(runtime["publication_receipt"]).resolve()
            or plan["publication_sha256"] != runtime["publication_sha256"]
            or Path(plan["run_root"]).resolve()
            != Path(runtime["runs_root"]).resolve() / plan["run_id"]
            or Path(plan["test_root"]).resolve()
            != Path(runtime["runs_root"]).resolve() / (plan["run_id"] + "-test")
        ):
            raise ValueError("native BR400 launch binding changed")
        for field in ("config", "publication"):
            if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                raise ValueError("native launch dependency changed")
        if (
            plan.get("genept_receipt")
            and sha256_file(Path(plan["genept_receipt"])) != plan["genept_sha256"]
        ):
            raise ValueError("gene prior receipt changed")
        if any(
            not Path(plan[key]).resolve().is_relative_to(SERVER_ROOT)
            for key in ("data_root", "run_root", "test_root")
        ):
            raise ValueError("scientific artifacts must stay on server")
        identity = inspect_source_identity(
            source,
            formal=True,
            expected_repository=plan["resolved_config"]["source_code"]["repository"],
            publication_receipt=plan["publication"],
            expected_publication_receipt_sha256=plan["publication_sha256"],
        )
        if identity.dirty or identity.commit != queue["source_commit"]:
            raise ValueError("BR400 source must be clean and published")
        for gpu in ("0", "1"):
            bindings.append(
                {
                    "row_name": row["name"],
                    "gpu": gpu,
                    "cell_batch": matrix["eval_batch"],
                    "reference_batch": 128,
                    "output": str(output / f"{row['name']}-eval-b{matrix['eval_batch']}-g{gpu}"),
                }
            )
    if bindings != queue["eval_bindings"] or any(
        len({r["plan"][key] for r in queue["rows"]}) != len(matrix["rows"])
        for key in ("run_id", "run_root", "test_root")
    ):
        raise ValueError("capacity bindings or unique fresh IDs changed")


def validate_staged_joint_gate(gate: dict) -> None:
    """Require real short engineering evidence for both R9 phases and retained state."""
    checks = {
        "phase_order",
        "teacher_centers_continuously_updated",
        "teacher_updated_in_both_phases",
        "centers_updated_in_both_phases",
        "prefix_keeps_ssl_and_aux",
        "prefix_head_has_no_grad_or_moments",
        "optimizer_never_reset",
        "boundary_moments_retained",
        "joint_keeps_ssl_prediction_and_aux",
        "prediction_head_learned_only_in_joint",
        "both_phase_resumes_exact",
    }
    if (
        gate.get("mode") != "ssl_then_joint"
        or gate.get("passed") is not True
        or gate.get("formal_pretrain_steps") != 150
        or gate.get("engineering_pretrain_steps") != 5
        or gate.get("engineering_joint_steps") != 5
        or set(gate.get("checks", {})) != checks
        or any(value is not True for value in gate["checks"].values())
    ):
        raise ValueError("R9 needs exact prefix/joint continuous-state short preflight evidence")


def validate_frozen_readout_parent(parent: dict) -> None:
    from generate_representation_round11 import (
        PARENT_CHECKPOINT_SHA256,
        PARENT_RUN,
        PARENT_TRAINING_SOURCE,
        metadata,
    )

    expected = metadata()["frozen_readout"]["parent_evidence_sha256"]
    if (
        parent.get("parent_run_id") != PARENT_RUN
        or parent.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent.get("parent_training_source", {}).get("commit") != PARENT_TRAINING_SOURCE
        or parent.get("parent_epoch") != 3
        or parent.get("parent_scientific_training_complete") is not True
        or parent.get("parent_full_test_complete") is not False
        or parent.get("optimizer_restored") is not False
        or parent.get("child_initial_steps") != 0
        or any(parent.get("parent_evidence_sha256", {}).get(k) != v for k, v in expected.items())
    ):
        raise ValueError("R11 exact scientific R9 parent evidence changed")


def validate_frozen_readout_gate(proof: dict) -> None:
    if (
        not all(
            proof.get(k) is True
            for k in (
                "passed",
                "exact_restore",
                "head_changed",
                "frozen_equal",
                "frozen_gradients_none",
                "all_four_head_tensors_updated",
                "prediction_changed",
                "native_softplus_preserved",
            )
        )
        or proof.get("steps") != 10
        or proof.get("teacher_updates") != 0
        or proof.get("center_updates") != 0
        or proof.get("trainable_parameters") != 131585
    ):
        raise ValueError("R11 needs actual head-only frozen-state exact10 restore proof")


def validate_precision_plan(plan: dict) -> None:
    """Admit only the three frozen scratch arms; no historical baseline dispatch."""
    source, config = Path(plan["repository_root"]), Path(plan["config"])
    matrix = verify_precision_manifest(source, config.parents[2] / "manifest.json")
    matching = [row for row in matrix["rows"] if source / row["config"] == config]
    parameters = plan["resolved_config"]["model"]["parameters"]
    if (
        len(matching) != 1
        or matching[0]["name"] not in PRECISION_ARMS
        or plan["resolved_config"] != load_experiment_config(config).model_dump(mode="json")
        or plan["resolved_config"]["training"]["max_epochs"]["value"] != 2
        or parameters["validation_mode"]["value"] != "disabled"
        or plan.get("postfit_policy") != "deferred"
    ):
        raise ValueError(
            "precision plan must preserve the exact three-arm two-epoch scratch family"
        )


def precision_coverage(plan: dict, name: str) -> tuple[set[str], set[str]]:
    """Expected observed paths from the sealed RI architecture, without model allocation."""
    from gradpert.config.v2 import V2Options

    architecture, _ = V2Options.parse_parameters(
        load_experiment_config(plan["config"]).model.parameters
    )
    if name == "PREC-SSL32":
        features = ("features.1.0", "features.1.2") if architecture.ssl2_cls_input_layernorm else ()
        heads = {
            f"{side}.{head}.{leaf}"
            for side in ("student", "teacher")
            for head in ("ssl1_cls", "ssl2_cls", "ssl2_node")
            for leaf in (
                (*features, "prototypes")
                if head == "ssl2_cls" and features
                else ("features.0", "features.2", "prototypes")
            )
        }
        return heads, set()
    if name == "PREC-PRED32":
        return {"student.prediction.0", "student.prediction.2"}, set()
    if (
        name != "PREC-FFN8"
        or architecture.graph_read_mode != "relay"
        or architecture.graph_mhc
        or architecture.attention != "relay_full"
    ):
        raise ValueError("unsupported precision architecture coverage")
    modules = {f"graph.layers.{i}.ffn" for i in range(4)}
    modules |= {f"cell.layers.{2 * i + 1}.sublayer" for i in range(architecture.kda_layers + 1)}
    modules |= {f"response.ffn_layers.{i}.sublayer" for i in range(architecture.kda_layers + 1)}
    return set(), {
        f"{side}.{module}.{projection}.weight"
        for side in ("student", "teacher")
        for module in modules
        for projection in ("up", "down")
    }


def validate_precision_gate(row: dict, receipt: dict) -> None:
    """Consume both native rank observations; a bare passed status is insufficient."""
    name = row["name"]
    if name not in PRECISION_ARMS:
        raise ValueError("precision gate rejects baseline or unknown arm")
    heads, weights = precision_coverage(row["plan"], name)
    ranks = receipt.get("rank_measurements", [])
    if (
        len(ranks) != 2
        or {item.get("rank") for item in ranks} != {0, 1}
        or not isinstance(receipt.get("resume_checkpoint_sha256"), str)
        or len(receipt["resume_checkpoint_sha256"]) != 64
    ):
        raise ValueError("precision gate requires both ranks and checkpoint restore identity")
    for item in ranks:
        audit = item.get("precision_audit", {})
        backend = audit.get("backend", {})
        if (
            item.get("physical_gpu") != str(item["rank"])
            or audit.get("schema") != "gradpert-precision-preflight-1"
            or audit.get("status") != "passed"
            or audit.get("exact_full_state_restore") is not True
            or not backend.get("torch_version")
            or not backend.get("cuda_version")
            or not backend.get("device_name")
            or backend.get("compute_capability") != [12, 0]
            or backend.get("matmul_allow_tf32") is not False
            or backend.get("float32_matmul_precision") != "highest"
        ):
            raise ValueError("precision gate missing native backend/dtype/fullrestore evidence")
        if weights and (
            backend["torch_version"].split("+")[0] != "2.13.0"
            or backend["cuda_version"] not in ("12.8", "12.9", "13.0")
            or "_scaled_mm" not in backend.get("scaled_mm_schema", "")
            or backend.get("recipe") != "per-tensor current scaling, E4M3 forward / E5M2 gradient"
        ):
            raise ValueError("precision gate requires the pinned real FP8 backend")
        observed_heads = set()
        for call in audit.get("head_linear_calls", []):
            key = (call.get("phase"), call.get("path"))
            if type(call.get("calls")) is not int or call["calls"] <= 0 or key in observed_heads:
                raise ValueError("invalid precision head call evidence")
            observed_heads.add(key)
        expected_heads = {
            (phase, path) for phase in ("before_restore", "after_restore") for path in heads
        }
        if observed_heads != expected_heads:
            raise ValueError("precision head coverage differs from exact active paths")
        observed_gemms = set()
        for call in audit.get("fp8_gemm_calls", []):
            op = call.get("operation")
            key = (call.get("phase"), call.get("path"), op)
            if (
                type(call.get("calls")) is not int
                or call["calls"] <= 0
                or key in observed_gemms
                or call.get("a")
                != ("torch.float8_e4m3fn" if op == "forward" else "torch.float8_e5m2")
                or call.get("b") != "torch.float8_e4m3fn"
                or call.get("output") != ("torch.bfloat16" if op == "forward" else "torch.float32")
                or call.get("use_fast_accum") is not False
            ):
                raise ValueError("invalid real FP8 GEMM recipe evidence")
            observed_gemms.add(key)
        expected_gemms = {
            (phase, path, op)
            for phase in ("before_restore", "after_restore")
            for path in weights
            for op in (
                ("forward", "dgrad", "wgrad") if path.startswith("student.") else ("forward",)
            )
        }
        if observed_gemms != expected_gemms:
            raise ValueError("precision GEMM coverage differs from both owners/phases/FFNs")
    if receipt.get("precision_audit") != next(
        item["precision_audit"] for item in ranks if item["rank"] == 0
    ):
        raise ValueError("precision primary and rank0 audit disagree")


def validate_training_gate(queue: dict, row: dict, entry: dict) -> dict:
    validate_preflight(row["plan"], entry)
    receipt = read_json(Path(entry["receipt"]))
    joint = receipt.get("joint_validation_check", {})
    protected = {
        "student",
        "teacher",
        "centers",
        "optimizer",
        "gradients",
        "numpy_rng",
        "python_rng",
        "torch_rng",
        "pending_statistics",
        "auxiliary_statistics",
        "module_modes",
    }
    before, after = joint.get("before_sha256", {}), joint.get("after_sha256", {})
    if (
        receipt.get("kind") != "preflight_only"
        or receipt.get("steps_requested") != 10
        or receipt.get("steps_completed") != 10
        or receipt.get("inference_shape") != [300, 5000]
        or receipt.get("evaluation_cell_batch")
        != row["plan"]["resolved_config"]["training"]["eval_batch_size"]["value"]
        or receipt.get("gpu") != "0,1"
        or receipt.get("world_size") != 2
        or receipt.get("shared_gpu_policy") != queue["shared_gpu_policy"]
        or receipt.get("shared_queue_sha256") != sha256_file(Path(queue["output"]) / "queue.json")
        or receipt["source"].get("publication_receipt_sha256") != row["plan"]["publication_sha256"]
        or not receipt.get("last_terms")
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in receipt["last_terms"].values()
        )
    ):
        raise ValueError("exact10 restore/inference/read-only joint-validation gate missing")
    if queue["family"] == PRECISION_SCHEMA:
        validate_precision_gate(row, receipt)
    if queue["family"] in LAST_ONLY_FAMILIES:
        if queue["family"] == RS13_SCHEMA:
            from generate_representation_round13 import validate_amplitude_gate

            validate_amplitude_gate(
                receipt.get("spectral_amplitude_gate", {}),
                arm=row["name"],
                resume_checkpoint_sha256=receipt.get("resume_checkpoint_sha256"),
            )
        if queue["family"] == RS12_SCHEMA:
            spectral = receipt.get("spectral_pair_gate", {})
            required = {
                "ten_native_updates",
                "exact_restore",
                "complete_population",
                "correct_warmup",
                "first_update_zero",
                "finite_losses",
            }
            if (
                spectral.get("passed") is not True
                or set(spectral.get("checks", {})) != required
                or not all(v is True for v in spectral["checks"].values())
                or len(spectral.get("updates", [])) != 10
                or any(
                    r.get("completed_before") != i
                    or r.get("population") != 320
                    or r.get("warmup") != min(1.0, i / 30)
                    or any(
                        not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k])
                        for k in ("rank", "cosine", "weighted")
                    )
                    for i, r in enumerate(spectral.get("updates", []))
                )
                or (spectral.get("updates") and spectral["updates"][0].get("weighted") != 0.0)
                or spectral.get("resume_checkpoint_sha256")
                != receipt.get("resume_checkpoint_sha256")
            ):
                raise ValueError("R12 exact10 native spectral/warmup/restore proof missing")
        if queue["family"] == RS11_SCHEMA:
            validate_frozen_readout_parent(receipt.get("readout_parent", {}))
            validate_frozen_readout_gate(receipt.get("frozen_readout_gate", {}))
        if queue["family"] == RS9_SCHEMA:
            validate_staged_joint_gate(receipt.get("staged_training_gate", {}))
        if (
            queue["family"] in (PT1_SCHEMA, PT2_SCHEMA, RS9_SCHEMA)
            and receipt.get("staged_training_gate", {}).get("passed") is not True
        ):
            raise ValueError(
                "staged training requires actual SSL/MSE transition and restore evidence"
            )
        if (
            row["plan"]["resolved_config"]["model"]["parameters"]["validation_mode"]["value"]
            != "disabled"
            or receipt.get("joint_validation_check_requested") is not False
            or "joint_validation_check" in receipt
            or "joint_validation_progress" in receipt
        ):
            raise ValueError("last-only gate must not perform joint validation")
        if queue["family"] in (FFN_SCHEMA, *FGPO_FAMILIES):
            for key in ("initial_loss_diagnostic", "updated_loss_diagnostic"):
                observed = receipt.get(key, {}).get("process", {})
                if (
                    observed.get("split") != "train"
                    or observed.get("selection_use") != "none"
                    or observed.get("schema") != "training-process-diagnostics-1"
                    or (
                        not observed.get("distributions")
                        and not (
                            queue["family"] in (PT1_SCHEMA, PT2_SCHEMA)
                            and key == "updated_loss_diagnostic"
                            and observed.get("training_stage") == "mse_finetune"
                            and not any(k.startswith("ssl") for k in observed.get("components", {}))
                        )
                    )
                    or not observed.get("features")
                    or not observed.get("outputs")
                ):
                    raise ValueError("FFN training-process probes missing from capacity gate")
    elif (
        joint.get("status") != "passed"
        or joint.get("validation_mode") != "joint_only"
        or joint.get("split") != "val"
        or joint.get("finite") is not True
        or not math.isfinite(joint.get("joint_loss", float("nan")))
        or set(joint.get("state_unchanged", {})) != protected
        or set(before) != protected
        or set(after) != protected
        or before != after
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in before.values()
        )
        or joint.get("batch_size")
        != min(100, row["plan"]["resolved_config"]["training"]["eval_batch_size"]["value"])
        or type(joint.get("batch_count")) is not int
        or joint["batch_count"] != 1
        or joint.get("scope") != "one_complete_native_validation_batch_engineering_only"
        or joint.get("full_validation") is not False
        or not all(value is True for value in joint["state_unchanged"].values())
    ):
        raise ValueError("exact read-only joint-validation gate missing")
    return receipt


def eval_capacity_command(queue: dict, row: dict, binding: dict) -> list[str]:
    plan = row["plan"]
    return [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/eval_batch_capacity.py"),
        "--config",
        plan["config"],
        "--data-root",
        plan["data_root"],
        "--output",
        binding["output"],
        "--publication",
        plan["publication"],
        "--publication-sha256",
        plan["publication_sha256"],
        "--gpu",
        binding["gpu"],
        "--cell-batch",
        str(binding["cell_batch"]),
        "--reference-batch",
        "128",
        "--shared-queue",
        str(Path(queue["output"]) / "queue.json"),
        "--resource-owner-pid",
        str(os.getpid()),
        "--row-name",
        row["name"],
    ]


def validate_eval_gate(queue: dict, row: dict, binding: dict, entry: dict) -> dict:
    from eval_batch_capacity import validate_receipt

    receipt = validate_receipt(queue, row, binding, entry)
    if queue["family"] == RS11_SCHEMA:
        validate_frozen_readout_parent(receipt.get("readout_parent", {}))
    return receipt


DATA_HASHES = (
    "canonical_sha256",
    "split_sha256",
    "expression_gene_order_sha256",
    "graph_gene_order_sha256",
    "graph_manifest_sha256",
    "genept_sha256",
)


def validate_common_gates(queue: dict, state: dict) -> dict:
    if queue["family"] == RS13_SCHEMA:
        from generate_representation_round13 import validate_calibration_binding

        if queue.get("relative_amplitude_binding") != validate_calibration_binding(
            Path(queue["source"]),
            read_json(Path(queue["runtime"])),
            formal_source_commit=queue["source_commit"],
        ):
            raise ValueError(
                "R13 fit requires accepted original calibration and source compatibility"
            )
    if queue["family"] == RS12_SCHEMA:
        from calibrate_spectral_pair import verify_endpoints

        spec = read_json(Path(queue["runtime"])).get("spectral_pair_calibration", {})
        required = {"init", "ri_last", "init_complete_sha256", "ri_last_complete_sha256"}
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError("R12 fit requires both sealed zero-update calibration endpoints")
        for endpoint in ("init", "ri_last"):
            root = Path(spec[endpoint]).resolve()
            if (
                not root.is_relative_to(SERVER_ROOT)
                or sha256_file(root / "COMPLETE.json") != spec[endpoint + "_complete_sha256"]
            ):
                raise ValueError("R12 calibration endpoint left server or checksum changed")
        verify_endpoints(spec["init"], spec["ri_last"], source_commit=queue["source_commit"])
    priority = [row["name"] for row in queue["rows"]]
    if set(state["preflights"]) != set(priority) or set(state["eval_capacity"]) != {
        f"{name}-g{gpu}" for name in priority for gpu in ("0", "1")
    }:
        raise ValueError("all configured update and dual eval gates required")
    indexed = {row["name"]: row for row in queue["rows"]}
    expected_data, expected_control = None, None
    for name in priority:
        receipt = validate_training_gate(queue, indexed[name], state["preflights"][name])
        data = {key: receipt.get("data", {}).get(key) for key in DATA_HASHES}
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in data.values()
        ):
            raise ValueError("update gate lacks exact frozen data hashes")
        if expected_data is None:
            expected_data = data
        if (
            data != expected_data
            or receipt.get("inference_expression_gene_order_sha256")
            != data["expression_gene_order_sha256"]
        ):
            raise ValueError("updated inference/common data identity differs")
        controls = {
            "manifest": receipt.get("inference_control_manifest_sha256"),
            "ordered_ids": receipt.get("ordered_control_row_ids_sha256"),
        }
        if expected_control is None:
            expected_control = controls
        if (
            any(not isinstance(value, str) or len(value) != 64 for value in controls.values())
            or controls != expected_control
        ):
            raise ValueError("updated control population differs across arms")
        for binding in (item for item in queue["eval_bindings"] if item["row_name"] == name):
            leaf = validate_eval_gate(
                queue, indexed[name], binding, state["eval_capacity"][f"{name}-g{binding['gpu']}"]
            )
            if {key: leaf["data"].get(key) for key in DATA_HASHES} != expected_data or {
                "manifest": leaf.get("inference_control_manifest_sha256"),
                "ordered_ids": leaf.get("ordered_control_row_ids_sha256"),
            } != controls:
                raise ValueError("fresh init and updated controls/data differ across GPU leaves")
    return {"data_sha256": expected_data, "controls_sha256": expected_control}


def training_command(path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import json,sys;from gradpert.execution.train_entry import execute_plan;"
        "execute_plan(json.load(open(sys.argv[1])))",
        str(path),
    ]


def evaluation_command(path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import json,sys;from gradpert.execution.v2_deferred_postfit import run_deferred_postfit;"
        "p=json.load(open(sys.argv[1]));"
        "run_deferred_postfit(p,gpu='0,1',cuda_memory_fraction=p['cuda_memory_fraction'])",
        str(path),
    ]


def validate_training_contract(plan: dict) -> dict:
    training = validate_training_stage(plan)
    epochs = expected_epochs(plan)
    if plan["resolved_config"]["model"]["parameters"]["validation_mode"]["value"] == "disabled":
        root = Path(plan["run_root"])
        if (
            training["epoch"] != epochs
            or training["last"]["epoch"] != epochs
            or training["test_roles"] != ["last"]
            or training["best"] is not None
            or (root / "fit/best-test.json").exists()
            or any(
                row["validation"] != {"validation_mode": "disabled", "performed": False}
                for row in read_json(root / "fit/history.json")
            )
        ):
            raise ValueError("last-only training must contain no best or actual validation")
    if precision_plan(plan):
        journal = read_json(Path(plan["run_root"]) / "fit/epoch_state.json")
        history = read_json(Path(plan["run_root"]) / "fit/history.json")
        if journal.get("budget") != [2, 150] or [
            item.get("optimizer_steps") for item in history
        ] != [150, 300]:
            raise ValueError(
                "precision training requires exactly 300 successful updates in two epochs"
            )
    return training


def precision_plan(plan: dict) -> bool:
    path = Path(plan["config"]).parents[2] / "manifest.json"
    return path.is_file() and read_json(path).get("schema") == PRECISION_SCHEMA


def expected_epochs(plan: dict) -> int:
    """Short research and staged budgets require exact frozen family configurations."""
    from generate_pt2 import configuration

    if plan["resolved_config"]["training"]["max_epochs"]["value"] == 3:
        source = Path(plan["repository_root"])
        target = Path(plan["config"])
        family = verify_rs9_manifest(source, target.parents[2] / "manifest.json")
        if len([r for r in family["rows"] if source / r["config"] == target]) != 1 or plan[
            "resolved_config"
        ] != load_experiment_config(target).model_dump(mode="json"):
            raise ValueError("three-epoch dispatch requires the exact sealed R9 family")
        return 3
    if plan["resolved_config"]["training"]["max_epochs"]["value"] == 2:
        import yaml
        from generate_representation_round1 import configs
        from generate_representation_round2 import configs as configs2

        source = Path(plan["repository_root"])
        raw = yaml.safe_load(Path(plan["config"]).read_text())
        registered = [*configs(source).values(), *configs2(source).values()]
        family_manifest = Path(plan["config"]).parents[2] / "manifest.json"
        if family_manifest.is_file() and read_json(family_manifest).get("schema") in (
            PRECISION_SCHEMA,
            RS3_SCHEMA,
            RS4_SCHEMA,
            RS5_SCHEMA,
            RS6_SCHEMA,
            RS7_SCHEMA,
            RS8_SCHEMA,
            RS11_SCHEMA,
            RS12_SCHEMA,
            RS13_SCHEMA,
        ):
            family = verify_manifest(source, family_manifest)
            matching = [r for r in family["rows"] if source / r["config"] == Path(plan["config"])]
            if len(matching) != 1:
                raise ValueError("research dispatch requires its exact sealed family leaf")
            registered.append(raw)
        if precision_plan(plan):
            validate_precision_plan(plan)
        if raw not in registered or plan["resolved_config"] != load_experiment_config(
            plan["config"]
        ).model_dump(mode="json"):
            raise ValueError("two-epoch dispatch requires an exact representation research arm")
        return 2
    if plan["resolved_config"]["training"]["max_epochs"]["value"] == 6:
        expected = load_experiment_config(
            Path(plan["repository_root"])
            / "configs/v2/pt2_native_io_20261017/PT2/gradpert_v2/nadig_jurkat.yaml"
        ).model_dump(mode="json")
        # Check the file against its generator as well as the plan's resolved identity.
        import yaml

        if (
            yaml.safe_load(Path(plan["config"]).read_text())
            != configuration(Path(plan["repository_root"]))
            or plan["resolved_config"] != expected
        ):
            raise ValueError("six-epoch dispatch is restricted to the exact PT2 configuration")
        return 6
    return 4


def validate_complete(plan: dict) -> dict:
    root = Path(plan["run_root"])
    if not (root / "COMPLETE.json").is_file() or any(
        (root / name).exists() for name in ("FAILURE.json", "EVALUATION_FAILURE.json")
    ):
        raise ValueError("scientific run has no successful native terminal")
    if (
        read_json(root / "launch.json") != plan
        or read_json(root / "resolved_config.json") != plan["resolved_config"]
    ):
        raise ValueError("completed launch/config changed")
    training = validate_training_contract(plan)
    last_only = (
        plan["resolved_config"]["model"]["parameters"]["validation_mode"]["value"] == "disabled"
    )
    roles = ["last"] if last_only else ["best", "last"]
    if (
        training["epoch"] != expected_epochs(plan)
        or training["last"]["epoch"] != expected_epochs(plan)
        or training["test_roles"] != roles
        or (last_only and (training["best"] is not None or (root / "fit/best-test.json").exists()))
    ):
        raise ValueError("exact epoch budget and checkpoint roles required")
    original_complete = read_json(root / "COMPLETE.json")
    if finalize_deferred_run(plan, training) != original_complete:
        raise ValueError("canonical completion changed")
    rows = collect_run(root)
    if [row["role"] for row in rows] != roles:
        raise ValueError("both role metrics required")
    for row in rows:
        role = row["role"]
        if (
            row["status"] != "complete"
            or row["training_sha"] != plan["source_commit"]
            or row["evaluation_sha"] != plan["source_commit"]
            or row["config_sha256"] != plan["config_sha256"]
        ):
            raise ValueError("role source/config/completion changed")
        result = read_json(root / "fit" / f"{role}-test.json")["result"]
        groups = result.get("metric_gene_groups", {})
        if set(groups) != {"seen_expression", "unseen_expression"}:
            raise ValueError("role lacks both exposure groups")
        for metrics in [result["metrics"], *(group["metrics"] for group in groups.values())]:
            if len(metrics) != 6 or {metric["metric_id"] for metric in metrics} != METRICS:
                raise ValueError("eighteen grouped metrics required per role")
            for metric in metrics:
                finite, total, value = (
                    metric["finite_condition_count"],
                    metric["total_condition_count"],
                    metric["macro_mean"],
                )
                if (
                    type(finite) is not int
                    or type(total) is not int
                    or not 0 <= finite <= total
                    or (finite == 0 and value is not None)
                    or (finite > 0 and (value is None or not math.isfinite(value)))
                ):
                    raise ValueError("metric counts and availability differ")
    if any(root.rglob("*.pkl")) or any(Path(plan["test_root"]).rglob("*.pkl")):
        raise ValueError("scientific completion requires whole-root zero PKL")
    return {
        "run_id": plan["run_id"],
        "complete_sha256": sha256_file(root / "COMPLETE.json"),
        "roles": rows,
        "test_sha256": {role: sha256_file(root / "fit" / f"{role}-test.json") for role in roles},
    }


def run_stage(
    queue: dict, row: dict, stage: str, command: list[str], gpu: str, state: dict, guard
) -> None:
    directory = Path(queue["output"])
    child, identity, known = None, None, {}

    def remember():
        parents = parent_map()
        for pid in parents:
            if child is not None and descendant(pid, child.pid, parents):
                actual = process_info(pid)
                if actual is not None:
                    if actual["uid"] != os.getuid():
                        raise RuntimeError("child descendant UID changed")
                    known.setdefault(pid, actual)
        for pid, expected in known.items():
            actual = process_info(pid)
            if actual is not None and any(
                actual[key] != expected[key]
                for key in ("pid", "uid", "start_ticks", "command_sha256")
            ):
                raise RuntimeError("active child identity changed")

    try:
        verify(queue)
        guard()
        with (directory / f"{row['name']}-{stage}.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=queue["source"],
                env=child_environment(queue["source"], gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = _capture_process_identity(child)
            known[child.pid] = {**identity, "uid": os.getuid()}
            state["active"] = {"name": row["name"], "stage": stage, "identity": identity}
            atomic_json(directory / "state.json", state)
            while child.poll() is None:
                remember()
                guard()
                time.sleep(POLL_SECONDS)
            code = child.wait()
        remember()
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("child ranks remain after launcher exit")
            time.sleep(POLL_SECONDS)
            remember()
        if any(p["owned"] for p in guard()["processes"]):
            raise RuntimeError("owned CUDA process remains after child exit")
        atomic_json(
            directory / f"{row['name']}-{stage}.exit.json",
            {"identity": identity, "exit_code": code, "command": command},
        )
        if code:
            raise RuntimeError(f"{row['name']} {stage} exited {code}; no fallback/retry")
        verify(queue)
        state["active"] = None
        atomic_json(directory / "state.json", state)
    except BaseException:
        if identity is not None:
            errors = []
            for pid, expected in reversed(list(known.items())):
                try:
                    terminate_tree(pid, expected)
                except BaseException as error:
                    errors.append(str(error))
            if errors:
                state["cleanup_errors"] = errors
        elif child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        raise


def gate_entry(path: Path) -> dict:
    return {"receipt": str(path), "sha256": sha256_file(path)}


def validate_precision_zero_pkl(queue: dict) -> None:
    """Reject whole-root PKL residue without deleting scientific artifacts."""
    if queue.get("family") != PRECISION_SCHEMA:
        return
    roots = {Path(queue["output"])}
    roots.update(
        Path(row["plan"][key]) for row in queue["rows"] for key in ("run_root", "test_root")
    )
    for root in roots:
        if any(root.rglob("*.pkl")):
            raise ValueError(f"precision whole-root zero-PKL contract failed: {root}")


def precision_fit_evidence(queue: dict, row: dict) -> dict:
    validate_precision_zero_pkl(queue)
    training = validate_training_contract(row["plan"])
    root = Path(row["plan"]["run_root"])
    return dict(
        schema="precision-fit-complete-1",
        arm=row["name"],
        queue_sha256=sha256_file(Path(queue["output"]) / "queue.json"),
        source_commit=queue["source_commit"],
        launch_plan_sha256=sha256_file(Path(queue["output"]) / f"{row['name']}.launch-plan.json"),
        train_complete=gate_entry(root / "TRAIN_COMPLETE.json"),
        last=training["last"],
        budget=dict(PRECISION_BUDGET),
        scientific_complete=False,
        postfit_mode=PRECISION_POSTFIT,
        test_status="deferred_to_main",
    )


def validate_fit_promotion(queue, directory, state, *, fit_only_row=None):
    """Explicit later-stage R13 promotion; never rerun or weaken sealed preflights."""
    precision = queue.get("family") == PRECISION_SCHEMA
    validate_precision_zero_pkl(queue)
    valid_state = (state.get("status"), state.get("phase")) == (
        "preflight_complete",
        "awaiting_main_acceptance",
    ) or (
        precision
        and (state.get("status"), state.get("phase"))
        == ("fit_complete", "awaiting_main_precision_postfit")
    )
    if fit_only_row is not None and (not precision or fit_only_row not in PRECISION_ARMS):
        raise ValueError("fit-only-row requires an exact precision arm")
    if (directory / "FAILURE.json").exists():
        raise ValueError("R13 fit promotion rejects a retained controller FAILURE")
    if (
        queue.get("family") not in PREFLIGHT_STAGE_FAMILIES
        or queue.get("preflight_only") is not True
        or not valid_state
        or state.get("active") is not None
        or state.get("scientific_complete") is not False
        or state.get("completed") != {}
        or state.get("queue_sha256") != sha256_file(directory / "queue.json")
        or state.get("source_commit") != queue["source_commit"]
    ):
        raise ValueError(
            "R13 resume-fit requires the exact successful preflight-only terminal state"
        )
    matrix = selected_matrix(
        verify_manifest(Path(queue["source"]), Path(queue["manifest"])), queue.get("only_arms")
    )
    common = validate_common_gates(queue, state)
    expected = dict(
        common_identity=common,
        queue_sha256=state["queue_sha256"],
        source_commit=queue["source_commit"],
        preflights=state["preflights"],
        eval_capacity=state["eval_capacity"],
        formal_order=matrix["priority"],
        preflight_only=True,
    )
    ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
    if ready != expected or read_json(directory / "MAIN_ACCEPTANCE.json") != {
        **expected,
        "approved": True,
    }:
        raise ValueError("R13 resume-fit requires main acceptance of every exact retained gate")
    promotion_file = f"FIT_PROMOTION.{fit_only_row}.json" if precision else "FIT_PROMOTION.json"
    if (directory / promotion_file).exists():
        raise ValueError(
            "R13 fit promotion already attempted; reconcile immutable execution evidence"
        )
    fitted = state.get("precision_fits", {}) if precision else {}
    if not isinstance(fitted, dict) or set(fitted) - set(PRECISION_ARMS):
        raise ValueError("unknown precision fitted arm")
    if fit_only_row in fitted:
        raise ValueError("precision fit cannot retrain a completed arm")
    for row in queue["rows"]:
        if row["name"] in fitted:
            entry = fitted[row["name"]]
            path = directory / f"FIT_COMPLETE.{row['name']}.json"
            if entry != gate_entry(path) or read_json(path) != precision_fit_evidence(queue, row):
                raise ValueError("retained precision fit evidence changed")
        elif any(Path(row["plan"][key]).exists() for key in ("run_root", "test_root")):
            raise FileExistsError("R13 fit promotion cannot overwrite an existing run/test root")
        if read_json(directory / f"{row['name']}.launch-plan.json") != row["plan"]:
            raise ValueError("R13 retained formal launch plan changed")
    return ready


def execute(
    queue: dict,
    directory: Path,
    *,
    eval_only_row: str | None = None,
    preflight_only: bool = False,
    resume_fit: bool = False,
    fit_only_row: str | None = None,
) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR
        or Path(queue["source"]).resolve() != Path(__file__).resolve().parents[2]
        or directory.resolve() != Path(queue["output"]).resolve()
        or ((directory / "state.json").exists() and not resume_fit)
        or read_json(directory / "queue.json") != queue
    ):
        raise ValueError("fresh CPU-only execution must match sealed source/root/queue")
    if (preflight_only and resume_fit) or (
        resume_fit and queue.get("family") not in PREFLIGHT_STAGE_FAMILIES
    ):
        raise ValueError("resume-fit is an explicit later R13 stage, separate from preflight-only")
    if queue.get("preflight_only", False) is not (preflight_only or resume_fit):
        raise ValueError("execution preflight-only flag differs from sealed queue")
    if queue.get("family") == PRECISION_SCHEMA:
        if (resume_fit and fit_only_row not in PRECISION_ARMS) or (
            not resume_fit and fit_only_row is not None
        ):
            raise ValueError("precision resume-fit requires exactly one explicit fit-only-row")
    elif fit_only_row is not None:
        raise ValueError("fit-only-row is restricted to precision promotion")
    verify(queue)
    matrix = selected_matrix(
        verify_manifest(Path(queue["source"]), Path(queue["manifest"])), queue.get("only_arms")
    )
    if eval_only_row is not None and (
        queue["family"] != FFN_SCHEMA or eval_only_row not in matrix["priority"]
    ):
        raise ValueError("evaluation-only engineering requires one exact FFN arm")
    queue_sha = sha256_file(directory / "queue.json")
    state = {
        "status": "running",
        "queue_sha256": queue_sha,
        "source_commit": queue["source_commit"],
        "phase": "preflight",
        "pid": os.getpid(),
        "active": None,
        "scientific_complete": False,
        "preflights": {},
        "eval_capacity": {},
        "completed": {},
    }
    if resume_fit:
        state = read_json(directory / "state.json")
        promotion_state_sha256 = sha256_file(directory / "state.json")
        validate_fit_promotion(queue, directory, state, fit_only_row=fit_only_row)
    if eval_only_row is not None:
        state["engineering_only_row"] = eval_only_row

    def guard():
        if sha256_file(directory / "queue.json") != queue_sha:
            raise ValueError("immutable queue changed")
        observed = snapshot(os.getpid())
        with (directory / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(observed) + "\n")
        admit(queue["shared_gpu_policy"], observed)
        return observed

    def abort(_sig, _frame):
        raise KeyboardInterrupt("BR400 controller interrupted")

    promotion_started = False
    handlers = {sig: signal.signal(sig, abort) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(lock(directory / "controller.lock"))
            if queue["family"] == FFN_SCHEMA:
                state["phase"] = "waiting_official_baselines"
                atomic_json(directory / "state.json", state)
                while not official_dependency_ready(queue["official_dependency"]):
                    if sha256_file(directory / "queue.json") != queue_sha:
                        raise ValueError("sealed dependent FFN queue changed")
                    time.sleep(60)
                state["phase"] = "preflight"
                atomic_json(directory / "state.json", state)
            gpu_leases = stack.enter_context(contextlib.ExitStack())
            for gpu in ("0", "1"):
                gpu_leases.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
            if any(p["owned"] for p in guard()["processes"]):
                raise RuntimeError("other owned CUDA work exists")
            if resume_fit:
                verify(queue)
                disk_state = read_json(directory / "state.json")
                if (
                    sha256_file(directory / "state.json") != promotion_state_sha256
                    or disk_state != state
                ):
                    raise ValueError(
                        "R13 terminal state changed while waiting for controller/GPU locks"
                    )
                state = disk_state
                ready = validate_fit_promotion(queue, directory, state, fit_only_row=fit_only_row)
                write_promotion = dict(
                    queue_sha256=queue_sha,
                    source_commit=queue["source_commit"],
                    preflight_complete_sha256=sha256_file(directory / "PREFLIGHT_COMPLETE.json"),
                    main_acceptance_sha256=sha256_file(directory / "MAIN_ACCEPTANCE.json"),
                    explicit_resume_fit=True,
                    preflights_reused=True,
                    preflight_state_sha256=promotion_state_sha256,
                )
                if queue["family"] == PRECISION_SCHEMA:
                    write_promotion["fit_only_row"] = fit_only_row
                    write_promotion["postfit_mode"] = PRECISION_POSTFIT
                promotion_file = (
                    f"FIT_PROMOTION.{fit_only_row}.json"
                    if queue["family"] == PRECISION_SCHEMA
                    else "FIT_PROMOTION.json"
                )
                atomic_json(directory / promotion_file, write_promotion)
                promotion_started = True
                state.update(
                    status="running",
                    phase="fit_promotion",
                    pid=os.getpid(),
                    active=None,
                    formal_fit_authorized=True,
                    explicit_fit_promotion=True,
                )
                atomic_json(directory / "state.json", state)
            else:
                indexed = {row["name"]: row for row in queue["rows"]}
                priority = [eval_only_row] if eval_only_row else matrix["preflight_priority"]
                for name in priority:
                    row = indexed[name]
                    if eval_only_row is None:
                        command = probe(row, directory, shared=True)
                        if queue["family"] not in LAST_ONLY_FAMILIES:
                            command = [*command, "--joint-validation-check"]
                        run_stage(queue, row, "preflight", command, "0,1", state, guard)
                        entry = gate_entry(directory / f"{name}-preflight/receipt.json")
                        validate_training_gate(queue, row, entry)
                        state["preflights"][name] = entry
                        atomic_json(directory / "state.json", state)
                    for binding in (b for b in queue["eval_bindings"] if b["row_name"] == name):
                        run_stage(
                            queue,
                            row,
                            f"eval-capacity-g{binding['gpu']}",
                            eval_capacity_command(queue, row, binding),
                            binding["gpu"],
                            state,
                            guard,
                        )
                        entry = gate_entry(Path(binding["output"]) / "receipt.json")
                        validate_eval_gate(queue, row, binding, entry)
                        state["eval_capacity"][f"{name}-g{binding['gpu']}"] = entry
                if eval_only_row is not None:
                    state.update(status="completed", phase="engineering_eval_complete", active=None)
                    atomic_json(directory / "state.json", state)
                    atomic_json(directory / "EVAL_CAPACITY_COMPLETE.json", state)
                    return
                common = validate_common_gates(queue, state)
                ready = {
                    "common_identity": common,
                    "queue_sha256": queue_sha,
                    "source_commit": queue["source_commit"],
                    "preflights": state["preflights"],
                    "eval_capacity": state["eval_capacity"],
                    "formal_order": matrix["priority"],
                }
                if queue["family"] == FFN_SCHEMA:
                    ready["official_dependency"] = queue["official_dependency"]
                if preflight_only:
                    ready["preflight_only"] = True
                validate_precision_zero_pkl(queue)
                atomic_json(directory / "PREFLIGHT_COMPLETE.json", ready)
                if preflight_only:
                    state.update(
                        status="preflight_complete",
                        phase="awaiting_main_acceptance",
                        active=None,
                        formal_fit_authorized=False,
                    )
                    atomic_json(directory / "state.json", state)
                    return
                state["phase"] = "waiting_main_acceptance"
                atomic_json(directory / "state.json", state)
                acceptance = directory / "MAIN_ACCEPTANCE.json"
                while not acceptance.exists():
                    guard()
                    time.sleep(POLL_SECONDS)
                accepted = read_json(acceptance)
                if accepted != {**ready, "approved": True}:
                    raise ValueError(
                        "main acceptance must bind all exact gates and published source"
                    )
            for row in queue["rows"]:
                if fit_only_row is not None and row["name"] != fit_only_row:
                    continue
                if resume_fit and (
                    sha256_file(directory / "MAIN_ACCEPTANCE.json")
                    != write_promotion["main_acceptance_sha256"]
                    or sha256_file(directory / "PREFLIGHT_COMPLETE.json")
                    != write_promotion["preflight_complete_sha256"]
                ):
                    raise ValueError(
                        "R13 main acceptance/preflight receipt changed after promotion"
                    )
                verify(queue)
                for key in ("run_root", "test_root"):
                    if Path(row["plan"][key]).exists():
                        raise FileExistsError("fresh formal run cannot overwrite prior artifacts")
                if validate_common_gates(queue, state) != ready["common_identity"]:
                    raise ValueError("accepted common gate identity changed before formal fit")
                path = directory / f"{row['name']}.launch-plan.json"
                if read_json(path) != row["plan"]:
                    raise ValueError("sealed formal plan changed")
                state["phase"] = f"{row['name']}_fit"
                run_stage(queue, row, "fit", training_command(path), "0,1", state, guard)
                validate_training_contract(row["plan"])
                if queue["family"] == PRECISION_SCHEMA:
                    fit_path = directory / f"FIT_COMPLETE.{row['name']}.json"
                    atomic_json(fit_path, precision_fit_evidence(queue, row))
                    state.setdefault("precision_fits", {})[row["name"]] = gate_entry(fit_path)
                    state.update(
                        status="fit_complete",
                        phase="awaiting_main_precision_postfit",
                        active=None,
                        scientific_complete=False,
                        formal_fit_authorized=False,
                        test_status="deferred_to_main",
                        postfit_mode=PRECISION_POSTFIT,
                    )
                    atomic_json(directory / "state.json", state)
                    return
                if queue["family"] in RESEARCH_FAMILIES:
                    from run_representation_postfit import (
                        postfit_command,
                        validate_research_complete,
                    )

                    state["phase"] = f"{row['name']}_postfit_validation"
                    run_stage(
                        queue,
                        row,
                        "postfit-validation",
                        postfit_command(queue, row, "validation"),
                        "0,1",
                        state,
                        guard,
                    )
                    # Analysis workers acquire GPU0 themselves; preserve the same
                    # lease lifecycle as the existing medium worker below.
                    gpu_leases.close()
                    state["phase"] = f"{row['name']}_research_analysis"
                    run_stage(
                        queue,
                        row,
                        "research-analysis",
                        postfit_command(queue, row, "analysis"),
                        "0",
                        state,
                        guard,
                    )
                    state.setdefault("research_completed", {})[row["name"]] = (
                        validate_research_complete(queue, row)
                    )
                    atomic_json(directory / "state.json", state)
                    for gpu in ("0", "1"):
                        gpu_leases.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
                    continue
                eval_stage = "last" if queue["family"] in LAST_ONLY_FAMILIES else "best-last"
                phase = "last" if queue["family"] in LAST_ONLY_FAMILIES else "best_last"
                state["phase"] = f"{row['name']}_{phase}"
                run_stage(queue, row, eval_stage, evaluation_command(path), "0,1", state, guard)
                state["completed"][row["name"]] = validate_complete(row["plan"])
                atomic_json(directory / "state.json", state)
                if queue["family"] in FGPO_FAMILIES:
                    from run_fgpo_medium import medium_command, validate_medium_complete

                    # The child acquires the normal GPU0 lease itself. Do not
                    # bypass the worker's lock or hold the same lock in its parent.
                    gpu_leases.close()
                    state["phase"] = f"{row['name']}_medium_analysis"
                    run_stage(
                        queue,
                        row,
                        "medium-analysis",
                        medium_command(queue, row),
                        "0",
                        state,
                        guard,
                    )
                    state.setdefault("medium_analysis", {})[row["name"]] = validate_medium_complete(
                        queue, row
                    )
                    atomic_json(directory / "state.json", state)
                    for gpu in ("0", "1"):
                        gpu_leases.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
            if queue["family"] in RESEARCH_FAMILIES:
                state.update(
                    status="research_complete",
                    phase="awaiting_main_analysis",
                    scientific_complete=False,
                    test_status="deferred_until_design_lock",
                )
                atomic_json(directory / "state.json", state)
                atomic_json(directory / "RESEARCH_COMPLETE.json", state)
                return
            state.update(status="complete", phase="complete", scientific_complete=True)
            atomic_json(directory / "state.json", state)
            atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        if resume_fit and not promotion_started:
            atomic_json(
                directory / f"FIT_PROMOTION_REJECTED-{time.time_ns()}.json",
                dict(
                    error_type=type(error).__name__,
                    error=str(error),
                    pid=os.getpid(),
                    source_commit=queue["source_commit"],
                    queue_sha256=queue_sha,
                    state_preserved=True,
                    fit_started=False,
                ),
            )
            raise
        state.update(
            status="failed", phase="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "FAILURE.json", state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "manifest", "runtime", "output", "shared-policy", "queue"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-fit", action="store_true")
    parser.add_argument("--fit-only-row", choices=tuple(PRECISION_ARMS))
    parser.add_argument("--eval-only-row", choices=("DP", "FN", "FG", "FB", "PO"))
    parser.add_argument(
        "--only-arms",
        nargs="+",
        choices=(*PENDING_FGPO_ARMS, *R3_CONTINUATION_ARMS, "R13-SRAMPNC"),
    )
    args = parser.parse_args()
    fields = ("source", "manifest", "runtime", "output", "shared_policy")
    if (
        args.prepare
        and not args.queue
        and not args.eval_only_row
        and not args.resume_fit
        and args.fit_only_row is None
        and all(getattr(args, key) is not None for key in fields)
    ):
        prepare(
            args.source,
            args.manifest,
            args.runtime,
            args.output,
            read_json(args.shared_policy),
            only_arms=args.only_arms,
            preflight_only=args.preflight_only,
        )
    elif (
        args.queue
        and not args.prepare
        and args.only_arms is None
        and all(getattr(args, key) is None for key in fields)
    ):
        execute(
            read_json(args.queue),
            args.queue.resolve().parent,
            eval_only_row=args.eval_only_row,
            preflight_only=args.preflight_only,
            resume_fit=args.resume_fit,
            fit_only_row=args.fit_only_row,
        )
    else:
        parser.error(
            "choose --prepare with source/manifest/runtime/output/shared-policy or --queue"
        )


if __name__ == "__main__":
    main()
