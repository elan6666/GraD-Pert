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
from generate_pt1 import SCHEMA as PT1_SCHEMA
from generate_pt1 import verify_manifest as verify_pt1_manifest
from generate_pt2 import SCHEMA as PT2_SCHEMA
from generate_pt2 import verify_manifest as verify_pt2_manifest
from generate_representation_round1 import SCHEMA as RS_SCHEMA
from generate_representation_round1 import verify_manifest as verify_rs_manifest
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

SERVER_ROOT = Path("/data/yilangliu")
POLL_SECONDS = 5
FGPO_FAMILIES = (
    FGPO_SCHEMA,
    DINO_SCHEMA,
    DP2_SCHEMA,
    NO_SSL_SCHEMA,
    PT1_SCHEMA,
    PT2_SCHEMA,
    RS_SCHEMA,
)
LAST_ONLY_FAMILIES = (REMAINING_SCHEMA, FFN_SCHEMA, *FGPO_FAMILIES)
DEFERRED_SCDFM_POLICY = "scdfm_fit_gears_terminal_evaluation_deferred_v1"
PENDING_FGPO_ARMS = ("S12-L4", "CG1", "U24", "MR1", "P1", "C1", "O1", "K2-fixed-prefix")


def selected_matrix(matrix: dict, only_arms: list[str] | None) -> dict:
    """Select an ordered pending scope without changing the sealed family configs."""
    if only_arms is None:
        return matrix
    if (
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
    if read_json(manifest).get("schema") == RS_SCHEMA:
        return verify_rs_manifest(source, manifest)
    if read_json(manifest).get("schema") == PT2_SCHEMA:
        return verify_pt2_manifest(source, manifest)
    if read_json(manifest).get("schema") == PT1_SCHEMA:
        return verify_pt1_manifest(source, manifest)
    if read_json(manifest).get("schema") == NO_SSL_SCHEMA:
        return verify_no_ssl_manifest(source, manifest)
    if read_json(manifest).get("schema") in (DINO_SCHEMA, DP2_SCHEMA):
        return verify_dino_manifest(source, manifest)
    if read_json(manifest).get("schema") == FGPO_SCHEMA:
        return verify_fgpo_manifest(source, manifest)
    if read_json(manifest).get("schema") == FFN_SCHEMA:
        return verify_ffn_manifest(source, manifest)
    return verify_br_manifest(source, manifest)


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
) -> dict:
    source, manifest, runtime, output = (
        path.resolve() for path in (source, manifest, runtime, output)
    )
    if output.exists() or not output.is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server queue root required")
    matrix = selected_matrix(verify_manifest(source, manifest), only_arms)
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
    if queue["family"] in LAST_ONLY_FAMILIES:
        if (
            queue["family"] in (PT1_SCHEMA, PT2_SCHEMA)
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

    return validate_receipt(queue, row, binding, entry)


DATA_HASHES = (
    "canonical_sha256",
    "split_sha256",
    "expression_gene_order_sha256",
    "graph_gene_order_sha256",
    "graph_manifest_sha256",
    "genept_sha256",
)


def validate_common_gates(queue: dict, state: dict) -> dict:
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
    return training


def expected_epochs(plan: dict) -> int:
    """Short research and staged budgets require exact frozen family configurations."""
    from generate_pt2 import configuration

    if plan["resolved_config"]["training"]["max_epochs"]["value"] == 2:
        import yaml
        from generate_representation_round1 import configs

        source = Path(plan["repository_root"])
        raw = yaml.safe_load(Path(plan["config"]).read_text())
        if raw not in configs(source).values() or plan["resolved_config"] != load_experiment_config(
            plan["config"]
        ).model_dump(mode="json"):
            raise ValueError("two-epoch dispatch requires an exact representation round1 arm")
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


def execute(queue: dict, directory: Path, *, eval_only_row: str | None = None) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR
        or Path(queue["source"]).resolve() != Path(__file__).resolve().parents[2]
        or directory.resolve() != Path(queue["output"]).resolve()
        or (directory / "state.json").exists()
        or read_json(directory / "queue.json") != queue
    ):
        raise ValueError("fresh CPU-only execution must match sealed source/root/queue")
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
            atomic_json(directory / "PREFLIGHT_COMPLETE.json", ready)
            state["phase"] = "waiting_main_acceptance"
            atomic_json(directory / "state.json", state)
            acceptance = directory / "MAIN_ACCEPTANCE.json"
            while not acceptance.exists():
                guard()
                time.sleep(POLL_SECONDS)
            accepted = read_json(acceptance)
            if accepted != {**ready, "approved": True}:
                raise ValueError("main acceptance must bind all exact gates and published source")
            for row in queue["rows"]:
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
                if queue["family"] == RS_SCHEMA:
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
            if queue["family"] == RS_SCHEMA:
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
    parser.add_argument("--eval-only-row", choices=("DP", "FN", "FG", "FB", "PO"))
    parser.add_argument("--only-arms", nargs="+", choices=PENDING_FGPO_ARMS)
    args = parser.parse_args()
    fields = ("source", "manifest", "runtime", "output", "shared_policy")
    if (
        args.prepare
        and not args.queue
        and not args.eval_only_row
        and all(getattr(args, key) is not None for key in fields)
    ):
        prepare(
            args.source,
            args.manifest,
            args.runtime,
            args.output,
            read_json(args.shared_policy),
            only_arms=args.only_arms,
        )
    elif (
        args.queue
        and not args.prepare
        and args.only_arms is None
        and all(getattr(args, key) is None for key in fields)
    ):
        execute(
            read_json(args.queue), args.queue.resolve().parent, eval_only_row=args.eval_only_row
        )
    else:
        parser.error(
            "choose --prepare with source/manifest/runtime/output/shared-policy or --queue"
        )


if __name__ == "__main__":
    main()
