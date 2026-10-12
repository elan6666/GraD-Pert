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
    PREFLIGHT_PRIORITY,
    PRIORITY,
    verify_manifest,
)
from generate_br400_group import (
    SCHEMA as FAMILY_SCHEMA,
)
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


def prepare(source: Path, manifest: Path, runtime: Path, output: Path, policy: dict) -> dict:
    source, manifest, runtime, output = (
        path.resolve() for path in (source, manifest, runtime, output)
    )
    if output.exists() or not output.is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server queue root required")
    matrix = verify_manifest(source, manifest)
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
        "mode": FAMILY_SCHEMA,
        "family": FAMILY_SCHEMA,
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
        or queue.get("mode") != FAMILY_SCHEMA
        or queue.get("family") != FAMILY_SCHEMA
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
    matrix = verify_manifest(source, Path(queue["manifest"]))
    runtime = read_json(Path(queue["runtime"]))
    if Path(runtime["repository_root"]).resolve() != source.resolve():
        raise ValueError("runtime source binding changed")
    if [row["name"] for row in queue["rows"]] != list(PRIORITY):
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
            or plan["cuda_memory_fraction"] != 0.85
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
        len({r["plan"][key] for r in queue["rows"]}) != 4
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
        or joint.get("status") != "passed"
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
        or not receipt.get("last_terms")
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in receipt["last_terms"].values()
        )
    ):
        raise ValueError("exact10 restore/inference/read-only joint-validation gate missing")
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
    if set(state["preflights"]) != set(PRIORITY) or set(state["eval_capacity"]) != {
        f"{name}-g{gpu}" for name in PRIORITY for gpu in ("0", "1")
    }:
        raise ValueError("all four update and eight eval gates required")
    indexed = {row["name"]: row for row in queue["rows"]}
    expected_data, expected_control = None, None
    for name in PRIORITY:
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
        "run_deferred_postfit(json.load(open(sys.argv[1])),gpu='0,1',cuda_memory_fraction=0.85)",
        str(path),
    ]


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
    training = validate_training_stage(plan)
    if (
        training["epoch"] != 4
        or training["last"]["epoch"] != 4
        or training["test_roles"] != ["best", "last"]
    ):
        raise ValueError("four epochs and both checkpoint roles required")
    original_complete = read_json(root / "COMPLETE.json")
    if finalize_deferred_run(plan, training) != original_complete:
        raise ValueError("canonical completion changed")
    rows = collect_run(root)
    if [row["role"] for row in rows] != ["best", "last"]:
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
        "test_sha256": {
            role: sha256_file(root / "fit" / f"{role}-test.json") for role in ("best", "last")
        },
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


def execute(queue: dict, directory: Path) -> None:
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
            for gpu in ("0", "1"):
                stack.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
            if any(p["owned"] for p in guard()["processes"]):
                raise RuntimeError("other owned CUDA work exists")
            indexed = {row["name"]: row for row in queue["rows"]}
            for name in PREFLIGHT_PRIORITY:
                row = indexed[name]
                command = [*probe(row, directory, shared=True), "--joint-validation-check"]
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
            common = validate_common_gates(queue, state)
            ready = {
                "common_identity": common,
                "queue_sha256": queue_sha,
                "source_commit": queue["source_commit"],
                "preflights": state["preflights"],
                "eval_capacity": state["eval_capacity"],
                "formal_order": list(PRIORITY),
            }
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
                validate_training_stage(row["plan"])
                state["phase"] = f"{row['name']}_best_last"
                run_stage(queue, row, "best-last", evaluation_command(path), "0,1", state, guard)
                state["completed"][row["name"]] = validate_complete(row["plan"])
                atomic_json(directory / "state.json", state)
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
    args = parser.parse_args()
    fields = ("source", "manifest", "runtime", "output", "shared_policy")
    if args.prepare and not args.queue and all(getattr(args, key) is not None for key in fields):
        prepare(
            args.source, args.manifest, args.runtime, args.output, read_json(args.shared_policy)
        )
    elif args.queue and not args.prepare and all(getattr(args, key) is None for key in fields):
        execute(read_json(args.queue), args.queue.resolve().parent)
    else:
        parser.error(
            "choose --prepare with source/manifest/runtime/output/shared-policy or --queue"
        )


if __name__ == "__main__":
    main()
