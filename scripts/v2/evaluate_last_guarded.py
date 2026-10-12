"""Guard an independent native last-checkpoint test; preserve its training run."""

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

from run_group import lock
from run_unified_ddp import _capture_process_identity
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
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.evaluation import merge_evaluation_shards

SCHEMA = "gradpert-v2-last-guarded-1"
SERVER_ROOT = Path("/data/yilangliu")
POLL_SECONDS = 5


def seal(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_relative_to(SERVER_ROOT):
        raise ValueError("sealed inputs must remain on server")
    return {"path": str(path), "sha256": sha256_file(path)}


def check_seal(entry: dict) -> Path:
    path = Path(entry["path"])
    if (
        not path.is_absolute()
        or not path.resolve().is_relative_to(SERVER_ROOT)
        or sha256_file(path) != entry["sha256"]
    ):
        raise ValueError("sealed evaluation input changed")
    return path


def source_identity(source: Path, runtime: dict, repository: str):
    return inspect_source_identity(
        source,
        formal=True,
        expected_repository=repository,
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )


def scientific_contract(plan: dict) -> dict:
    source = Path(plan["repository_root"]).resolve()
    config = load_experiment_config(plan["config"]).model_dump(mode="json")
    for key in ("config", "runtime", "publication"):
        if sha256_file(Path(plan[key])) != plan[key + "_sha256"]:
            raise ValueError("scientific config/runtime/publication changed")
    runtime = read_json(Path(plan["runtime"]))
    if (
        read_json(Path(plan["run_root"]) / "resolved_config.json") != config
        or config != plan["resolved_config"]
        or config["training"]["max_epochs"]["value"] != 4
        or config["training"]["eval_batch_size"]["value"] != 300
        or Path(runtime["repository_root"]).resolve() != source
        or Path(runtime["data_root"]).resolve() != Path(plan["data_root"]).resolve()
        or Path(runtime["publication_receipt"]).resolve() != Path(plan["publication"]).resolve()
        or runtime["publication_sha256"] != plan["publication_sha256"]
        or Path(plan["run_root"]).resolve() != Path(runtime["runs_root"]).resolve() / plan["run_id"]
    ):
        raise ValueError("scientific runtime/data/config binding changed")
    if any(
        not Path(plan[k]).resolve().is_relative_to(SERVER_ROOT)
        for k in ("repository_root", "config", "run_root", "data_root")
    ):
        raise ValueError("scientific inputs stay on server")
    identity = source_identity(source, runtime, config["source_code"]["repository"])
    if identity.dirty or identity.commit != plan["source_commit"]:
        raise ValueError("scientific source must match clean published training")
    training = validate_training_stage(plan)
    checkpoint = Path(plan["run_root"]) / "fit" / training["last"]["file"]
    if (
        training["epoch"] != 4
        or training["last"]["epoch"] != 4
        or sha256_file(checkpoint) != training["last"]["sha256"]
        or sha256_file(Path(plan["run_root"]) / "fit/last.pt") != training["last"]["sha256"]
        or training["identity"].get("resolved_config_sha256") != sha256_json(config)
        or training["identity"]["data"]
        != read_json(Path(plan["run_root"]) / "run_manifest.json")["data"]
    ):
        raise ValueError("last checkpoint must be the complete epoch4 training boundary")
    return training


def absent_outputs(plan: dict) -> None:
    # Interrupted native postfit outputs are historical evidence, preserved in place.
    root = Path(plan["run_root"])
    if any(root.rglob("*.pkl")):
        raise ValueError("training root requires zero PKL")


def native_args(plan: dict, training: dict, output: Path) -> dict:
    return {
        "config": plan["config"],
        "runtime": plan["runtime"],
        "training_run_root": plan["run_root"],
        "checkpoint": str(Path(plan["run_root"]) / "fit" / training["last"]["file"]),
        "checkpoint_sha256": training["last"]["sha256"],
        "checkpoint_role": "last",
        "output_root": str(output / "evaluation"),
        "gpu": "0,1",
        "split": "test",
    }


RESOLVE_CODE = (
    "import argparse,json,sys;from pathlib import Path;"
    "from gradpert.execution.v2_checkpoint_eval import resolve_evaluation_plan;"
    "a=json.loads(sys.argv[1]);"
    "a.update({k:Path(a[k]) for k in "
    "('config','runtime','training_run_root','checkpoint','output_root')});"
    "p=resolve_evaluation_plan(argparse.Namespace(**a));"
    "p.update(cpu_training_state=True,cuda_memory_fraction=0.9);print(json.dumps(p))"
)


def resolve_native(plan: dict, training: dict, output: Path) -> dict:
    result = subprocess.check_output(
        [sys.executable, "-c", RESOLVE_CODE, json.dumps(native_args(plan, training, output))],
        text=True,
        cwd=plan["repository_root"],
        env=child_environment(plan["repository_root"], ""),
    )
    return json.loads(result)


def child_command(queue: dict) -> list[str]:
    code = (
        "import argparse,json,sys;from pathlib import Path;"
        "from gradpert.execution.v2_checkpoint_eval import "
        "resolve_evaluation_plan,execute_evaluation_plan;"
        "from gradpert.evaluation.state import prepare_evaluation_state;"
        "q=json.load(open(sys.argv[1]));a=q['native_args'].copy();"
        "a.update({k:Path(a[k]) for k in "
        "('config','runtime','training_run_root','checkpoint','output_root')});"
        "p=resolve_evaluation_plan(argparse.Namespace(**a));"
        "p.update(cpu_training_state=True,cuda_memory_fraction=0.9);"
        "\nif p != q['evaluation_plan']:\n"
        "    raise ValueError('native evaluation binding changed')\n"
        "c=q['plan']['resolved_config'];"
        "prepare_evaluation_state(dataset_id=c['dataset_id'],protocol_id=c['data']['protocol_id'],"
        "data_root=Path(p['data_root']),evaluation_protocol='v2');execute_evaluation_plan(p)"
    )
    return [sys.executable, "-c", code, str(Path(queue["output"]) / "queue.json")]


def prepare(source: Path, runtime: Path, training_plan: Path, policy: dict, output: Path) -> dict:
    source, runtime, training_plan, output = (
        p.resolve() for p in (source, runtime, training_plan, output)
    )
    if output.exists() or not output.is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server controller output required")
    plan = read_json(training_plan)
    training = scientific_contract(plan)
    absent_outputs(plan)
    controller_runtime = read_json(runtime)
    if (
        source != Path(__file__).resolve().parents[2]
        or Path(controller_runtime["repository_root"]).resolve() != source
        or Path(controller_runtime["data_root"]).resolve() != Path(plan["data_root"]).resolve()
    ):
        raise ValueError("controller source/runtime/data binding changed")
    identity = source_identity(
        source, controller_runtime, plan["resolved_config"]["source_code"]["repository"]
    )
    if identity.dirty:
        raise ValueError("controller source must be clean and published")
    root = Path(plan["run_root"])
    paths = [
        root / p
        for p in (
            "TRAIN_COMPLETE.json",
            "launch.json",
            "resolved_config.json",
            "run_manifest.json",
            "fit/epoch_state.json",
            "fit/history.json",
            "fit/last.pt",
        )
    ]
    paths.append(root / "fit" / training["last"]["file"])
    paths.extend(Path(plan[k]) for k in ("config", "runtime", "publication"))
    queue = {
        "schema": SCHEMA,
        "output": str(output),
        "launch": seal(training_plan),
        "plan": plan,
        "training": training,
        "training_files": [seal(p) for p in paths],
        "original_receipts": [seal(p) for p in sorted(root.rglob("*.json"))],
        "native_args": native_args(plan, training, output),
        "evaluation_plan": resolve_native(plan, training, output),
        "shared_gpu_policy": copy.deepcopy(policy),
        "policy_sha256": sha256_json(policy),
        "controller": {
            "repository_root": str(source),
            "commit": identity.commit,
            "runtime": seal(runtime),
            "publication": seal(Path(controller_runtime["publication_receipt"])),
            "script": seal(Path(__file__)),
        },
    }
    verify(queue, fresh=True)
    output.mkdir()
    atomic_json(output / "queue.json", queue)
    return queue


def verify(queue: dict, *, fresh: bool) -> dict:
    if queue.get("schema") != SCHEMA or read_json(check_seal(queue["launch"])) != queue["plan"]:
        raise ValueError("sealed training launch changed")
    policy = queue["shared_gpu_policy"]
    validate_policy(policy)
    if (
        policy["schema"] != DYNAMIC_SCHEMA
        or policy["max_own_memory_fraction"] != 0.9
        or sha256_json(policy) != queue["policy_sha256"]
    ):
        raise ValueError("exact approved memory90 policy required")
    for entry in queue["training_files"]:
        check_seal(entry)
    originals = queue["original_receipts"]
    observed = sorted(Path(queue["plan"]["run_root"]).rglob("*.json"))
    if [str(p.resolve()) for p in observed] != [e["path"] for e in originals]:
        raise ValueError("original scientific receipts were added or removed")
    for entry in originals:
        check_seal(entry)
    plan = queue["plan"]
    training = scientific_contract(plan)
    if training != queue["training"]:
        raise ValueError("training boundary changed")
    controller = queue["controller"]
    runtime = read_json(check_seal(controller["runtime"]))
    publication = check_seal(controller["publication"])
    source = Path(controller["repository_root"]).resolve()
    script = check_seal(controller["script"])
    if (
        source != Path(__file__).resolve().parents[2]
        or script != Path(__file__).resolve()
        or Path(runtime["repository_root"]).resolve() != source
        or Path(runtime["data_root"]).resolve() != Path(plan["data_root"]).resolve()
        or Path(runtime["publication_receipt"]).resolve() != publication
        or runtime["publication_sha256"] != controller["publication"]["sha256"]
    ):
        raise ValueError("controller execution identity changed")
    identity = source_identity(
        source, runtime, plan["resolved_config"]["source_code"]["repository"]
    )
    if identity.dirty or identity.commit != controller["commit"]:
        raise ValueError("controller source changed")
    output = Path(queue["output"])
    protected = [
        source,
        Path(plan["repository_root"]).resolve(),
        Path(plan["run_root"]).resolve(),
        Path(plan["test_root"]).resolve(),
        Path(queue["launch"]["path"]).parent.resolve(),
    ]
    if (
        not output.is_absolute()
        or not output.resolve().is_relative_to(SERVER_ROOT)
        or any(output.resolve().is_relative_to(p) for p in protected)
    ):
        raise ValueError("new output must preserve scientific source/run/queue")
    expected = native_args(plan, training, output)
    ep = queue["evaluation_plan"]
    if (
        queue["native_args"] != expected
        or ep["checkpoint_role"] != "last"
        or ep["checkpoint_epoch"] != 4
        or ep["checkpoint_sha256"] != training["last"]["sha256"]
        or ep["checkpoint"] != expected["checkpoint"]
        or ep["training_identity"] != training["identity"]
        or ep["training_run_root"] != plan["run_root"]
        or ep["config"] != plan["config"]
        or ep["config_sha256"] != plan["config_sha256"]
        or ep["runtime_sha256"] != plan["runtime_sha256"]
        or ep["data_root"] != plan["data_root"]
        or ep["output_root"] != expected["output_root"]
        or ep["gpu"] != ["0", "1"]
        or ep["split"] != "test"
        or ep.get("cpu_training_state") is not True
        or ep.get("cuda_memory_fraction") != 0.9
        or ep["evaluation_source"]
        != source_identity(
            Path(plan["repository_root"]),
            read_json(Path(plan["runtime"])),
            plan["resolved_config"]["source_code"]["repository"],
        ).payload()
        or ep["evaluation_source"]["commit"] != plan["source_commit"]
        or ep["evaluation_source"]["dirty"]
        or ep["evaluation_source"]["repository_root"] != plan["repository_root"]
        or ep.get("diagnostic_only")
        or "diagnostic_condition_count" in ep
    ):
        raise ValueError("native last/epoch4/data/source/output/GPU contract changed")
    absent_outputs(plan)
    if fresh and Path(ep["output_root"]).exists():
        raise FileExistsError("independent evaluation output must be fresh")
    return plan


def validate_complete(queue: dict) -> dict:
    ep = queue["evaluation_plan"]
    root = Path(ep["output_root"])
    if (root / "FAILURE.json").exists() or not (root / "COMPLETE.json").is_file():
        raise ValueError("native evaluation lacks successful terminal")
    receipt = read_json(root / "COMPLETE.json")
    if (
        receipt["plan"] != ep
        or read_json(root / "plan.json") != ep
        or receipt.get("zero_pkl") is not True
    ):
        raise ValueError("native completion plan/zeroPKL changed")
    shards = [read_json(root / f"worker-{i}.json") for i in range(2)]
    order = shards[0]["frozen_condition_order"]
    if not order or any(s["frozen_condition_order"] != order for s in shards):
        raise ValueError("evaluation shard frozen order changed")
    for i, shard in enumerate(shards):
        if (
            [r["condition_id"] for r in shard["result"]["conditions"]] != order[i::2]
            or shard["evaluation_environment"] != receipt["evaluation_environment"]
            or shard["runtime_measurement"]["condition_count"] != len(order[i::2])
            or shard["runtime_measurement"]["cell_batch"] != 300
            or shard["runtime_measurement"]["cpu_training_state"] is not True
            or shard["runtime_measurement"]["cuda_memory_fraction"] != 0.9
        ):
            raise ValueError("native shard identity/coverage/runtime changed")
    merged = merge_evaluation_shards([s["result"] for s in shards], ordered_conditions=tuple(order))
    merged["expression_exposure"] = shards[0]["result"]["expression_exposure"]
    merged = json.loads(json.dumps(merged))
    if merged != receipt["result"] or merged["split"] != "test":
        raise ValueError("native merged metrics differ from exact shards")
    groups = merged["metric_gene_groups"]
    if set(groups) != {"seen_expression", "unseen_expression"}:
        raise ValueError("both expression exposure groups required")
    for metrics in [merged["metrics"], *(g["metrics"] for g in groups.values())]:
        if len(metrics) != 6 or {m["metric_id"] for m in metrics} != METRICS:
            raise ValueError("eighteen native metrics required")
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
                or total != len(order)
                or (finite == 0 and value is not None)
                or (finite > 0 and (value is None or not math.isfinite(value)))
            ):
                raise ValueError("metric effective counts/availability differ")
    for row in merged["conditions"]:
        controls, truth = row["control_row_ids"], row["truth_row_ids"]
        if (
            len(controls) != 300
            or not truth
            or row["control_row_ids_sha256"] != sha256_json(controls)
            or row["truth_row_ids_sha256"] != sha256_json(truth)
            or not math.isfinite(row["loss"])
        ):
            raise ValueError("native condition ordered row identity or loss changed")
    if any(root.rglob("*.pkl")) or any(Path(queue["plan"]["run_root"]).rglob("*.pkl")):
        raise ValueError("whole evaluation and training roots require zero PKL")
    return {
        "evaluation_complete_sha256": sha256_file(root / "COMPLETE.json"),
        "checkpoint_role": "last",
        "checkpoint_epoch": 4,
        "checkpoint_sha256": ep["checkpoint_sha256"],
        "result": merged,
    }


def execute(queue: dict, directory: Path) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR
    ):
        raise ValueError("CPU visibility and allocator contract required")
    if (
        directory.resolve() != Path(queue["output"]).resolve()
        or not directory.resolve().is_relative_to(SERVER_ROOT)
        or (directory / "state.json").exists()
    ):
        raise ValueError("sealed fresh controller root required")
    if read_json(directory / "queue.json") != queue:
        raise ValueError("evaluation queue differs from file")
    plan = verify(queue, fresh=True)
    queue_sha = sha256_file(directory / "queue.json")
    state = {
        "status": "running",
        "pid": os.getpid(),
        "scientific_complete": False,
        "queue_sha256": queue_sha,
        "command": child_command(queue),
        "active": None,
    }
    child = None
    identity = None
    known = {}

    def guard():
        if sha256_file(directory / "queue.json") != queue_sha:
            raise ValueError("evaluation queue file changed")
        observed = snapshot(os.getpid())
        with (directory / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(observed) + "\n")
        admit(queue["shared_gpu_policy"], observed)
        return observed

    def remember():
        parents = parent_map()
        for pid in parents:
            if child is not None and descendant(pid, child.pid, parents):
                info = process_info(pid)
                if info is not None:
                    if info["uid"] != os.getuid():
                        raise RuntimeError("evaluation descendant UID changed")
                    known.setdefault(pid, info)
        for pid, expected in known.items():
            current = process_info(pid)
            if current is not None and any(
                current[k] != expected[k] for k in ("pid", "uid", "start_ticks", "command_sha256")
            ):
                raise RuntimeError("evaluation process identity changed")

    def stop():
        errors = []
        for pid, expected in reversed(list(known.items())):
            try:
                if process_info(pid) is not None:
                    terminate_tree(pid, expected)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            state["cleanup_errors"] = errors

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("BR evaluation controller interrupted")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(lock(directory / "controller.lock"))
        for gpu in ("0", "1"):
            stack.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
        if any(p["owned"] for p in guard()["processes"]):
            raise RuntimeError("other owned CUDA work exists")
        verify(queue, fresh=True)
        with (directory / "evaluation.log").open("wb") as stream:
            child = subprocess.Popen(
                child_command(queue),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=child_environment(plan["repository_root"], ""),
                cwd=plan["repository_root"],
                start_new_session=True,
            )
            identity = _capture_process_identity(child)
            known[child.pid] = {**identity, "uid": os.getuid()}
            info = process_info(child.pid)
            if (
                info is None
                or info["uid"] != os.getuid()
                or any(info[k] != identity[k] for k in ("pid", "start_ticks", "command_sha256"))
            ):
                raise RuntimeError("evaluation child identity missing or changed")
            state["active"] = identity
            atomic_json(directory / "state.json", state)
            while child.poll() is None:
                remember()
                guard()
                time.sleep(POLL_SECONDS)
            code = child.wait()
        state.update(exit_code=code, active=None)
        remember()
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("evaluation descendants remain after native exit")
            time.sleep(POLL_SECONDS)
            remember()
        if any(p["owned"] for p in guard()["processes"]):
            raise RuntimeError("evaluation CUDA work remains after exit")
        if code != 0:
            raise RuntimeError(f"native capped BR evaluation failed with exit {code}; no retry")
        verify(queue, fresh=False)
        completed = validate_complete(queue)
        state.update(status="complete", scientific_complete=True, completed=completed)
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        if identity is not None:
            stop()
        elif child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "FAILURE.json", state)
        raise
    finally:
        stack.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    for field in ("source", "runtime", "training-plan", "policy", "output", "queue"):
        parser.add_argument("--" + field, type=Path)
    args = parser.parse_args()
    fields = ("source", "runtime", "training_plan", "policy", "output")
    if args.prepare and not args.queue and all(getattr(args, k) is not None for k in fields):
        prepare(args.source, args.runtime, args.training_plan, read_json(args.policy), args.output)
    elif args.queue and not args.prepare and all(getattr(args, k) is None for k in fields):
        execute(read_json(args.queue), args.queue.resolve().parent)
    else:
        parser.error(
            "choose --prepare --source --runtime --training-plan --policy --output, or --queue"
        )


if __name__ == "__main__":
    main()
