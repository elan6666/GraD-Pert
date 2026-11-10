"""Run the sealed server stages for a small-softmax platform workflow."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import select
import subprocess
import sys
from pathlib import Path

SERVER = Path("/data/yilangliu/GraD-Pert")
PYTHON = SERVER / "source/.venv/bin/python"
ANALYSIS = SERVER / "development/source-small-softmax-analysis-20c1bdf-20261109"
ANALYSIS_COMMIT = "20c1bdf2ca2e2424b13fa0a1f4060362f91a95ec"
ANALYSIS_SHA256 = "2f54a6a51d196fd3909f7dc8dcfbd49e01603ed1ed9ab5db95e2d7095dcb160f"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": stat[19], "command_sha256": digest(proc / "cmdline")}


def verified_inputs(spec):
    plan_file = Path(spec["training_plan"]).resolve(strict=True)
    root = plan_file.parent
    require(root.is_relative_to(SERVER), "training plan must stay on the research server")
    require(digest(plan_file) == spec["plan_sha256"], "training plan hash mismatch")
    plan = read(plan_file)
    require(plan["run_id"] == spec["run_id"], "run identity mismatch")
    require(plan["cuda_memory_fraction"] == 0.9, "training requires the sealed memory ceiling")
    run = Path(plan["run_root"]).resolve()
    require(run.is_relative_to(root), "run root must be inside the sealed stage")
    source = Path(plan["repository_root"]).resolve(strict=True)
    require(source.is_relative_to(SERVER), "training source must stay on the research server")
    require(digest(Path(plan["config"])) == plan["config_sha256"], "config hash mismatch")
    check_checkout(source, plan["source_commit"])
    for file, key in (
        ("run-expression-val.py", "evaluation_launcher_sha256"),
        ("verify-expression-val.py", "verifier_sha256"),
    ):
        require(digest(root / file) == spec[key], f"{file} changed")
    return plan_file, plan, root, run, source


def check_checkout(source, commit):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    require(
        git("rev-parse", "HEAD") == commit and not git("status", "--porcelain"),
        f"source checkout differs from the pinned commit at {source}",
    )


def accepted_training(spec, root, run):
    require(not (root / "train-failure.json").exists(), "training recorded a failure")
    done = read(root / "train-complete.json")
    require(done["plan_sha256"] == spec["plan_sha256"], "training terminal plan mismatch")
    terminal = run / "TRAIN_COMPLETE.json"
    train = read(terminal)
    require(
        train["identity"]["run_id"] == spec["run_id"] and train["epoch"] == 2,
        "training terminal is for another run or epoch",
    )
    checkpoint = (run / "fit" / train["last"]["file"]).resolve(strict=True)
    require(checkpoint.is_relative_to(run / "fit"), "last checkpoint escaped fit root")
    require(digest(checkpoint) == train["last"]["sha256"], "last checkpoint hash mismatch")
    return terminal, train


def condition_aggregates(completed, spec, receipt_sha):
    result = completed["result"]
    summary = result.get("response_diagnostics")
    enabled = spec.get("response_diagnostics", False)
    require(
        completed["plan"].get("response_diagnostics", False) is enabled
        and (summary is not None) is enabled,
        "frozen response diagnostics do not match the sealed workflow",
    )
    if summary is not None:
        require(
            summary["schema"] == "frozen-response-diagnostics-v1"
            and summary["candidate_count"] == 445
            and 0 <= summary["valid_candidate_count"] <= 445,
            "invalid response diagnostics summary",
        )

    def scalar(value):
        require(
            value is None or (type(value) in (int, float) and math.isfinite(value)),
            "invalid aggregate scalar",
        )
        return value

    def metrics(block):
        return (
            {}
            if block is None
            else {
                row["metric_id"]: {"value": scalar(row["value"]), "reason": row["reason"]}
                for row in block["results"]
                if row["metric_id"]
                in (
                    "txpert_macro_pearson_delta_all",
                    "trishift_pearson_delta_all",
                    "systema_pearson_all",
                )
            }
        )

    def pick(data, names):
        return {name: scalar(data[name]) for name in names}

    modes = ("model", "control_copy", "train_mean")
    rows = []
    for row in result["conditions"]:
        groups = row["metric_gene_groups"]
        item = {
            "condition": row["condition_id"],
            "truth_count": len(row["truth_row_ids"]),
            "mse": scalar(row["loss"]),
            "pearson": {
                "all": metrics(row["metrics"]),
                "seen": metrics(groups["seen_expression"]),
                "unseen": metrics(groups["unseen_expression"]),
            },
        }
        if enabled:
            diagnostic = row["response_diagnostics"]
            item["effects"] = {
                **pick(
                    diagnostic,
                    (
                        "truth_effect_rms",
                        "pred_effect_rms",
                        "effect_rms_ratio",
                        "effect_cosine",
                        "target_direction_accuracy",
                        "mse_gain_over_copy",
                        "mse_gain_over_train_mean",
                    ),
                ),
                "mse": pick(diagnostic["mse"], modes),
                "retrieval": {
                    mode: pick(
                        diagnostic["retrieval"][mode], ("rank", "reciprocal_rank", "top1", "top5")
                    )
                    for mode in modes
                },
                "top_effects": {
                    k: {
                        mode: pick(
                            diagnostic["top_effects"][k][mode],
                            (
                                "truth_top_count",
                                "predicted_top_count",
                                "recall",
                                "precision",
                                "direction_accuracy",
                            ),
                        )
                        for mode in modes
                    }
                    for k in ("20", "50")
                },
            }
        if enabled and "effect_groups" in diagnostic:
            geometry = (
                "truth_effect_rms",
                "pred_effect_rms",
                "effect_rms_ratio",
                "effect_cosine",
                "amplitude_mse",
                "direction_mse",
            )
            item["effects"].update(
                pick(diagnostic, ("amplitude_mse", "direction_mse")),
                effect_groups={
                    name: pick(values, geometry)
                    for name, values in diagnostic["effect_groups"].items()
                    if name in ("seen_expression", "unseen_expression")
                },
                retrieval_without_val_targets={
                    mode: pick(
                        diagnostic["retrieval_without_val_targets"][mode],
                        ("rank", "reciprocal_rank", "top1", "top5"),
                    )
                    for mode in modes
                },
            )
        rows.append(item)
    require(
        len({row["condition"] for row in rows}) == len(rows) == 445,
        "aggregate diagnostics do not cover frozen validation",
    )
    return {
        "schema": "frozen-condition-aggregates-v1",
        "run_id": spec["run_id"],
        "validation_sha256": receipt_sha,
        "response_diagnostics": summary,
        "rows": rows,
    }


def accepted_outputs(spec, root, run):
    train_file, train = accepted_training(spec, root, run)
    require(not (root / "evaluation-failure.json").exists(), "frozen evaluation recorded a failure")
    terminal = read(root / "evaluation-complete.json")
    plan = root / "EXPRESSION_VAL_PLAN.json"
    result = root / "expression-val/COMPLETE.json"
    require(
        digest(plan) == terminal["plan_sha256"] and digest(result) == terminal["receipt_sha256"],
        "evaluation terminal hashes differ",
    )
    val_file = root / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json"
    val = read(val_file)
    require(
        val["run_id"] == spec["run_id"]
        and val["status"] == "main_independently_accepted_frozen_expression_validation"
        and val["validation_split"] == "val"
        and val["validation_conditions"] == 445
        and val["checkpoint_sha256"] == train["last"]["sha256"]
        and val["training_sha256"] == digest(train_file)
        and val["evaluation_sha256"] == terminal["receipt_sha256"]
        and val["independent_verifier_sha256"] == spec["verifier_sha256"],
        "independent validation receipt differs from the sealed run",
    )
    analysis_dir = root / "analysis-20c1bdf"
    analysis_file = analysis_dir / "receipt.json"
    analysis = read(analysis_file)
    require(
        analysis["status"] == "completed"
        and analysis["run_id"] == spec["run_id"]
        and analysis["analysis_identity"]["commit"] == ANALYSIS_COMMIT
        and analysis["analysis_identity"]["script_sha256"] == ANALYSIS_SHA256
        and analysis["input_hashes"]["TRAIN_COMPLETE.json"] == digest(train_file)
        and terminal["receipt_sha256"] in analysis["input_hashes"].values()
        and analysis["model_loads"] == analysis["optimizer_updates"] == 0
        and not analysis["gpu_work"],
        "analysis receipt differs from the standard CPU analysis",
    )
    files = {
        str(train_file): digest(train_file),
        str(val_file): digest(val_file),
        str(analysis_file): digest(analysis_file),
        str(root / "PLAN.json"): spec["plan_sha256"],
    }
    for name, identity in analysis["outputs"].items():
        file = analysis_dir / name
        require(
            file.parent == analysis_dir and file.is_file() and not file.is_symlink(),
            "unexpected analysis artifact path",
        )
        require(
            0 < file.stat().st_size == identity["bytes"] <= 10 * 1024**2,
            f"invalid artifact size for {name}",
        )
        require(digest(file) == identity["sha256"], f"analysis artifact hash mismatch: {name}")
        files[str(file)] = identity["sha256"]
    require(
        [row["optimizer_steps"] for row in val["training_history"]]
        == [spec["expected_updates"] // 2, spec["expected_updates"]],
        "training updates differ from the workflow budget",
    )
    return {
        "status": "completed",
        "run_id": spec["run_id"],
        "files": files,
        "condition_aggregates": condition_aggregates(
            read(result), spec, terminal["receipt_sha256"]
        ),
    }


def run_stages(spec, workflow):
    plan_file, plan, root, run, source = verified_inputs(spec)
    identity = process_identity(os.getpid())
    stage = "training"

    def publish(status, **extra):
        write(
            workflow / "state.json",
            {
                "status": status,
                "stage": stage,
                "owner": identity,
                "run_id": spec["run_id"],
                **extra,
            },
        )

    def launch(arguments, gpu, cwd=source):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gpu,
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="8",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=f"{source / 'src'}:{source}",
            MPLBACKEND="Agg",
        )
        subprocess.run([str(PYTHON), *map(str, arguments)], cwd=cwd, env=env, check=True)

    try:
        publish("running")
        if not (root / "train-complete.json").exists():
            require(
                spec.get("response_diagnostics") is True,
                "new workflow runs require frozen response diagnostics in the sealed launcher",
            )
            require(
                not run.exists() and not (root / "train-owner.json").exists(),
                "training was already dispatched without a successful terminal",
            )
            require(not (root / "train-failure.json").exists(), "training recorded a failure")
            preflight = read(root / "MAIN_PREFLIGHT_ACCEPTANCE.json")
            require(
                preflight["status"] == "passed"
                and preflight["run_id"] == spec["run_id"]
                and preflight["plan_sha256"] == spec["plan_sha256"]
                and preflight["receipt_sha256"] == digest(root / "preflight/receipt.json"),
                "independent training preflight acceptance differs",
            )
            launch(
                [
                    source / "scripts/v2/run_small_softmax.py",
                    "--plan",
                    plan_file,
                    "--plan-sha256",
                    spec["plan_sha256"],
                    "--stage",
                    "train",
                    "--gpu-uuid",
                    ",".join(spec["training_gpu_uuids"]),
                ],
                plan["gpu"],
            )
        accepted_training(spec, root, run)

        stage = "frozen_validation"
        publish("running")
        if not (root / "evaluation-complete.json").exists():
            require(
                not any(
                    (root / name).exists()
                    for name in (
                        "evaluation-owner.json",
                        "evaluation-failure.json",
                        "expression-val",
                    )
                ),
                "an earlier validation attempt requires reconciliation",
            )
            launch([root / "run-expression-val.py"], spec["evaluation_gpu"])
        require(not (root / "evaluation-failure.json").exists(), "frozen validation failed")
        if not (root / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json").exists():
            launch([root / "verify-expression-val.py"], "")

        stage = "standard_analysis"
        publish("running")
        analyzer = ANALYSIS / "scripts/v2/analyze_completed_ablation.py"
        check_checkout(ANALYSIS, ANALYSIS_COMMIT)
        require(digest(analyzer) == ANALYSIS_SHA256, "standard analysis script changed")
        analysis_dir = root / "analysis-20c1bdf"
        if not analysis_dir.exists():
            launch(
                [
                    analyzer,
                    "--run-root",
                    run,
                    "--output",
                    analysis_dir,
                    "--name",
                    spec["experiment_id"],
                    "--expression-validation",
                    root / "expression-val/COMPLETE.json",
                ],
                "",
                cwd=ANALYSIS,
            )
        write(workflow / "ready.json", accepted_outputs(spec, root, run))
        publish("completed")
    except Exception as error:
        publish("failed", error=str(error))
        raise


def alive(owner):
    try:
        return process_identity(owner["pid"]) == owner
    except (FileNotFoundError, ProcessLookupError):
        return False


def start(spec, driver_bytes, retry):
    _, _, root, _, _ = verified_inputs(spec)
    workflow = root / "platform-workflow-v2"
    workflow.mkdir(exist_ok=True)
    require(workflow.resolve().parent == root, "workflow directory escaped its stage")
    with (workflow / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec_path = workflow / "spec.json"
        script = workflow / "driver.py"
        owner_path = workflow / "owner.json"
        state = workflow / "state.json"
        if spec_path.exists():
            require(read(spec_path) == spec, "a different workflow already owns this run")
        else:
            write(spec_path, spec)
        if script.exists():
            require(script.read_bytes() == driver_bytes, "sealed workflow driver changed")
        else:
            with script.open("xb") as stream:
                stream.write(driver_bytes)
        if owner_path.exists():
            owner = read(owner_path)
            if (
                alive(owner)
                or (state.exists() and read(state)["status"] == "completed")
                or not retry
            ):
                return owner
        log = workflow / "driver.log"
        with log.open("ab") as stream:
            process = subprocess.Popen(
                [str(PYTHON), "-u", str(script), "run", str(workflow)],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owner = process_identity(process.pid)
        write(owner_path, owner)
        return owner


def wait(spec, owner):
    _, _, root, _, _ = verified_inputs(spec)
    workflow = root / "platform-workflow-v2"
    require(read(workflow / "spec.json") == spec, "workflow spec changed while waiting")
    require(read(workflow / "owner.json") == owner, "workflow owner changed while waiting")
    if alive(owner):
        try:
            fd = os.pidfd_open(owner["pid"])
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                if alive(owner):
                    poller = select.poll()
                    poller.register(fd, select.POLLIN)
                    poller.poll()
            finally:
                os.close(fd)
    state = read(workflow / "state.json")
    require(
        state["status"] == "completed" and state["owner"] == owner,
        f"server workflow stopped at {state['stage']}: {state.get('error', state['status'])}",
    )
    return read(workflow / "ready.json")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        workflow = Path(sys.argv[2])
        run_stages(read(workflow / "spec.json"), workflow)
    else:
        spec = json.loads(sys.argv[2])
        if mode == "verify":
            _, _, root, run, _ = verified_inputs(spec)
            result = accepted_outputs(spec, root, run)
        elif mode == "start":
            result = start(
                spec, base64.b64decode(sys.argv[3], validate=True), sys.argv[4] == "retry"
            )
        elif mode == "wait":
            result = wait(spec, json.loads(sys.argv[3]))
        else:
            raise ValueError(f"unknown mode {mode}")
        print(json.dumps(result))
