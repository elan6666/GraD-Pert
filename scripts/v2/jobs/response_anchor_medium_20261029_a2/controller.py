"""Frozen job submission wrapper; science runs only in the published native CLI.

One real script-backed owner, two fixed physical GPUs, sequential endpoint
phases. Failure stops new phase submission and drains already running children.
No retries, process kills, resource-policy patches or scientific computation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/data/yilangliu/GraD-Pert")
DEV = BASE / "development"
ANALYSIS_SHA = "7ae8824c7e8c65bb34c89df302f57999cbf38cef"
ANALYSIS = DEV / "source-response-anchor-analysis-7ae8824-20261028-a1"
PUBLICATION = DEV / "response-anchor-analysis-7ae8824-20261028-a1/PUBLICATION.json"
PUBLICATION_SHA = "e7d20a22770cbc9c287c7acd13ae21b0032d7793a845d60204ef602cff09f78e"
JOB = DEV / "response-anchor-medium-parallel-7ae8824-20261029-a2"
ARTIFACTS = JOB / "artifacts"
CONTROL = JOB / "controller"
OUTPUT = JOB / "execution"
METADATA = JOB / "metadata"
SEALER = ARTIFACTS / "seal_medium_metadata_v3.py"
SEALER_SHA = "f5aef3cf53f9fa270dc6ccb1e66d7d3a68472c710f2a15deb31bd84fb41ce21a"
WORKER = ANALYSIS / "scripts/v2/response_anchor_medium.py"
WORKER_SHA = "4637ab7bb61bb6bb2696b342b19b18ac71a550bdfba53958682fa31e7aac3eed"
SHARED_GPU = ANALYSIS / "scripts/v2/shared_gpu.py"
SHARED_GPU_SHA = "4ae721777c941c8b60d93d4aa9569fc2896d28dfd944de3718d156b72827bae9"
PYTHON = BASE / "source/.venv/bin/python"
GPU_MAP = {"R3-RI": "0", "R14-FRGA": "1"}
NATIVE = {
    "R3-RI": DEV / "source-representation-round3-au-f0cb20e",
    "R14-FRGA": DEV / "source-response-anchor-dc5233e-20261028-a2",
}
NATIVE_SHA = {
    "R3-RI": "f0cb20ebb2efab749970f2bf5d205ed2f358592c",
    "R14-FRGA": "dc5233ef1a243b2144107e5441a5a8db4f800fc6",
}
MODES = ("seal", "medium", "panel", "validate")
IDENTITY_OBSERVE_SECONDS = 2.0
IDENTITY_RETRY_SECONDS = 0.02
OWNER_FIELDS = ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
COUNTS = dict(
    conditions=1335,
    rows=5317,
    deep_conditions=128,
    deep_rows=506,
    gradient_conditions=32,
    gradient_rows=125,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def progress(root, value):
    temporary = root / f".PROGRESS.{os.getpid()}.tmp"
    write_new(temporary, value)
    temporary.replace(root / "PROGRESS.json")


def verify_files(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("immutable job input changed: " + str(path))


def argv_hash(command):
    return hashlib.sha256(b"".join(os.fsencode(arg) + b"\0" for arg in command)).hexdigest()


def observe_child_identity(process, command, inspect_process, owner):
    """Retry only missing identities of a live child, never an observed mismatch.

    The existing published process_info remains the sole identity source. A child
    that exits without an observed identity fails even when its return code is0.
    Every actual observation is retained to diagnose the unconfirmed exec race.
    """
    started = time.monotonic()
    deadline = started + IDENTITY_OBSERVE_SECONDS
    samples = []
    while True:
        error = None
        try:
            identity = inspect_process(process.pid)
        except Exception as failure:
            identity, error = None, repr(failure)
        returncode = process.poll()
        elapsed = time.monotonic() - started
        samples.append(
            dict(
                observed_at=now(),
                elapsed_seconds=elapsed,
                actual_process_identity=identity,
                identity_lookup_error=error,
                child_poll_returncode=returncode,
            )
        )
        valid = (
            identity is not None
            and identity.get("pid") == process.pid
            and identity.get("ppid") == owner["pid"]
            and identity.get("uid") == owner["uid"]
            and identity.get("command_sha256") == argv_hash(command)
            and bool(identity.get("start_ticks"))
        )
        if identity is not None or error is not None or returncode is not None:
            status = (
                "verified"
                if valid
                else "observed_mismatch"
                if identity is not None
                else "lookup_error"
                if error is not None
                else "child_exited_without_observed_identity"
            )
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            status = "live_child_missing_identity_deadline"
            break
        time.sleep(min(IDENTITY_RETRY_SECONDS, remaining))
    return dict(
        verified=valid,
        status=status,
        actual_process_identity=identity,
        identity_lookup_error=error,
        duration_seconds=time.monotonic() - started,
        bound_seconds=IDENTITY_OBSERVE_SECONDS,
        retry_interval_seconds=IDENTITY_RETRY_SECONDS,
        samples=samples,
    )


def spawn_phase(root, label, command, environment, inspect_process, owner, *, cwd):
    started_at = now()
    with (
        (root / f"{label}.stdout.log").open("x") as stdout,
        (root / f"{label}.stderr.log").open("x") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(os.environ, **environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )
    observation = observe_child_identity(process, command, inspect_process, owner)
    identity = observation["actual_process_identity"]
    identity_error = observation["identity_lookup_error"]
    valid = observation["verified"]
    receipt = dict(
        label=label,
        command=command,
        environment=environment,
        started_at=started_at,
        pid=process.pid,
        actual_process_identity=identity,
        identity_lookup_error=identity_error,
        identity_observation=observation,
        expected_raw_nul_command_sha256=argv_hash(command),
        process_identity_verified=valid,
        owner_process=owner,
    )
    try:
        write_new(root / f"{label}.STARTED.json", receipt)
    except BaseException:
        process.wait()
        raise
    return process, receipt


def finish_phase(root, process, receipt):
    polled = process.poll()
    if polled is None:
        raise ValueError("finish requires observed child completion")
    waited = process.wait()
    result = dict(
        **receipt,
        finished_at=now(),
        poll_returncode=polled,
        wait_returncode=waited,
        wait_called=True,
        status="completed" if waited == 0 and receipt["process_identity_verified"] else "failed",
    )
    write_new(root / (receipt["label"] + ".EXIT.json"), result)
    return result


def _schedule(plan, root, inspect_process, owner, verify_frozen, running, *, cwd):
    states = {row["endpoint"]: dict(next=0, status="pending") for row in plan}
    phases, errors = [], []
    failed = False

    def report():
        progress(root, dict(updated_at=now(), failed=failed, owner=owner, endpoints=states))

    def submit(row):
        nonlocal failed
        name = row["endpoint"]
        state = states[name]
        phase = row["modes"][state["next"]]
        label = name + "." + phase["mode"]
        try:
            verify_frozen()
            process, receipt = spawn_phase(
                root, label, phase["command"], row["env"], inspect_process, owner, cwd=cwd
            )
            running[name] = (process, receipt)
            state.update(status="running", phase=phase["mode"], pid=process.pid)
            if not receipt["process_identity_verified"]:
                failed = True
                errors.append(
                    dict(endpoint=name, phase=phase["mode"], error="child identity differs")
                )
        except Exception as error:
            failed = True
            state.update(status="failed", phase=phase["mode"])
            errors.append(dict(endpoint=name, phase=phase["mode"], error=repr(error)))
        report()

    # Submit one phase per GPU before waiting; a launch failure blocks later submissions.
    for row in plan:
        if failed:
            break
        submit(row)
    while running:
        completed = []
        for name, (process, receipt) in list(running.items()):
            if process.poll() is None:
                continue
            result = finish_phase(root, process, receipt)
            phases.append(result)
            del running[name]
            state = states[name]
            state["next"] += 1
            state["last_wait_returncode"] = result["wait_returncode"]
            state["status"] = "phase_completed" if result["status"] == "completed" else "failed"
            if result["status"] != "completed":
                failed = True
                errors.append(dict(endpoint=name, phase=state["phase"], error="phase failed"))
            completed.append(name)
        # Observe every ready exit before deciding whether to submit another phase.
        for name in completed:
            state = states[name]
            if state["next"] == len(MODES) and state["status"] != "failed":
                state["status"] = "completed"
            elif not failed:
                row = next(row for row in plan if row["endpoint"] == name)
                submit(row)
        if completed:
            report()
        if running:
            time.sleep(0.2)
    for row in plan:
        state = states[row["endpoint"]]
        state["skipped_modes"] = [p["mode"] for p in row["modes"][state["next"] :]]
        if state["status"] not in ("completed", "failed"):
            state["status"] = "stopped_after_peer_failure"
    report()
    result = dict(
        status="failed" if failed else "completed", endpoints=states, phases=phases, errors=errors
    )
    write_new(root / "PHASES.json", result)
    return result


def run_phases(plan, root, inspect_process, owner, verify_frozen, *, cwd):
    """Schedule a validated frozen plan, draining owned children even on I/O errors."""
    running = {}
    try:
        return _schedule(plan, root, inspect_process, owner, verify_frozen, running, cwd=cwd)
    finally:
        # No terminate/kill. Wait for every already launched phase on any failure.
        for process, receipt in list(running.values()):
            process.wait()
            try:
                if not (root / (receipt["label"] + ".EXIT.json")).exists():
                    finish_phase(root, process, receipt)
            except Exception:
                # Preserve draining even if a receipt cannot be written.
                continue


def validate_plan(document, seal, owner_binding):
    if (
        document.get("execute") != "one_owner_parallel_endpoints_sequential_phases"
        or document.get("endpoint_gpu_map") != GPU_MAP
        or document.get("endpoint_order") != list(GPU_MAP)
        or len(document.get("commands", [])) != 2
    ):
        raise ValueError("exact two-GPU command plan required")
    specs = {}
    for name, row in zip(GPU_MAP, document["commands"], strict=True):
        spec_path = METADATA / (name + ".SPEC.json")
        binding = seal["specs"][name]
        if binding["path"] != str(spec_path) or sha(spec_path) != binding["sha256"]:
            raise ValueError("actual SPEC binding differs")
        spec = read(spec_path)
        if (
            spec["schema"] != "response-anchor-direct-medium-spec-v1"
            or spec["name"] != name
            or spec["gpu"] != GPU_MAP[name]
            or spec["output"] != str(OUTPUT / name)
            or spec["resource_owner"] != owner_binding
            or spec["analysis_source"]["root"] != str(ANALYSIS)
            or spec["analysis_source"]["commit"] != ANALYSIS_SHA
            or spec["native_source"]["root"] != str(NATIVE[name])
            or spec["native_source"]["commit"] != NATIVE_SHA[name]
        ):
            raise ValueError("fixed source/GPU/output/owner SPEC differs")
        expected_env = dict(
            CUDA_VISIBLE_DEVICES=GPU_MAP[name],
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTHONPATH=str(NATIVE[name] / "src") + ":" + str(NATIVE[name]),
        )
        expected_modes = [
            dict(
                mode=mode,
                command=[
                    str(PYTHON),
                    str(WORKER),
                    mode,
                    "--spec",
                    str(spec_path),
                    "--spec-sha256",
                    binding["sha256"],
                ],
            )
            for mode in MODES
        ]
        if row != dict(endpoint=name, env=expected_env, modes=expected_modes):
            raise ValueError("command argv/environment differs from frozen native CLI")
        specs[name] = dict(path=str(spec_path), sha256=binding["sha256"])
    return document["commands"], specs


def terminal(specs):
    native = {}
    for name, binding in specs.items():
        output = OUTPUT / name
        result = read(output / "COMPLETE.json")
        spec = read(binding["path"])
        summary = read(output / "analysis/summary.json")
        panel = read(output / "panel/COMPLETE.json")
        if (
            result.get("status") != "completed"
            or result.get("updates") != 0
            or result.get("scientific_complete") is not False
            or result.get("spec_sha256") != binding["sha256"]
            or result.get("counts") != COUNTS
            or result.get("panel_rows") != 128
            or result.get("native_evaluation_source", {}).get("commit") != NATIVE_SHA[name]
            or result.get("analysis_adapter_source", {}).get("commit") != ANALYSIS_SHA
            or result.get("training_source", {}).get("commit") != NATIVE_SHA[name]
            or result.get("endpoint", {}).get("role") != "last"
            or result.get("endpoint", {}).get("epoch") != 2
            or any(
                row.get("checkpoint_sha256") != spec["checkpoint"]["sha256"]
                for row in (summary, panel)
            )
        ):
            raise ValueError("native endpoint COMPLETE identity differs: " + name)
        native[name] = dict(
            path=str(output / "COMPLETE.json"), sha256=sha(output / "COMPLETE.json")
        )
    if any(p.is_file() for p in JOB.rglob("*.pkl")):
        raise ValueError("whole-job zero-PKL postcondition failed")
    return native


def load_process_info():
    module_spec = importlib.util.spec_from_file_location("sealed_shared_gpu", SHARED_GPU)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.process_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-sha256", required=True)
    args = parser.parse_args()
    controller = Path(__file__).resolve()
    if (
        controller != ARTIFACTS / "controller.py"
        or not Path.cwd().resolve().is_relative_to("/data/yilangliu")
        or any(p.resolve() != p.absolute() for p in (JOB, ARTIFACTS, CONTROL, OUTPUT, METADATA))
    ):
        raise ValueError("fixed canonical server job-artifact paths required")
    immutable = {
        controller: args.controller_sha256,
        SEALER: SEALER_SHA,
        WORKER: WORKER_SHA,
        SHARED_GPU: SHARED_GPU_SHA,
        PUBLICATION: PUBLICATION_SHA,
    }
    verify_files(immutable)
    if read(PUBLICATION)["published_commit"] != ANALYSIS_SHA:
        raise ValueError("analysis publication commit differs")
    if any(path.exists() for path in (CONTROL, OUTPUT, METADATA)):
        raise ValueError("fresh controller/execution/metadata directories required; no overwrite")
    inspect_process = load_process_info()
    actual_owner = inspect_process(os.getpid())
    if actual_owner is None or actual_owner["script_basename"] != "controller.py":
        raise ValueError("real script-backed controller process required")
    owner = {key: actual_owner[key] for key in OWNER_FIELDS}
    CONTROL.mkdir()
    OUTPUT.mkdir()
    owner_path = OUTPUT / "DISPATCH_OWNER.json"
    write_new(owner_path, dict(schema="response-anchor-analysis-owner-v1", process=owner))
    owner_binding = dict(path=str(owner_path), sha256=sha(owner_path))
    immutable[owner_path] = owner_binding["sha256"]
    manifest = dict(
        schema="response-anchor-medium-parallel-job-v3",
        started_at=now(),
        owner=owner_binding,
        process=owner,
        analysis_commit=ANALYSIS_SHA,
        endpoint_gpu_map=GPU_MAP,
        controller=dict(path=str(controller), sha256=args.controller_sha256),
        sealer=dict(path=str(SEALER), sha256=SEALER_SHA),
        worker=dict(path=str(WORKER), sha256=WORKER_SHA),
        scientific_complete=False,
    )
    write_new(CONTROL / "MANIFEST.json", manifest)
    try:
        sealer_command = [
            str(PYTHON),
            str(SEALER),
            "--seal-dir",
            str(METADATA),
            "--output-root",
            str(OUTPUT),
            "--owner-path",
            str(owner_path),
            "--owner-sha256",
            owner_binding["sha256"],
        ]
        process, started = spawn_phase(
            CONTROL,
            "metadata-seal",
            sealer_command,
            {"CUDA_VISIBLE_DEVICES": ""},
            inspect_process,
            owner,
            cwd=ANALYSIS,
        )
        process.wait()
        sealed_exit = finish_phase(CONTROL, process, started)
        if sealed_exit["status"] != "completed":
            raise ValueError("metadata sealer failed")
        seal = read(METADATA / "METADATA_SEAL.json")
        commands = read(METADATA / "COMMANDS.json")
        plan, specs = validate_plan(commands, seal, owner_binding)
        for path in (
            METADATA / "COMMANDS.json",
            METADATA / "METADATA_SEAL.json",
            *(Path(v["path"]) for v in specs.values()),
        ):
            immutable[path] = sha(path)
        write_new(
            CONTROL / "SEALED_JOB_MANIFEST.json",
            dict(
                **manifest,
                specs=specs,
                immutable_bindings=[dict(path=str(p), sha256=h) for p, h in immutable.items()],
            ),
        )

        def verify_frozen():
            verify_files(immutable)
            current = inspect_process(os.getpid())
            if current is None or any(current.get(k) != owner[k] for k in OWNER_FIELDS):
                raise ValueError("controller owner process identity changed")

        result = run_phases(plan, CONTROL, inspect_process, owner, verify_frozen, cwd=ANALYSIS)
        if result["status"] != "completed":
            raise ValueError("endpoint phase failed; already running peer phases drained")
        verify_frozen()
        native = terminal(specs)
        write_new(
            CONTROL / "COMPLETE.json",
            dict(
                status="completed",
                completed_at=now(),
                updates=0,
                scientific_complete=False,
                analysis_commit=ANALYSIS_SHA,
                controller=manifest["controller"],
                sealer=manifest["sealer"],
                specs=specs,
                native_complete=native,
                whole_job_zero_pkl=True,
                phases_sha256=sha(CONTROL / "PHASES.json"),
            ),
        )
        return 0
    except Exception as error:
        write_new(
            CONTROL / "FAILURE.json",
            dict(
                status="failed",
                failed_at=now(),
                error=repr(error),
                scientific_complete=False,
                retries=0,
                killed_processes=[],
                analysis_commit=ANALYSIS_SHA,
            ),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
