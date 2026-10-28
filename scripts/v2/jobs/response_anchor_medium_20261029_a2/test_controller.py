"""Bounded synthetic subprocess checks of submission, never model/scientific tests."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch, tmp_path):
    file = Path(__file__).with_name("controller.py")
    spec = importlib.util.spec_from_file_location("synthetic_controller", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = tmp_path / "job"
    job.mkdir()
    metadata, output, root = (job / name for name in ("metadata", "execution", "controller"))
    for path in (metadata, output, root):
        path.mkdir()
    worker = job / "synthetic_worker.py"
    worker.write_text(
        """import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--spec');p.add_argument('--spec-sha256');a=p.parse_args()
name=Path(a.spec).name.split('.SPEC')[0]
def event(kind):
 value=dict(kind=kind,endpoint=name,mode=a.mode,time=time.time(),pid=os.getpid(),ppid=os.getppid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'])
 fd=os.open(os.environ['SYNTHETIC_EVENTS'],os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
 os.write(fd,(json.dumps(value)+'\\n').encode());os.close(fd)
event('start')
fail=os.environ.get('SYNTHETIC_FAIL')=='1'
delay=0.3
if fail and a.mode=='medium':delay=0.6 if name=='R14-FRGA' else 0.12
time.sleep(delay)
event('end')
raise SystemExit(17 if fail and name=='R3-RI' and a.mode=='medium' else 0)
"""
    )
    monkeypatch.setattr(module, "JOB", job)
    monkeypatch.setattr(module, "METADATA", metadata)
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(module, "WORKER", worker)
    monkeypatch.setattr(module, "PYTHON", Path(sys.executable))
    monkeypatch.setenv("SYNTHETIC_EVENTS", str(job / "events.jsonl"))
    owner = dict(
        pid=os.getpid(),
        uid=os.getuid(),
        start_ticks="synthetic-owner",
        command_sha256="owner",
        script_basename="test_controller.py",
    )
    owner_binding = dict(path=str(output / "DISPATCH_OWNER.json"), sha256="owner-json")
    commands, seals = [], {}
    for name, gpu in module.GPU_MAP.items():
        spec_path = metadata / (name + ".SPEC.json")
        module.write_new(
            spec_path,
            dict(
                schema="response-anchor-direct-medium-spec-v1",
                name=name,
                gpu=gpu,
                output=str(output / name),
                resource_owner=owner_binding,
                analysis_source=dict(root=str(module.ANALYSIS), commit=module.ANALYSIS_SHA),
                native_source=dict(root=str(module.NATIVE[name]), commit=module.NATIVE_SHA[name]),
                checkpoint=dict(sha256="synthetic-cp"),
            ),
        )
        seals[name] = dict(path=str(spec_path), sha256=module.sha(spec_path))
        commands.append(
            dict(
                endpoint=name,
                env=dict(
                    CUDA_VISIBLE_DEVICES=gpu,
                    PYTORCH_ALLOC_CONF="expandable_segments:True",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    PYTHONPATH=str(module.NATIVE[name] / "src") + ":" + str(module.NATIVE[name]),
                ),
                modes=[
                    dict(
                        mode=mode,
                        command=[
                            sys.executable,
                            str(worker),
                            mode,
                            "--spec",
                            str(spec_path),
                            "--spec-sha256",
                            seals[name]["sha256"],
                        ],
                    )
                    for mode in module.MODES
                ],
            )
        )
    document = dict(
        execute="one_owner_parallel_endpoints_sequential_phases",
        endpoint_gpu_map=module.GPU_MAP,
        endpoint_order=list(module.GPU_MAP),
        commands=commands,
    )
    plan, specs = module.validate_plan(document, dict(specs=seals), owner_binding)
    real_popen = subprocess.Popen
    processes, waits = {}, {}

    def tracked_popen(argv, **kwargs):
        assert kwargs["shell"] is False
        assert argv in [p["command"] for row in plan for p in row["modes"]]
        process = real_popen(argv, **kwargs)
        processes[process.pid] = process
        waits[process.pid] = 0
        real_wait = process.wait

        def wait(*args, **kw):
            waits[process.pid] += 1
            return real_wait(*args, **kw)

        process.wait = wait
        return process

    monkeypatch.setattr(module.subprocess, "Popen", tracked_popen)

    def inspect(pid):
        proc = processes[pid]
        return dict(
            pid=pid,
            ppid=os.getpid(),
            uid=os.getuid(),
            start_ticks="synthetic-child-" + str(pid),
            command_sha256=module.argv_hash(proc.args),
            script_basename=worker.name,
        )

    return (
        module,
        root,
        document,
        seals,
        owner_binding,
        plan,
        specs,
        owner,
        inspect,
        processes,
        waits,
    )


def events(module):
    return [json.loads(line) for line in (module.JOB / "events.jsonl").read_text().splitlines()]


def test_two_gpu_parallel_exact_phase_order_same_owner_and_wait_codes(controller):
    m, root, _, _, _, plan, specs, owner, inspect, processes, waits = controller
    frozen = {Path(v["path"]): v["sha256"] for v in specs.values()}
    result = m.run_phases(plan, root, inspect, owner, lambda: m.verify_files(frozen), cwd=m.JOB)
    assert result["status"] == "completed" and len(result["phases"]) == 8
    rows = events(m)
    for name, gpu in m.GPU_MAP.items():
        starts = [r for r in rows if r["endpoint"] == name and r["kind"] == "start"]
        assert [r["mode"] for r in starts] == list(m.MODES)
        assert {r["gpu"] for r in starts} == {gpu}
        assert {r["ppid"] for r in starts} == {owner["pid"]}
    first = {
        name: next(
            r
            for r in rows
            if r["endpoint"] == name and r["mode"] == "seal" and r["kind"] == "start"
        )
        for name in m.GPU_MAP
    }
    ends = {
        name: next(
            r for r in rows if r["endpoint"] == name and r["mode"] == "seal" and r["kind"] == "end"
        )
        for name in m.GPU_MAP
    }
    assert max(r["time"] for r in first.values()) < min(r["time"] for r in ends.values())
    assert all(p.poll() == 0 for p in processes.values())
    assert set(waits.values()) == {1}
    for phase in result["phases"]:
        assert phase["wait_called"] and phase["poll_returncode"] == phase["wait_returncode"] == 0
        assert phase["actual_process_identity"]["pid"] == phase["pid"]
        assert phase["actual_process_identity"]["command_sha256"] == m.argv_hash(phase["command"])
        assert phase["process_identity_verified"]


def test_failed_endpoint_stops_new_phases_and_drains_live_peer(controller, monkeypatch):
    m, root, _, _, _, plan, _, owner, inspect, processes, waits = controller
    monkeypatch.setenv("SYNTHETIC_FAIL", "1")
    result = m.run_phases(plan, root, inspect, owner, lambda: None, cwd=m.JOB)
    assert result["status"] == "failed"
    rows = events(m)
    assert all(row["mode"] in ("seal", "medium") for row in rows)
    assert len([r for r in rows if r["kind"] == "end"]) == 4
    assert all(p.poll() is not None for p in processes.values()) and set(waits.values()) == {1}
    exits = [p["wait_returncode"] for p in result["phases"]]
    assert exits.count(17) == 1 and exits.count(0) == 3
    assert result["endpoints"]["R3-RI"]["skipped_modes"] == ["panel", "validate"]
    assert result["endpoints"]["R14-FRGA"]["skipped_modes"] == ["panel", "validate"]


@pytest.mark.parametrize("mutation", ["argv", "env", "gpu_map", "endpoint_order"])
def test_reject_unfrozen_argv_env_gpu_and_order_before_launch(controller, mutation):
    m, _, doc, seals, owner, _, _, _, _, processes, _ = controller
    bad = copy.deepcopy(doc)
    if mutation == "argv":
        bad["commands"][0]["modes"][0]["command"].append("--limit=3")
    if mutation == "env":
        bad["commands"][0]["env"]["PYTHONPATH"] = "untrusted"
    if mutation == "gpu_map":
        bad["endpoint_gpu_map"]["R14-FRGA"] = "0"
    if mutation == "endpoint_order":
        bad["endpoint_order"].reverse()
    with pytest.raises(ValueError):
        m.validate_plan(bad, dict(specs=seals), owner)
    assert not processes


def test_immutable_logs_and_input_bytes_reject_overwrite(controller):
    m, root, _, _, _, plan, specs, owner, inspect, processes, _ = controller
    path = root / "R3-RI.seal.stdout.log"
    path.write_text("old immutable output")
    result = m.run_phases(plan, root, inspect, owner, lambda: None, cwd=m.JOB)
    assert result["status"] == "failed" and not processes
    assert path.read_text() == "old immutable output"
    with pytest.raises(FileExistsError):
        m.write_new(path, {"replace": True})
    binding = next(iter(specs.values()))
    Path(binding["path"]).write_text("changed")
    with pytest.raises(ValueError, match="immutable job input changed"):
        m.verify_files({Path(binding["path"]): binding["sha256"]})


def test_receipt_io_failure_still_waits_for_already_running_peer(controller, monkeypatch):
    m, root, _, _, _, plan, _, owner, inspect, processes, waits = controller
    original = m.finish_phase
    raised = False

    def fail_once(*args):
        nonlocal raised
        if not raised:
            raised = True
            raise OSError("synthetic receipt failure")
        return original(*args)

    monkeypatch.setattr(m, "finish_phase", fail_once)
    with pytest.raises(OSError, match="synthetic receipt failure"):
        m.run_phases(plan, root, inspect, owner, lambda: None, cwd=m.JOB)
    assert len(processes) == 2 and all(p.poll() == 0 for p in processes.values())
    assert all(count >= 1 for count in waits.values())
    assert all(row["mode"] == "seal" for row in events(m))


def test_terminal_requires_native_bindings_and_whole_job_zero_pkl(controller):
    m, _, _, _, _, _, specs, _, _, processes, _ = controller
    for name, binding in specs.items():
        out = m.OUTPUT / name
        (out / "analysis").mkdir(parents=True)
        (out / "panel").mkdir()
        m.write_new(
            out / "COMPLETE.json",
            dict(
                status="completed",
                updates=0,
                scientific_complete=False,
                spec_sha256=binding["sha256"],
                counts=m.COUNTS,
                panel_rows=128,
                native_evaluation_source=dict(commit=m.NATIVE_SHA[name]),
                analysis_adapter_source=dict(commit=m.ANALYSIS_SHA),
                training_source=dict(commit=m.NATIVE_SHA[name]),
                endpoint=dict(role="last", epoch=2),
            ),
        )
        for path in (out / "analysis/summary.json", out / "panel/COMPLETE.json"):
            m.write_new(path, dict(checkpoint_sha256="synthetic-cp"))
    assert set(m.terminal(specs)) == set(m.GPU_MAP)
    p = m.JOB / "synthetic.pkl"
    p.write_bytes(b"synthetic-marker")
    with pytest.raises(ValueError, match="zero-PKL"):
        m.terminal(specs)
    assert not processes


def test_missing_then_real_identity_retried_and_each_observation_retained(controller):
    m, root, _, _, _, plan, _, owner, inspect, _, _ = controller
    calls = 0

    def transient(pid):
        nonlocal calls
        calls += 1
        return None if calls <= 2 else inspect(pid)

    command = plan[0]["modes"][0]["command"]
    process, receipt = m.spawn_phase(
        root, "transient-identity", command, plan[0]["env"], transient, owner, cwd=m.JOB
    )
    observation = receipt["identity_observation"]
    assert observation["verified"] and observation["status"] == "verified"
    assert calls == len(observation["samples"]) == 3
    assert all(row["actual_process_identity"] is None for row in observation["samples"][:2])
    assert observation["samples"][-1]["actual_process_identity"] == inspect(process.pid)
    assert observation["samples"][-1]["elapsed_seconds"] >= m.IDENTITY_RETRY_SECONDS
    process.wait()
    assert m.finish_phase(root, process, receipt)["status"] == "completed"


def test_always_missing_live_identity_hits_bound_without_fabrication(controller, monkeypatch):
    m, root, _, _, _, plan, _, owner, _, _, _ = controller
    monkeypatch.setattr(m, "IDENTITY_OBSERVE_SECONDS", 0.06)
    command = plan[0]["modes"][0]["command"]
    process, receipt = m.spawn_phase(
        root, "identity-deadline", command, plan[0]["env"], lambda _: None, owner, cwd=m.JOB
    )
    observation = receipt["identity_observation"]
    assert observation["status"] == "live_child_missing_identity_deadline"
    assert not receipt["process_identity_verified"] and receipt["actual_process_identity"] is None
    assert len(observation["samples"]) >= 2
    assert 0.06 <= observation["duration_seconds"] < 0.2
    assert process.poll() is None
    process.wait()
    result = m.finish_phase(root, process, receipt)
    assert result["wait_returncode"] == 0 and result["status"] == "failed"


def test_child_exits_without_identity_fails_even_returncode_zero(controller):
    m, root, _, _, _, plan, _, owner, _, processes, _ = controller

    def exited(pid):
        processes[pid].wait()
        return None

    command = plan[0]["modes"][0]["command"]
    process, receipt = m.spawn_phase(
        root, "exited-unobserved", command, plan[0]["env"], exited, owner, cwd=m.JOB
    )
    observation = receipt["identity_observation"]
    assert observation["status"] == "child_exited_without_observed_identity"
    assert len(observation["samples"]) == 1
    assert observation["samples"][0]["child_poll_returncode"] == 0
    result = m.finish_phase(root, process, receipt)
    assert result["wait_returncode"] == 0 and result["status"] == "failed"


@pytest.mark.parametrize("field", ["pid", "ppid", "uid", "command_sha256", "start_ticks"])
def test_observed_identity_mismatch_is_never_retried_or_replaced(controller, field):
    m, root, _, _, _, plan, _, owner, inspect, _, _ = controller
    calls = 0

    def wrong(pid):
        nonlocal calls
        calls += 1
        identity = inspect(pid)
        identity[field] = "" if field == "start_ticks" else "wrong-actual-value"
        return identity

    command = plan[0]["modes"][0]["command"]
    process, receipt = m.spawn_phase(
        root, "mismatch-" + field, command, plan[0]["env"], wrong, owner, cwd=m.JOB
    )
    observation = receipt["identity_observation"]
    assert observation["status"] == "observed_mismatch" and calls == 1
    assert len(observation["samples"]) == 1
    assert not receipt["process_identity_verified"]
    assert receipt["actual_process_identity"][field] == (
        "" if field == "start_ticks" else "wrong-actual-value"
    )
    process.wait()
    assert m.finish_phase(root, process, receipt)["status"] == "failed"


def test_identity_lookup_error_is_recorded_and_never_retried(controller):
    m, root, _, _, _, plan, _, owner, _, _, _ = controller
    calls = 0

    def denied(_):
        nonlocal calls
        calls += 1
        raise PermissionError("synthetic proc inspection denied")

    process, receipt = m.spawn_phase(
        root,
        "lookup-error",
        plan[0]["modes"][0]["command"],
        plan[0]["env"],
        denied,
        owner,
        cwd=m.JOB,
    )
    assert receipt["identity_observation"]["status"] == "lookup_error" and calls == 1
    assert "synthetic proc inspection denied" in receipt["identity_lookup_error"]
    process.wait()
    assert m.finish_phase(root, process, receipt)["status"] == "failed"
