"""Bounded R50 training-only concurrency probe; never a scientific run."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def save(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(solo: dict, pair: list[dict]) -> dict:
    rows = [solo, *pair]
    if len(pair) != 2 or any(r["status"] != "complete" for r in rows):
        raise ValueError("require three complete probes")
    for field in ("commit", "config_sha", "gpu_uuid", "steps", "warmup"):
        if any(r[field] != solo[field] for r in pair):
            raise ValueError(f"unmatched probe {field}")
    for r in rows:
        if [s["global_step"] for s in r["samples"]] != list(range(r["steps"])):
            raise ValueError("probe step sequence differs")
        if r["evaluation_access"] or r["persistent_pkl"]:
            raise ValueError("probe crossed artifact/evaluation boundary")
    durations = [sum(s["seconds"] for s in r["samples"][r["warmup"] :]) for r in rows]
    count = solo["steps"] - solo["warmup"]
    starts = [r["samples"][r["warmup"]]["start"] for r in pair]
    ends = [r["samples"][-1]["end"] for r in pair]
    overlap = max(0, min(ends) - max(starts)) / max(ends[i] - starts[i] for i in range(2))
    speedup = (2 * count / (max(ends) - min(starts))) / (count / durations[0])
    memory_ok = all(
        s["free_bytes"] >= 4 * 1024**3 and s["oom"] == 0 and s["retry"] == 0
        for r in pair
        for s in r["samples"]
    )
    return {
        "status": "complete",
        "scientific_completion": False,
        "conditional_on_other_gpu_load": True,
        "aggregate_throughput_ratio": speedup,
        "overlap_fraction": overlap,
        "median_step_seconds": [
            statistics.median(s["seconds"] for s in r["samples"][r["warmup"] :]) for r in rows
        ],
        "memory_pass": memory_ok,
        "concurrency_candidate": speedup >= 1.15 and overlap >= 0.8 and memory_ok,
        "formal_launch_authorized_by_this_receipt": False,
        "next": (
            "Require full-epoch capacity including validation and checkpoint peaks "
            "before shared formal fits; retain validation-only LR selection."
        ),
    }


def worker(a: argparse.Namespace) -> None:
    import torch

    import gradpert.execution.native as native
    from gradpert.training.step import GraDPertStepEngine
    from scripts.performance.profile_native_a0 import (
        ProfileComplete,
        _make_bounded_train_step,
        _training_only_evaluator_factory,
    )

    root = a.root / a.worker
    root.mkdir(exist_ok=False)
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("allocator missing")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("bind one GPU")
    # Fixed disjoint budgets, not a race on whichever process sees free memory first.
    torch.cuda.set_per_process_memory_fraction(0.40, 0)
    state = {"guard_bindings": [], "truth_access_attempts": []}
    native.CanonicalEvaluationData = _training_only_evaluator_factory(state)

    def forbidden(*args, **kwargs):
        state["truth_access_attempts"].append("validation_callback")
        raise RuntimeError("training-only probe reached validation")

    native.evaluate_validation_macro_delta = forbidden
    samples = []
    started = 0.0

    def before(engine):
        nonlocal started
        if not samples:
            save(root / "READY.json", {"pid": os.getpid()})
            deadline = time.monotonic() + 600
            while not (a.root / ("GO-solo" if a.worker == "solo" else "GO-pair")).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("paired worker readiness timeout")
                time.sleep(1)
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()

    def after(engine, metrics, global_step):
        torch.cuda.synchronize()
        end = time.monotonic()
        free, _ = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        samples.append(
            {
                "global_step": global_step,
                "start": started,
                "end": end,
                "seconds": end - started,
                "free_bytes": free,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "oom": stats.get("num_ooms", 0),
                "retry": stats.get("num_alloc_retries", 0),
                "metrics": dataclasses.asdict(metrics),
            }
        )

    GraDPertStepEngine.train_step = _make_bounded_train_step(
        GraDPertStepEngine.train_step, total_steps=32, before_step=before, after_step=after
    )
    result = dict(
        status="failed",
        scientific_completion=False,
        commit=a.commit,
        config_sha=a.config_sha,
        gpu_uuid=a.gpu_uuid,
        steps=32,
        warmup=4,
    )
    try:
        native.run_native_experiment(
            config_path=a.source / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml",
            data_root=a.data_root,
            run_root=root / "native",
            run_id=f"r50-probe/{a.root.name}/{a.worker}",
            run_seed=1,
            mode="full",
            device_name="cuda:0",
            repository_root=a.source,
            formal=True,
            source_publication_receipt=a.publication,
            source_publication_receipt_sha256=a.publication_sha,
            genept_preflight_receipt=a.genept_receipt,
            genept_preflight_receipt_sha256=a.genept_sha,
        )
        raise RuntimeError("probe unexpectedly completed training")
    except ProfileComplete:
        result["status"] = "complete"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result.update(
            samples=samples,
            evaluation_access=state["truth_access_attempts"],
            persistent_pkl=len(list(root.rglob("*.pkl"))),
        )
        save(root / "PROBE.json", result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "root", "data-root", "publication", "genept-receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("commit", "config-sha", "publication-sha", "genept-sha", "gpu-uuid"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--worker", choices=("solo", "pair0", "pair1"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(a.source), "status", "--porcelain"], text=True
    ).strip()
    if head != a.commit or dirty:
        raise RuntimeError("source drift")
    for path, expected in (
        (a.source / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml", a.config_sha),
        (a.publication, a.publication_sha),
        (a.genept_receipt, a.genept_sha),
    ):
        if digest(path) != expected:
            raise RuntimeError("immutable input drift")
    if a.worker:
        worker(a)
        return
    if a.root.exists() or not a.root.is_relative_to("/data/yilangliu/GraD-Pert/development"):
        raise ValueError("fresh server development root required")
    if a.dry_run:
        print("PREFLIGHT_OK: solo32 then pair32+32; no validation/test/formal completion")
        return
    a.root.mkdir()
    for group, names in (("solo", ["solo"]), ("pair", ["pair0", "pair1"])):
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        )
        if a.gpu_uuid in apps:
            raise RuntimeError("selected GPU occupied; preserve root, no retry")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu_uuid)
        processes = []
        for name in names:
            with (a.root / f"{name}.log").open("x") as log:
                processes.append(
                    subprocess.Popen(
                        [sys.executable, __file__, *sys.argv[1:], "--worker", name],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        deadline = time.monotonic() + 900
        try:
            while not all((a.root / name / "READY.json").exists() for name in names):
                if (
                    any(proc.poll() is not None for proc in processes)
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError("worker failed/timed out before ready")
                time.sleep(2)
            (a.root / f"GO-{group}").touch(exist_ok=False)
            deadline = time.monotonic() + 1800
            while any(proc.poll() is None for proc in processes):
                if (
                    any(proc.poll() not in (None, 0) for proc in processes)
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError("probe failed; stop its own peer only")
                time.sleep(5)
            if any(proc.returncode != 0 for proc in processes):
                raise RuntimeError("probe failed")
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
    records = [
        json.loads((a.root / name / "PROBE.json").read_text())
        for name in ("solo", "pair0", "pair1")
    ]
    save(a.root / "COMPLETE.json", compare(records[0], records[1:]))


if __name__ == "__main__":
    main()
