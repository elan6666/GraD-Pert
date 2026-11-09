"""Resume only the interrupted 2026-11-09 B arm under its original GPU0 gate."""

import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path("/data/yilangliu/GraD-Pert")
F = BASE / "development/llm-norm-standard-attention-20261109-a2"
S = BASE / "development/source-llm-norm-c57c139-20261109-a2"
R = F / "b-resume-controller-a1"
EVIDENCE = F / "b-resume-evidence-a1"
SOURCE_COMMIT = "c57c139858fa4b911cbafdeee957736bc33b7497"
PROTOCOL = "2b4add9fdfd0f00d2c61ac9e821ec9a9d6ffb03db7892a250eaf79fe9ab71bbf"
PLAN = "bc893b281aac96032f333f348269ddd98a3b606bef86f5210b35646d009b91c3"
FAILURE = "08a123c8975d0af784cb3aced820bd48a6ec45ab46c6dfa2dba4eedb32b6bb40"
CP = "49f5960cd255163997c58dc7975866755c3e88eae5fe52f9f4d0b08a45605503"
ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def matches(path, expected, sha256_file):
    if sha256_file(path) != expected:
        raise ValueError(f"sealed receipt differs: {path.name}")


def command(python, source, plan):
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        str(source / "scripts/v2/distributed_train.py"),
        "--plan",
        str(plan),
        "--resume",
    ]


def verify_epoch1(run, evidence, sha256_file, read_json):
    journal = read_json(run / "fit/epoch_state.json")
    history = read_json(run / "fit/history.json")
    if (
        journal["epoch"] != 1
        or journal["budget"] != [2, 150]
        or [row["optimizer_steps"] for row in history] != [150]
        or (run / "FAILURE.json").exists()
        or (run / "TRAIN_COMPLETE.json").exists()
        or (run / "fit/epoch-0002.pt").exists()
    ):
        raise ValueError("expected only one committed epoch and no native terminal")
    for relative in (
        "fit/epoch_state.json",
        "fit/history.json",
        "fit/live_progress.json",
        "run_manifest.json",
        "launch.json",
    ):
        path = run / relative
        matches(path, sha256_file(evidence / path.name), sha256_file)
    matches(run / "fit/epoch-0001.pt", CP, sha256_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-sha256", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only and any(os.environ.get(k) != v for k, v in ENV.items()):
        raise ValueError("original physical GPU, allocator and cuBLAS environment required")
    os.environ["PYTHONPATH"] = str(S / "src")
    sys.path[:0] = [str(S / "src"), str(S)]
    os.chdir(S)

    from gradpert.data._io import atomic_json, read_json
    from gradpert.execution.identity import inspect_source_identity
    from gradpert.execution.v2_training_stage import validate_training_stage
    from gradpert.hashing import sha256_file
    from scripts.v2.run_dataset_transfer_300 import native_run
    from scripts.v2.run_group import lock
    from scripts.v2.shared_gpu import admit_selected_device, process_info, snapshot

    matches(F / "PROTOCOL.json", PROTOCOL, sha256_file)
    matches(F / "CONTROLLER_FAILURE.json", FAILURE, sha256_file)
    evidence_path = EVIDENCE / "MAIN_INTERRUPTION_EVIDENCE.json"
    matches(evidence_path, args.evidence_sha256, sha256_file)
    evidence = read_json(evidence_path)
    for name in (
        "CONTROLLER_FAILURE.json",
        "epoch_state.json",
        "history.json",
        "live_progress.json",
        "run_manifest.json",
        "launch.json",
    ):
        matches(EVIDENCE / name, evidence["archived"][name], sha256_file)
    protocol = read_json(F / "PROTOCOL.json")
    source = inspect_source_identity(
        S,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=F / "PUBLICATION.json",
        expected_publication_receipt_sha256=protocol["publication_sha256"],
    )
    if source.dirty or source.commit != SOURCE_COMMIT:
        raise ValueError("recovery must use the original clean numerical source")
    matches(F / "RUNTIME.json", protocol["runtime_sha256"], sha256_file)
    policy_file = Path(protocol["policy_path"])
    matches(policy_file, protocol["policy_sha256"], sha256_file)
    policy = read_json(policy_file)["shared_gpu_policy"]
    plan_path = F / "B_NORM_STANDARD_PLAN.json"
    matches(plan_path, PLAN, sha256_file)
    plan = read_json(plan_path)
    run = Path(plan["run_root"])
    matches(Path(plan["config"]), plan["config_sha256"], sha256_file)
    if plan["gpu"] != "0" or plan["cuda_memory_fraction"] != 0.9:
        raise ValueError("original GPU0 allocation required")
    owner_pid = os.getpid()

    def guard():
        admit_selected_device(policy, snapshot(owner_pid), 0)

    cmd = command(sys.executable, S, plan_path)
    with lock(BASE / "runtime/v2-gpu-0.lock"):
        guard()
        verify_epoch1(run, EVIDENCE, sha256_file, read_json)
        if R.exists():
            raise FileExistsError("new recovery controller root required")
        if args.check_only:
            print("Resume inputs and original GPU0 policy verified; no CUDA launched.")
            return
        owner = process_info(owner_pid)
        if owner is None:
            raise RuntimeError("resume controller identity unavailable")
        R.mkdir()
        binding = {
            "identity": owner,
            "phase": "resume_B_epoch2",
            "run_id": plan["run_id"],
            "run_root": str(run),
            "gpu": 0,
            "command": cmd,
            "parent_failure_sha256": FAILURE,
            "source_commit": SOURCE_COMMIT,
            "launcher_sha256": sha256_file(Path(__file__)),
            "evidence_sha256": args.evidence_sha256,
            "plan_sha256": PLAN,
            "start_checkpoint_sha256": CP,
            "start_optimizer_updates": 150,
        }
        atomic_json(R / "OWNER.json", binding)
        start = time.monotonic()
        try:
            guard()
            native_run(cmd, R / "B-resume.log", policy, admission_check=guard)
            elapsed = time.monotonic() - start
            stage = validate_training_stage(plan)
            history = read_json(run / "fit/history.json")
            old_history = read_json(EVIDENCE / "history.json")
            if (
                history[:1] != old_history
                or [row["optimizer_steps"] for row in history] != [150, 300]
                or stage["last"]["epoch"] != 2
                or any(p.is_file() and p.suffix.lower() == ".pkl" for p in F.rglob("*"))
            ):
                raise ValueError("resumed native terminal does not preserve the committed prefix")
            matches(run / "fit/epoch-0001.pt", CP, sha256_file)
            atomic_json(
                R / "CONTROLLER_COMPLETE.json",
                {
                    **binding,
                    "status": "training_complete_main_acceptance_pending",
                    "epochs": 2,
                    "optimizer_steps": 300,
                    "scientific_complete": False,
                    "TRAIN_COMPLETE_sha256": sha256_file(run / "TRAIN_COMPLETE.json"),
                    "epoch_training_seconds": [row["training_seconds"] for row in history],
                    "resume_subprocess_wall_seconds": elapsed,
                },
            )
        except BaseException as error:
            atomic_json(
                R / "CONTROLLER_FAILURE.json",
                {
                    **binding,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "preserve_results": True,
                },
            )
            raise


if __name__ == "__main__":
    main()
