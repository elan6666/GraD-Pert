"""Seal U4 then U3 after the preserved U2 run; never rewrite the old queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from run_cap40_ablations import execute
from run_group import next_action
from run_unseen_group import validate_reference, verify_manifest

from gradpert.data._io import atomic_json
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file


def dependency_from_queue(path: Path, expected_sha: str) -> tuple[dict, dict]:
    if not path.resolve().is_relative_to("/data/yilangliu") or sha256_file(path) != expected_sha:
        raise ValueError("previous queue identity changed")
    old = json.loads(path.read_text())
    if [r["name"] for r in old["rows"]] != ["U1", "U2", "U3"]:
        raise ValueError("expected original U1/U2/U3 queue")
    if next_action(old["rows"][0]["plan"]) != "skip_complete":
        raise ValueError("U1 must already be terminal")
    order_change = json.loads((path.parent / "USER_ORDER_CHANGE.json").read_text())
    try:
        old_command = Path(f"/proc/{order_change['pid']}/cmdline").read_bytes()
    except FileNotFoundError:
        old_command = b""
    if old_command:
        raise ValueError("old controller must be retired before follow-up")
    plan = old["rows"][1]["plan"]
    root = Path(plan["run_root"])
    if not root.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("dependency stays on server")
    if json.loads((root / "launch.json").read_text()) != plan:
        raise ValueError("U2 launch differs from sealed plan")
    state = json.loads((path.parent / "state.json").read_text())
    if state["row"] != "U2" or state["phase"] != "formal":
        raise ValueError("previous queue is no longer the expected U2 stage")
    if Path(old["rows"][2]["plan"]["run_root"]).exists():
        raise ValueError("old U3 already started; explicit reconciliation required")
    if not (path.parent / "USER_ORDER_CHANGE.json").exists():
        raise ValueError("old controller order-change evidence required")
    pid = state["child_pid"]
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\x00", " ")
    except FileNotFoundError:
        command = ""
    if not command and not (root / "COMPLETE.json").exists():
        raise ValueError("U2 process missing without completion")
    return old, {
        "name": "U2",
        "plan": plan,
        "launch_sha256": sha256_file(root / "launch.json"),
        "pid": pid,
        "pid_cmdline_sha256": hashlib.sha256(command.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--previous-queue", type=Path, required=True)
    parser.add_argument("--previous-queue-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--defer-tests", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator contract is missing")
    if not args.queue_root.resolve().is_relative_to("/data/yilangliu"):
        parser.error("queue stays on server")
    source = Path(__file__).resolve().parents[2]
    manifest = verify_manifest(source, args.manifest)
    if manifest["schema"] != "gradpert-v2-unseen-four-1":
        raise ValueError("U4 follow-up needs the new four-arm design")
    old, dependency = dependency_from_queue(args.previous_queue, args.previous_queue_sha256)
    validate_reference(source, Path(old["baseline"]), old["baseline_complete_sha256"])
    rows = []
    for name in ("U4", "U3"):
        row = next(r for r in manifest["rows"] if r["name"] == name)
        plan = resolve_plan(
            argparse.Namespace(
                config=source / row["config"],
                runtime=args.runtime,
                data_root=None,
                gpu="0,1",
                seed=1,
                defer_test=args.defer_tests,
            )
        )
        rows.append({"name": name, "plan": plan, "probe_kind": "preflight_only"})
    queue = {
        "schema": "gradpert-v2-unseen-U4-U3-followup-1",
        "preflight_steps": 10,
        "baseline": old["baseline"],
        "baseline_complete_sha256": old["baseline_complete_sha256"],
        "dependencies": [dependency],
        "rows": rows,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "previous_queue": str(args.previous_queue),
        "previous_queue_sha256": args.previous_queue_sha256,
        "overall_order": ["U1", "U2", "U4", "U3"],
        "postfit_policy": "deferred" if args.defer_tests else "inline",
        "resource_policy": "training_priority_idle_gpu_postfit",
        "shared_training_evaluation_gpu": False,
        "preserved_U1": old["rows"][0]["plan"],
        "superseded_U3": old["rows"][2]["plan"],
    }
    args.queue_root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.queue_root / "queue.json", queue)
    if args.execute:
        execute(queue, args.queue_root)
    else:
        print(json.dumps({"queue_root": str(args.queue_root), "status": "planned_not_launched"}))


if __name__ == "__main__":
    main()
