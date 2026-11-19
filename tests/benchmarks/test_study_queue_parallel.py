import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.crosscell import study_queue as queue

DUMMY = """
import json
import os
import sys
import time
from pathlib import Path
root, peer, predecessor, fail = sys.argv[1:]
root = Path(root)
root.mkdir()
(root / "started").write_text(os.environ["CUDA_VISIBLE_DEVICES"])
if predecessor and not (Path(predecessor) / "result.json").exists():
    sys.exit("jobs overlapped on the same GPU")
if peer:
    deadline = time.monotonic() + 4
    while not (Path(peer) / "started").exists():
        if time.monotonic() > deadline:
            sys.exit("jobs on distinct GPUs did not overlap")
        time.sleep(0.01)
if fail:
    sys.exit(1)
(root / "result.json").write_text(json.dumps({"valid": True}))
"""


def fixture(root):
    root.mkdir()
    (root / "ready").mkdir()
    (root / "plan.json").write_text(
        json.dumps({"order": queue.CROSS_ONLY_ORDER, "cohort_sha256": "c"})
    )
    predecessor = root / "txpert-result.json"
    predecessor.write_text("{}")
    queue.write(
        root / "predecessor_handoff.json",
        {"c2_result": str(predecessor), "sha256": queue.digest(predecessor)},
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@local",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Initial",
        ],
        check=True,
    )
    # Queue outputs sit outside the clean runner checkout in production.
    subprocess.run(
        ["git", "-C", str(root), "config", "status.showUntrackedFiles", "no"], check=True
    )
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def descriptor(root, arm, commit, gpu, *, label=None, peer="", predecessor="", fail=""):
    label = label or arm
    stages = []
    for stage in ("smoke", "formal"):
        output = root / f"{label}-{stage}"
        arguments = (
            [str(output), str(peer), str(predecessor), fail]
            if stage == "formal"
            else [str(output), "", "", ""]
        )
        stages.append(
            {
                "name": stage,
                "command": [sys.executable, "-c", DUMMY, *arguments],
                "log": str(output.with_suffix(".log")),
                "run_root": str(output),
                "receipt": str(output / "result.json"),
                "expected": {"valid": True},
            }
        )
    queue.write(
        root / "ready" / f"{arm}.json",
        {
            "arm_id": arm,
            "cohort_sha256": "c",
            "repository": str(root),
            "source_commit": commit,
            "env": {"CUDA_VISIBLE_DEVICES": gpu},
            "stages": stages,
        },
    )


def wait_for(root, predicate):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        state = queue.read(root / "state.json") if (root / "state.json").exists() else {}
        if predicate(state):
            return state
        time.sleep(0.01)
    raise AssertionError(f"queue never reached expected state: {state}")


def test_parallel_runs_ready_arms_on_distinct_gpus_and_isolates_failure(tmp_path):
    root = tmp_path / "queue"
    commit = fixture(root)
    descriptor(root, "scbig-c2", commit, "0", peer=root / "prescribe-c2-formal")
    descriptor(root, "perturbdiff-c2", commit, "0", predecessor=root / "scbig-c2-formal")
    descriptor(root, "prescribe-c2", commit, "1", peer=root / "scbig-c2-formal", fail="yes")
    with ThreadPoolExecutor() as pool:
        task = pool.submit(queue.run, root, queue.digest(root / "plan.json"), 0.01, parallel=2)
        try:
            state = wait_for(
                root,
                lambda s: s.get("arms", {}).get("perturbdiff-c2", {}).get("status") == "completed",
            )
            assert state["arms"]["scfoundation-c2"]["status"] == "awaiting_verified_runner"
            assert state["arms"]["scbig-c2"]["status"] == "completed"
            assert wait_for(
                root, lambda s: s.get("arms", {}).get("prescribe-c2", {}).get("status") == "blocked"
            )
            assert (root / "scbig-c2-formal/started").read_text() == "0"
            assert (root / "prescribe-c2-formal/started").read_text() == "1"
        finally:
            descriptor(root, "scdfm-c2", commit, "0")
            descriptor(root, "scfoundation-c2", commit, "1")
            descriptor(root, "prescribe-c2", commit, "1", label="prescribe-a2")
        task.result(timeout=6)
    assert queue.read(root / "state.json")["status"] == "completed"


def test_serial_queue_does_not_accept_parallel_mode(tmp_path):
    root = tmp_path / "queue"
    fixture(root)
    plan = root / "plan.json"
    queue.write(plan, {"order": queue.ORDER})
    with pytest.raises(ValueError, match="Cross-cell C2"):
        queue.run(root, queue.digest(plan), 0.01, parallel=2)
