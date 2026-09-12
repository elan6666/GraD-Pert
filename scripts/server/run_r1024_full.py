"""Run one reviewed R1024 coordinate for fifty epochs then best/last tests."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.server.run_r50_selection import command, sha, validate

ROWS = tuple(
    "r1024_" + x for x in ("s1", "s2", "u1", "t1", "t2", "p1", "p2", "c1", "l1", "l2", "l3")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=ROWS, required=True)
    for key in ("source", "root", "data-root", "publication", "genept-receipt", "step-receipt"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("commit", "config-sha", "publication-sha", "genept-sha", "step-sha"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.4)
    a = parser.parse_args()
    if a.memory_fraction != 0.4:
        raise ValueError("R1024 shared tasks require disjoint 40-percent allocator budgets")
    cfg = a.source / f"configs/r50/{a.row}/gradpert_b2/nadig_jurkat.yaml"
    if a.root.exists() or not a.root.resolve().is_relative_to("/data/yilangliu/GraD-Pert/runs"):
        raise ValueError("fresh server run root required")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator missing")
    if os.environ.get("GRADPERT_SPARSE_UNION_IMPL") != "cpu_array":
        raise ValueError("explicit union implementation required")
    for path, expected in (
        (cfg, a.config_sha),
        (a.publication, a.publication_sha),
        (a.genept_receipt, a.genept_sha),
        (a.step_receipt, a.step_sha),
    ):
        if sha(path) != expected:
            raise ValueError("input hash differs")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.source, text=True).strip()
        != a.commit
    ):
        raise ValueError("source commit differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=a.source, text=True).strip():
        raise ValueError("source dirty")
    probe = json.loads(a.step_receipt.read_text())
    identity = json.loads((a.step_receipt.parent / "source_identity.json").read_text())
    if sha(a.step_receipt.parent / "source_identity.json") != probe["source_identity_sha256"]:
        raise ValueError("step identity hash differs")
    if (
        probe["status"] != "complete"
        or probe["config_sha256"] != a.config_sha
        or identity["commit"] != a.commit
        or identity["dirty"]
        or probe["test_evaluations"] != 0
        or len(probe["steps"]) != 1
        or probe["steps"][0]["completed_steps"] != 1
        or not probe["steps"][0]["checkpoint_roundtrip_exact"]
        or not probe["steps"][0]["resources"]["cuda_acceptance"]
    ):
        raise ValueError("matching successful one-step receipt required")
    a.root.mkdir(parents=True)
    with (a.root / "full.log").open("x") as log:
        rc = subprocess.run(
            command(a, "full"), cwd=a.source, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    (a.root / "fit_exit.json").write_text(json.dumps({"rc": rc, "source_commit": a.commit}))
    if rc:
        raise RuntimeError("fit failed; preserve and do not relaunch")
    result = validate(a.root / "full", epochs=50, commit=a.commit, config_sha=a.config_sha)
    from gradpert.execution.postfit import evaluate_best_last

    evaluate_best_last(
        training_root=a.root / "full",
        output_root=a.root.parent / (a.row + "-test"),
        data_root=a.data_root,
        repository_root=a.source,
        publication=a.publication,
        publication_sha256=a.publication_sha,
        device_name="cuda:0",
        memory_fraction=a.memory_fraction,
    )
    (a.root / "COMPLETE.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
