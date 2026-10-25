"""Repair R13 SRAMP diagnostics without repeating fitting or validation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cap40_posthoc import read, sha, write
from generate_representation_round13 import ARMS, SCHEMA
from run_readout_analysis_repair import (
    git,
    preserved_files,
    repair_panel_command,
    validate_controller,
    validate_output,
    validate_repair_publication,
)

TRAINING_SHA = "dc79777827fb763e7b37b709d0149bbc1e9e4a9e"
QUEUE_SHA = "d6ba86a54466cb579b449fb7f4682045a317f96aab64e9ba60f4a7df5d240dac"
FAILURE_SHA = "79fe26797e366316f413b534eec07dcdfd8787032f9e92df7a3a883915426bfe"
CHECKPOINT_SHA = "df5ff956a0e806c1f4c4ee7c1ec442197e28ac639a3a6b2e1ee2051c382450fa"
ALLOWED_CHANGES = {
    "scripts/v2/run_fgpo_medium.py": "M",
    "scripts/v2/run_representation_postfit.py": "M",
    "scripts/v2/run_relative_amplitude_analysis_repair.py": "A",
    "tests/v2/test_representation_round13.py": "M",
    "tests/v2/test_relative_amplitude_analysis_repair.py": "A",
}


def validate_repair_source(queue, source):
    if (
        queue.get("family") != SCHEMA
        or queue.get("source_commit") != TRAINING_SHA
        or [r.get("name") for r in queue.get("rows", [])] != list(ARMS)
    ):
        raise ValueError("repair requires the original two-arm R13 queue/source")
    source = Path(source).resolve()
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("repair checkout must be clean")
    changes = dict(
        (path, status)
        for status, path in (
            line.split("\t")
            for line in git(
                source, "diff", "--name-status", "--no-renames", TRAINING_SHA, "HEAD"
            ).splitlines()
        )
    )
    if changes != ALLOWED_CHANGES:
        raise ValueError("R13 repair changed undeclared training/config/observer code")
    return {"commit": git(source, "rev-parse", "HEAD"), "dirty": False, "diff": changes}


def validate_original(queue_path, queue):
    if (
        Path(queue_path).resolve() != (Path(queue["output"]) / "queue.json").resolve()
        or sha(queue_path) != QUEUE_SHA
        or sha(Path(queue["output"]) / "FAILURE.json") != FAILURE_SHA
        or sha(Path(queue["output"]) / "state.json") != FAILURE_SHA
    ):
        raise ValueError("exact immutable failed attempt is required")
    if Path(queue["rows"][1]["plan"]["run_root"]).exists():
        raise ValueError("queued NC must remain unstarted during SRAMP analysis recovery")


def execute(queue_path, output, *, analysis_sha, publication, publication_sha256):
    from run_fgpo_medium import execute as medium_execute
    from run_fgpo_medium import validate_medium_complete
    from run_group import lock
    from run_representation_postfit import context, validate_panel, validate_validation

    import gradpert

    source = Path(__file__).resolve().parents[2]
    queue_path = Path(queue_path).resolve()
    queue = read(queue_path)
    validate_original(queue_path, queue)
    output = validate_output(queue, output)
    source_proof = validate_repair_source(queue, source)
    identity = validate_repair_publication(queue, source, publication, publication_sha256)
    if (
        identity.commit != analysis_sha
        or identity.commit != source_proof["commit"]
        or identity.dirty
        or not Path(gradpert.__file__).resolve().is_relative_to(Path(queue["source"]) / "src")
    ):
        raise ValueError("published analysis source or original native imports differ")
    row = queue["rows"][0]
    training = context(queue, row)
    validation = validate_validation(queue, row)
    if training["last"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("completed SRAMP checkpoint changed")
    original = preserved_files(queue)
    admission = validate_controller(queue)
    if output.exists():
        raise ValueError("fresh repair output required")
    with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-r13-analysis-repair.lock")):
        validate_original(queue_path, queue)
        validate_controller(queue)
        output.mkdir(parents=True, exist_ok=False)
        inputs = dict(
            queue=str(queue_path),
            queue_sha256=QUEUE_SHA,
            original_failure_sha256=FAILURE_SHA,
            training_source_commit=TRAINING_SHA,
            analysis_source=identity.payload(),
            source_diff=source_proof["diff"],
            analysis_publication=str(publication),
            analysis_publication_sha256=publication_sha256,
            training=training,
            validation=validation,
            original_files=original,
            admission=admission,
        )
        write(output / "INPUTS.json", inputs)
        try:
            medium_execute(
                queue_path,
                row["name"],
                repair_output=output / "medium",
                repair_publication=publication,
                repair_publication_sha256=publication_sha256,
            )
            medium = validate_medium_complete(queue, row, root=output / "medium")
            validate_controller(queue)
            command = repair_panel_command(queue, row, source, output / "panel")
            with (output / "panel.log").open("x") as log:
                subprocess.run(
                    command,
                    cwd=queue["source"],
                    env=dict(
                        os.environ,
                        PYTHONPATH=str(Path(queue["source"]) / "src"),
                        CUDA_VISIBLE_DEVICES="0",
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            panel = validate_panel(queue, row, root=output / "panel")
            expected = {"commit": analysis_sha, "dirty": False}
            if (
                read(output / "medium/MEDIUM_COMPLETE.json")["report_controller_commit"]
                != analysis_sha
                or read(output / "medium/analysis/summary.json")["analysis"] != expected
                or read(output / "panel/COMPLETE.json")["analysis_source"] != expected
                or preserved_files(queue) != original
                or validate_validation(queue, row) != validation
                or validate_repair_source(queue, source) != source_proof
                or validate_repair_publication(
                    queue, source, publication, publication_sha256
                ).payload()
                != identity.payload()
            ):
                raise ValueError("repair provenance or original artifacts changed")
            validate_original(queue_path, queue)
            roots = [output, Path(queue["output"]), Path(row["plan"]["run_root"])]
            if any(any(root.rglob("*.pkl")) for root in roots):
                raise ValueError("zero-PKL postcondition failed")
            write(
                output / "ANALYSIS_REPAIR_COMPLETE.json",
                dict(
                    status="completed",
                    scope="R13-SRAMP-analysis-only-repair-1",
                    updates=0,
                    retrained=False,
                    validation_rerun=False,
                    scientific_complete=False,
                    test_status="deferred_until_design_lock",
                    zero_pkl=True,
                    original_artifacts_unchanged=True,
                    inputs=inputs,
                    medium=medium,
                    panel=panel,
                    pending_diagnostics=["gamma/affine-free/fixed-row-RMS supplement"],
                    outputs={
                        str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in sorted(output.rglob("*"))
                        if p.is_file()
                    },
                ),
            )
        except BaseException as error:
            write(
                output / "FAILURE.json",
                dict(
                    status="failed",
                    error=repr(error),
                    queue_sha256=QUEUE_SHA,
                    analysis_sha=analysis_sha,
                    updates=0,
                ),
            )
            raise
    return output / "ANALYSIS_REPAIR_COMPLETE.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "output", "publication"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("analysis-sha", "publication-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    execute(
        args.queue,
        args.output,
        analysis_sha=args.analysis_sha,
        publication=args.publication,
        publication_sha256=args.publication_sha256,
    )


if __name__ == "__main__":
    main()
