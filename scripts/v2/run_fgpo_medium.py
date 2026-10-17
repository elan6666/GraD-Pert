"""One completed new-family arm: exact medium rows, parity, diagnostics and report."""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from cap40_posthoc import population_identity, read, sha, write
from generate_dino_pair import ARMS as DINO_ARMS
from generate_dino_pair import REMAINING_SCHEMA as DP2_SCHEMA
from generate_dino_pair import SCHEMA as DINO_SCHEMA
from generate_fgpo_family import ARMS, SCHEMA
from generate_no_ssl import ARMS as NO_SSL_ARMS
from generate_no_ssl import SCHEMA as NO_SSL_SCHEMA
from generate_pt1 import ARMS as PT1_ARMS
from generate_pt1 import SCHEMA as PT1_SCHEMA

FAMILY_ARMS = {
    SCHEMA: ARMS,
    DINO_SCHEMA: DINO_ARMS,
    DP2_SCHEMA: {"DP2": DINO_ARMS["DP2"]},
    NO_SSL_SCHEMA: NO_SSL_ARMS,
    PT1_SCHEMA: PT1_ARMS,
}
AUTHORIZED_ARMS = tuple(dict.fromkeys(name for arms in FAMILY_ARMS.values() for name in arms))

SERVER_ROOT = Path("/data/yilangliu")
COUNTS = dict(
    conditions=1335,
    rows=5317,
    deep_conditions=128,
    deep_rows=506,
    gradient_conditions=32,
    gradient_rows=125,
)


def validate_analysis_spec(spec):
    fields = {
        "parent_manifest",
        "parent_manifest_sha256",
        "performance_receipt",
        "performance_receipt_sha256",
    }
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError("medium analysis needs exact sampling and performance receipts")
    for name in ("parent_manifest", "performance_receipt"):
        path = Path(spec[name]).resolve()
        if not path.is_relative_to(SERVER_ROOT) or sha(path) != spec[name + "_sha256"]:
            raise ValueError("medium sampling/performance receipt changed or left server")
    parent = read(spec["parent_manifest"])
    if parent.get("analysis_scope") != "medium_cap40_stratified" or any(
        parent.get(k) != v for k, v in COUNTS.items()
    ):
        raise ValueError("only the original comparable medium population is authorized")
    from cap40_posthoc import reuse_medium_schedule

    reuse_medium_schedule(parent)
    performance = read(spec["performance_receipt"])
    parity = performance.get("full_parity", {})
    fast, reference = (
        performance.get(k, float("nan")) for k in ("full_fast_seconds", "full_reference_seconds")
    )
    if (
        performance.get("status") != "completed"
        or performance.get("same_sample_scope") is not True
        or performance.get("deterministic_algorithms") is not True
        or performance.get("model_updates") != 0
        or parity.get("passed") is not True
        or parity.get("atol") != 1e-8
        or not 0 <= parity.get("max_abs", float("inf")) <= 1e-8
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (fast, reference))
        or not 0 < fast < reference
    ):
        raise ValueError("full-medium correctness/stability and end-to-end gain are required")
    return parent


def medium_command(queue, row):
    return [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/run_fgpo_medium.py"),
        "--queue",
        str(Path(queue["output"]) / "queue.json"),
        "--arm",
        row["name"],
    ]


def validate_medium_complete(queue, row, *, root=None):
    root = Path(root) if root is not None else Path(queue["output"]) / f"{row['name']}-medium"
    path = root / "MEDIUM_COMPLETE.json"
    result = read(path)
    plan = row["plan"]
    journal = read(Path(plan["run_root"]) / "fit/epoch_state.json")
    expected = {
        "status": "completed",
        "arm": row["name"],
        "source_commit": queue["source_commit"],
        "queue_sha256": sha(Path(queue["output"]) / "queue.json"),
        "checkpoint_sha256": journal["last"]["sha256"],
        "updates": 0,
        **COUNTS,
    }
    parent = validate_analysis_spec(queue["medium_analysis"])
    expected["diagnostic_population_sha256"] = population_identity(parent["schedule"])
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("completed medium arm identity, population or counts differ")
    summary = read(root / "analysis/summary.json")
    complete = read(root / "analysis/COMPLETE.json")
    if (
        complete.get("state_unchanged") is not True
        or summary.get("updates") != 0
        or summary.get("state_before") != summary.get("state_after")
        or summary.get("method_sha") != queue["source_commit"]
        or summary.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or summary.get("config_sha256") != plan["config_sha256"]
        or summary.get("diagnostic_population_sha256") != expected["diagnostic_population_sha256"]
        or any(summary.get(k) != v for k, v in COUNTS.items())
        or result.get("parity", {}).get("passed") is not True
        or result["parity"].get("atol") != 1e-8
        or not 0 <= result["parity"].get("max_abs", float("inf")) <= 1e-8
    ):
        raise ValueError("medium analysis state, method, population or parity changed")
    for name, entry in result["outputs"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target) != entry["sha256"]:
            raise ValueError("sealed medium artifact changed")
    return {"receipt": str(path), "sha256": sha(path), **COUNTS}


def validate_report_repair(q, arm, output, report_source):
    """Postfit D0/PT1 report repair; freeze numerical code and the parent queue."""
    if (arm, q.get("family")) not in (("D0", NO_SSL_SCHEMA), ("PT1", PT1_SCHEMA)):
        raise ValueError("analysis-only report repair is scoped to D0 or PT1")
    output = Path(output).resolve()
    protected = [Path(q["output"]).resolve()] + [
        Path(row["plan"]["run_root"]).resolve() for row in q["rows"]
    ]
    if not output.is_relative_to(SERVER_ROOT) or any(
        output.is_relative_to(root) for root in protected
    ):
        raise ValueError("repair output must be a new independent server root")
    paths = [
        "src",
        "scripts/v2/cap40_posthoc*.py",
        "scripts/v2/profile_medium_observer.py",
        "scripts/v2/shared_gpu.py",
        "scripts/v2/run_br400_group.py",
        "scripts/v2/generate_no_ssl.py",
        "scripts/v2/generate_pt1.py",
    ]
    if (
        subprocess.check_output(
            ["git", "-C", str(report_source), "status", "--porcelain"], text=True
        ).strip()
        or subprocess.check_output(
            [
                "git",
                "-C",
                str(report_source),
                "diff",
                "--name-only",
                q["source_commit"],
                "HEAD",
                "--",
                *paths,
            ],
            text=True,
        ).strip()
    ):
        raise ValueError("repair source is dirty or numerical method/analysis code changed")
    return output


def pt1_parent_binding(parent, training_identity, *, parent_path, parent_sha256):
    """Adapt only PT1's declared phase metadata, retaining every sampling field.

    Native sealing still compares the complete live runtime identity with the
    checkpoint identity. This does not relax that check or change numerical code.
    """
    trained = training_identity["data"]
    comparable = copy.deepcopy(trained)
    protocol = comparable["loss_protocol"]
    staged = protocol.pop("staged_training", None)
    if (
        not isinstance(staged, dict)
        or staged.get("mode") != "ssl_then_mse"
        or comparable != parent["data"]
    ):
        raise ValueError("PT1 medium parent may differ only by declared staged_training metadata")
    return {
        **copy.deepcopy(parent),
        "data": copy.deepcopy(trained),
        "training_protocol_binding": {
            "original_parent_manifest": str(parent_path),
            "original_parent_sha256": parent_sha256,
            "changed_field": "data.loss_protocol.staged_training",
            "previous_value": None,
            "training_value": copy.deepcopy(staged),
            "training_run_id": training_identity["run_id"],
            "diagnostic_population_sha256": population_identity(parent["schedule"]),
            "sampling_changed": False,
        },
    }


def execute(queue_path, arm, *, repair_output=None):
    from profile_medium_observer import compare
    from run_br400_group import validate_complete, verify

    q = read(queue_path)
    if arm not in FAMILY_ARMS.get(q.get("family"), {}):
        raise ValueError("medium postfit work is limited to the authorized new family")
    verify(q)
    source = Path(q["source"])
    report_source = Path(__file__).resolve().parents[2]
    if repair_output is None and source.resolve() != report_source:
        raise ValueError("medium worker must use the exact published queue source")
    parent = validate_analysis_spec(q["medium_analysis"])
    row = next(r for r in q["rows"] if r["name"] == arm)
    validate_complete(row["plan"])
    root = (
        validate_report_repair(q, arm, repair_output, report_source)
        if repair_output is not None
        else Path(q["output"]) / f"{arm}-medium"
    )
    if root.exists() or not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server-only medium run required")
    root.mkdir()
    start = time.perf_counter()
    plan = row["plan"]
    env = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    stages = {}

    def call(label, command, gpu):
        begin = time.perf_counter()
        with (root / (label + ".log")).open("x") as log:
            subprocess.run(
                command,
                cwd=source,
                env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        stages[label] = time.perf_counter() - begin

    base = [sys.executable, str(source / "scripts/v2/cap40_posthoc.py")]
    common = [
        "--run-root",
        plan["run_root"],
        "--config",
        plan["config"],
        "--data-root",
        plan["data_root"],
        "--method-sha",
        q["source_commit"],
    ]
    try:
        call(
            "saved-training-report",
            [
                sys.executable,
                str(report_source / "scripts/v2/analyze_completed_ablation.py"),
                "--run-root",
                plan["run_root"],
                "--output",
                str(root / "saved"),
                "--name",
                arm,
            ],
            "",
        )
        parent_path = q["medium_analysis"]["parent_manifest"]
        parent_sha256 = q["medium_analysis"]["parent_manifest_sha256"]
        if arm == "PT1":
            bound_parent = pt1_parent_binding(
                parent,
                read(Path(plan["run_root"]) / "run_manifest.json"),
                parent_path=parent_path,
                parent_sha256=parent_sha256,
            )
            parent_path = root / "pt1-medium-parent.json"
            write(parent_path, bound_parent)
            parent_sha256 = sha(parent_path)
        call(
            "seal",
            [
                *base,
                "seal-medium",
                *common,
                "--reuse-medium-schedule",
                "--parent-manifest",
                str(parent_path),
                "--parent-manifest-sha256",
                parent_sha256,
                "--output",
                str(root / "manifest"),
            ],
            "",
        )
        manifest = root / "manifest/manifest.json"
        derived = read(manifest)
        if population_identity(derived["schedule"]) != population_identity(parent["schedule"]):
            raise ValueError("resealing changed the common diagnostic population")
        command = [
            *base,
            "run",
            *common,
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            sha(manifest),
            "--resource-queue",
            str(queue_path),
            "--arm",
            arm,
            "--deterministic",
            "--checkpoint-conditions",
            "100",
        ]
        for label, backend in (("reference-probe", "reference"), ("fast-probe", "fast")):
            call(
                label,
                [
                    *command,
                    "--output",
                    str(root / label),
                    "--stats-backend",
                    backend,
                    "--limit",
                    "3",
                    "--gate-indices",
                    "0,1,15",
                ],
                "0",
            )
        parity = compare(root / "reference-probe", root / "fast-probe")
        call(
            "medium", [*command, "--output", str(root / "analysis"), "--stats-backend", "fast"], "0"
        )
        summary = read(root / "analysis/summary.json")
        if any(summary.get(k) != v for k, v in COUNTS.items()):
            raise ValueError("medium output coverage incomplete")
        call(
            "report",
            [
                sys.executable,
                str(source / "scripts/v2/cap40_posthoc_report.py"),
                "--input",
                str(root / "analysis"),
                "--original",
                str(root / "saved"),
                "--output",
                str(root / "report"),
            ],
            "",
        )
        checkpoint = read(Path(plan["run_root"]) / "fit/epoch_state.json")["last"]
        write(
            root / "MEDIUM_COMPLETE.json",
            {
                "status": "completed",
                "arm": arm,
                "source_commit": q["source_commit"],
                "report_controller_commit": subprocess.check_output(
                    ["git", "-C", str(report_source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "parent_failed_output_preserved": repair_output is not None,
                "queue_sha256": sha(queue_path),
                "checkpoint_sha256": checkpoint["sha256"],
                "diagnostic_population_sha256": population_identity(parent["schedule"]),
                "manifest_sha256": sha(manifest),
                "updates": 0,
                **COUNTS,
                "parity": parity,
                "end_to_end_seconds": time.perf_counter() - start,
                "stage_seconds": stages,
                "outputs": {
                    str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                    and p.suffix in (".json", ".csv", ".png", ".md")
                    and "recovery" not in p.parts
                },
            },
        )
        validate_medium_complete(q, row, root=root)
    except BaseException as exc:
        write(
            root / "FAILURE.json",
            {
                "status": "failed",
                "error": repr(exc),
                "queue_sha256": sha(queue_path),
                "stage_seconds": stages,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--arm", choices=AUTHORIZED_ARMS, required=True)
    parser.add_argument("--repair-output", type=Path)
    arguments = parser.parse_args()
    execute(arguments.queue, arguments.arm, repair_output=arguments.repair_output)
