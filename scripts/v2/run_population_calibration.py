"""One persistent resource owner around the native first-eight calibration CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from scripts.v2.jobs.response_anchor_medium_20261029_a2.controller import (
    OWNER_FIELDS,
    finish_phase,
    spawn_phase,
    write_new,
)
from scripts.v2.run_response_anchor import checked
from scripts.v2.shared_gpu import (
    admit,
    descendant,
    parent_map,
    process_info,
    snapshot,
    terminate_tree,
)


def calibration_command(spec, root, owner):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(root / "scripts/v2/calibrate_response_anchor.py"),
        "--config",
        spec["config"]["path"],
        "--data-root",
        spec["data_root"],
        "--output",
        str(Path(spec["output"]) / "calibration"),
        "--publication",
        spec["publication"]["path"],
        "--publication-sha256",
        spec["publication"]["sha256"],
        "--shared-policy",
        spec["policy"]["path"],
        "--shared-policy-sha256",
        spec["policy"]["sha256"],
        "--source-commit",
        spec["source_commit"],
        "--run-id",
        spec["run_id"],
        "--gpu",
        "0,1",
        "--seed",
        "1",
        "--bf16",
        "--resource-owner-pid",
        str(owner),
    ]


IDENTITY_FIELDS = ("pid", "uid", "start_ticks", "command_sha256")
DRAIN_SECONDS = 30.0


def require_deterministic_route_audit(audit: dict[str, Any]) -> None:
    metadata = audit.get("deterministic_audit", {})
    if (
        metadata.get("scope") != "route_audit_only_not_norms_or_fit"
        or metadata.get("cuda") is not True
        or metadata.get("cublas_workspace_config") not in (":4096:8", ":16:8")
        or metadata.get("inside") != {"enabled": True, "warn_only": False}
        or set(metadata.get("before", {})) != {"enabled", "warn_only"}
        or any(type(v) is not bool for v in metadata.get("before", {}).values())
        or metadata.get("restored") != metadata.get("before")
        or metadata.get("settings_restored") is not True
        or "failure" in metadata
        or audit.get("additional_audit_replays") != 2
        or audit.get("group_replays") != 5
        or audit.get("replay_norms_included_in_first_eight") is not False
    ):
        raise ValueError("protected population deterministic audit metadata differs")
    repeats = audit.get("repeatability_comparisons", {})
    if set(repeats) != {"true_true", "false_false"}:
        raise ValueError("protected population repeatability routes differ")
    for reports in repeats.values():
        if set(reports) != {"input", "response"}:
            raise ValueError("protected population repeatability scopes differ")
        for report in reports.values():
            if (
                report.get("atol") != 2e-6
                or report.get("rtol") != 0
                or report.get("tensor_count", 0) <= 0
                or any(
                    report.get(k) != 0
                    for k in ("missing_mismatch", "shape_mismatch", "nonfinite_tensors")
                )
                or not math.isfinite(report.get("max_abs", math.inf))
                or not 0 <= report["max_abs"] <= 2e-6
            ):
                raise ValueError("protected population repeatability gradient gate differs")


def require_head_route_audit(result, options):
    """Historical default receipts stay valid; protected heads require all-rank proof."""
    if getattr(options, "response_anchor_prediction_head_grad", True) is not False:
        return
    route = "gpm_head_parameters_detached_preserve_input_jacobian"
    audits = result.get("population_head_route_audit_ranks")
    if isinstance(audits, list):
        for audit in audits:
            require_deterministic_route_audit(audit)
    if (
        result.get("response_anchor_prediction_head_grad") is not False
        or result.get("gradient_routing") != route
        or not isinstance(audits, list)
        or len(audits) != 2
        or any(
            audit.get("schema") != "population-head-route-audit-v1"
            or audit.get("gradient_routing") != route
            or any(
                audit.get(field) != 0
                for field in (
                    "score_atol",
                    "score_rtol",
                    "loss_atol",
                    "loss_rtol",
                    "cuda_response_gradient_rtol",
                    "cuda_input_gradient_rtol",
                )
            )
            or audit.get("cuda_response_gradient_atol") != 2e-6
            or audit.get("cuda_input_gradient_atol") != 2e-6
            or any(
                not isinstance(audit.get("gradient_comparisons", {}).get(scope), dict)
                or audit["gradient_comparisons"][scope].get("atol") != tolerance
                or audit["gradient_comparisons"][scope].get("rtol") != 0
                or audit["gradient_comparisons"][scope].get("tensor_count", 0) <= 0
                or any(
                    audit["gradient_comparisons"][scope].get(field) != 0
                    for field in (
                        "missing_mismatch",
                        "shape_mismatch",
                        "nonfinite_tensors",
                    )
                )
                or not math.isfinite(audit["gradient_comparisons"][scope].get("max_abs", math.inf))
                or not 0 <= audit["gradient_comparisons"][scope]["max_abs"] <= tolerance
                for scope, tolerance in (("input", 2e-6), ("response", 2e-6))
            )
            or any(
                audit.get(field) is not True
                for field in (
                    "complete",
                    "head_score_and_loss_equal",
                    "input_gradient_equal",
                    "response_gradient_equal",
                    "auxiliary_head_gradient_zero",
                    "primary_head_gradient_preserved",
                    "state_restored",
                )
            )
            or any(
                audit.get(field) != 0
                for field in (
                    "optimizer_updates",
                    "teacher_updates",
                    "center_updates",
                )
            )
            for audit in audits
        )
    ):
        raise ValueError("protected population head calibration audit differs")


def same_process(current, expected):
    return current is not None and all(
        current.get(key) == expected.get(key) for key in IDENTITY_FIELDS
    )


def observe_owned(known, owner):
    """Extend only currently identity-matched owned roots to their own-UID descendants."""
    roots = {
        identity["pid"]
        for identity in known.values()
        if same_process(process_info(identity["pid"]), identity)
    }
    if not roots:
        return
    parents = parent_map()
    for pid in parents:
        if not any(descendant(pid, root, parents) for root in roots):
            continue
        identity = process_info(pid)
        if (
            identity is not None
            and identity.get("pid") == pid
            and identity.get("uid") == owner["uid"]
            and identity.get("start_ticks")
            and identity.get("command_sha256")
        ):
            # Keep each observed identity, including a child's legitimate exec transition.
            known[tuple(identity[key] for key in IDENTITY_FIELDS)] = identity


def active_owned(known):
    return [
        identity
        for identity in known.values()
        if same_process(process_info(identity["pid"]), identity)
    ]


def execute(path, digest):
    path = checked(dict(path=str(path), sha256=digest))
    spec = json.loads(path.read_text())
    if (
        set(spec)
        != {
            "schema",
            "run_id",
            "source_commit",
            "config",
            "publication",
            "policy",
            "data_root",
            "output",
        }
        or spec["schema"] != "population-initialization-calibration-v1"
    ):
        raise ValueError("exact population calibration metadata spec required")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    root = Path(__file__).resolve().parents[2]
    config_path, publication, policy_path = (
        checked(spec[k]) for k in ("config", "publication", "policy")
    )
    config = load_experiment_config(config_path)
    _, options = V2Options.parse_parameters(config.model.parameters)
    if (
        options.response_anchor_loss != "population_prediction"
        or options.response_anchor_weight != 1.0
        or not config_path.resolve().is_relative_to(root)
    ):
        raise ValueError("published unit-weight population probe required")
    output = Path(spec["output"]).resolve()
    if (
        not output.is_relative_to("/data/yilangliu")
        or output.is_relative_to(root)
        or output.is_relative_to(Path(spec["data_root"]).resolve())
        or output.exists()
    ):
        raise ValueError("fresh independent server calibration output required")
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=publication,
        expected_publication_receipt_sha256=spec["publication"]["sha256"],
    )
    if source.commit != spec["source_commit"]:
        raise ValueError("calibration source differs")
    policy = json.loads(policy_path.read_text())

    def guard():
        checked(dict(path=str(path), sha256=digest))
        for key in ("config", "publication", "policy"):
            checked(spec[key])
        admit(policy, snapshot(os.getpid()))

    guard()
    output.mkdir(parents=True, exist_ok=False)
    child = None
    started = None
    known = {}
    try:
        observed_owner = process_info(os.getpid())
        if observed_owner is None:
            raise RuntimeError("resource owner identity unavailable")
        owner = {key: observed_owner[key] for key in OWNER_FIELDS}
        write_new(output / "OWNER.json", owner)
        command = calibration_command(spec, root, os.getpid())
        environment = dict(
            PYTHONPATH=f"{root}/src:{root}/scripts/v2:{root}",
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            CUDA_VISIBLE_DEVICES="0,1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        child, started = spawn_phase(
            output, "native-calibration", command, environment, process_info, owner, cwd=root
        )
        if not started["process_identity_verified"]:
            raise RuntimeError("native calibration child identity was not verified")
        identity = started["actual_process_identity"]
        known[tuple(identity[key] for key in IDENTITY_FIELDS)] = identity
        observe_owned(known, owner)
        while child.poll() is None:
            observe_owned(known, owner)
            guard()
            time.sleep(5)
        terminal = finish_phase(output, child, started)
        if terminal["status"] != "completed":
            raise RuntimeError("native calibration failed; preserve actual exit receipt")
        deadline = time.monotonic() + DRAIN_SECONDS
        while active_owned(known):
            observe_owned(known, owner)
            guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("observed calibration descendants remain after launcher exit")
            time.sleep(min(1.0, remaining))
        complete = output / "calibration/COMPLETE.json"
        result = json.loads(complete.read_text())
        if (
            result.get("run_id") != spec["run_id"]
            or result.get("response_anchor_loss") != "population_prediction"
            or result.get("source", {}).get("commit") != spec["source_commit"]
            or not result.get("calibration_complete")
            or not result.get("state_restored")
            or result.get("world_size") != 2
            or any(
                result.get(k) != 0
                for k in ("optimizer_updates", "teacher_updates", "center_updates")
            )
            or list(output.rglob("*.pkl"))
        ):
            raise ValueError("native population calibration closure differs")
        require_head_route_audit(result, options)
        guard()
        if (
            inspect_source_identity(
                root,
                formal=True,
                expected_repository=config.source_code.repository,
                publication_receipt=publication,
                expected_publication_receipt_sha256=spec["publication"]["sha256"],
            )
            != source
        ):
            raise ValueError("source changed during calibration")
        write_new(
            output / "COMPLETE.json",
            dict(
                status="completed",
                run_id=spec["run_id"],
                spec=dict(path=str(path), sha256=digest),
                owner=owner,
                observed_process_identities=list(known.values()),
                observed_owned_processes_drained=True,
                native_complete=dict(path=str(complete), sha256=sha256_file(complete)),
                zero_pkl=True,
                scientific_complete=False,
            ),
        )
    except BaseException as error:
        cleanup_errors = []
        # A reparented rank may outlive torchrun. Only the captured exact identity is eligible.
        for identity in reversed(list(known.values())):
            try:
                if same_process(process_info(identity["pid"]), identity):
                    terminate_tree(identity["pid"], identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(repr(cleanup_error))
        exit_observation = {}
        if child is not None:
            try:
                exit_observation["poll_returncode"] = child.poll()
                exit_observation["wait_called"] = True
                exit_observation["wait_returncode"] = child.wait(timeout=DRAIN_SECONDS)
            except BaseException as cleanup_error:
                exit_observation["wait_error"] = repr(cleanup_error)
                cleanup_errors.append(repr(cleanup_error))
            try:
                if (
                    child.poll() is not None
                    and not (output / "native-calibration.EXIT.json").exists()
                ):
                    finish_phase(output, child, started)
            except BaseException as cleanup_error:
                cleanup_errors.append(repr(cleanup_error))
        write_new(
            output / "FAILURE.json",
            dict(
                status="failed",
                run_id=spec["run_id"],
                error=repr(error),
                cleanup_errors=cleanup_errors,
                child_exit_observation=exit_observation,
                observed_process_identities=list(known.values()),
                remaining_identity_matched_owned=active_owned(known),
                unverified_child_never_signalled=bool(
                    child is not None
                    and started is not None
                    and not started["process_identity_verified"]
                ),
                scientific_complete=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    execute(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
