"""Two retained PT2 checkpoints, one physical sampling/view manifest, no updates."""

from __future__ import annotations

import time
from pathlib import Path

from cap40_posthoc import read, sha, write
from checkpoint_endpoint import select_checkpoint
from generate_pt2 import ENDPOINTS, SCHEMA


def validate_complete(queue, row, *, root=None):
    from run_fgpo_medium import COUNTS, validate_medium_complete

    root = Path(root) if root is not None else Path(queue["output"]) / "PT2-medium"
    result = read(root / "MEDIUM_COMPLETE.json")
    if (
        queue.get("family") != SCHEMA
        or row["name"] != "PT2"
        or result.get("schema") != "pt2-dual-endpoint-medium-complete-1"
        or result.get("status") != "completed"
        or result.get("source_commit") != queue["source_commit"]
        or result.get("queue_sha256") != sha(Path(queue["output"]) / "queue.json")
        or result.get("updates") != 0
        or set(result.get("endpoints", {})) != set(ENDPOINTS)
        or any(result.get(k) != v for k, v in COUNTS.items())
    ):
        raise ValueError("PT2 requires both exact, completed medium endpoints")
    manifests = set()
    checkpoints = set()
    for label, spec in ENDPOINTS.items():
        child = root / label
        proof = validate_medium_complete(queue, row, root=child, checkpoint_role=spec["role"])
        if result["endpoints"][label] != proof:
            raise ValueError("PT2 endpoint receipt changed")
        r = read(child / "MEDIUM_COMPLETE.json")
        _, endpoint = select_checkpoint(Path(row["plan"]["run_root"]), spec["role"])
        if r["checkpoint_endpoint"] != endpoint or endpoint["epoch"] != spec["epoch"]:
            raise ValueError("PT2 checkpoint epoch or stage changed")
        manifest = Path(r["manifest"]).resolve()
        if not manifest.is_relative_to(root.resolve()) or sha(manifest) != r["manifest_sha256"]:
            raise ValueError("PT2 shared manifest changed or escaped this analysis")
        manifests.add((str(manifest), r["manifest_sha256"]))
        checkpoints.add(r["checkpoint_sha256"])
    if len(manifests) != 1 or len(checkpoints) != 2:
        raise ValueError("PT2 needs distinct checkpoints and one identical physical manifest")
    return dict(
        receipt=str(root / "MEDIUM_COMPLETE.json"),
        sha256=sha(root / "MEDIUM_COMPLETE.json"),
        **COUNTS,
    )


def execute(queue_path, arm, *, repair_output=None):
    from run_br400_group import validate_complete as validate_training
    from run_br400_group import verify
    from run_fgpo_medium import (
        COUNTS,
        SERVER_ROOT,
        validate_medium_complete,
        validate_report_repair,
    )
    from run_fgpo_medium import execute as endpoint

    queue = read(queue_path)
    if arm != "PT2" or queue.get("family") != SCHEMA:
        raise ValueError("dual endpoint analysis is limited to the exact PT2 family")
    verify(queue)
    row = next(r for r in queue["rows"] if r["name"] == arm)
    validate_training(row["plan"])
    root = (
        validate_report_repair(queue, arm, repair_output, Path(__file__).resolve().parents[2])
        if repair_output is not None
        else Path(queue["output"]) / "PT2-medium"
    )
    if root.exists() or not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server-only dual-endpoint output required")
    root.mkdir()
    started = time.perf_counter()
    proofs, manifest = {}, None
    try:
        for label, spec in ENDPOINTS.items():
            child = root / label
            endpoint(
                queue_path,
                arm,
                checkpoint_role=spec["role"],
                endpoint_root=child,
                shared_manifest=manifest,
                repair_output=repair_output,
            )
            proofs[label] = validate_medium_complete(
                queue, row, root=child, checkpoint_role=spec["role"]
            )
            manifest = Path(read(child / "MEDIUM_COMPLETE.json")["manifest"])
        write(
            root / "MEDIUM_COMPLETE.json",
            dict(
                schema="pt2-dual-endpoint-medium-complete-1",
                status="completed",
                arm=arm,
                source_commit=queue["source_commit"],
                queue_sha256=sha(queue_path),
                updates=0,
                endpoints=proofs,
                end_to_end_seconds=time.perf_counter() - started,
                **COUNTS,
            ),
        )
        return validate_complete(queue, row)
    except BaseException as exc:
        write(
            root / "FAILURE.json",
            dict(status="failed", error=repr(exc), completed_endpoints=proofs),
        )
        raise
