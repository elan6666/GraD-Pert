"""R9 frozen endpoints share one physical medium recipe and fixed panel."""

from __future__ import annotations

from pathlib import Path

from cap40_posthoc import read, sha, write
from checkpoint_endpoint import select_checkpoint
from generate_representation_round9 import ARMS, ENDPOINTS, SCHEMA


def validate_complete(queue, row, *, root=None):
    from run_fgpo_medium import COUNTS, validate_medium_complete

    root = Path(root) if root is not None else Path(queue["output"]) / f"{row['name']}-medium"
    receipt = read(root / "MEDIUM_COMPLETE.json")
    if (
        queue.get("family") != SCHEMA
        or row["name"] not in ARMS
        or receipt.get("schema") != "staged-joint-dual-endpoint-medium-1"
        or receipt.get("status") != "completed"
        or receipt.get("updates") != 0
        or receipt.get("source_commit") != queue["source_commit"]
        or receipt.get("queue_sha256") != sha(Path(queue["output"]) / "queue.json")
        or set(receipt.get("endpoints", {})) != set(ENDPOINTS)
    ):
        raise ValueError("R9 needs both frozen medium endpoints")
    manifests, checkpoints = set(), set()
    for label, spec in ENDPOINTS.items():
        child = root / label
        proof = validate_medium_complete(queue, row, root=child, checkpoint_role=spec["role"])
        cp, endpoint = select_checkpoint(Path(row["plan"]["run_root"]), spec["role"])
        result = read(child / "MEDIUM_COMPLETE.json")
        if (
            receipt["endpoints"][label] != proof
            or endpoint != {**spec, "full_training_epochs": 3}
            or result.get("checkpoint_endpoint") != endpoint
        ):
            raise ValueError("R9 endpoint receipt changed")
        manifest = Path(result["manifest"]).resolve()
        if (
            not manifest.is_relative_to(root.resolve())
            or sha(manifest) != result["manifest_sha256"]
        ):
            raise ValueError("R9 shared physical manifest changed")
        manifests.add((str(manifest), result["manifest_sha256"]))
        checkpoints.add(cp["sha256"])
    if len(manifests) != 1 or len(checkpoints) != 2:
        raise ValueError("R9 needs distinct checkpoints and the same physical medium manifest")
    return dict(
        receipt=str(root / "MEDIUM_COMPLETE.json"),
        sha256=sha(root / "MEDIUM_COMPLETE.json"),
        **COUNTS,
    )


def execute(queue_path, arm):
    from run_br400_group import validate_training_contract, verify
    from run_fgpo_medium import COUNTS, SERVER_ROOT, validate_medium_complete
    from run_fgpo_medium import execute as endpoint

    queue = read(queue_path)
    if queue.get("family") != SCHEMA or arm not in ARMS:
        raise ValueError("dual endpoint runner requires the exact R9 family")
    verify(queue)
    row = next(row for row in queue["rows"] if row["name"] == arm)
    validate_training_contract(row["plan"])
    root = Path(queue["output"]) / f"{arm}-medium"
    if root.exists() or not root.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("fresh server-only R9 output required")
    root.mkdir()
    manifest, proofs = None, {}
    try:
        for label, spec in ENDPOINTS.items():
            child = root / label
            endpoint(
                queue_path,
                arm,
                checkpoint_role=spec["role"],
                endpoint_root=child,
                shared_manifest=manifest,
            )
            proofs[label] = validate_medium_complete(
                queue, row, root=child, checkpoint_role=spec["role"]
            )
            manifest = Path(read(child / "MEDIUM_COMPLETE.json")["manifest"])
        write(
            root / "MEDIUM_COMPLETE.json",
            dict(
                schema="staged-joint-dual-endpoint-medium-1",
                status="completed",
                updates=0,
                source_commit=queue["source_commit"],
                queue_sha256=sha(queue_path),
                endpoints=proofs,
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
