"""Read-only structural validation of four frozen-coordinate timing receipts."""

import argparse
import json
from pathlib import Path

from gradpert.hashing import sha256_file
from scripts.performance.abba_statistics import summarize_abba
from scripts.performance.profile_native_a0 import _abba_isolation


def validate_receipts(receipts: list[dict]) -> dict:
    if len(receipts) != 4:
        raise ValueError("requires four ordered ABBA receipts")
    reference = receipts[0]
    uuid = reference["physical_gpu_identity"]["selected_physical_gpu"]["uuid"]
    arms = []
    for receipt, impl in zip(
        receipts, ("cpu_vectorized", "cpu_array", "cpu_array", "cpu_vectorized"), strict=True
    ):
        required = {
            "status": "complete",
            "phase": "timing",
            "timing_protocol": "abba_5_20",
            "coordinate": "r50_e3_batch512",
            "capture_exact_state": False,
            "scientific_completion": False,
            "warmup_steps": 5,
            "measured_steps": 20,
            "observed_step_count": 25,
            "primary_failure": None,
            "teardown_failures": [],
            "initial_exact_state": None,
            "cuda_retry_or_oom_counter_total": 0,
            "sparse_union_implementation": impl,
        }
        for key, expected in required.items():
            if receipt[key] != expected:
                raise ValueError(f"invalid ABBA field: {key}")
        for key in (
            "source_commit",
            "source_identity",
            "exact_a0_identity",
            "native_architecture",
            "run_seed",
            "instrumentation",
            "ordered_batch_identities",
        ):
            if receipt[key] != reference[key]:
                raise ValueError(f"ABBA identity differs: {key}")
        if receipt["source_identity"]["dirty"]:
            raise ValueError("dirty ABBA source")
        if receipt["physical_gpu_identity"]["selected_physical_gpu"]["uuid"] != uuid:
            raise ValueError("ABBA physical GPU differs")
        for key in ("preflight_predicates", "runtime_predicates", "completion_resource_predicates"):
            if not receipt[key] or not all(v is True for v in receipt[key].values()):
                raise ValueError(f"ABBA predicates failed: {key}")
        steps = receipt["steps"]
        identities = receipt["ordered_batch_identities"]
        if [b["global_step"] for b in identities] != list(range(25)):
            raise ValueError("missing ordered batch identities")
        for batch in identities:
            for key in ("perturbed_row_ids_sha256", "control_row_ids_sha256"):
                value = batch[key]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise ValueError("invalid batch identity hash")
        if [s["global_step"] for s in steps] != list(range(25)):
            raise ValueError("ABBA ordered steps differ")
        for i, step in enumerate(steps):
            if "exact_state" in step or step["profile_phase"] != (
                "warmup" if i < 5 else "measured"
            ):
                raise ValueError("ABBA instrumentation differs")
            _abba_isolation(step["isolation_snapshot"], uuid)
        for key in ("started_host_snapshot", "completed_host_snapshot"):
            _abba_isolation(receipt[key], uuid)
        wall = [s["metrics"]["step_wall_ms"] for s in steps[5:]]
        if wall != receipt["measured_step_wall_ms"]:
            raise ValueError("ABBA raw timings differ")
        arms.append(
            {
                "wall_ms": wall,
                "peak_allocated": receipt["peak_allocated_gpu_bytes"],
                "peak_reserved": receipt["peak_reserved_gpu_bytes"],
            }
        )
    return summarize_abba(arms)


def validate_files(root: Path, expected_sha256: list[str]) -> dict:
    paths = [
        root / arm / "profile_evidence/profile-receipt.json" for arm in ("A1", "B1", "B2", "A2")
    ]
    if len(expected_sha256) != 4 or len(set(expected_sha256)) != 4:
        raise ValueError("requires four distinct pinned receipt hashes")
    for path, expected in zip(paths, expected_sha256, strict=True):
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError("receipt file hash mismatch")
    if list(root.rglob("*.pkl")):
        raise ValueError("ABBA root contains PKL")
    result = validate_receipts([json.loads(p.read_text()) for p in paths])
    result["receipt_sha256"] = {str(p): h for p, h in zip(paths, expected_sha256, strict=True)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-sha256", nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("ABBA acceptance evidence already exists")
    result = validate_files(args.root, args.receipt_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
