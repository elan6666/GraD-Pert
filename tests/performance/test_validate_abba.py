import json
from copy import deepcopy

import pytest

from gradpert.hashing import sha256_file
from scripts.performance.profile_native_a0 import ProfileGateError
from scripts.performance.validate_abba import validate_files, validate_receipts


def receipts():
    snapshot = {
        "process_id": 123,
        "load_average": [0.5],
        "nvidia_smi": {
            "gpus": {"returncode": 0, "rows": [{"uuid": "gpu0", "utilization.gpu": "50"}]},
            "compute_apps": {"returncode": 0, "rows": [{"pid": "123"}]},
        },
    }
    result = []
    for impl, wall in zip(
        ("cpu_vectorized", "cpu_array", "cpu_array", "cpu_vectorized"),
        (1000, 800, 800, 1000),
        strict=True,
    ):
        result.append(
            {
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
                "source_commit": "a" * 40,
                "source_identity": {"dirty": False},
                "exact_a0_identity": {"config": "fixed"},
                "native_architecture": {"dim": 128},
                "run_seed": 1,
                "instrumentation": {},
                "physical_gpu_identity": {"selected_physical_gpu": {"uuid": "gpu0"}},
                "preflight_predicates": {"ok": True},
                "runtime_predicates": {"ok": True},
                "completion_resource_predicates": {"ok": True},
                "steps": [
                    {
                        "global_step": i,
                        "profile_phase": "warmup" if i < 5 else "measured",
                        "metrics": {"step_wall_ms": wall},
                        "isolation_snapshot": deepcopy(snapshot),
                    }
                    for i in range(25)
                ],
                "started_host_snapshot": deepcopy(snapshot),
                "completed_host_snapshot": deepcopy(snapshot),
                "measured_step_wall_ms": [wall] * 20,
                "peak_allocated_gpu_bytes": 1000,
                "peak_reserved_gpu_bytes": 2000,
            }
        )
    for receipt in result:
        receipt["ordered_batch_identities"] = [
            {
                "global_step": i,
                "perturbed_row_ids_sha256": "a" * 64,
                "control_row_ids_sha256": "b" * 64,
            }
            for i in range(25)
        ]
    return result


def test_valid_receipts_pass():
    assert validate_receipts(receipts())["timing_thresholds_passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "b" * 40),
        ("warmup_steps", 2),
        ("capture_exact_state", True),
        ("runtime_predicates", {}),
        ("cuda_retry_or_oom_counter_total", 1),
        ("run_seed", 2),
        ("measured_step_wall_ms", [2] * 20),
    ],
)
def test_tampered_receipt_rejected(field, value):
    rows = receipts()
    rows[1][field] = value
    with pytest.raises(ValueError):
        validate_receipts(rows)


def test_mid_run_competitor_rejected():
    rows = receipts()
    rows[2]["steps"][12]["isolation_snapshot"]["nvidia_smi"]["compute_apps"]["rows"].append(
        {"pid": "999"}
    )
    with pytest.raises(ProfileGateError, match="competing GPU"):
        validate_receipts(rows)


def test_live_files_hash_and_pkl_gate(tmp_path):
    hashes = []
    for arm, receipt in zip(("A1", "B1", "B2", "A2"), receipts(), strict=True):
        receipt["run_id"] = arm
        path = tmp_path / arm / "profile_evidence/profile-receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        hashes.append(sha256_file(path))
    assert validate_files(tmp_path, hashes)["timing_thresholds_passed"]
    stray = tmp_path / "stray.pkl"
    stray.write_bytes(b"test")
    with pytest.raises(ValueError, match="PKL"):
        validate_files(tmp_path, hashes)
    stray.unlink()
    hashes[0] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_files(tmp_path, hashes)
