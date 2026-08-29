from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/performance/profile_native_a0.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("profile_native_a0_test_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load A0 profiler script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def profiler_script() -> ModuleType:
    return _load_script()


def test_training_only_guard_never_constructs_or_materializes_evaluation_data(
    profiler_script: ModuleType,
) -> None:
    state = {
        "real_canonical_evaluation_constructor_count": 0,
        "guard_bindings": [],
        "validation_cache_requested": False,
        "validation_cache_materialized": False,
        "truth_access_attempts": [],
    }
    factory = profiler_script._training_only_evaluator_factory(state)

    with factory(split_name="val") as guard:
        assert guard.configure_expression_cache(enabled=True) == 0.0

    assert state == {
        "real_canonical_evaluation_constructor_count": 0,
        "guard_bindings": ["val"],
        "validation_cache_requested": True,
        "validation_cache_materialized": False,
        "truth_access_attempts": [],
    }
    assert profiler_script._evaluation_access_summary(state) == {
        "validation_guard_bound": True,
        "test_guard_bound": False,
        "validation_accessed": False,
        "test_truth_accessed": False,
    }


def test_training_only_guard_fails_closed_on_truth_access(profiler_script: ModuleType) -> None:
    state: dict[str, object] = {}
    guard = profiler_script._training_only_evaluator_factory(state)(split_name="test")

    with pytest.raises(profiler_script.EvaluationAccessError, match=r"test\.control_manifest"):
        _ = guard.control_manifest

    assert state["truth_access_attempts"] == ["test.control_manifest"]


def test_bounded_step_stops_on_nth_update_without_n_plus_one(
    profiler_script: ModuleType,
) -> None:
    original_calls: list[int] = []
    before_calls: list[int] = []
    after_calls: list[int] = []

    def original(engine, batch, *, global_step: int):
        original_calls.append(global_step)
        return SimpleNamespace(step_wall_ms=float(global_step))

    bounded = profiler_script._make_bounded_train_step(
        original,
        total_steps=2,
        before_step=lambda engine: before_calls.append(len(original_calls)),
        after_step=lambda engine, metrics, global_step: after_calls.append(global_step),
    )

    assert bounded(object(), object(), global_step=0).step_wall_ms == 0.0
    with pytest.raises(profiler_script.ProfileComplete, match="final step"):
        bounded(object(), object(), global_step=1)

    assert original_calls == [0, 1]
    assert before_calls == [0, 1]
    assert after_calls == [0, 1]


def test_percentiles_are_linear_and_explicit(profiler_script: ModuleType) -> None:
    assert profiler_script._percentiles([]) is None
    assert profiler_script._percentiles([1.0, 2.0, 3.0, 4.0]) == {
        "p50": 2.5,
        "p90": pytest.approx(3.7),
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
    }


def test_capacity_requires_max_of_four_gib_and_fifteen_percent(
    profiler_script: ModuleType,
) -> None:
    clean_evaluation = {
        "validation_callback_count": 0,
        "truth_access_attempts": [],
        "validation_cache_materialized": False,
    }

    def predicates(*, free_gib: int, total_gib: int) -> dict[str, bool]:
        observed = [
            {
                "gpu_free_bytes_after_step": free_gib * profiler_script.GIB,
                "gpu_total_bytes": total_gib * profiler_script.GIB,
                "cuda_memory_stats": {},
            }
        ]
        result, _ = profiler_script._runtime_capacity_predicates(
            observed,
            total_steps=1,
            minimum_gpu_headroom_fraction=0.15,
            minimum_gpu_free_bytes=4 * profiler_script.GIB,
            evaluation_state=clean_evaluation,
        )
        return result

    absolute_binds = predicates(free_gib=3, total_gib=16)
    assert absolute_binds["gpu_headroom_fraction_at_least_limit"] is True
    assert absolute_binds["gpu_free_bytes_at_least_absolute_limit"] is False

    fraction_binds = predicates(free_gib=10, total_gib=80)
    assert fraction_binds["gpu_headroom_fraction_at_least_limit"] is False
    assert fraction_binds["gpu_free_bytes_at_least_absolute_limit"] is True

    assert all(predicates(free_gib=12, total_gib=80).values())


def test_physical_gpu_resolution_honors_cuda_visible_devices(
    profiler_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    snapshot = {
        "nvidia_smi": {
            "gpus": {
                "available": True,
                "rows": [
                    {"index": "0", "uuid": "GPU-A"},
                    {"index": "1", "uuid": "GPU-B"},
                ],
            },
            "compute_apps": {
                "available": True,
                "rows": [{"gpu_uuid": "GPU-A", "pid": "123", "process_name": "other"}],
            },
        }
    }

    selected, competing = profiler_script._selected_physical_gpu(
        snapshot,
        device_name="cuda:0",
    )

    assert selected["uuid"] == "GPU-B"
    assert competing == []


def test_preflight_failure_still_writes_atomic_receipt(
    profiler_script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_root = tmp_path / "failed-profile"
    missing_config = tmp_path / "missing.yaml"
    frozen_hash = "a" * 64
    arguments = [
        "--config",
        str(missing_config),
        "--data-root",
        str(tmp_path),
        "--run-root",
        str(run_root),
        "--run-id",
        "failure-evidence",
        "--repository-root",
        str(tmp_path),
        "--development-commit",
        "b" * 40,
        "--phase",
        "capacity",
        "--expected-config-sha256",
        frozen_hash,
        "--expected-canonical-data-sha256",
        frozen_hash,
        "--expected-split-content-sha256",
        frozen_hash,
        "--expected-source-h5ad-sha256",
        frozen_hash,
        "--expected-source-registry-sha256",
        frozen_hash,
        "--expected-graph-gene-order-sha256",
        frozen_hash,
        "--expected-topology-content-sha256",
        frozen_hash,
        "--minimum-disk-free-bytes",
        "0",
        "--minimum-host-available-bytes",
        "0",
    ]

    assert profiler_script.main(arguments) == 2
    receipt_path = run_root / "profile_evidence/profile-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["primary_failure"]["type"] == "FileNotFoundError"
    assert receipt["teardown_failures"] == []
