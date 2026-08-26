from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradpert.contracts import RunManifest
from gradpert.execution.matrix import (
    DATASET_IDS,
    LEARNED_MODEL_IDS,
    MatrixRuntime,
    build_experiment_tasks,
    require_completed_task,
    require_learned_smoke_gate,
)
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
HASH = "b" * 64


def _runtime(tmp_path: Path, *, runs_root: Path | None = None) -> MatrixRuntime:
    return MatrixRuntime(
        project_root=ROOT,
        config_root=ROOT / "configs" / "experiments",
        data_root=tmp_path / "data",
        runs_root=runs_root or tmp_path / "runs",
        native_python=Path("/runtime/native/python"),
        gears_python=Path("/runtime/gears/python"),
        gears_checkout=Path("/upstreams/gears"),
        gears_data_root=tmp_path / "gears-data",
        txpert_python=Path("/runtime/txpert/python"),
        txpert_checkout=Path("/upstreams/txpert"),
        devices=("cuda:0", "cuda:1"),
    )


def _manifest(task: object, *, formal_eligible: bool = True) -> RunManifest:
    return RunManifest(
        schema_version="run-manifest-v1",
        run_id=task.run_id,
        model_id=task.model_id,
        dataset_id=task.dataset_id,
        protocol_id="datasets-v2",
        run_seed=task.run_seed,
        source_commit=COMMIT,
        source_dirty=not formal_eligible,
        formal_eligible=formal_eligible,
        config_sha256=task.config_sha256,
        environment_sha256=HASH,
        canonical_data_sha256=HASH,
        split_content_sha256=HASH,
        control_manifest_sha256=HASH,
        status="evaluated",
        best_checkpoint_sha256=HASH if task.model_id in LEARNED_MODEL_IDS else None,
        test_evaluations=1,
    )


def _write_training_receipt(task: object, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "test-training-receipt-v1",
                "model_id": task.model_id,
                "dataset_id": task.dataset_id,
                "epochs_requested": 1,
                "epochs_completed": 1,
                "canonical_test_truth_present_during_fit": False,
                "checkpoint_sha256": HASH,
            }
        ),
        encoding="utf-8",
    )


def test_exact_phase_counts_seeds_and_official_isolation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    smoke = build_experiment_tasks(
        phase="smoke",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )
    nonlearned = build_experiment_tasks(
        phase="nonlearned",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )
    full = build_experiment_tasks(
        phase="full",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )

    assert (len(smoke), len(nonlearned), len(full)) == (15, 15, 20)
    assert {task.run_seed for task in smoke + nonlearned} == {1}
    assert {task.run_seed for task in full} == {1, 2, 3, 4}
    assert {task.model_id for task in full} == {"gradpert_b2"}
    assert {task.dataset_id for task in full} == set(DATASET_IDS)
    assert {task.expected_epochs for task in smoke} == {1}
    assert {task.expected_epochs for task in full} == {200}
    assert len({task.task_id for task in smoke + nonlearned + full}) == 50

    gears = next(task for task in smoke if task.model_id == "gears")
    txpert = next(task for task in smoke if task.model_id == "txpert_public")
    txpert_rpe1 = next(
        task
        for task in smoke
        if task.model_id == "txpert_public" and task.dataset_id == "replogle_rpe1_essential"
    )
    assert gears.command[:3] == (
        "/runtime/gears/python",
        "-m",
        "benchmarks.gears.runner",
    )
    assert txpert.command[:3] == (
        "/runtime/txpert/python",
        "-m",
        "benchmarks.txpert.runner",
    )
    assert dict(gears.environment)["PYTHONPATH"] == f"{ROOT / 'src'}:{ROOT}"
    assert "--formal" in gears.command and "--official-data-root" in gears.command
    assert "CUDA_VISIBLE_DEVICES" not in dict(gears.environment)
    assert txpert_rpe1.device == "cuda:1"
    assert dict(txpert_rpe1.environment)["CUDA_VISIBLE_DEVICES"] == "1"
    txpert_device_position = txpert_rpe1.command.index("--device") + 1
    assert txpert_rpe1.command[txpert_device_position] == "cuda:0"
    for task in (item for item in smoke if item.model_id == "txpert_public"):
        physical_index = task.device.removeprefix("cuda:")
        assert dict(task.environment)["CUDA_VISIBLE_DEVICES"] == physical_index
        command_device_position = task.command.index("--device") + 1
        assert task.command[command_device_position] == "cuda:0"


def test_txpert_matrix_rejects_ambiguous_process_device(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime = MatrixRuntime(**{**runtime.__dict__, "devices": ("cuda",)})

    with pytest.raises(ValueError, match="explicit cuda:<index>"):
        build_experiment_tasks(
            phase="smoke",
            runtime=runtime,
            namespace="formal-v1",
            expected_commit=COMMIT,
            formal=True,
        )


def test_full_phase_is_formal_only_and_resume_is_explicit(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="formal-only"):
        build_experiment_tasks(
            phase="full",
            runtime=runtime,
            namespace="development-v2",
            expected_commit=COMMIT,
            formal=False,
        )
    tasks = build_experiment_tasks(
        phase="full",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
        resume_native_full=True,
    )
    assert all(task.command[-1] == "--resume" for task in tasks)


def test_completed_task_must_match_every_identity(tmp_path: Path) -> None:
    task = build_experiment_tasks(
        phase="smoke",
        runtime=_runtime(tmp_path),
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )[0]
    path = task.run_root / "small_results" / "run_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(_manifest(task).model_dump_json(), encoding="utf-8")
    assert require_completed_task(task) == path

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_sha256"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        require_completed_task(task)


def test_full_gate_requires_all_15_exact_smoke_receipts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    tasks = build_experiment_tasks(
        phase="smoke",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )
    for task in tasks:
        path = task.run_root / "small_results" / "run_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(_manifest(task).model_dump_json(), encoding="utf-8")
        _write_training_receipt(task, path.parent / "training_receipt.json")
    verified = require_learned_smoke_gate(
        runtime=runtime,
        expected_commit=COMMIT,
        require_formal=True,
    )
    assert len(verified) == 15

    verified[-1].unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        require_learned_smoke_gate(
            runtime=runtime,
            expected_commit=COMMIT,
            require_formal=True,
        )


def test_full_gate_rejects_cross_model_fairness_drift(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    tasks = build_experiment_tasks(
        phase="smoke",
        runtime=runtime,
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )
    for task in tasks:
        path = task.run_root / "small_results" / "run_manifest.json"
        path.parent.mkdir(parents=True)
        manifest = _manifest(task)
        if task.model_id == "txpert_public" and task.dataset_id == DATASET_IDS[0]:
            manifest = manifest.model_copy(update={"control_manifest_sha256": "c" * 64})
        path.write_text(manifest.model_dump_json(), encoding="utf-8")
        _write_training_receipt(task, path.parent / "training_receipt.json")
    with pytest.raises(ValueError, match="fairness hashes differ"):
        require_learned_smoke_gate(
            runtime=runtime,
            expected_commit=COMMIT,
            require_formal=True,
        )


def test_task_config_hashes_are_the_committed_files(tmp_path: Path) -> None:
    tasks = build_experiment_tasks(
        phase="nonlearned",
        runtime=_runtime(tmp_path),
        namespace="formal-v1",
        expected_commit=COMMIT,
        formal=True,
    )
    assert all(task.config_sha256 == sha256_file(task.config_path) for task in tasks)
