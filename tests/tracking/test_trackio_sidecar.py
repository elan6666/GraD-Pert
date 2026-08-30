from __future__ import annotations

import csv
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gradpert.tracking.trackio_sidecar as sidecar_module
from gradpert.tracking.trackio_sidecar import (
    TrackingGateError,
    TrackioSidecarConfig,
    run_trackio_sidecar,
)


class _RepositoryNotFoundError(Exception):
    pass


class _BucketNotFoundError(Exception):
    pass


class _FakeHub:
    def __init__(
        self,
        *,
        exists: bool = False,
        private: bool = True,
        bucket_exists: bool = False,
        bucket_private: bool = True,
    ) -> None:
        self.exists = exists
        self.private = private
        self.bucket_exists = bucket_exists
        self.bucket_private = bucket_private
        self.mounted_bucket: str | None = None
        self.utils = SimpleNamespace(RepositoryNotFoundError=_RepositoryNotFoundError)
        self.errors = SimpleNamespace(BucketNotFoundError=_BucketNotFoundError)

    @staticmethod
    def get_token() -> str:
        return "private-test-token"

    def HfApi(self, *, token: str) -> Any:
        assert token == "private-test-token"
        owner = self

        class _Api:
            @staticmethod
            def space_info(*, repo_id: str) -> Any:
                assert repo_id == "owner/private-space"
                if not owner.exists:
                    raise _RepositoryNotFoundError
                return SimpleNamespace(private=owner.private)

            @staticmethod
            def bucket_info(*, bucket_id: str) -> Any:
                assert bucket_id == "owner/private-bucket"
                if not owner.bucket_exists:
                    raise _BucketNotFoundError
                return SimpleNamespace(
                    id="owner/private-bucket",
                    private=owner.bucket_private,
                )

            @staticmethod
            def create_bucket(*, bucket_id: str, private: bool, exist_ok: bool) -> Any:
                assert bucket_id == "owner/private-bucket"
                assert private is True
                assert exist_ok is False
                owner.bucket_exists = True
                owner.bucket_private = True
                return SimpleNamespace(bucket_id=bucket_id)

            @staticmethod
            def get_space_runtime(*, repo_id: str) -> Any:
                assert repo_id == "owner/private-space"
                volumes = []
                if owner.mounted_bucket is not None:
                    volumes.append(
                        SimpleNamespace(
                            type="bucket",
                            source=owner.mounted_bucket,
                            mount_path="/data",
                        )
                    )
                return SimpleNamespace(volumes=volumes)

        return _Api()


class _FakeTrackio:
    def __init__(self, hub: _FakeHub) -> None:
        self.hub = hub
        self.init_kwargs: dict[str, Any] | None = None
        self.logged: list[tuple[int, dict[str, int | float]]] = []
        self.gpu_devices: list[int] = []
        self.finished = False

    def init(self, **kwargs: Any) -> Any:
        self.init_kwargs = kwargs
        self.hub.exists = True
        self.hub.mounted_bucket = kwargs["bucket_id"]
        return SimpleNamespace(id="trackio-run-1")

    def log(self, metrics: dict[str, int | float], *, step: int) -> None:
        self.logged.append((step, metrics))

    def log_gpu(self, *, run: Any, device: int) -> dict[str, float]:
        assert run.id == "trackio-run-1"
        self.gpu_devices.append(device)
        return {"gpu/0/utilization": 50.0}

    def finish(self) -> None:
        self.finished = True


class _NoMountTrackio(_FakeTrackio):
    def init(self, **kwargs: Any) -> Any:
        self.init_kwargs = kwargs
        self.hub.exists = True
        return SimpleNamespace(id="trackio-run-1")


class _FilesystemTrackio(_FakeTrackio):
    def init(self, **kwargs: Any) -> Any:
        store = Path(os.environ["TRACKIO_DIR"])
        (store / "trackio-test.sqlite").write_text("private\n", encoding="utf-8")
        return super().init(**kwargs)


class _RaisingFinishTrackio(_FakeTrackio):
    def __init__(self, hub: _FakeHub) -> None:
        super().__init__(hub)
        self.finish_calls = 0

    def finish(self) -> None:
        self.finish_calls += 1
        raise RuntimeError("finish-error")


class _RaisingFinishNoMountTrackio(_NoMountTrackio):
    def finish(self) -> None:
        raise RuntimeError("cleanup-error")


def _architecture() -> dict[str, Any]:
    return {
        "graph_axis_policy": "recomputed_hvg_union_candidate_targets",
        "graph_hvg_count": 512,
        "graph_encoder_family": "multi_source_sparse_transformer",
        "graph_sources": ["string", "go"],
        "local_view_builder": "ring_induced",
        "local_view_count": 4,
        "local_view_node_budget_ratio_numerator": 1,
        "local_view_node_budget_ratio_denominator": 2,
        "local_anchor_mask_view_ratio_numerator": 0,
        "local_anchor_mask_view_ratio_denominator": 1,
        "gene_feature_mode": "learned_id",
        "decoder_mode": "additive",
    }


def _train_row(global_step: int) -> dict[str, Any]:
    return {
        "epoch": 0,
        "global_step": global_step,
        "total_loss": 5.0 - global_step,
        "prediction_loss": 1.0,
        "condition_consistency_loss": 2.0,
        "masked_node_loss": 3.0,
        "spread_loss": 4.0,
        "spread_available": True,
        "teacher_momentum": 0.996,
        "prediction_graph_gradient_norm": 0.1,
        "auxiliary_graph_gradient_norm": 0.2,
        "prediction_to_auxiliary_gradient_ratio": 0.5,
        "condition_target_entropy": 1.1,
        "masked_node_target_entropy": None,
        "condition_prototypes_used": 8,
        "masked_node_prototypes_used": 7,
        "condition_center_norm": 0.3,
        "masked_node_center_norm": None,
        "unique_condition_count": 2,
        "masked_node_count": 3,
        "batch_cell_count": 256,
        "data_read_ms": 10.0,
        "host_to_device_ms": 20.0,
        "view_build_ms": 30.0,
        "teacher_forward_ms": 40.0,
        "student_global_ms": 50.0,
        "student_local_ms": 60.0,
        "prediction_ms": 70.0,
        "backward_update_ms": 80.0,
        "step_wall_ms": 970.0,
        "local_view_realization_count": 4,
        "local_node_count_sum": 5616,
        "local_node_count_min": 1404,
        "local_node_count_max": 1404,
        "local_budget_hit_count": 4,
        "local_node_counts_sha256": "sensitive-local-hash",
        "masked_local_assignment_count": 0,
        "masked_local_index_counts_json": "sensitive-local-json",
        "masked_local_assignments_sha256": "sensitive-mask-hash",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(tmp_path: Path, *, formal_eligible: bool = True) -> Path:
    run_root = tmp_path / "scientific-run"
    small = run_root / "small_results"
    small.mkdir(parents=True)
    (small / "run_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "native-run-meta-v1",
                "run_id": "ablation/nadig_jurkat/a0_ratio_ring_half/seed-1",
                "mode": "pilot",
                "model_id": "gradpert_b2",
                "dataset_id": "nadig_jurkat",
                "protocol_id": "within_cell_unseen_single",
                "run_seed": 1,
                "source": {
                    "commit": "a" * 40,
                    "dirty": False,
                    "formal_eligible": formal_eligible,
                    "secret_path": "/private/server/path",
                },
                "config_path": "/private/config/path",
                "config_sha256": "b" * 64,
                "max_epochs": 1,
                "steps_per_epoch": 2,
                "validation_monitor": "val/txpert_macro_pearson_delta",
                "native_architecture": _architecture(),
                "runtime_graph_gene_order_sha256": "sensitive-gene-order-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(small / "train_steps.csv", [_train_row(0), _train_row(1)])
    _write_csv(
        small / "validation.csv",
        [
            {
                "epoch": 0,
                "global_step": 2,
                "val_txpert_macro_pearson_delta": 0.42,
                "improved": True,
                "consecutive_non_improvements": 0,
            }
        ],
    )
    (small / "training_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "native-training-receipt-v1",
                "epochs_completed": 1,
                "optimizer_steps": 2,
                "canonical_test_truth_present_during_fit": False,
                "checkpoint_sha256": "sensitive-checkpoint-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (small / "metrics_summary.json").write_text('{"sealed_test_metric": 999}\n', encoding="utf-8")
    return run_root


def _config(tmp_path: Path, run_root: Path) -> TrackioSidecarConfig:
    return TrackioSidecarConfig(
        run_root=run_root,
        trackio_dir=tmp_path / "trackio-store",
        state_path=tmp_path / "tracking-state.json",
        receipt_path=tmp_path / "tracking-receipt.json",
        project="grad-pert-vnext-ablations",
        run_name="a0-ratio-ring-half-seed1",
        group="nadig-jurkat-four-local-v1",
        space_id="owner/private-space",
        bucket_id="owner/private-bucket",
        variant_id="a0_ratio_ring_half",
        expected_run_id="ablation/nadig_jurkat/a0_ratio_ring_half/seed-1",
        expected_source_commit="a" * 40,
        expected_config_sha256="b" * 64,
        expected_model_id="gradpert_b2",
        expected_dataset_id="nadig_jurkat",
        expected_protocol_id="within_cell_unseen_single",
        expected_seed=1,
        expected_optimizer_steps=2,
        expected_validations=1,
        follow=False,
    )


@pytest.fixture(autouse=True)
def _single_visible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")


def test_sidecar_logs_only_allowlisted_curves_and_system_settings(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    hub = _FakeHub()
    trackio = _FakeTrackio(hub)

    receipt = run_trackio_sidecar(
        config,
        trackio_module=trackio,
        hub_module=hub,
    )

    assert trackio.finished is True
    assert trackio.init_kwargs is not None
    assert trackio.init_kwargs["private"] is True
    assert trackio.init_kwargs["bucket_id"] == "owner/private-bucket"
    assert trackio.init_kwargs["name"].endswith("-aaaaaaa")
    assert trackio.init_kwargs["resume"] == "never"
    assert trackio.init_kwargs["auto_log_gpu"] is False
    assert trackio.init_kwargs["auto_log_cpu"] is True
    assert trackio.gpu_devices == [0]
    assert [step for step, _ in trackio.logged] == [1, 2, 2]
    assert trackio.logged[0][1]["train/total_loss"] == 5.0
    assert trackio.logged[0][1]["performance/end_to_end_ms"] == 1000.0
    assert trackio.logged[0][1]["performance/cells_per_second"] == 256.0
    assert trackio.logged[-1][1]["validation/txpert_macro_pearson_delta"] == 0.42
    captured = json.dumps(
        {"config": trackio.init_kwargs["config"], "logs": trackio.logged},
        sort_keys=True,
    )
    for forbidden in (
        "sealed_test_metric",
        "sensitive-local-hash",
        "sensitive-local-json",
        "sensitive-mask-hash",
        "sensitive-gene-order-hash",
        "sensitive-checkpoint-hash",
        "/private/",
        "private-test-token",
    ):
        assert forbidden not in captured
    assert receipt["status"] == "local_client_complete"
    assert receipt["trackio_run_id"] == "trackio-run-1"
    assert receipt["optimizer_steps_enqueued"] == 2
    assert receipt["validations_enqueued"] == 1
    assert receipt["gpu_samples_enqueued"] == 1
    assert receipt["visible_cuda_device_selector"] == "1"
    assert receipt["test_metrics_uploaded"] is False
    assert receipt["artifacts_uploaded"] is False
    assert receipt["performance_timing_lineage"] is False
    assert receipt["telemetry_authority"] is False
    assert receipt["live_dashboard_provisional"] is True
    assert receipt["remote_sync_verified"] is False


def test_sidecar_creates_owner_only_local_tracking_state_and_restores_umask(
    tmp_path: Path,
) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    hub = _FakeHub()
    previous_umask = os.umask(0o022)
    try:
        run_trackio_sidecar(
            config,
            trackio_module=_FilesystemTrackio(hub),
            hub_module=hub,
        )
        observed_umask = os.umask(0o022)
        assert observed_umask == 0o022
    finally:
        os.umask(previous_umask)

    private_paths = (
        config.trackio_dir,
        config.trackio_dir / "trackio-test.sqlite",
        config.state_path,
        config.receipt_path,
        config.state_path.with_suffix(f"{config.state_path.suffix}.lock"),
    )
    for path in private_paths:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        assert mode & 0o077 == 0, (path, oct(mode))
    assert stat.S_IMODE(config.trackio_dir.stat().st_mode) == 0o700


def test_sidecar_rejects_nonformal_source(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path, formal_eligible=False)
    with pytest.raises(TrackingGateError, match="non-formal"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(_FakeHub()),
            hub_module=_FakeHub(),
        )


def test_sidecar_rejects_public_existing_space(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub(exists=True, private=False)
    with pytest.raises(TrackingGateError, match="non-private"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_sidecar_rejects_public_existing_bucket(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub(
        exists=True,
        private=True,
        bucket_exists=True,
        bucket_private=False,
    )
    with pytest.raises(TrackingGateError, match="non-private Bucket"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_sidecar_refuses_to_replace_existing_space_bucket(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub(exists=True, private=True, bucket_exists=True)
    hub.mounted_bucket = "owner/other-bucket"
    with pytest.raises(TrackingGateError, match="refusing to replace"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_sidecar_requires_postflight_mount_and_finishes_on_error(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub()
    trackio = _NoMountTrackio(hub)
    with pytest.raises(TrackingGateError, match="not mounted"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=trackio,
            hub_module=hub,
        )
    assert trackio.finished is True


def test_sidecar_rejects_discontinuous_or_nonfinite_training_rows(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = [_train_row(0), _train_row(2)]
    rows[1]["total_loss"] = "nan"
    _write_csv(run_root / "small_results" / "train_steps.csv", rows)
    hub = _FakeHub()
    with pytest.raises(TrackingGateError, match="global_step"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_sidecar_requires_tracking_files_outside_scientific_root(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    inside = TrackioSidecarConfig(
        **{
            **config.__dict__,
            "trackio_dir": run_root / "trackio",
        }
    )
    with pytest.raises(TrackingGateError, match="outside"):
        run_trackio_sidecar(
            inside,
            trackio_module=_FakeTrackio(_FakeHub()),
            hub_module=_FakeHub(),
        )


def test_sidecar_rejects_colliding_tracking_paths(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    collision = TrackioSidecarConfig(
        **{
            **config.__dict__,
            "receipt_path": config.state_path,
        }
    )
    with pytest.raises(TrackingGateError, match="must be distinct"):
        run_trackio_sidecar(
            collision,
            trackio_module=_FakeTrackio(_FakeHub()),
            hub_module=_FakeHub(),
        )


def test_sidecar_rejects_seed_mismatch(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    wrong_seed = TrackioSidecarConfig(**{**config.__dict__, "expected_seed": 2})
    with pytest.raises(TrackingGateError, match="run seed"):
        run_trackio_sidecar(
            wrong_seed,
            trackio_module=_FakeTrackio(_FakeHub()),
            hub_module=_FakeHub(),
        )


def test_sidecar_rejects_validation_monitor_mismatch(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    wrong_monitor = TrackioSidecarConfig(
        **{**config.__dict__, "expected_validation_monitor": "val/other_metric"}
    )
    with pytest.raises(TrackingGateError, match="validation_monitor"):
        run_trackio_sidecar(
            wrong_monitor,
            trackio_module=_FakeTrackio(_FakeHub()),
            hub_module=_FakeHub(),
        )


def test_sidecar_rejects_preimported_trackio_with_wrong_store(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub()
    trackio = _FakeTrackio(hub)
    trackio.utils = SimpleNamespace(TRACKIO_DIR=tmp_path / "wrong-store")
    with pytest.raises(TrackingGateError, match="imported before TRACKIO_DIR"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=trackio,
            hub_module=hub,
        )


def test_sidecar_binds_store_before_optional_trackio_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    config = _config(tmp_path, run_root)
    hub = _FakeHub()
    trackio = _FakeTrackio(hub)

    def import_after_binding(name: str) -> Any:
        assert name == "trackio"
        bound = Path(os.environ["TRACKIO_DIR"])
        assert bound == config.trackio_dir.resolve()
        trackio.utils = SimpleNamespace(TRACKIO_DIR=bound)
        return trackio

    monkeypatch.setattr(sidecar_module, "_import_optional", import_after_binding)
    receipt = run_trackio_sidecar(config, hub_module=hub)
    assert receipt["status"] == "local_client_complete"


def test_sidecar_requires_one_visible_logical_gpu_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    hub = _FakeHub()
    with pytest.raises(TrackingGateError, match="exactly one visible"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_sidecar_rejects_non_decimal_visible_gpu_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-01234567")
    hub = _FakeHub()
    with pytest.raises(TrackingGateError, match="decimal physical index"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_finish_failure_is_not_retried(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub()
    trackio = _RaisingFinishTrackio(hub)
    with pytest.raises(RuntimeError, match="finish-error"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=trackio,
            hub_module=hub,
        )
    assert trackio.finish_calls == 1


def test_cleanup_failure_does_not_mask_primary_error(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    hub = _FakeHub()
    trackio = _RaisingFinishNoMountTrackio(hub)
    with pytest.raises(TrackingGateError, match="not mounted"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=trackio,
            hub_module=hub,
        )


def test_sidecar_requires_pretest_training_receipt(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    (run_root / "small_results" / "training_receipt.json").unlink()
    hub = _FakeHub()
    with pytest.raises(TrackingGateError, match="training receipt"):
        run_trackio_sidecar(
            _config(tmp_path, run_root),
            trackio_module=_FakeTrackio(hub),
            hub_module=hub,
        )


def test_trackio_is_absent_from_training_and_performance_processes() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden_roots = (
        root / "src/gradpert/training",
        root / "src/gradpert/execution/native.py",
        root / "scripts/performance",
    )
    for forbidden_root in forbidden_roots:
        paths = (
            [forbidden_root] if forbidden_root.is_file() else sorted(forbidden_root.glob("*.py"))
        )
        for path in paths:
            assert "trackio" not in path.read_text(encoding="utf-8").lower(), path


def test_wrapper_has_no_token_argument() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts/tracking/sync_trackio.py").read_text(encoding="utf-8")
    assert "--token" not in wrapper
    assert "hf_token" not in wrapper.lower()
