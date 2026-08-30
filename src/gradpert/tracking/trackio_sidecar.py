"""Private Trackio telemetry sourced only from native scalar training receipts.

This module deliberately stays outside the training process.  It tails the
append-only scalar CSV receipts that native training already produces and sends
an explicit allowlist to a private Hugging Face Trackio Space.  It never opens
evaluation summaries, predictions, checkpoints, H5AD files, or PKL artifacts.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import importlib
import io
import json
import math
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from gradpert.hashing import sha256_file

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SPACE_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_FORMAL_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9._/+:-]+$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TRAIN_FLOAT_FIELDS: dict[str, str] = {
    "total_loss": "train/total_loss",
    "prediction_loss": "train/prediction_loss",
    "condition_consistency_loss": "train/condition_consistency_loss",
    "masked_node_loss": "train/masked_node_loss",
    "spread_loss": "train/spread_loss",
    "teacher_momentum": "distillation/teacher_momentum",
    "prediction_graph_gradient_norm": "gradients/prediction_graph_norm",
    "auxiliary_graph_gradient_norm": "gradients/auxiliary_graph_norm",
    "condition_target_entropy": "distillation/condition_target_entropy",
    "condition_center_norm": "distillation/condition_center_norm",
    "data_read_ms": "performance/data_read_ms",
    "host_to_device_ms": "performance/host_to_device_ms",
    "view_build_ms": "performance/view_build_ms",
    "teacher_forward_ms": "performance/teacher_forward_ms",
    "student_global_ms": "performance/student_global_ms",
    "student_local_ms": "performance/student_local_ms",
    "prediction_ms": "performance/prediction_ms",
    "backward_update_ms": "performance/backward_update_ms",
    "step_wall_ms": "performance/step_wall_ms",
}

_TRAIN_OPTIONAL_FLOAT_FIELDS: dict[str, str] = {
    "prediction_to_auxiliary_gradient_ratio": "gradients/prediction_to_auxiliary_ratio",
    "masked_node_target_entropy": "distillation/masked_node_target_entropy",
    "masked_node_center_norm": "distillation/masked_node_center_norm",
}

_TRAIN_INTEGER_FIELDS: dict[str, str] = {
    "condition_prototypes_used": "distillation/condition_prototypes_used",
    "masked_node_prototypes_used": "distillation/masked_node_prototypes_used",
    "unique_condition_count": "batch/unique_condition_count",
    "masked_node_count": "batch/masked_node_count",
    "batch_cell_count": "batch/cell_count",
    "local_view_realization_count": "local/view_count",
    "local_node_count_sum": "local/node_count_sum",
    "local_node_count_min": "local/node_count_min",
    "local_node_count_max": "local/node_count_max",
    "local_budget_hit_count": "local/budget_hit_count",
    "masked_local_assignment_count": "local/masked_assignment_count",
}

_TRAIN_BOOLEAN_FIELDS: dict[str, str] = {
    "spread_available": "train/spread_available",
}

_SAFE_ARCHITECTURE_FIELDS = (
    "graph_axis_policy",
    "graph_hvg_count",
    "graph_encoder_family",
    "graph_sources",
    "local_view_builder",
    "local_view_count",
    "local_view_node_budget_ratio_numerator",
    "local_view_node_budget_ratio_denominator",
    "local_anchor_mask_view_ratio_numerator",
    "local_anchor_mask_view_ratio_denominator",
    "gene_feature_mode",
    "decoder_mode",
)


class TrackingGateError(RuntimeError):
    """A privacy, identity, lifecycle, or receipt predicate rejected tracking."""


class TransientReceiptSnapshot(RuntimeError):
    """A concurrently appended CSV changed while the sidecar was reading it."""


@dataclass(frozen=True)
class TrackioSidecarConfig:
    run_root: Path
    trackio_dir: Path
    state_path: Path
    receipt_path: Path
    project: str
    run_name: str
    group: str
    space_id: str
    bucket_id: str
    variant_id: str
    expected_run_id: str
    expected_source_commit: str
    expected_config_sha256: str
    expected_model_id: str
    expected_dataset_id: str
    expected_protocol_id: str
    expected_seed: int
    expected_optimizer_steps: int
    expected_validations: int
    expected_validation_monitor: str = "val/txpert_macro_pearson_delta"
    poll_seconds: float = 30.0
    follow: bool = True
    log_gpu: bool = True
    gpu_device: int = 0
    auto_log_cpu: bool = True
    system_log_interval: float = 30.0

    def __post_init__(self) -> None:
        for label, value in (
            ("project", self.project),
            ("run_name", self.run_name),
            ("group", self.group),
            ("variant_id", self.variant_id),
            ("expected_source_commit", self.expected_source_commit),
            ("expected_model_id", self.expected_model_id),
            ("expected_dataset_id", self.expected_dataset_id),
            ("expected_protocol_id", self.expected_protocol_id),
        ):
            if not _SAFE_NAME.fullmatch(value):
                raise ValueError(f"{label} must match {_SAFE_NAME.pattern}")
        if not _SPACE_ID.fullmatch(self.space_id):
            raise ValueError("space_id must be an explicit owner/name")
        if not _SPACE_ID.fullmatch(self.bucket_id):
            raise ValueError("bucket_id must be an explicit owner/name")
        if not _FORMAL_RUN_ID.fullmatch(self.expected_run_id):
            raise ValueError("expected_run_id must be a slash-delimited formal run ID")
        if not _GIT_COMMIT.fullmatch(self.expected_source_commit):
            raise ValueError("expected_source_commit must be a full lowercase Git commit")
        if not _SHA256.fullmatch(self.expected_config_sha256):
            raise ValueError("expected_config_sha256 must be a lowercase SHA-256")
        if not _SAFE_METADATA_VALUE.fullmatch(self.expected_validation_monitor):
            raise ValueError("expected_validation_monitor contains unsafe characters")
        if self.expected_optimizer_steps <= 0:
            raise ValueError("expected_optimizer_steps must be positive")
        if self.expected_validations <= 0:
            raise ValueError("expected_validations must be positive")
        if self.expected_seed < 0:
            raise ValueError("expected_seed must be non-negative")
        if self.poll_seconds < 5:
            raise ValueError("poll_seconds must be at least 5 seconds")
        if self.system_log_interval < 10:
            raise ValueError("system_log_interval must be at least 10 seconds")
        if self.gpu_device < 0:
            raise ValueError("gpu_device must be non-negative")


@dataclass(frozen=True)
class _MetricEvent:
    source: str
    source_index: int
    trackio_step: int
    metrics: dict[str, int | float]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _require_separate_tracking_paths(config: TrackioSidecarConfig) -> None:
    run_root = config.run_root.resolve(strict=True)
    if not run_root.is_dir() or config.run_root.is_symlink():
        raise TrackingGateError("run_root must be a real directory")
    for label, path in (
        ("trackio_dir", config.trackio_dir),
        ("state_path", config.state_path),
        ("receipt_path", config.receipt_path),
    ):
        if _is_within(path, run_root):
            raise TrackingGateError(f"{label} must stay outside the scientific run root")
        if path.exists() and path.is_symlink():
            raise TrackingGateError(f"{label} must not be a symlink")
    resolved = {
        config.trackio_dir.resolve(strict=False),
        config.state_path.resolve(strict=False),
        config.receipt_path.resolve(strict=False),
    }
    if len(resolved) != 3:
        raise TrackingGateError("tracking store, state and receipt paths must be distinct")
    if _is_within(config.state_path, config.trackio_dir) or _is_within(
        config.receipt_path, config.trackio_dir
    ):
        raise TrackingGateError("tracking state/receipt must stay outside the Trackio store")
    if config.state_path.exists() or config.receipt_path.exists():
        raise TrackingGateError("tracking state/receipt already exists; use a fresh lineage")


def _stable_text(path: Path) -> str:
    try:
        before = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise TrackingGateError(f"tracking input must be a regular file: {path.name}")
    text = path.read_text(encoding="utf-8")
    after = path.stat(follow_symlinks=False)
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise TransientReceiptSnapshot(f"receipt changed during read: {path.name}")
    return text


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = _stable_text(path)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TrackingGateError(f"CSV has no header: {path.name}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise TransientReceiptSnapshot(f"CSV has an incomplete appended row: {path.name}")
    return list(reader.fieldnames), rows


def _finite_float(value: str, *, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise TrackingGateError(f"invalid float in scalar receipt field {field}") from error
    if not math.isfinite(parsed):
        raise TrackingGateError(f"non-finite scalar receipt field {field}")
    return parsed


def _integer(value: str, *, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise TrackingGateError(f"invalid integer in scalar receipt field {field}") from error


def _boolean(value: str, *, field: str) -> int:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1
    if normalized == "false":
        return 0
    raise TrackingGateError(f"invalid boolean in scalar receipt field {field}")


def _train_events(path: Path, *, steps_per_epoch: int) -> list[_MetricEvent]:
    fieldnames, rows = _read_csv(path)
    required = {
        "epoch",
        "global_step",
        *_TRAIN_FLOAT_FIELDS,
        *_TRAIN_OPTIONAL_FLOAT_FIELDS,
        *_TRAIN_INTEGER_FIELDS,
        *_TRAIN_BOOLEAN_FIELDS,
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise TrackingGateError("train_steps.csv lacks scalar fields: " + ", ".join(missing))
    events: list[_MetricEvent] = []
    for index, row in enumerate(rows):
        global_step = _integer(row["global_step"], field="global_step")
        epoch = _integer(row["epoch"], field="epoch")
        if global_step != index:
            raise TrackingGateError("train global_step values are not contiguous from zero")
        if epoch != global_step // steps_per_epoch:
            raise TrackingGateError("train epoch/global_step relation differs from run metadata")
        metrics: dict[str, int | float] = {
            "progress/epoch_index": epoch,
            "progress/optimizer_step": global_step + 1,
        }
        for field, key in _TRAIN_FLOAT_FIELDS.items():
            metrics[key] = _finite_float(row[field], field=field)
        for field, key in _TRAIN_OPTIONAL_FLOAT_FIELDS.items():
            if row[field].strip() not in {"", "None", "null"}:
                metrics[key] = _finite_float(row[field], field=field)
        for field, key in _TRAIN_INTEGER_FIELDS.items():
            metrics[key] = _integer(row[field], field=field)
        for field, key in _TRAIN_BOOLEAN_FIELDS.items():
            metrics[key] = _boolean(row[field], field=field)
        end_to_end_ms = (
            float(metrics["performance/data_read_ms"])
            + float(metrics["performance/host_to_device_ms"])
            + float(metrics["performance/step_wall_ms"])
        )
        if end_to_end_ms <= 0:
            raise TrackingGateError("end-to-end step time must be positive")
        metrics["performance/end_to_end_ms"] = end_to_end_ms
        metrics["performance/steps_per_second"] = 1000.0 / end_to_end_ms
        metrics["performance/cells_per_second"] = (
            int(metrics["batch/cell_count"]) * 1000.0 / end_to_end_ms
        )
        events.append(
            _MetricEvent(
                source="train",
                source_index=index,
                trackio_step=global_step + 1,
                metrics=metrics,
            )
        )
    return events


def _validation_events(path: Path, *, steps_per_epoch: int) -> list[_MetricEvent]:
    fieldnames, rows = _read_csv(path)
    required = {
        "epoch",
        "global_step",
        "val_txpert_macro_pearson_delta",
        "improved",
        "consecutive_non_improvements",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise TrackingGateError("validation.csv lacks scalar fields: " + ", ".join(missing))
    events: list[_MetricEvent] = []
    for index, row in enumerate(rows):
        epoch = _integer(row["epoch"], field="epoch")
        global_step = _integer(row["global_step"], field="global_step")
        if epoch != index:
            raise TrackingGateError("validation epoch values are not contiguous from zero")
        if global_step != (epoch + 1) * steps_per_epoch:
            raise TrackingGateError("validation global_step differs from the epoch boundary")
        events.append(
            _MetricEvent(
                source="validation",
                source_index=index,
                trackio_step=global_step,
                metrics={
                    "progress/epoch_index": epoch,
                    "progress/completed_epochs": epoch + 1,
                    "progress/optimizer_step": global_step,
                    "validation/txpert_macro_pearson_delta": _finite_float(
                        row["val_txpert_macro_pearson_delta"],
                        field="val_txpert_macro_pearson_delta",
                    ),
                    "validation/improved": _boolean(row["improved"], field="improved"),
                    "validation/consecutive_non_improvements": _integer(
                        row["consecutive_non_improvements"],
                        field="consecutive_non_improvements",
                    ),
                },
            )
        )
    return events


def _scalar(value: object, *, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        if not _SAFE_METADATA_VALUE.fullmatch(value):
            raise TrackingGateError(f"unsafe string run metadata field: {field}")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TrackingGateError(f"unsafe non-scalar run metadata field: {field}")


def _safe_run_config(
    run_meta: Mapping[str, Any],
    config: TrackioSidecarConfig,
) -> tuple[int, dict[str, str | int | float | bool | None]]:
    if run_meta.get("schema_version") != "native-run-meta-v1":
        raise TrackingGateError("run metadata schema differs from native-run-meta-v1")
    exact = {
        "run_id": config.expected_run_id,
        "model_id": config.expected_model_id,
        "dataset_id": config.expected_dataset_id,
        "protocol_id": config.expected_protocol_id,
        "config_sha256": config.expected_config_sha256,
        "validation_monitor": config.expected_validation_monitor,
    }
    for field, expected in exact.items():
        if run_meta.get(field) != expected:
            raise TrackingGateError(f"run metadata {field} differs from tracking contract")
    if run_meta.get("mode") != "pilot":
        raise TrackingGateError("Trackio sidecar accepts formal pilot runs only")
    source = run_meta.get("source")
    if not isinstance(source, Mapping):
        raise TrackingGateError("run metadata lacks source identity")
    if source.get("formal_eligible") is not True or source.get("dirty") is not False:
        raise TrackingGateError("Trackio sidecar rejects non-formal or dirty source runs")
    if source.get("commit") != config.expected_source_commit:
        raise TrackingGateError("run source commit differs from tracking contract")
    if run_meta.get("run_seed") != config.expected_seed:
        raise TrackingGateError("run seed differs from tracking contract")
    max_epochs = run_meta.get("max_epochs")
    steps_per_epoch = run_meta.get("steps_per_epoch")
    if not isinstance(max_epochs, int) or not isinstance(steps_per_epoch, int):
        raise TrackingGateError("run metadata lacks integer epoch/step counts")
    if max_epochs != config.expected_validations:
        raise TrackingGateError("formal epoch count differs from expected validations")
    if steps_per_epoch * max_epochs != config.expected_optimizer_steps:
        raise TrackingGateError("formal optimizer-step count differs from tracking contract")
    architecture = run_meta.get("native_architecture")
    if not isinstance(architecture, Mapping):
        raise TrackingGateError("run metadata lacks native architecture")
    safe: dict[str, str | int | float | bool | None] = {
        "tracking_schema": "gradpert-trackio-sidecar-v1",
        "tracking_purpose": "formal_ablation_curves_only",
        "test_metrics_uploaded": False,
        "artifacts_uploaded": False,
        "variant_id": config.variant_id,
        "run_id": config.expected_run_id,
        "model_id": config.expected_model_id,
        "dataset_id": config.expected_dataset_id,
        "protocol_id": _scalar(run_meta.get("protocol_id"), field="protocol_id"),
        "run_seed": _scalar(run_meta.get("run_seed"), field="run_seed"),
        "source_commit": config.expected_source_commit,
        "max_epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": config.expected_optimizer_steps,
        "validation_monitor": config.expected_validation_monitor,
    }
    for field in _SAFE_ARCHITECTURE_FIELDS:
        value = architecture.get(field)
        if field == "graph_sources":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise TrackingGateError("native architecture graph_sources is invalid")
            items = [_scalar(item, field=f"{field}[]") for item in value]
            safe[f"architecture/{field}"] = "+".join(str(item) for item in items)
        else:
            safe[f"architecture/{field}"] = _scalar(value, field=field)
    return steps_per_epoch, safe


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(_stable_text(path))
    if not isinstance(payload, Mapping):
        raise TrackingGateError(f"JSON input must be an object: {path.name}")
    return payload


def _hub_api(hub: ModuleType) -> Any:
    token = hub.get_token()
    if not token:
        raise TrackingGateError(
            "Hugging Face authentication is missing; run `hf auth login` privately"
        )
    return hub.HfApi(token=token)


def _require_private_hugging_face_space(hub: ModuleType, api: Any, space_id: str) -> bool:
    try:
        info = api.space_info(repo_id=space_id)
    except hub.utils.RepositoryNotFoundError:
        return False
    if getattr(info, "private", None) is not True:
        raise TrackingGateError("refusing to send experiment metrics to a non-private Space")
    return True


def _require_space_bucket_mount(
    api: Any,
    space_id: str,
    bucket_id: str,
    *,
    require_present: bool,
) -> bool:
    runtime = api.get_space_runtime(repo_id=space_id)
    volumes = getattr(runtime, "volumes", None) or []
    mounted = [
        volume
        for volume in volumes
        if getattr(volume, "type", None) == "bucket"
        and getattr(volume, "mount_path", None) == "/data"
    ]
    if len(mounted) > 1:
        raise TrackingGateError("Hugging Face Space has multiple Bucket mounts at /data")
    if mounted and getattr(mounted[0], "source", None) != bucket_id:
        raise TrackingGateError("refusing to replace the Space's existing /data Bucket")
    present = bool(mounted)
    if require_present and not present:
        raise TrackingGateError("private Trackio Bucket is not mounted at Space /data")
    return present


def _require_private_hugging_face_bucket(
    hub: ModuleType,
    api: Any,
    bucket_id: str,
    *,
    create_if_missing: bool,
) -> bool:
    try:
        info = api.bucket_info(bucket_id=bucket_id)
    except hub.errors.BucketNotFoundError:
        if not create_if_missing:
            return False
        api.create_bucket(bucket_id=bucket_id, private=True, exist_ok=False)
        info = api.bucket_info(bucket_id=bucket_id)
    if getattr(info, "id", bucket_id) != bucket_id:
        raise TrackingGateError("Hugging Face Bucket identity differs from tracking contract")
    if getattr(info, "private", None) is not True:
        raise TrackingGateError("refusing to send experiment metrics to a non-private Bucket")
    return True


def _import_optional(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise TrackingGateError(
            "Trackio support is not installed; install the `tracking` project extra"
        ) from error


def _require_trackio_storage_dir(trackio: ModuleType | Any, expected: Path) -> None:
    utils = getattr(trackio, "utils", None)
    actual = getattr(utils, "TRACKIO_DIR", None)
    if actual is None:
        return
    if Path(actual).expanduser().resolve(strict=False) != expected.resolve(strict=False):
        raise TrackingGateError(
            "Trackio was imported before TRACKIO_DIR was bound to this fresh lineage"
        )


def _require_single_visible_gpu(config: TrackioSidecarConfig) -> str | None:
    if not config.log_gpu:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise TrackingGateError("CUDA_VISIBLE_DEVICES must bind the sidecar to exactly one GPU")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise TrackingGateError("Trackio sidecar requires exactly one visible CUDA device")
    if re.fullmatch(r"0|[1-9][0-9]*", devices[0]) is None:
        raise TrackingGateError(
            "CUDA_VISIBLE_DEVICES must use one non-negative decimal physical index"
        )
    if config.gpu_device != 0:
        raise TrackingGateError(
            "gpu_device is a logical CUDA index and must be 0 in one-GPU sidecars"
        )
    return devices[0]


def _wait_for_file(path: Path, *, follow: bool, poll_seconds: float) -> None:
    while not path.is_file():
        if not follow:
            raise TrackingGateError(f"required tracking input is absent: {path.name}")
        time.sleep(poll_seconds)


def _state_payload(
    config: TrackioSidecarConfig,
    *,
    status: str,
    last_train_global_step: int,
    last_validation_epoch: int,
    trackio_run_id: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "gradpert-trackio-sidecar-state-v1",
        "status": status,
        "project": config.project,
        "run_name": config.run_name,
        "group": config.group,
        "space_id": config.space_id,
        "bucket_id": config.bucket_id,
        "variant_id": config.variant_id,
        "run_id": config.expected_run_id,
        "last_train_global_step": last_train_global_step,
        "last_validation_epoch": last_validation_epoch,
        "trackio_run_id": trackio_run_id,
    }


def _run_trackio_sidecar(
    config: TrackioSidecarConfig,
    *,
    trackio_module: ModuleType | Any | None = None,
    hub_module: ModuleType | Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Tail one formal run into a fresh private Trackio lineage."""

    _require_separate_tracking_paths(config)
    small_root = config.run_root / "small_results"
    run_meta_path = small_root / "run_meta.json"
    train_path = small_root / "train_steps.csv"
    validation_path = small_root / "validation.csv"
    training_receipt_path = small_root / "training_receipt.json"
    _wait_for_file(run_meta_path, follow=config.follow, poll_seconds=config.poll_seconds)
    run_meta = _load_json(run_meta_path)
    steps_per_epoch, safe_config = _safe_run_config(run_meta, config)

    config.trackio_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    config.trackio_dir.chmod(0o700)
    if stat.S_IMODE(config.trackio_dir.stat(follow_symlinks=False).st_mode) != 0o700:
        raise TrackingGateError("Trackio storage directory is not owner-only")
    os.environ["TRACKIO_DIR"] = str(config.trackio_dir.resolve())
    os.environ.setdefault(
        "TRACKIO_PLOT_ORDER",
        "train/total_loss,train/*,validation/*,performance/*,gpu/*,cpu/*",
    )
    trackio = trackio_module if trackio_module is not None else _import_optional("trackio")
    _require_trackio_storage_dir(trackio, config.trackio_dir)
    hub = hub_module if hub_module is not None else _import_optional("huggingface_hub")
    api = _hub_api(hub)
    space_existed = _require_private_hugging_face_space(hub, api, config.space_id)
    if space_existed:
        _require_space_bucket_mount(
            api,
            config.space_id,
            config.bucket_id,
            require_present=False,
        )
    _require_private_hugging_face_bucket(
        hub,
        api,
        config.bucket_id,
        create_if_missing=True,
    )
    visible_gpu = _require_single_visible_gpu(config)
    display_name = f"{config.run_name}-{config.expected_source_commit[:7]}"
    run: Any | None = None
    finish_attempted = False
    try:
        run = trackio.init(
            project=config.project,
            name=display_name,
            group=config.group,
            space_id=config.space_id,
            bucket_id=config.bucket_id,
            config=safe_config,
            resume="never",
            private=True,
            embed=False,
            # Trackio 0.37's background monitor observes every physical GPU.  Log
            # one explicit logical device below to preserve the one-GPU scope.
            auto_log_gpu=False,
            gpu_log_interval=config.system_log_interval,
            auto_log_cpu=config.auto_log_cpu,
            cpu_log_interval=config.system_log_interval,
        )
        if not space_existed and not _require_private_hugging_face_space(hub, api, config.space_id):
            raise TrackingGateError("new Trackio Space was not verifiably created as private")
        _require_space_bucket_mount(
            api,
            config.space_id,
            config.bucket_id,
            require_present=True,
        )
        if not _require_private_hugging_face_bucket(
            hub,
            api,
            config.bucket_id,
            create_if_missing=False,
        ):
            raise TrackingGateError("Trackio Bucket disappeared after initialization")
        trackio_run_id = getattr(run, "id", None)
        if not isinstance(trackio_run_id, str) or not _SAFE_NAME.fullmatch(trackio_run_id):
            raise TrackingGateError("Trackio returned an invalid run identifier")

        last_train = -1
        last_validation = -1
        last_gpu_log = -math.inf
        gpu_samples_enqueued = 0
        _atomic_json(
            config.state_path,
            _state_payload(
                config,
                status="active",
                last_train_global_step=last_train,
                last_validation_epoch=last_validation,
                trackio_run_id=trackio_run_id,
            ),
        )
        while True:
            now = time.monotonic()
            if config.log_gpu and now - last_gpu_log >= config.system_log_interval:
                gpu_metrics = trackio.log_gpu(run=run, device=config.gpu_device)
                if not isinstance(gpu_metrics, Mapping) or not gpu_metrics:
                    raise TrackingGateError("Trackio returned no metrics for the selected GPU")
                gpu_samples_enqueued += 1
                last_gpu_log = now
            try:
                train_events = (
                    _train_events(train_path, steps_per_epoch=steps_per_epoch)
                    if train_path.is_file()
                    else []
                )
                validation_events = (
                    _validation_events(validation_path, steps_per_epoch=steps_per_epoch)
                    if validation_path.is_file()
                    else []
                )
            except TransientReceiptSnapshot:
                if not config.follow:
                    raise
                sleep(config.poll_seconds)
                continue
            if len(train_events) > config.expected_optimizer_steps:
                raise TrackingGateError("training receipt exceeds the formal optimizer-step budget")
            if len(validation_events) > config.expected_validations:
                raise TrackingGateError("validation receipt exceeds the formal epoch budget")
            pending = [event for event in train_events if event.source_index > last_train] + [
                event for event in validation_events if event.source_index > last_validation
            ]
            pending.sort(
                key=lambda event: (
                    event.trackio_step,
                    0 if event.source == "train" else 1,
                )
            )
            for event in pending:
                trackio.log(event.metrics, step=event.trackio_step)
                if event.source == "train":
                    last_train = event.source_index
                else:
                    last_validation = event.source_index
            if pending:
                _atomic_json(
                    config.state_path,
                    _state_payload(
                        config,
                        status="active",
                        last_train_global_step=last_train,
                        last_validation_epoch=last_validation,
                        trackio_run_id=trackio_run_id,
                    ),
                )
            complete = (
                last_train + 1 == config.expected_optimizer_steps
                and last_validation + 1 == config.expected_validations
            )
            if complete:
                if not training_receipt_path.is_file():
                    if not config.follow:
                        raise TrackingGateError("formal training receipt is not complete")
                    sleep(config.poll_seconds)
                    continue
                training_receipt = _load_json(training_receipt_path)
                expected_training_receipt = {
                    "schema_version": "native-training-receipt-v1",
                    "epochs_completed": config.expected_validations,
                    "optimizer_steps": config.expected_optimizer_steps,
                    "canonical_test_truth_present_during_fit": False,
                }
                if any(
                    training_receipt.get(field) != expected
                    for field, expected in expected_training_receipt.items()
                ):
                    raise TrackingGateError(
                        "training receipt does not prove the formal pre-test lifecycle"
                    )
                break
            if not config.follow:
                raise TrackingGateError("formal scalar receipts are not complete")
            sleep(config.poll_seconds)

        finish_attempted = True
        trackio.finish()
    finally:
        if run is not None and not finish_attempted:
            # Trackio delivery is explicitly best effort, but its monitor and
            # pending-buffer threads still need deterministic shutdown.  This
            # branch already has a primary body exception, so cleanup cannot
            # replace the fail-closed cause.
            with contextlib.suppress(Exception):
                trackio.finish()

    receipt: dict[str, object] = {
        "schema_version": "gradpert-trackio-sidecar-receipt-v1",
        "status": "local_client_complete",
        "project": config.project,
        "run_name": display_name,
        "group": config.group,
        "space_id": config.space_id,
        "bucket_id": config.bucket_id,
        "variant_id": config.variant_id,
        "run_id": config.expected_run_id,
        "trackio_run_id": trackio_run_id,
        "optimizer_steps_enqueued": last_train + 1,
        "validations_enqueued": last_validation + 1,
        "gpu_samples_enqueued": gpu_samples_enqueued,
        "visible_gpu_count": 1 if visible_gpu is not None else 0,
        "visible_cuda_device_selector": visible_gpu,
        "logical_gpu_device": config.gpu_device if visible_gpu is not None else None,
        "train_steps_csv_sha256": sha256_file(train_path),
        "validation_csv_sha256": sha256_file(validation_path),
        "metric_policy": "allowlisted_train_validation_scalars_only",
        "telemetry_authority": False,
        "live_dashboard_provisional": True,
        "remote_sync_verified": False,
        "test_metrics_uploaded": False,
        "artifacts_uploaded": False,
        "prediction_content_uploaded": False,
        "performance_timing_lineage": False,
        "trackio_config": asdict(config)
        | {
            "run_root": None,
            "trackio_dir": None,
            "state_path": None,
            "receipt_path": None,
        },
    }
    _atomic_json(config.receipt_path, receipt)
    _atomic_json(
        config.state_path,
        _state_payload(
            config,
            status="local_client_complete",
            last_train_global_step=last_train,
            last_validation_epoch=last_validation,
            trackio_run_id=trackio_run_id,
        ),
    )
    return receipt


def run_trackio_sidecar(
    config: TrackioSidecarConfig,
    *,
    trackio_module: ModuleType | Any | None = None,
    hub_module: ModuleType | Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run one fail-closed sidecar process under an exclusive lineage lock."""

    # Trackio owns SQLite/JSONL files below its private store.  Keep their
    # creation owner-only for the full sidecar lifetime, then restore the
    # process setting for in-process callers and tests.
    previous_umask = os.umask(0o077)
    lock_path = config.state_path.with_suffix(f"{config.state_path.suffix}.lock")
    descriptor: int | None = None
    previous_trackio_dir = os.environ.get("TRACKIO_DIR")
    previous_plot_order = os.environ.get("TRACKIO_PLOT_ORDER")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.is_symlink():
            raise TrackingGateError("tracking lock must not be a symlink")
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | no_follow, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrackingGateError("another Trackio sidecar owns this lineage") from error
        return _run_trackio_sidecar(
            config,
            trackio_module=trackio_module,
            hub_module=hub_module,
            sleep=sleep,
        )
    finally:
        if previous_trackio_dir is None:
            os.environ.pop("TRACKIO_DIR", None)
        else:
            os.environ["TRACKIO_DIR"] = previous_trackio_dir
        if previous_plot_order is None:
            os.environ.pop("TRACKIO_PLOT_ORDER", None)
        else:
            os.environ["TRACKIO_PLOT_ORDER"] = previous_plot_order
        try:
            if descriptor is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        finally:
            os.umask(previous_umask)
