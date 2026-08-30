#!/usr/bin/env python3
"""Stream one formal GraD-Pert run's safe scalar receipts to private Trackio."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradpert.tracking import TrackioSidecarConfig, run_trackio_sidecar  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trackio-dir", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--project", default="grad-pert-vnext-ablations")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--space-id",
        default=os.environ.get("GRADPERT_TRACKIO_SPACE_ID"),
        required=os.environ.get("GRADPERT_TRACKIO_SPACE_ID") is None,
    )
    parser.add_argument(
        "--bucket-id",
        default=os.environ.get("GRADPERT_TRACKIO_BUCKET_ID"),
        required=os.environ.get("GRADPERT_TRACKIO_BUCKET_ID") is None,
    )
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-model-id", default="gradpert_b2")
    parser.add_argument("--expected-dataset-id", default="nadig_jurkat")
    parser.add_argument("--expected-protocol-id", default="within_cell_unseen_single")
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument(
        "--expected-validation-monitor",
        default="val/txpert_macro_pearson_delta",
    )
    parser.add_argument("--expected-optimizer-steps", type=int, default=5820)
    parser.add_argument("--expected-validations", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--system-log-interval", type=float, default=30.0)
    parser.add_argument("--no-follow", action="store_false", dest="follow")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--no-log-gpu", action="store_false", dest="log_gpu")
    parser.add_argument("--no-auto-log-cpu", action="store_false", dest="auto_log_cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = TrackioSidecarConfig(
        run_root=args.run_root,
        trackio_dir=args.trackio_dir,
        state_path=args.state_path,
        receipt_path=args.receipt_path,
        project=args.project,
        run_name=args.run_name,
        group=args.group,
        space_id=args.space_id,
        bucket_id=args.bucket_id,
        variant_id=args.variant_id,
        expected_run_id=args.expected_run_id,
        expected_source_commit=args.expected_source_commit,
        expected_config_sha256=args.expected_config_sha256,
        expected_model_id=args.expected_model_id,
        expected_dataset_id=args.expected_dataset_id,
        expected_protocol_id=args.expected_protocol_id,
        expected_seed=args.expected_seed,
        expected_validation_monitor=args.expected_validation_monitor,
        expected_optimizer_steps=args.expected_optimizer_steps,
        expected_validations=args.expected_validations,
        poll_seconds=args.poll_seconds,
        follow=args.follow,
        log_gpu=args.log_gpu,
        gpu_device=args.gpu_device,
        auto_log_cpu=args.auto_log_cpu,
        system_log_interval=args.system_log_interval,
    )
    run_trackio_sidecar(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
