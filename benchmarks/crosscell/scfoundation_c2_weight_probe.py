"""Verify the released scFoundation GEARS encoder on a synthetic server-side input."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.crosscell.remaining_c2_readiness import (
    FOUNDATION_WEIGHTS,
    FOUNDATION_WEIGHTS_SHA,
)
from benchmarks.crosscell.study_paths import SERVER, STUDY
from gradpert.data._io import atomic_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file
from scripts.v2.shared_gpu import process_info, snapshot

UPSTREAM = SERVER / "upstreams/scfoundation-c2-study"
COMMIT = "397631c495eddf9ad6644fc00c6ea8139e651245"
WEIGHTS_REPOSITORY = "genbio-ai/scFoundation@cb434153a1acfacd215eefc956ea445f7cc39cc3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.parent != STUDY
        or os.getenv("CUDA_VISIBLE_DEVICES") != "0"
        or os.getenv("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.getenv("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("scFoundation requires the admitted GPU 0 and a fresh server output")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
        or sha256_file(FOUNDATION_WEIGHTS) != FOUNDATION_WEIGHTS_SHA
    ):
        raise ValueError("the released scFoundation checkpoint identity or source changed")

    with lock(SERVER / "runtime/v2-gpu-0.lock"):
        admit_selected_idle(snapshot(os.getpid()), 0, args.gpu_uuid)
        output.mkdir()
        atomic_json(
            output / "worker.json",
            {
                "stage": "scfoundation_released_weight_probe",
                "owner": process_info(os.getpid()),
                "physical_gpu": 0,
            },
        )
        started = time.monotonic()
        os.chdir(UPSTREAM / "GEARS")
        sys.path.insert(0, str(UPSTREAM / "GEARS"))
        from modules.encoders import MAEAutobinencoder

        # Match the released GEARS caller's frozen expression encoder.
        cfg = {
            "load_path": str(FOUNDATION_WEIGHTS),
            "bin_set": "autobin_resolution_append",
            "highres": 0,
            "finetune_method": "frozen",
        }
        model = MAEAutobinencoder(cfg, hidden_size=512).to("cuda").eval()
        model.requires_grad_(False)
        values = torch.zeros((2, 19265), device="cuda", dtype=torch.float32)
        values[0, :256] = 1.0
        values[1, 512:896] = 1.5
        values[:, -1] = torch.tensor([1000.0, 1500.0], device="cuda")
        with torch.inference_mode():
            embedding = model(values)
        if list(embedding.shape) != [2, 19264, 512] or not torch.isfinite(embedding).all().item():
            raise ValueError("the released GEARS encoder produced invalid embeddings")
        atomic_json(
            output / "result.json",
            {
                "model": "scfoundation",
                "scenario": "C2",
                "status": "released_weight_and_encoder_smoke_passed",
                "upstream_commit": COMMIT,
                "source_commit": source_commit,
                "weights_mirror": WEIGHTS_REPOSITORY,
                "weights_sha256": FOUNDATION_WEIGHTS_SHA,
                "input": "synthetic non-test inputs and synthetic library depths",
                "shape": list(embedding.shape),
                "gpu": 0,
                "elapsed_seconds": time.monotonic() - started,
                "test_conditions_accessed": 0,
                "optimizer_updates": 0,
                "formal_eligible": False,
            },
        )


if __name__ == "__main__":
    main()
