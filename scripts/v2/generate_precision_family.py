"""Seal three precision-only two-epoch candidates against the retained R3-RI."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import representation_family as sealed
import yaml
from generate_representation_round1 import metadata as first_metadata

from gradpert.hashing import sha256_file

SCHEMA = "representation-precision-family-1"
FAMILY = "precision_ri_20261031"
BASELINE = "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
BASELINE_SHA256 = "65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7"
REFERENCE = "docs/experiments/PRECISION_RI_20261031.md"
CHANGES = {
    "PREC-SSL32": {"ssl_head_precision": "float32"},
    "PREC-PRED32": {"prediction_precision": "float32"},
    "PREC-FFN8": {"ffn_compute_precision": "fp8_hybrid"},
}


def configs(source: Path) -> dict:
    path = source / BASELINE
    if sha256_file(path) != BASELINE_SHA256:
        raise ValueError("retained R3-RI configuration changed")
    anchor = yaml.safe_load(path.read_text())
    result = {}
    for arm, changes in CHANGES.items():
        config = copy.deepcopy(anchor)
        for key, value in changes.items():
            config["model"]["parameters"][key] = dict(
                value=value, source="project_preregistered", reference=REFERENCE
            )
        result[arm] = config
    return result


def metadata() -> dict:
    value = first_metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(CHANGES),
        preflight_priority=list(CHANGES),
        baseline=dict(
            experiment="R3-RI",
            config=BASELINE,
            sha256=BASELINE_SHA256,
            training_source_commit="f0cb20ebb2efab749970f2bf5d205ed2f358592c",
            checkpoint_sha256="96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4",
        ),
        main_reference="R3-RI",
        reused_controls=["R3-RI"],
        repeat_baseline=False,
        initialization="from_scratch; common RI initialization/seed; no checkpoint warm start",
        completed_optimizer_updates=300,
        test_timing="deferred; no test-based precision selection",
        scientific_complete=False,
    )
    value["research_iteration"].update(
        round="precision",
        authorized_rounds=None,
        hypothesis="separate sensitive head precision from FFN throughput without changing losses",
        future_rounds="analyze these three candidates before any combination or additional arm",
    )
    value["precision"] = dict(
        changes=CHANGES,
        fp8_recipe="stateless per-tensor current scaling; E4M3 forward / E5M2 output gradient",
        fp8_backend="native torch._scaled_mm; no quantize/dequantize BF16 simulation or fallback",
        ds_v3_alignment="principle only; not the all-E4M3 1x128/128x128 DeepSeek-V3 recipe",
        master_parameters="unchanged FP32",
        gradient_storage="unchanged FP32",
        optimizer="unchanged Muon/AdamW routes and state precision",
        teacher="same module precision policy as Student, no teacher backward",
        kda_state="unchanged FP32",
        inactive_heads="SSL1 node remains inactive",
        primary_evaluation=(
            "common_ri_native: original unautocast RI validation; "
            "training-precision sensitivity separate"
        ),
        analysis="same unified medium/crossed template and fixed population manifests",
        preflight=(
            "10 native updates/fullrestore/300-control inference plus precision dtype/kernel audit"
        ),
        fp8_capacity="must retain micro80/global320; unsupported backend or shape fails closed",
    )
    value["comparison_limits"] = [
        "retained R3-RI is historical; prove source parity or label cross-version descriptive",
        "single seed and exploratory two epochs; no causal claim from loss or rank alone",
        "FP8 FLOPs/peak conventions differ; measure complete-update throughput directly",
    ]
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    return sealed.verify(source, path, metadata(), configs(source), CHANGES)


def generate(source: Path, output: Path) -> dict:
    return sealed.generate(source, output, metadata(), configs(source), CHANGES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2") / FAMILY)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
