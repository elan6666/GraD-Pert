"""Seal the original soft-rank/positive-cosine pair; reuse historical RI controls."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import representation_family as sealed
from generate_representation_round1 import configs as first_configs
from generate_representation_round1 import metadata as first_metadata

SCHEMA = "representation-round12-soft-rank-medium-1"
FAMILY = "representation_round12_20261025"
REFERENCE = "docs/design/SOFT_RANK_PAIR_RESEARCH.md"
ARMS = ("R12-SRBAL", "R12-SRAGG")
STRENGTHS = {
    "R12-SRBAL": dict(target=0.20, rank_weight=0.02, cosine_weight=0.10),
    "R12-SRAGG": dict(target=0.35, rank_weight=0.04, cosine_weight=0.20),
}
CHANGES = {
    arm: dict(
        spectral_rank_target=strength["target"],
        spectral_rank_weight=strength["rank_weight"],
        spectral_cosine_weight=strength["cosine_weight"],
        spectral_softness=0.25,
        spectral_warmup_steps=30,
        au_alignment_weight=0.0,
        au_uniformity_weight=0.0,
        cr_weight=0.0,
    )
    for arm, strength in STRENGTHS.items()
}


def configs(source: Path) -> dict:
    anchor = first_configs(source)["RS-RI"]
    values = {}
    for arm, changes in CHANGES.items():
        value = copy.deepcopy(anchor)
        for key, setting in changes.items():
            value["model"]["parameters"][key] = dict(
                value=setting, source="project_preregistered", reference=REFERENCE
            )
        values[arm] = value
    return values


def metadata() -> dict:
    value = first_metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        initialization="from_scratch; original RI architecture and seed1; no checkpoint warm start",
        test_timing="after_research_design_lock",
        reused_controls=["R3-RI", "RS-B0", "RS-RI"],
        main_reference="R3-RI",
        repeat_baseline=False,
        scientific_complete=False,
        completed_optimizer_updates=300,
    )
    value["schedule_semantics"].update(
        comparison="original RI two-epoch300-update budget; historical cross-version single seed",
        auxiliary_warmup="min(1, completed_successful_updates/30); first update zero",
    )
    value["research_iteration"].update(
        round=12,
        authorized_rounds=None,
        stage_review_after_round=12,
        hypothesis="soft spectral floor and positive cosine preserve diversity and view stability",
        future_rounds="evidence-led; no automatic extra arms or baseline rerun",
        original_design="archived round3-design-20261022/DESIGN_V2_CN.md",
    )
    value["spectral_pair"] = dict(
        representation="two Student Global Response CLS after final RMSNorm before projector",
        population="complete unique effective population across devices and microbatches",
        covariance="raw centered per Global; no row L2 or diagonal jitter",
        rho="exp(spectral_entropy)/min(D,N-1)",
        rank="mean_views(.25*softplus((log(target)-log(rho))/.25))",
        cosine="mean_same_input(1-cos(Global0,Global1)); both Student branches differentiate",
        strengths=STRENGTHS,
        negative_pairs=False,
        partial_diagnostics="NA: complete effective population required",
    )
    value["calibration_required"] = dict(
        endpoints=["init", "ri_last"],
        training_populations=8,
        effective_batch=320,
        updates=0,
        same_ordered_ids_and_rng=True,
        losses=["joint", "MSE", "rank_bal", "rank_agg", "cosine"],
        groups=["response", "graph"],
        decision="retain original coefficients unless an explicit versioned evidence revision",
    )
    value["comparison_limits"] = [
        "R3-RI is historical and cross-version, not an identical-trajectory replay",
        "RS-B0 and RS-RI provide historical context only; no baseline reruns",
        "single seed; no validation/test calibration; tests deferred",
        "rank and cosine changed together; individual contributions are not identified",
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
