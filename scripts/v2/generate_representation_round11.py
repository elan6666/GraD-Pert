"""Seal frozen R9 prediction MLP adaptation with an explicit extra budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round9 as previous
import representation_family as sealed

SCHEMA = "representation-round11-frozen-readout-medium-1"
FAMILY = "representation_round11_20261025"
REFERENCE = "docs/design/FROZEN_READOUT_RESEARCH.md"
ARMS = ("R11-READOUT",)
PARENT_RUN = "nadig_jurkat-seed1-20261024T160937Z-352f0995e12243be9f8be24f1f95defa"
PARENT_ROOT = (
    "/data/yilangliu/GraD-Pert/development/representation-r9-staged-joint-4f04c5f-20261025-a2/runs/"
    + PARENT_RUN
)
PARENT_TRAINING_SOURCE = "4f04c5f3a6a5d735b5337f3bd1ccc9337b155246"
PARENT_CHECKPOINT_SHA256 = "33c9a2fa27d9aaea96e667ee7d3542e8d923812b45f4bd3cf256e0e8d7ece595"
CHANGES = dict(
    training_mode="frozen_readout",
    ssl_pretrain_epochs=0,
    lambda1=0.0,
    lambda2=0.0,
    au_alignment_weight=0.0,
    au_uniformity_weight=0.0,
    cr_weight=0.0,
    readout_parent_run_root=PARENT_ROOT,
    readout_parent_run_id=PARENT_RUN,
    readout_parent_source_commit=PARENT_TRAINING_SOURCE,
    readout_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
    readout_parent_epoch=3,
)


def configs(source: Path) -> dict:
    value = previous.configs(source)["R9-SSL1J2"]
    value["training"]["formal_run_policy"] = "v2_fixed_2"
    value["training"]["max_epochs"] = dict(value=2, source="user_locked", reference=REFERENCE)
    for key, setting in CHANGES.items():
        value["model"]["parameters"][key] = dict(
            value=setting, source="project_preregistered", reference=REFERENCE
        )
    return {ARMS[0]: value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        epochs=2,
        reused_controls=["R3-RI", "R5-PROJ", "R6-EMA99", "R9-SSL1J2"],
        scientific_complete=False,
    )
    value["initialization"] = (
        "exact R9 epoch3 all-state warm start; original prediction MLP; fresh child RNG/optimizer"
    )
    value.pop("coding_rate", None)
    value["research_iteration"]["success"] = (
        "final validation prediction MSE and full18; frozen representation preserved; "
        "no new SSL learning"
    )
    value.pop("staged_joint", None)
    value.pop("diagnostic_endpoints", None)
    value["research_iteration"].update(
        round=11,
        stage_review_after_round=11,
        hypothesis="frozen R9 features may retain prediction MLP adaptation capacity",
    )
    value["schedule_semantics"].update(
        comparison=(
            "parent450 total/300 MSE + child300 head MSE =750 total/600 MSE; unmatched extra budget"
        ),
        sampler_epochs=[0, 1],
        teacher_centers_optimizer="Teacher/centers/backbone frozen; fresh native head optimizer",
        view_rng="fresh seeded child RNG; no original replay claim",
        ema="inactive",
        auxiliary_warmup="inactive",
        lr="original two-epoch300-update endpoint cosine",
    )
    value["coefficient_decision"] = dict(
        weights=CHANGES,
        parent_arm="R9-SSL1J2",
        intervention="only existing prediction MLP trainable; no SSL/A/U/CR/EMA",
    )
    value["frozen_readout"] = dict(
        parent_run_root=PARENT_ROOT,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        parent_source_commit=PARENT_TRAINING_SOURCE,
        parent_terminal="TRAIN_COMPLETE only; tests deferred",
        parent_evidence_sha256={
            "fit/epoch_state.json": (
                "ea9a9ca9593ef81b707638969a4dece8720e4750a9033a2988f86dce670241da"
            ),
            "TRAIN_COMPLETE.json": (
                "b9f5d5c44960d2b293398d611991f0a98739397231ced4e08e2bbb775dc9aa36"
            ),
            "resolved_config.json": (
                "0ff2c230123622e1ea404fd5388f91af2360f01207fe96c5018a064c53635f85"
            ),
            "run_manifest.json": "7670e88549a98ab6285668f48b73126443f643a281fb14f408c571c5295cab4f",
        },
        parent_resolved_config_canonical_sha256="5d7534762fcb0c8682e0a9e15d432dc1c101cfffb5b04b618634f83762ad7cf2",
        child_epochs=2,
        child_head_updates=300,
        total_updates=750,
        total_mse_updates=600,
        total_budget_matched=False,
        teacher_updates=0,
        center_updates=0,
        prediction_parameters=131585,
        prediction_tensors=4,
        teacher_geometry="unchanged R9 reference requires frozen digest proof",
        final_validation_conditions=445,
        final_validation_metrics=18,
    )
    value["teacher_tracking"].update(
        implementation="native frozen readout; zero Teacher/center commits",
        native_source_baseline_commit="c4c825a20cf3e71ca878079235e337374f9d5f0a",
        training_views="backbone eval; only native prediction MLP trains",
        center_temperature_prototypes="frozen R9; inactive SSL",
        limits=[
            "existing nonlinear head; not standard DINO linear evaluation",
            "extra head-update budget; not matched compute",
            "one seed; test deferred",
        ],
    )
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
