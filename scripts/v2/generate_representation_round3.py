"""Seal calibrated alignment/uniformity strengths against an unchanged RI control."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import yaml
from generate_representation_round1 import configs as first_configs
from generate_representation_round1 import metadata as first_metadata
from report_au_calibration import load_endpoint as load_calibration_endpoint

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file, sha256_json

SCHEMA = "representation-round3-medium-1"
FAMILY = "representation_round3_20261023"
DECISION_SCHEMA = "representation-round3-au-decision-1"
CALIBRATION_SOURCE = "288fe94232611a32f86f7fa92539a6032c41026d"
CALIBRATION_CONFIG = "configs/v2/representation_round2_20261022/R2-RI/gradpert_v2/nadig_jurkat.yaml"
REFERENCE = "scripts/v2/generate_representation_round3.py#calibrated-au"
ARMS = ("R3-RI", "R3-AUB", "R3-AUS")
EVIDENCE_FILES = (
    "PLAN.json",
    "COMPLETE.json",
    "CENSUS.json",
    "BATCH_INDEX.json",
    *(f"batch-{i:02d}.json" for i in range(8)),
)


def _sha(value, length=64):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _contained_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError("calibration evidence must be a source-contained regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("calibration evidence escaped its sealed bundle")
    return resolved


def load_decision(source: Path, path: Path) -> dict:
    """Independently verify both eight-batch observation endpoints, without mutation."""
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("decision must remain source contained")
    decision = json.loads(path.read_text())
    required = {
        "schema",
        "calibration_source_commit",
        "balanced",
        "strong",
        "strength_multiplier",
        "rationale",
        "calibration",
    }
    if (
        not isinstance(decision, dict)
        or set(decision) != required
        or (
            decision["schema"] != DECISION_SCHEMA
            or decision["calibration_source_commit"] != CALIBRATION_SOURCE
            or not isinstance(decision["rationale"], str)
            or not decision["rationale"].strip()
        )
    ):
        raise ValueError("round3 requires an explicit calibrated coefficient decision")
    for name in ("balanced", "strong"):
        weights = decision[name]
        if not isinstance(weights, dict) or set(weights) != {
            "alignment_weight",
            "uniformity_weight",
        }:
            raise ValueError("both AU strengths require explicit alignment and uniformity weights")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in weights.values()
        ):
            raise ValueError("AU coefficients must be finite and strictly positive")
    scale = decision["strength_multiplier"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 1
    ):
        raise ValueError("strong AU must use one finite multiplier greater than one")
    if any(
        not math.isclose(decision["strong"][key], scale * value, rel_tol=1e-12, abs_tol=0)
        for key, value in decision["balanced"].items()
    ):
        raise ValueError("strong AU must preserve the balanced alignment:uniformity ratio")
    if not isinstance(decision["calibration"], dict) or set(decision["calibration"]) != {
        "init",
        "ri_last",
    }:
        raise ValueError("both init and RI-last complete calibration endpoints required")
    populations, plans = [], []
    for endpoint in ("init", "ri_last"):
        reference = decision["calibration"][endpoint]
        if (
            not isinstance(reference, dict)
            or set(reference) != {"root", "complete_sha256"}
            or (
                reference["root"] != f"calibration/{endpoint}"
                or not _sha(reference["complete_sha256"])
            )
        ):
            raise ValueError("calibration roots and COMPLETE checksums must be explicitly bound")
        files = {
            name: _contained_file(path.parent, f"{reference['root']}/{name}")
            for name in EVIDENCE_FILES
        }
        if sha256_file(files["COMPLETE.json"]) != reference["complete_sha256"]:
            raise ValueError("calibration COMPLETE checksum changed")
        records = {name: json.loads(file.read_text()) for name, file in files.items()}
        plan, complete, census, index = (records[name] for name in EVIDENCE_FILES[:4])
        plan_hash = sha256_json(plan)
        if any(
            plan.get(key) != value
            for key, value in {
                "schema": "gradpert-au-calibration-1",
                "mode": "gradients",
                "batches": 8,
                "effective_population": 320,
                "microbatch": 80,
                "temperature": 2.0,
                "updates": 0,
                "split": "train_only",
                "seed": 1,
                "config_sha256": sha256_file(source / CALIBRATION_CONFIG),
            }.items()
        ) or (
            plan.get("source", {}).get("commit") != CALIBRATION_SOURCE
            or plan["source"].get("dirty") is not False
            or plan["source"].get("published_commit") != CALIBRATION_SOURCE
            or plan["source"].get("formal_eligible") is not True
        ):
            raise ValueError("calibration source, population or no-update training scope changed")
        if endpoint == "init":
            if (
                plan.get("checkpoint_sha256") is not None
                or plan.get("training_identity") is not None
            ):
                raise ValueError("init calibration must have no checkpoint")
        elif (
            not _sha(plan.get("checkpoint_sha256"))
            or not isinstance(plan.get("training_identity"), dict)
            or not plan["training_identity"]
        ):
            raise ValueError("RI-last calibration requires its original checkpoint identity")
        hashes = {name: sha256_file(files[name]) for name in EVIDENCE_FILES[4:]}
        if complete != {
            "calibration_complete": True,
            "scientific_complete": False,
            "updates": 0,
            "plan_sha256": plan_hash,
            "batches_completed": 8,
            "expected": 8,
            "batch_receipts": hashes,
        } or index != {"plan_sha256": plan_hash, "receipts": hashes}:
            raise ValueError("calibration requires all eight checksum-bound complete batches")
        if (
            any(
                census.get(k) != v
                for k, v in {
                    "plan_sha256": plan_hash,
                    "complete": True,
                    "model_forward": False,
                    "updates": 0,
                }.items()
            )
            or not isinstance(census.get("batches"), list)
            or len(census["batches"]) != 8
        ):
            raise ValueError("calibration needs the full frozen eight-batch metadata census")
        verified = load_calibration_endpoint(files["PLAN.json"].parent, endpoint, {})
        if verified["plan"] != plan or verified["census"] != census["batches"]:
            raise ValueError("calibration plan or populations changed during verification")
        populations.append(census["batches"])
        plans.append(plan)
    if populations[0] != populations[1] or plans[0].get("rng_seed") != plans[1].get("rng_seed"):
        raise ValueError("init and RI-last must use the same eight training populations and RNG")
    return decision


def changes(decision: dict) -> dict:
    values = {ARMS[0]: {}}
    for arm, strength in zip(ARMS[1:], ("balanced", "strong"), strict=True):
        values[arm] = {
            "au_alignment_weight": decision[strength]["alignment_weight"],
            "au_uniformity_weight": decision[strength]["uniformity_weight"],
            "au_temperature": 2.0,
            "au_warmup_steps": 30,
        }
    return values


def configs(source: Path, decision: dict) -> dict:
    anchor = first_configs(source)["RS-RI"]
    values = {}
    for name, settings in changes(decision).items():
        value = copy.deepcopy(anchor)
        for key, setting in settings.items():
            value["model"]["parameters"][key] = {
                "value": setting,
                "source": "project_preregistered",
                "reference": REFERENCE,
            }
        values[name] = value
    return values


def metadata(decision: dict, decision_hash: str) -> dict:
    value = first_metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=["R3-AUS", "R3-RI", "R3-AUB"],
    )
    value["test_timing"] = "after_research_design_lock"
    value["research_iteration"].update(
        round=3,
        authorized_rounds=None,
        stage_review_after_round=4,
        continuation="until task-relevant representation problem is resolved",
        future_rounds=(
            "adaptive after deep analysis; four rounds are a review, not automatic success"
        ),
        original_anchor="representation_round1_20261021/RS-B0",
        repeated_candidate="R3-RI exactly repeats the unchanged two-epoch RI configuration",
        hypothesis=(
            "same-input alignment and compatible-context dispersion improve response "
            "representation without setting target rank or cosine margins"
        ),
    )
    value["coefficient_decision"] = {"path": "decision.json", "sha256": decision_hash}
    value["calibration"] = copy.deepcopy(decision["calibration"])
    value["source_roles"] = {
        "calibration_commit": CALIBRATION_SOURCE,
        "training_commit": (
            "bound separately by the native clean published queue and launch source_commit"
        ),
        "ri_last_training_commit": "preserved in calibration/ri_last/PLAN.json training_identity",
    }
    value["alignment_uniformity"] = {
        "representation": "two Student Global Response CLS after final RMSNorm before projector",
        "normalization": "auxiliary FP32 L2 only; main representations unchanged",
        "alignment": "mean(1-cos) for the same control and perturbation across two Globals",
        "uniformity": (
            "logmeanexp(4*(cos-1)) across eligible unordered pairs separately per Global, then mean"
        ),
        "eligible_pairs": "same control cell_type::batch context and different perturbation",
        "population": (
            "complete differentiable cross-rank and accumulation population; "
            "deduplicate(control,p) before sharding"
        ),
        "temperature": 2.0,
        "warmup_successful_updates": 30,
        "balanced": decision["balanced"],
        "strong": decision["strong"],
        "strength_multiplier": decision["strength_multiplier"],
        "empty_pairs": "fail explicitly",
        "partial_diagnostics": "AU unavailable; never fake a full-population loss",
    }
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    decision_path = _contained_file(path.parent, "decision.json")
    decision = load_decision(source, decision_path)
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata(
        decision, sha256_file(decision_path)
    ):
        raise ValueError("round3 registered protocol or original decision checksum changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("round3 requires exactly RI, balanced AU and stronger proportional AU")
    expected, axes = configs(source, decision), changes(decision)
    for row in value["rows"]:
        target = _contained_file(path.parent, f"{row['name']}/gradpert_v2/nadig_jurkat.yaml")
        if (
            row
            != {
                "name": row["name"],
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": axes[row["name"]],
            }
            or yaml.safe_load(target.read_text()) != expected[row["name"]]
        ):
            raise ValueError("round3 must preserve RI and its calibrated AU coefficients")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path, decision_path: Path) -> dict:
    source, output, decision_path = (p.resolve() for p in (source, output, decision_path))
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    decision = load_decision(source, decision_path)
    output.mkdir(parents=True)
    (output / "decision.json").write_bytes(decision_path.read_bytes())
    for endpoint in ("init", "ri_last"):
        for name in EVIDENCE_FILES:
            relative = f"calibration/{endpoint}/{name}"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_contained_file(decision_path.parent, relative), target)
    rows = []
    axes = changes(decision)
    for name, value in configs(source, decision).items():
        target = output / name / "gradpert_v2/nadig_jurkat.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        rows.append(
            {
                "name": name,
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": axes[name],
            }
        )
    path = output / "manifest.json"
    path.write_text(
        json.dumps({**metadata(decision, sha256_file(decision_path)), "rows": rows}, indent=2)
        + "\n"
    )
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/v2/representation_round3_20261023")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output, root / args.decision), indent=2))
