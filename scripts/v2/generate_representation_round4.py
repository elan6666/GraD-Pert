"""Seal one coding-rate replacement; completed controls are references, never jobs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_representation_round1 import configs as first_configs
from generate_representation_round1 import metadata as first_metadata

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "representation-round4-cr-medium-1"
FAMILY = "representation_round4_20261023"
REFERENCE = ".byte-os/plans/representation-round4-cr-20261023.md"
ARMS = ("R4-CR",)
CHANGES = {
    "au_alignment_weight": 0.05,
    "au_uniformity_weight": 0.0,
    "au_temperature": 2.0,
    "au_warmup_steps": 30,
    "cr_weight": 0.00004,
    "cr_epsilon_squared": 0.05,
    "cr_warmup_steps": 30,
}


def configs(source: Path) -> dict:
    value = copy.deepcopy(first_configs(source)["RS-RI"])
    for key, setting in CHANGES.items():
        value["model"]["parameters"][key] = {
            "value": setting,
            "source": "project_preregistered",
            "reference": REFERENCE,
        }
    return {"R4-CR": value}


def metadata() -> dict:
    value = first_metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        initialization="from_scratch; unchanged RI architecture and seed; no checkpoint resume",
        test_timing="after_research_design_lock",
        reused_controls=["R3-RI", "R3-AUB", "R3-AUS"],
        repeat_baseline=False,
    )
    value["schedule_semantics"]["comparison"] = (
        "same two-epoch relative schedules as existing RI; no added fine-tuning epochs"
    )
    value["research_iteration"].update(
        round=4,
        authorized_rounds=None,
        stage_review_after_round=4,
        future_rounds="redesign from evidence until task-relevant representation is improved",
        hypothesis=(
            "spectral coding-rate expansion may preserve prediction better than pair repulsion"
        ),
    )
    value["coefficient_decision"] = {
        "weights": CHANGES,
        "diagnostic_source_commit": "49d4ac7b73767143b56a3286eb3bb78cc0d94c1a",
        "diagnostic_run_id": "representation-r4-mechanism-49d4ac7-20261023-a1",
        "complete_sha256": {
            "R3-RI": "1914eee277c55fc2ad2a26888118304ed14c14d5088ead0d6c89bfc3d1a22ac5",
            "R3-AUS": "e95c13621e9050df6197bfff4ccd8e1eb7dfa985156155bd553acf575e0246fe",
        },
        "populations_per_endpoint": 8,
        "diagnostic_updates": 0,
        "rationale": (
            "CR4e-5 roughly matches AUB U.005 gradient scale at RI endpoint; "
            "directions are mixed and AUS scale is weaker, so this is exploration, not superiority"
        ),
    }
    value["coding_rate"] = {
        "representation": "two Student Global Response CLS after final RMSNorm before projector",
        "formula": "mean_views(-0.5*logdet(I+d/(N*epsilon_squared)*U.T@U))",
        "normalization": "auxiliary FP64 L2; uncentered second moment; main tensors unchanged",
        "population": "full deduplicated effective batch across ranks and microbatches",
        "limitations": [
            "trained endpoint calibration is not initialization or Muon/AdamW update evidence",
            "full-population CR differs from context-filtered U; "
            "context differences can be rewarded",
            "adapted backbone-level loss, not official SimDINO replication",
        ],
        "partial_diagnostics": "unavailable; training loss requires full population",
    }
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata():
        raise ValueError("R4 frozen coding-rate research protocol changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("R4 dispatch is exactly one CR arm, with no repeated baseline")
    target = path.parent / "R4-CR/gradpert_v2/nadig_jurkat.yaml"
    if target.is_symlink() or not target.resolve().is_relative_to(source):
        raise ValueError("configuration must remain source contained")
    expected = {
        "name": "R4-CR",
        "config": str(target.relative_to(source)),
        "sha256": sha256_file(target),
        "changes": CHANGES,
    }
    if (
        value["rows"] != [expected]
        or yaml.safe_load(target.read_text()) != configs(source)["R4-CR"]
    ):
        raise ValueError("R4 must preserve RI and the exact frozen A/CR coefficients")
    load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    target = output / "R4-CR/gradpert_v2/nadig_jurkat.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(configs(source)["R4-CR"], sort_keys=False, allow_unicode=True))
    row = dict(
        name="R4-CR",
        config=str(target.relative_to(source)),
        sha256=sha256_file(target),
        changes=CHANGES,
    )
    path = output / "manifest.json"
    path.write_text(json.dumps({**metadata(), "rows": [row]}, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2") / FAMILY)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
