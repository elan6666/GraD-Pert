"""Five explicit last-only arms anchored to completed BR400; never dispatch BR."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "ffn-five-last-1"
PRIORITY = ("DP", "FN", "FG", "FB", "PO")
PARENT = "configs/v2/br400_best_last_checkpoint_20261012/BR/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA = "d7877c03839645901f930817d5258638ca8738c650bd23fc42e5da33415e6f50"
REFERENCE = ".byte-os/plans/OFFICIAL_RECOVERY_FFN_DIAGNOSTICS_20261013.plan.md"
CHANGES = {
    "FB": {"ffn_linear_bias": "none"},
    "FG": {"ffn_variant": "gelu"},
    "FN": {"ffn_input_norm": "layernorm"},
    "DP": {"prototype_weight_norm": True, "prototype_freeze_epochs": 1},
    "PO": {"prediction_parameterization": "softplus"},
}


def configurations(source: Path) -> dict:
    parent = source / PARENT
    if sha256_file(parent) != PARENT_SHA:
        raise ValueError("completed BR400 anchor changed")
    result = {}
    for name in PRIORITY:
        cfg = copy.deepcopy(yaml.safe_load(parent.read_text()))
        for key, value in {"validation_mode": "disabled", **CHANGES[name]}.items():
            cfg["model"]["parameters"][key] = {
                "value": value,
                "source": "user_locked",
                "reference": REFERENCE,
            }
        cfg["training"].update(monitor="none", monitor_mode="none")
        cfg["model"]["parameters"]["process_diagnostics"] = {
            "value": True,
            "source": "user_locked",
            "reference": REFERENCE,
        }
        result[name] = cfg
    return result


def metadata() -> dict:
    return {
        "schema": SCHEMA,
        "parents": {"BR": {"config": PARENT, "sha256": PARENT_SHA}},
        "priority": list(PRIORITY),
        "preflight_priority": list(PRIORITY),
        "common_microbatch": 100,
        "world_size": 2,
        "accumulation": 2,
        "global_batch": 400,
        "epochs": 4,
        "seed": 1,
        "eval_batch": 300,
        "validation": "disabled",
        "selection": "final_epoch",
        "test_roles": ["last"],
        "preflight_updates": 10,
        "eval_capacity_reference_batch": 128,
        "execution": "official_dependency_then_dual_preflight_then_dual_fit_last",
        "requires_official_baselines": ["scdfm", "gears"],
        "rerun_BR": False,
        "process_diagnostics": "fixed_training_only_epoch_boundaries",
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("FFN manifest must be source-contained")
    manifest = json.loads(path.read_text())
    if {k: v for k, v in manifest.items() if k != "rows"} != metadata():
        raise ValueError("FFN five-arm protocol changed")
    if [row["name"] for row in manifest["rows"]] != list(PRIORITY):
        raise ValueError("FFN scope/order must exclude BR")
    expected = configurations(source)
    for row in manifest["rows"]:
        cfg_path = path.parent / row["name"] / "gradpert_v2/nadig_jurkat.yaml"
        if (
            row["config"] != str(cfg_path.relative_to(source))
            or sha256_file(cfg_path) != row["sha256"]
            or yaml.safe_load(cfg_path.read_text()) != expected[row["name"]]
        ):
            raise ValueError("FFN configuration identity changed")
        load_experiment_config(cfg_path)
    return manifest


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained config directory required")
    rows = []
    for name, cfg in configurations(source).items():
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        rows.append(
            {"name": name, "config": str(path.relative_to(source)), "sha256": sha256_file(path)}
        )
    path = output / "manifest.json"
    path.write_text(json.dumps({**metadata(), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[2], args.output)))
