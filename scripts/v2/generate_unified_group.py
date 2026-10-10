"""Seal the ten authorized unified-MLP arms without changing historical configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.hashing import sha256_file

PARENT = "configs/v2/cap40_genept_gelu_20261010/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "12fbec67fd027969a45111832c412f1a500f433a0e328ec800b4342b498a6c3d"
BASELINE_PRECHANGE_SHA = "40fb032365a7d61b8de6caae7e4fb15b78c2ee24"
REFERENCE = ".byte-os/plans/GRADPERT_V2_UNIFIED_MLP_FIRST_BATCH_20261010.plan.md"
SCHEMA = "unified-first-1"
FAMILY = "unified_first_20261010"
DEFAULT_OUTPUT = f"configs/v2/{FAMILY}"

# Order is part of the authorization; no paused or supplementary group is generated.
ARMS = {
    "N0": {},
    "U24": {"gene_conditioned_readout": True},
    "MR1": {"masked_response_ratio": 0.25, "lambda_masked_response": 1.0},
    "P1": {"population_response": True, "max_conditions": 1},
    "C1": {"perturbation_injection": "entry"},
    "O1": {"random_gene_order": False},
    "VH": {
        "global_min_ratio": 0.8,
        "global_max_ratio": 1.0,
        "local_min_ratio": 0.4,
        "local_max_ratio": 0.65,
    },
    "S1-L4": {"ssl1_local_views": 4, "ssl1_local_layout": "go_string_random"},
    "CG1": {"control_conditioned_graph": True},
    "S12-L4": {
        "ssl1_local_views": 4,
        "ssl2_local_views": 4,
        "ssl1_local_layout": "go_go_string_string",
    },
}

BASE_PARAMETERS = {
    "mlp_profile": "unified",
    "genept_projection_activation": "none",
    "learned_genept_projection": True,
    "control_conditioned_graph": False,
    "perturbation_injection": "per_layer",
    "random_gene_order": True,
    "ssl1_local_views": 2,
    "ssl2_local_views": 2,
    "ssl1_local_layout": "go_string",
    "independent_view_rng": True,
    "masked_response_ratio": 0.0,
    "lambda_masked_response": 0.0,
    "population_response": False,
    "lambda_mmd": 1.0,
    "world_size": 1,
    "accumulation": 1,
    "relay_scan_chunk_size": 32,
    "relay_graph_chunk_rows": 64,
    "relay_sequence_chunk_size": 16,
}


def _parameter(value: object, *, source: str = "user_locked") -> dict:
    return {"value": value, "source": source, "reference": REFERENCE}


def _batch(batch: object) -> int:
    if type(batch) is not int or batch < 2:
        raise ValueError("common batch must be an integer of at least two cells")
    return batch


def baseline(source: Path, batch: int) -> dict:
    """Return the complete new baseline derived from the frozen published parent."""
    batch = _batch(batch)
    if sha256_file(source / PARENT) != PARENT_SHA256:
        raise ValueError("frozen prechange parent checksum changed")
    value = yaml.safe_load((source / PARENT).read_text())
    parameters = value["model"]["parameters"]
    for name, setting in {**BASE_PARAMETERS, "microbatch": batch}.items():
        setting_source = (
            "project_preregistered" if name in ("microbatch", "lambda_mmd") else "user_locked"
        )
        parameters[name] = _parameter(setting, source=setting_source)
    training = value["training"]
    for name, setting in (("train_batch_size", batch), ("eval_batch_size", batch)):
        training[name] = _parameter(setting, source="project_preregistered")
    # The parent already fixes six epochs, disabled validation, seed1 and last-only
    # selection. Keep all its data, scheduler, loss and artifact fields byte-traceable.
    ExperimentConfig.model_validate(value)
    return value


def _arm(baseline_value: dict, changes: dict) -> dict:
    value = copy.deepcopy(baseline_value)
    for name, setting in changes.items():
        value["model"]["parameters"][name] = _parameter(setting)
    ExperimentConfig.model_validate(value)
    return value


def _metadata(batch: int) -> dict:
    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "parent": PARENT,
        "parent_sha256": PARENT_SHA256,
        "baseline_prechange_sha": BASELINE_PRECHANGE_SHA,
        "epochs": 6,
        "validation": "disabled",
        "test_roles": ["last"],
        "common_microbatch": batch,
        "global_batch": batch,
        "world_size": 1,
        "accumulation": 1,
        "seed": 1,
        "concurrent_experiments": 2,
        "primary_metric": "txpert_macro_pearson_delta_deg",
        "preflight_updates": 10,
    }


def generate(source: Path, output: Path, batch: int = 32, *, replace: bool = False) -> dict:
    """Generate all ten self-contained configs; rebatching always applies to all arms."""
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(source):
        raise ValueError("configuration output must stay within the source checkout")
    base = baseline(source, batch)
    configs = {name: _arm(base, changes) for name, changes in ARMS.items()}
    if output.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output}")
        # Only replace an intact generated matrix, never a hand-edited directory.
        verify_manifest(source, output / "manifest.json")
    else:
        output.mkdir(parents=True)
    rows = []
    for name, value in configs.items():
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        load_experiment_config(path)
        rows.append(
            {
                "name": name,
                # Config schema retains its canonical model/dataset experiment_id.
                # This identity distinguishes the arm in queue and result receipts.
                "experiment_id": f"{FAMILY}__{name}",
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": ARMS[name],
            }
        )
    manifest = {**_metadata(batch), "rows": rows}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_manifest(source, path)


def verify_manifest(source: Path, path: Path) -> dict:
    """Fail closed on scope, checksums or any undeclared config/provenance difference."""
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest escaped source checkout")
    manifest = json.loads(path.read_text())
    batch = _batch(manifest.get("common_microbatch"))
    rows = manifest.get("rows", [])
    if (
        {key: value for key, value in manifest.items() if key != "rows"} != _metadata(batch)
        or not isinstance(rows, list)
        or len(rows) != len(ARMS)
        or [row.get("name") for row in rows] != list(ARMS)
    ):
        raise ValueError("manifest differs from the authorized first batch")
    base = baseline(source, batch)
    for row in rows:
        name = row["name"]
        expected_path = path.parent / name / "gradpert_v2/nadig_jurkat.yaml"
        config = (source / row["config"]).resolve()
        if not config.is_relative_to(source) or config != expected_path.resolve():
            raise ValueError("config path escaped or changed arm identity")
        if sha256_file(config) != row["sha256"]:
            raise ValueError("config checksum changed")
        if (
            set(row) != {"name", "experiment_id", "config", "sha256", "changes"}
            or row["experiment_id"] != f"{FAMILY}__{name}"
            or row["changes"] != ARMS[name]
        ):
            raise ValueError("declared arm changes differ from the authorized first batch")
        load_experiment_config(config)
        if yaml.safe_load(config.read_text()) != _arm(base, ARMS[name]):
            raise ValueError("arm changed an unrelated data/model/training/provenance setting")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else source / args.output
    print(json.dumps(generate(source, output, args.batch, replace=args.replace), indent=2))


if __name__ == "__main__":
    main()
