"""Prepare dataset-specific capacity configs or measured initial ablation groups."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from capacity_report import collect
from generate_group import generate, row_configuration, verify_group

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    "nadig_jurkat",
    "nadig_hepg2",
    "replogle_k562_essential",
    "replogle_rpe1_essential",
    "norman",
)


def template(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset")
    return ROOT / "configs/v2/integration/gradpert_v2" / f"{dataset}.yaml"


def prepare_probe(dataset: str, batch: int, output: Path) -> Path:
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be positive")
    raw = yaml.safe_load(template(dataset).read_text())
    config = row_configuration(raw, {"microbatch": batch, "train_batch_size": batch})
    output.mkdir(parents=True, exist_ok=False)
    path = output / "gradpert_v2" / f"{dataset}.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    load_experiment_config(path)
    return path


def prepare_initial(
    dataset: str, config: Path, receipt: Path, receipt_sha256: str, output: Path
) -> dict:
    if sha256_file(receipt) != receipt_sha256:
        raise ValueError("capacity receipt checksum mismatch")
    observed = collect(receipt, config)
    expected = yaml.safe_load(template(dataset).read_text())
    raw = yaml.safe_load(config.read_text())
    if observed["dataset"] != dataset or raw["data"] != expected["data"]:
        raise ValueError("capacity dataset or canonical split differs")
    if observed["world_size"] != 1 or observed["accumulation"] != 1:
        raise ValueError("initial groups require measured single-rank physical batch")
    actual_parameters = raw["model"]["parameters"]
    expected_parameters = expected["model"]["parameters"]
    if set(actual_parameters) != set(expected_parameters) or any(
        actual_parameters[key]["value"] != value["value"]
        for key, value in expected_parameters.items()
        if key != "microbatch"
    ):
        raise ValueError("capacity method differs from the initial joint-SSL design")
    output.mkdir(parents=True, exist_ok=False)
    parent = output / "parent/gradpert_v2" / f"{dataset}.yaml"
    parent.parent.mkdir(parents=True)
    shutil.copyfile(config, parent)
    groups = {}
    for group in ("B0", "H1"):
        generate(parent, group, output / group)
        groups[group] = verify_group(output / group / "manifest.json", parent)
    result = {
        "dataset": dataset,
        "capacity": observed,
        "groups": groups,
        "status": "generated_pending_exact_source_config_seed_launch_preflight",
    }
    (output / "dataset_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="nadig_jurkat")
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--batch", type=int, required=True)
    initial = commands.add_parser("initial")
    initial.add_argument("--capacity-config", type=Path, required=True)
    initial.add_argument("--receipt", type=Path, required=True)
    initial.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    if args.action == "probe":
        print(prepare_probe(args.dataset, args.batch, args.output))
    else:
        result = prepare_initial(
            args.dataset, args.capacity_config, args.receipt, args.receipt_sha256, args.output
        )
        print(json.dumps({"dataset": result["dataset"], "status": result["status"]}))


if __name__ == "__main__":
    main()
