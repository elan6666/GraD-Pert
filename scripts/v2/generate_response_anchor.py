"""Write one self-contained RI+fixed-response anchor configuration, never a baseline run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file

PARENT = "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7"
REFERENCE = "docs/experiments/RESPONSE_GROUP_ANCHOR_20261028.md"
DIRECTION_REFERENCE = "docs/experiments/RESPONSE_ANCHOR_DIRECTION_20261029.md"
HEAD_PROTECTION_REFERENCE = "docs/experiments/RESPONSE_POPULATION_HEAD_PROTECTION_20261030.md"


def configuration(
    root: Path,
    *,
    bank_path: str,
    manifest_sha256: str,
    arrays_sha256: str,
    weight: float,
    loss_mode: str = "mse",
    population_path: str = "",
    population_sha256: str = "",
    population_arrays_sha256: str = "",
    prediction_head_grad: bool = True,
) -> dict:
    if loss_mode not in {"mse", "cosine_direction", "population_prediction"}:
        raise ValueError("unknown response anchor objective")
    if type(prediction_head_grad) is not bool or (
        not prediction_head_grad and loss_mode != "population_prediction"
    ):
        raise ValueError(
            "prediction_head_grad must be boolean; False requires population_prediction"
        )
    if loss_mode != "population_prediction" and any(
        (population_path, population_sha256, population_arrays_sha256)
    ):
        raise ValueError("population fields require population_prediction")
    parent = root / PARENT
    if sha256_file(parent) != PARENT_SHA256:
        raise ValueError("R3-RI reference recipe changed")
    value = yaml.safe_load(parent.read_text())
    for key, item in {
        "response_anchor_path": bank_path,
        "response_anchor_sha256": manifest_sha256,
        "response_anchor_arrays_sha256": arrays_sha256,
        "response_anchor_weight": weight,
        "response_anchor_seed": 20261028,
    }.items():
        value["model"]["parameters"][key] = {
            "value": item,
            "source": "project_preregistered",
            "reference": REFERENCE,
        }
    if loss_mode == "cosine_direction":
        value["model"]["parameters"]["response_anchor_loss"] = {
            "value": loss_mode,
            "source": "project_preregistered",
            "reference": DIRECTION_REFERENCE,
        }
    if loss_mode == "population_prediction":
        for key, item in {
            "response_anchor_loss": loss_mode,
            "response_anchor_population_path": population_path,
            "response_anchor_population_sha256": population_sha256,
            "response_anchor_population_arrays_sha256": population_arrays_sha256,
        }.items():
            value["model"]["parameters"][key] = {
                "value": item,
                "source": "project_preregistered",
                "reference": "docs/experiments/RESPONSE_POPULATION_PREDICTION_20261029.md",
            }
        if not prediction_head_grad:
            value["model"]["parameters"]["response_anchor_weight"]["reference"] = (
                HEAD_PROTECTION_REFERENCE + "#first8-calibration-only"
            )
            value["model"]["parameters"]["response_anchor_prediction_head_grad"] = {
                "value": False,
                "source": "project_preregistered",
                "reference": HEAD_PROTECTION_REFERENCE,
            }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-path", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--arrays-sha256", required=True)
    parser.add_argument("--weight", required=True, type=float)
    parser.add_argument(
        "--loss-mode", choices=("mse", "cosine_direction", "population_prediction"), default="mse"
    )
    parser.add_argument("--population-path", default="")
    parser.add_argument("--population-sha256", default="")
    parser.add_argument("--population-arrays-sha256", default="")
    parser.add_argument(
        "--no-prediction-head-grad", action="store_false", dest="prediction_head_grad"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("never replace a sealed probe or fit configuration")
    value = configuration(
        Path(__file__).resolve().parents[2],
        bank_path=args.bank_path,
        manifest_sha256=args.manifest_sha256,
        arrays_sha256=args.arrays_sha256,
        weight=args.weight,
        loss_mode=args.loss_mode,
        population_path=args.population_path,
        population_sha256=args.population_sha256,
        population_arrays_sha256=args.population_arrays_sha256,
        prediction_head_grad=args.prediction_head_grad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    config = load_experiment_config(args.output)
    V2Options.parse_parameters(config.model.parameters)
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
