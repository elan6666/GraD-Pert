"""Materialize a new self-contained training recipe with authorized throughput defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gradpert.config.schema import ExperimentConfig
from gradpert.config.training_defaults import with_training_throughput_defaults
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file


def prepare(source: Path, destination: Path) -> str:
    value = with_training_throughput_defaults(yaml.safe_load(source.read_text()))
    config = ExperimentConfig.model_validate(value)
    V2Options.parse_parameters(config.model.parameters)
    if destination.parts[-2:] != (config.model_id, f"{config.dataset_id}.yaml"):
        raise ValueError("new training config path must match model and dataset identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A new path is required even when the old content happens to match.
    with destination.open("x") as stream:
        stream.write(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    return sha256_file(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.input, args.output))
