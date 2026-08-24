"""Fail-closed YAML loader for one resolved experiment file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from gradpert.config.schema import ExperimentConfig

_YAML_REFERENCE_TOKEN = re.compile(r"(^|[\s\[{,:])([&*])[A-Za-z0-9_.-]+", re.MULTILINE)


def _reject_hidden_composition(text: str) -> None:
    if _YAML_REFERENCE_TOKEN.search(text):
        raise ValueError("YAML anchors and aliases are forbidden in experiment configs")
    if re.search(r"(^|\n)\s*<<\s*:", text):
        raise ValueError("YAML merge keys are forbidden in experiment configs")


def _reject_defaults_key(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "defaults":
                raise ValueError(f"Hidden defaults key is forbidden at {path}")
            _reject_defaults_key(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_defaults_key(item, f"{path}[{index}]")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load exactly one standalone YAML experiment config."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    _reject_hidden_composition(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    _reject_defaults_key(payload)
    config = ExperimentConfig.model_validate(payload)
    expected_path = Path(config.model_id) / f"{config.dataset_id}.yaml"
    if config_path.parts[-2:] != expected_path.parts:
        raise ValueError(
            f"Config identity/path mismatch: expected suffix {expected_path}, got {config_path}"
        )
    return config
