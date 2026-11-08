"""Seal two normalization-layout arms without scheduling a baseline rerun."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("configs/v2/graph_prenorm_20261108/baseline/gradpert_v2/nadig_jurkat.yaml")
FAMILY = Path("configs/v2/llm_norm_attention_20261109")
REFERENCE = "docs/experiments/LLM_NORM_ATTENTION_20261109.md"


def generate(root: Path = ROOT) -> dict:
    parent = yaml.safe_load((root / PARENT).read_text())
    rows = []
    for arm, attention in (("A_norm_mla", "latent"), ("B_norm_standard", "standard")):
        value = copy.deepcopy(parent)
        for name, setting in (
            ("normalization_layout", "llm_pre"),
            ("terminal_attention", attention),
        ):
            value["model"]["parameters"][name] = {
                "value": setting,
                "source": "user_locked",
                "reference": REFERENCE,
            }
        path = FAMILY / arm / "gradpert_v2/nadig_jurkat.yaml"
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        if target.exists() and target.read_text() != content:
            raise ValueError("refuse to overwrite changed experiment config")
        target.write_text(content)
        arch, _ = V2Options.parse_parameters(load_experiment_config(target).model.parameters)
        assert arch.normalization_layout == "llm_pre" and arch.terminal_attention == attention
        rows.append({"arm": arm, "config": str(path), "sha256": sha256_file(target)})
    manifest = {
        "schema": "gradpert-llm-norm-terminal-attention-1",
        "parent_config": str(PARENT),
        "parent_sha256": sha256_file(root / PARENT),
        "changed_model_parameters": ["normalization_layout", "terminal_attention"],
        "epochs": 2,
        "new_baseline": False,
        "checkpoint_role": "last",
        "evaluation_partition": "validation",
        "deg": False,
        "test_tuning": False,
        "rows": rows,
    }
    target = root / FAMILY / "manifest.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if target.exists() and target.read_text() != content:
        raise ValueError("refuse to overwrite changed manifest")
    target.write_text(content)
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
