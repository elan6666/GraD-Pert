from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "external_baseline_dispatch", ROOT / "scripts/v2/run_external_baseline.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments(model="gears", **overrides):
    values = dict(
        config=ROOT / f"configs/r50-rerun/{model}/nadig_jurkat.yaml",
        anchor_config=ROOT / "configs/v2/capacity/ddp_m74_eval128/gradpert_v2/nadig_jurkat.yaml",
        python=Path("/data/yilangliu/env/bin/python"),
        official_checkout=Path("/data/yilangliu/upstream"),
        run_root=Path("/data/yilangliu/fresh-run"),
        run_id="baseline-step",
        mode="step-smoke",
        smoke_run_root=None,
        official_data_root=Path("/data/yilangliu/official-data"),
        genept_seed=Path("/data/yilangliu/seed.npz"),
        environment_lock=Path("/data/yilangliu/lock.json"),
    )
    return argparse.Namespace(**(values | overrides))


RUNTIME = dict(
    data_root="/data/yilangliu/canonical",
    publication_receipt="/data/yilangliu/publication.json",
    publication_sha256="a" * 64,
)


@pytest.mark.parametrize("model", MODULE.MODELS)
def test_official_dispatch_and_full_gate(model):
    args = arguments(model)
    original = args.config.read_bytes()
    result = MODULE.command(args, RUNTIME)
    assert result[2:4] == ["gradpert", "benchmark"]
    assert result[result.index("--model") + 1] == model
    assert "--step-smoke" in result and "train" not in result
    assert args.config.read_bytes() == original
    args.mode = "full"
    with pytest.raises(ValueError, match="smoke run root"):
        MODULE.command(args, RUNTIME)
    args.smoke_run_root = Path("/data/yilangliu/passed-step")
    result = MODULE.command(args, RUNTIME)
    assert "--step-smoke" not in result and "--smoke-run-root" in result


def test_reject_different_canonical_split(tmp_path):
    args = arguments()
    raw = yaml.safe_load(args.config.read_text())
    raw["data"]["split_seed"] += 1
    args.config = tmp_path / "gears/nadig_jurkat.yaml"
    args.config.parent.mkdir()
    args.config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="canonical data/evaluation"):
        MODULE.command(args, RUNTIME)


def test_reject_native_config_as_external():
    args = arguments()
    args.config = args.anchor_config
    with pytest.raises(ValueError, match="official baseline"):
        MODULE.command(args, RUNTIME)


def test_model_specific_inputs_are_required():
    with pytest.raises(ValueError, match="official data root"):
        MODULE.command(arguments(official_data_root=None), RUNTIME)
    with pytest.raises(ValueError, match="environment lock"):
        MODULE.command(arguments("scouter_genept_seed", environment_lock=None), RUNTIME)
