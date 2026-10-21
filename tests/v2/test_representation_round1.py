"""Two-epoch research protocol, immutable anchor, and single-axis RI candidate."""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options
from gradpert.execution import train_entry
from gradpert.execution.v2 import _run_v2
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as family  # noqa: E402

MANIFEST = ROOT / "configs/v2/representation_round1_20261021/manifest.json"


def test_two_epoch_anchor_and_ri_have_exactly_one_model_difference():
    before = (ROOT / family.BASELINE).read_bytes()
    manifest = family.verify_manifest(ROOT, MANIFEST)
    assert manifest["priority"] == ["RS-B0", "RS-RI"]
    assert (manifest["epochs"], manifest["global_batch"], manifest["preflight_updates"]) == (
        2,
        320,
        10,
    )
    assert manifest["test_roles"] == ["last"] and manifest["validation"] == "disabled"
    assert manifest["research_iteration"]["test_results_for_selection"] is False
    assert manifest["fixed_panel_diagnostics"]["conditions"] == 32
    assert manifest["fixed_panel_diagnostics"]["controls_per_condition"] == 4
    configs = family.configs(ROOT)
    parent = yaml.safe_load(before)
    anchor = copy.deepcopy(configs["RS-B0"])
    for name in ("formal_run_policy", "max_epochs"):
        anchor["training"][name] = parent["training"][name]
    assert anchor == parent
    candidate = copy.deepcopy(configs["RS-RI"])
    switch = candidate["model"]["parameters"].pop("response_injection_residual")
    assert switch["value"] is True
    assert switch["source"] == "project_preregistered"
    assert configs["RS-B0"]["training"]["max_epochs"]["source"] == "user_locked"
    assert candidate == configs["RS-B0"]
    for row in manifest["rows"]:
        config = load_experiment_config(ROOT / row["config"])
        arch, options = V2Options.parse_parameters(config.model.parameters)
        assert arch.response_injection_residual == (row["name"] == "RS-RI")
        assert arch.ffn_variant == "gelu" and arch.prediction_parameterization == "softplus"
        assert options.validation_mode == "disabled"
        assert (options.microbatch, options.world_size, options.accumulation) == (80, 2, 2)
        assert config.training.formal_run_policy == "v2_fixed_2"
        assert config.training.max_epochs.value == 2 and config.training.monitor == "none"
        assert config.training.scheduler.value == parent["training"]["scheduler"]["value"]
        assert config.training.run_seeds == [1]
        assert config.continuation is None
    assert (ROOT / family.BASELINE).read_bytes() == before
    assert sha256_file(ROOT / family.BASELINE) == family.BASELINE_SHA256


@pytest.fixture
def generated(tmp_path):
    # Standalone temporary source avoids writes to historical fixtures.
    source = tmp_path / "source"
    anchor = source / family.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / family.BASELINE).read_bytes())
    output = source / "configs/v2/round1"
    manifest = family.generate(source, output)
    return source, output, manifest


@pytest.mark.parametrize(
    "mutation", ["extra_arm", "epoch", "batch", "test_selection", "panel_size", "changes", "path"]
)
def test_manifest_rejects_modified_scope(generated, mutation):
    source, output, manifest = generated
    if mutation == "extra_arm":
        manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
    elif mutation == "epoch":
        manifest["epochs"] = 4
    elif mutation == "batch":
        manifest["global_batch"] = 400
    elif mutation == "test_selection":
        manifest["research_iteration"]["test_results_for_selection"] = True
    elif mutation == "panel_size":
        manifest["fixed_panel_diagnostics"]["conditions"] = 128
    elif mutation == "changes":
        manifest["rows"][1]["changes"]["lambda1"] = 0
    else:
        manifest["rows"][0]["config"] = family.BASELINE
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        family.verify_manifest(source, path)


@pytest.mark.parametrize(
    "mutation", ["epoch", "loss", "activation", "seed", "lr", "validation", "residual"]
)
def test_config_tampering_fails_even_with_recomputed_manifest_hash(generated, mutation):
    source, output, manifest = generated
    row = manifest["rows"][1]
    path = source / row["config"]
    value = yaml.safe_load(path.read_text())
    if mutation == "epoch":
        value["training"]["max_epochs"]["value"] = 3
    elif mutation == "seed":
        value["training"]["run_seeds"] = [2]
    elif mutation == "lr":
        value["training"]["scheduler"]["value"]["warmup_fraction"] = 0.2
    else:
        key, setting = {
            "loss": ("ssl1_node", 1),
            "activation": ("ffn_variant", "swiglu"),
            "validation": ("validation_mode", "joint_only"),
            "residual": ("response_injection_residual", False),
        }[mutation]
        value["model"]["parameters"][key]["value"] = setting
    path.write_text(yaml.safe_dump(value))
    row["sha256"] = sha256_file(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="change only"):
        family.verify_manifest(source, output / "manifest.json")


def test_anchor_drift_and_reused_or_outside_output_rejected(generated, tmp_path):
    source, output, _ = generated
    with pytest.raises(ValueError, match="fresh"):
        family.generate(source, output)
    with pytest.raises(ValueError, match="source-contained"):
        family.generate(source, tmp_path / "outside")
    anchor = source / family.BASELINE
    anchor.write_text(anchor.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen B0"):
        family.configs(source)


@pytest.mark.parametrize("mutation", ["epochs", "best", "stopping", "legacy_model"])
def test_fixed2_policy_is_strict_and_does_not_reinterpret_old_policies(mutation):
    value = family.configs(ROOT)["RS-B0"]
    if mutation == "epochs":
        value["training"]["max_epochs"]["value"] = 4
    elif mutation == "best":
        value["training"]["monitor"] = "val/joint_loss"
        value["training"]["monitor_mode"] = "min"
        value["model"]["parameters"]["validation_mode"]["value"] = "joint_only"
    elif mutation == "stopping":
        value["training"]["early_stopping"] = True
    else:
        value["model_id"] = value["model"]["model_id"] = "gradpert_b2"
        value["model"]["version"] = "v1"
        value["experiment_id"] = "gradpert_b2__nadig_jurkat"
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(value)
    old = yaml.safe_load((ROOT / family.BASELINE).read_text())
    old["training"]["max_epochs"]["value"] = 2
    with pytest.raises(ValueError, match="exactly 4"):
        ExperimentConfig.model_validate(old)


@pytest.mark.parametrize("deferred", [False, True])
def test_native_plan_uses_two_epochs_no_validation_last_only(tmp_path, monkeypatch, deferred):
    row = family.verify_manifest(ROOT, MANIFEST)["rows"][0]
    monkeypatch.setattr(train_entry, "repository_root", lambda: ROOT)
    monkeypatch.setattr(train_entry, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(
        "gradpert.execution.identity.inspect_source_identity",
        lambda *a, **kw: SimpleNamespace(commit="a" * 40),
    )
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "data_root": str(tmp_path),
                "runs_root": str(tmp_path / "runs"),
                "publication_receipt": str(publication),
                "publication_sha256": sha256_file(publication),
                "genept_receipt": str(publication),
                "genept_sha256": sha256_file(publication),
            }
        )
    )
    args = argparse.Namespace(
        config=ROOT / row["config"],
        seed=1,
        gpu="0,1",
        data_root=None,
        runtime=runtime,
        dry_run=True,
        defer_test=deferred,
    )
    plan = train_entry.resolve_plan(args)
    assert plan["resolved_config"]["training"]["max_epochs"]["value"] == 2
    assert plan["report_all_validation_metrics"] is False
    assert not any("validation" in step or "best" in step for step in plan["stages"])
    assert plan["stages"][-1] == ("TRAIN_COMPLETE; last test pending" if deferred else "last test")
    assert not (tmp_path / "runs").exists()
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(ValueError, match="configuration changed"):
        _run_v2({"config": plan["config"], "config_sha256": "0" * 64})
