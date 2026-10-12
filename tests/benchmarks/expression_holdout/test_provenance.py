"""Actual environment drift and post-inference provenance must fail closed."""

import json
import sys
from importlib import metadata
from types import SimpleNamespace

import pytest

from benchmarks.expression_holdout.provenance import (
    persist_inference_receipt,
    verify_cpu_assets,
    verify_environment,
    verify_evaluation_files,
)
from gradpert.hashing import sha256_file


@pytest.fixture
def environment(tmp_path, monkeypatch):
    lock = tmp_path / "freeze.txt"
    lock.write_text("first==1.2\nsecond==3.4\n")
    receipt = tmp_path / "environment.json"
    receipt.write_text(
        json.dumps(
            {
                "freeze_sha256": sha256_file(lock),
                "distribution_versions": {"first": "1.2", "second": "3.4"},
                "python": sys.version,
            }
        )
    )
    versions = {"first": "1.2", "second": "3.4"}

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", version)
    runtime = {
        "environment_lock": str(lock),
        "environment_lock_sha256": sha256_file(lock),
        "environment_receipt": str(receipt),
        "environment_receipt_sha256": sha256_file(receipt),
        "python": sys.executable,
    }
    return runtime, versions


def test_installed_distributions_and_python_are_recorded(environment):
    runtime, versions = environment
    result = verify_environment(runtime)
    assert result["verified_distribution_versions"] == versions
    assert result["python_executable"] == sys.executable
    assert result["environment_receipt_sha256"] == runtime["environment_receipt_sha256"]


@pytest.mark.parametrize("change", ["different", "missing"])
def test_all_declared_distribution_versions_are_checked(environment, change):
    runtime, versions = environment
    if change == "different":
        versions["second"] = "3.5"
    else:
        del versions["second"]
    with pytest.raises(ValueError, match="distribution"):
        verify_environment(runtime)


def test_changed_lock_fails_even_if_receipt_is_unchanged(environment):
    runtime, _ = environment
    from pathlib import Path

    Path(runtime["environment_lock"]).write_text("first==999\n")
    with pytest.raises(ValueError, match="lock seal"):
        verify_environment(runtime)


def test_receipt_must_bind_lock_not_just_its_own_hash(environment):
    runtime, _ = environment
    from pathlib import Path

    receipt = Path(runtime["environment_receipt"])
    content = json.loads(receipt.read_text())
    content["freeze_sha256"] = "0" * 64
    receipt.write_text(json.dumps(content))
    runtime["environment_receipt_sha256"] = sha256_file(receipt)
    with pytest.raises(ValueError, match="sealed freeze"):
        verify_environment(runtime)


def test_wrong_interpreter_rejected(environment):
    runtime, _ = environment
    runtime["python"] = "/another/venv/bin/python"
    with pytest.raises(ValueError, match="interpreter"):
        verify_environment(runtime)


def test_inference_noise_added_after_fit_is_persisted(tmp_path):
    fitted = SimpleNamespace(receipt={"inference_noise": [{"sha256": "noise", "seed": 12}]})
    identity = {"source_commit": "source", "checkpoint_sha256": "checkpoint"}
    bound = persist_inference_receipt(tmp_path, fitted, identity)
    receipt = tmp_path / "inference-receipt.json"
    content = json.loads(receipt.read_text())
    assert content["inference_noise"] == fitted.receipt["inference_noise"]
    assert content["checkpoint_sha256"] == "checkpoint"
    assert bound["inference_receipt_sha256"] == sha256_file(receipt)


def test_evaluation_fail_fast_needs_only_manifests_without_test_matrix(tmp_path):
    root = tmp_path / "dataset" / "protocol"
    control = root / "manifests/evaluation_controls.test.json"
    reference = root / "evaluation_v2/state_manifest.json"
    control.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    control.write_text("frozen ordered controls")
    reference.write_text("frozen DEG and references")
    expected = {
        "control_manifest_sha256": sha256_file(control),
        "evaluation_state_sha256": sha256_file(reference),
    }
    config = {
        "data": {"dataset_id": "dataset", "protocol_id": "protocol"},
        "evaluation": {"expected_hashes": expected},
    }
    assert verify_evaluation_files(config, tmp_path) == expected
    # No expression files were ever created or required.
    reference.write_text("valid but regenerated references")
    with pytest.raises(ValueError, match="before fit"):
        verify_evaluation_files(config, tmp_path)


def test_cpu_assets_are_verified_before_deserialization(tmp_path):
    prior = tmp_path / "prior.pkl"
    prior.write_bytes(b"hashed diagnostic prior; not loaded here")
    names = ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    runtime = {
        "prior_path": str(prior),
        "prior_sha256": sha256_file(prior),
        "gears_assets": str(tmp_path),
        "gears_asset_sha256": {name: sha256_file(tmp_path / name) for name in names},
    }
    assert verify_cpu_assets(runtime)["prior_sha256"] == runtime["prior_sha256"]
    prior.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="prior identity"):
        verify_cpu_assets(runtime)
    runtime["prior_sha256"] = sha256_file(prior)
    (tmp_path / names[0]).write_bytes(b"different graph")
    with pytest.raises(ValueError, match="graph asset identity"):
        verify_cpu_assets(runtime)
