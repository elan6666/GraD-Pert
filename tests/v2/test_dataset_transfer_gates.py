"""Promotion cannot substitute a weaker probe or a negative main decision."""

import json
import runpy
from pathlib import Path

import pytest
from test_run_group import preflight

from gradpert.hashing import sha256_file


@pytest.fixture
def api(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/v2"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_dataset_transfer_300.py"))


def family(tmp_path):
    receipt, entry, plan = preflight(tmp_path)
    receipt.update(
        kind="preflight_only",
        steps_requested=10,
        steps_completed=10,
        inference_exercised=True,
        inference_shape=[300, 4883],
    )
    return receipt, entry, plan


def save(receipt, entry):
    Path(entry["receipt"]).write_text(json.dumps(receipt))
    entry["sha256"] = sha256_file(entry["receipt"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "integration_only"),
        ("steps_requested", 1),
        ("steps_completed", 11),
        ("inference_exercised", False),
        ("inference_shape", [1, 4883]),
    ],
)
def test_family_rejects_weaker_or_different_native_evidence(tmp_path, api, field, value):
    receipt, entry, plan = family(tmp_path)
    save(receipt, entry)
    api["validate_family_preflight"](plan, entry)
    receipt[field] = value
    save(receipt, entry)
    with pytest.raises(ValueError):
        api["validate_family_preflight"](plan, entry)


@pytest.mark.parametrize("approval", [False, None, 1, "true"])
def test_affirmative_main_acceptance_required(tmp_path, api, approval):
    _, _, plan = family(tmp_path)
    plan["run_id"] = "native-run"
    ready = tmp_path / "PREFLIGHT_READY.json"
    ready.write_text("{}")
    accepted = dict(
        approved=True,
        preflight_ready_sha256=sha256_file(ready),
        source_commit=plan["source_commit"],
        config_sha256=plan["config_sha256"],
        run_id=plan["run_id"],
        optimizer_updates=300,
    )
    api["validate_acceptance"](plan, ready, accepted)
    accepted["approved"] = approval
    with pytest.raises(ValueError):
        api["validate_acceptance"](plan, ready, accepted)
