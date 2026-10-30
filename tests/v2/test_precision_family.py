"""Precision candidates retain the sealed two-epoch control and data protocol."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.v2.generate_precision_family import (
    BASELINE,
    CHANGES,
    FAMILY,
    configs,
    generate,
    metadata,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_only_one_precision_factor_changes_per_candidate():
    anchor = yaml.safe_load((ROOT / BASELINE).read_text())
    candidates = configs(ROOT)
    assert list(candidates) == ["PREC-SSL32", "PREC-PRED32", "PREC-FFN8"]
    for arm, candidate in candidates.items():
        restored = copy.deepcopy(candidate)
        for field in CHANGES[arm]:
            del restored["model"]["parameters"][field]
        assert restored == anchor
        assert candidate["training"]["max_epochs"]["value"] == 2
        assert candidate["model"]["parameters"]["microbatch"]["value"] == 80
        assert candidate["model"]["exclude_test_target_expression"] is True


def test_manifest_preserves_reference_and_does_not_launch_baseline():
    value = metadata()
    assert value["repeat_baseline"] is False
    assert value["main_reference"] == "R3-RI"
    assert value["completed_optimizer_updates"] == 300
    assert value["priority"] == list(CHANGES)
    assert value["precision"]["ds_v3_alignment"].startswith("principle only")
    assert value["scientific_complete"] is False


def test_precision_controller_requires_main_acceptance_and_defers_test():
    from scripts.v2.run_unified_ddp import is_precision_queue, needs_preflight_acceptance

    queue = {"rows": [{"name": arm} for arm in CHANGES]}
    assert is_precision_queue(queue)
    assert needs_preflight_acceptance(queue)
    assert "deferred" in metadata()["test_timing"]
    queue["rows"].append({"name": "R3-RI"})
    assert not is_precision_queue(queue)


def test_generated_self_contained_configs_and_negative_tamper(tmp_path):
    anchor = tmp_path / BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / BASELINE).read_bytes())
    output = tmp_path / "configs/v2" / FAMILY
    result = generate(tmp_path, output)
    assert [r["name"] for r in result["rows"]] == list(CHANGES)
    with pytest.raises(ValueError, match="fresh"):
        generate(tmp_path, output)
    manifest = output / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["repeat_baseline"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol changed"):
        verify_manifest(tmp_path, manifest)
