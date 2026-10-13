"""No BR rerun, single mechanism changes, and real official-terminal dependency."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
family = importlib.import_module("generate_ffn_five")
runner = importlib.import_module("run_br400_group")


def test_five_configs_keep_parent_data_protocol_and_only_declared_changes():
    parent = yaml.safe_load((ROOT / family.PARENT).read_text())
    configs = family.configurations(ROOT)
    assert list(configs) == ["DP", "FN", "FG", "FB", "PO"]
    for name, cfg in configs.items():
        assert cfg["data"] == parent["data"]
        assert cfg["evaluation"] == parent["evaluation"]
        assert cfg["model"]["exclude_test_target_expression"] is True
        changed = copy.deepcopy(cfg)
        params = changed["model"]["parameters"]
        for field in family.CHANGES[name]:
            if field in parent["model"]["parameters"]:
                params[field] = parent["model"]["parameters"][field]
            else:
                params.pop(field)
        params.pop("process_diagnostics")
        params["validation_mode"] = parent["model"]["parameters"]["validation_mode"]
        for field in ("monitor", "monitor_mode"):
            changed["training"][field] = parent["training"][field]
        assert changed == parent


def test_published_five_manifest_has_no_hidden_baseline_training():
    manifest = family.verify_manifest(ROOT, ROOT / "configs/v2/ffn_five_20261013/manifest.json")
    assert manifest["rerun_BR"] is False
    assert manifest["epochs"] == 4 and manifest["test_roles"] == ["last"]
    assert manifest["requires_official_baselines"] == ["scdfm", "gears"]
    assert (
        runner.verify_manifest(ROOT, ROOT / "configs/v2/ffn_five_20261013/manifest.json")
        == manifest
    )


def test_official_dependency_requires_both_evaluated_terminals(tmp_path, monkeypatch):
    from benchmarks.expression_holdout import config as official_config

    monkeypatch.setattr(runner, "SERVER_ROOT", tmp_path)
    root = tmp_path / "official"
    root.mkdir()
    rows = []
    completed = {}
    for name in ("scdfm", "gears"):
        run = root / name
        run.mkdir()
        (run / "receipt.json").write_text("{}")
        rows.append({"name": name, "run_root": str(run)})
        completed[name] = {
            "root": str(run),
            "receipt_sha256": sha256_file(run / "receipt.json"),
            "last_test_sha256": name + "-test-hash",
        }
    queue = {"source_commit": "a" * 40, "rows": rows}
    (root / "queue.json").write_text(json.dumps(queue))
    spec = {
        "root": str(root),
        "queue_sha256": sha256_file(root / "queue.json"),
        "source_commit": "a" * 40,
    }
    monkeypatch.setattr(
        official_config,
        "validate_terminal",
        lambda path, row, sha: {"last_test_sha256": row["name"] + "-test-hash"},
    )
    assert runner.official_dependency_ready(spec) is False
    terminal = {"status": "complete", "scientific_complete": True, "completed": completed}
    (root / "COMPLETE.json").write_text(json.dumps(terminal))
    assert runner.official_dependency_ready(spec) is True
    terminal["completed"].pop("gears")
    (root / "COMPLETE.json").write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="both required"):
        runner.official_dependency_ready(spec)
    (root / "FAILURE.json").write_text("{}")
    with pytest.raises(RuntimeError, match="remain unstarted"):
        runner.official_dependency_ready(spec)
