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


def deferred_fixture(tmp_path, monkeypatch):
    from benchmarks.expression_holdout import config as official_config

    monkeypatch.setattr(runner, "SERVER_ROOT", tmp_path)
    root = tmp_path / "official"
    root.mkdir()
    rows = []
    for name in ("scdfm", "gears"):
        run = root / name
        (run / "fit").mkdir(parents=True)
        config = run / "config.yaml"
        config.write_text(
            (ROOT / f"configs/v2/official_expression_20261013/{name}_jurkat.yaml").read_text()
        )
        runtime = run / "runtime.json"
        runtime.write_text("{}")
        rows.append(
            {
                "name": name,
                "run_root": str(run),
                "config": str(config),
                "config_sha256": sha256_file(config),
                "runtime": str(runtime),
                "runtime_sha256": sha256_file(runtime),
                "scientific_source_commit": name + "-source",
            }
        )
    queue = {"source_commit": "a" * 40, "rows": rows}
    (root / "queue.json").write_text(json.dumps(queue))
    row = rows[0]
    config = official_config.load_config(Path(row["config"]))
    identity = {
        "source_commit": "scdfm-source",
        "source_clean": True,
        "config_sha256": row["config_sha256"],
        "runtime_sha256": row["runtime_sha256"],
        "checkpoint_role": "last",
        "official_commit": config["source_code"]["commit"],
        **config["evaluation"]["expected_hashes"],
        "scientific_complete": False,
    }
    scdfm = Path(row["run_root"])
    checkpoint = scdfm / "fit/last.pt"
    checkpoint.write_bytes(b"preserved-checkpoint")
    (scdfm / "RUN.json").write_text(json.dumps(identity))
    (scdfm / "fit-receipt.json").write_text(
        json.dumps(
            {
                **identity,
                "training_complete": True,
                "epochs_completed": 4,
                "completed_updates": 3988,
                "total_optimizer_updates": 3988,
                "checkpoint": str(checkpoint),
            }
        )
    )
    gears = Path(rows[1]["run_root"])
    (gears / "receipt.json").write_text("{}")
    (gears / "COMPLETE.json").write_text("{}")
    # The old parent remains failed; no synthetic COMPLETE is ever written.
    (root / "FAILURE.json").write_text("{}")
    calls = []

    def terminal(path, row, sha):
        calls.append((path, row["name"], sha))
        return {"last_test_sha256": "gears-test-hash"}

    monkeypatch.setattr(official_config, "validate_terminal", terminal)
    spec = {
        "root": str(root),
        "queue_sha256": sha256_file(root / "queue.json"),
        "source_commit": queue["source_commit"],
        "completion_policy": runner.DEFERRED_SCDFM_POLICY,
        "authorization_reference": "user:2026-10-14-native-five-scdfm-eval-deferred",
        "scdfm_fit": {
            "evaluation_status": "deferred_by_user",
            "run_receipt_sha256": sha256_file(scdfm / "RUN.json"),
            "fit_receipt_sha256": sha256_file(scdfm / "fit-receipt.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "gears_receipt_sha256": sha256_file(gears / "receipt.json"),
        "gears_last_test_sha256": "gears-test-hash",
    }
    return root, scdfm, gears, spec, calls


def test_explicit_deferral_releases_only_with_preserved_fit_and_real_gears(tmp_path, monkeypatch):
    root, scdfm, _, spec, calls = deferred_fixture(tmp_path, monkeypatch)
    assert runner.official_dependency_ready(spec) is True
    assert [entry[1:] for entry in calls] == [("gears", "gears-source")]
    assert not (scdfm / "COMPLETE.json").exists()
    assert not (root / "COMPLETE.json").exists()
    assert json.loads((scdfm / "RUN.json").read_text())["scientific_complete"] is False
    strict = {k: v for k, v in spec.items() if k in ("root", "queue_sha256", "source_commit")}
    with pytest.raises(RuntimeError, match="remain unstarted"):
        runner.official_dependency_ready(strict)


@pytest.mark.parametrize("change", ["checkpoint", "fit", "source", "authorization", "gears"])
def test_deferred_dependency_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, change):
    _, scdfm, _, spec, _ = deferred_fixture(tmp_path, monkeypatch)
    if change == "checkpoint":
        (scdfm / "fit/last.pt").write_bytes(b"different")
    elif change == "fit":
        path = scdfm / "fit-receipt.json"
        fit = json.loads(path.read_text())
        fit["completed_updates"] = 3987
        path.write_text(json.dumps(fit))
        spec["scdfm_fit"]["fit_receipt_sha256"] = sha256_file(path)
    elif change == "source":
        path = scdfm / "RUN.json"
        record = json.loads(path.read_text())
        record["source_commit"] = "wrong"
        path.write_text(json.dumps(record))
        spec["scdfm_fit"]["run_receipt_sha256"] = sha256_file(path)
    elif change == "authorization":
        spec.pop("authorization_reference")
    else:
        spec["gears_last_test_sha256"] = "wrong"
    with pytest.raises(ValueError):
        runner.official_dependency_ready(spec)


def test_deferred_dependency_waits_for_gears_and_rejects_unknown_policy(tmp_path, monkeypatch):
    _, _, gears, spec, _ = deferred_fixture(tmp_path, monkeypatch)
    (gears / "COMPLETE.json").unlink()
    assert runner.official_dependency_ready(spec) is False
    spec["completion_policy"] = "skip_everything"
    with pytest.raises(ValueError, match="unknown"):
        runner.official_dependency_ready(spec)
