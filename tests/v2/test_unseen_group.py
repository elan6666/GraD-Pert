"""The three new arms cannot change the frozen experiment protocol."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
spec = importlib.util.spec_from_file_location(
    "unseen_queue", ROOT / "scripts/v2/run_unseen_group.py"
)
assert spec and spec.loader
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_checked_in_three_independent_configs_keep_l0_settings():
    m = queue.verify_manifest(ROOT, ROOT / "configs/v2/cap40_unseen_20261008/manifest.json")
    assert [row["name"] for row in m["rows"]] == ["U1", "U2", "U3"]
    assert all(sum(row["changes"].values()) == 1 for row in m["rows"])
    assert (m["epochs"], m["validation"], m["test_roles"]) == (6, "disabled", ["last"])


def generated(tmp_path):
    from generate_unseen_group import generate

    parent = tmp_path / queue.PARENT
    parent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / queue.PARENT, parent)
    out = tmp_path / "configs/new-unseen"
    generate(tmp_path, out)
    return out / "manifest.json"


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 20),
        ("global_batch", 128),
        ("test_roles", ["best", "last"]),
        ("validation", "joint_only"),
    ],
)
def test_changed_experiment_scope_is_rejected(tmp_path, field, value):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    m[field] = value
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="authorized"):
        queue.verify_manifest(tmp_path, path)


def test_single_arm_cannot_change_an_unrelated_loss_weight(tmp_path):
    import yaml

    path = generated(tmp_path)
    m = json.loads(path.read_text())
    cfg = tmp_path / m["rows"][0]["config"]
    v = yaml.safe_load(cfg.read_text())
    v["model"]["parameters"]["lambda2"]["value"] = 0
    cfg.write_text(yaml.safe_dump(v))
    m["rows"][0]["sha256"] = queue.sha256_file(cfg)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="unrelated"):
        queue.verify_manifest(tmp_path, path)


def test_prior_must_be_finished_frozen_l0_not_l1_or_new_run(tmp_path):
    from gradpert.config import load_experiment_config

    root = tmp_path / "run"
    (root / "fit").mkdir(parents=True)
    files = {
        "COMPLETE.json": {},
        "fit/epoch_state.json": {"epoch": 6, "best": None, "last": {"epoch": 6}},
        "fit/last-test.json": {
            "identity": {"role": "last"},
            "result": {"split": "test", "conditions": [{}] * 592},
        },
        "run_manifest.json": {
            "config_sha256": queue.PARENT_SHA256,
            "source": {"commit": "ab022caa57a3b45dc5a14c6ae38bc280702112e4", "dirty": False},
        },
        "resolved_config.json": load_experiment_config(ROOT / queue.PARENT).model_dump(mode="json"),
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    sha = queue.sha256_file(root / "COMPLETE.json")
    queue.validate_reference(ROOT, root, sha)
    files["run_manifest.json"]["source"]["dirty"] = True
    (root / "run_manifest.json").write_text(json.dumps(files["run_manifest.json"]))
    with pytest.raises(ValueError, match="frozen completed L0"):
        queue.validate_reference(ROOT, root, sha)


def test_four_arm_design_raw_projection_and_order(tmp_path):
    from generate_unseen_group import RAW_PRIOR, RAW_SHA, generate

    parent = tmp_path / queue.PARENT
    parent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / queue.PARENT, parent)
    out = tmp_path / "configs/new-four"
    generate(tmp_path, out, with_u4=True)
    m = queue.verify_manifest(tmp_path, out / "manifest.json")
    assert [r["name"] for r in m["rows"]] == ["U1", "U2", "U4", "U3"]
    assert all(sum(r["changes"].values()) == 1 for r in m["rows"])
    import yaml

    p = tmp_path / m["rows"][2]["config"]
    v = yaml.safe_load(p.read_text())
    assert v["model"]["parameters"]["genept_artifact_path"]["value"] == RAW_PRIOR
    assert v["model"]["parameters"]["genept_sha256"]["value"] == RAW_SHA
    v["model"]["parameters"]["genept_artifact_path"]["value"] = "pca.npz"
    p.write_text(yaml.safe_dump(v))
    m["rows"][2]["sha256"] = queue.sha256_file(p)
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="raw GenePT"):
        queue.verify_manifest(tmp_path, out / "manifest.json")
