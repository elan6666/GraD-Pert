import runpy
from pathlib import Path

import pytest
import yaml

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
API = runpy.run_path(str(ROOT / "scripts/v2/generate_group.py"))
PARENT = ROOT / "configs/v2/capacity/m16/gradpert_v2/nadig_jurkat.yaml"


def test_new_groups_default_to_current_full_v2_jurkat_parent():
    raw = yaml.safe_load(API["DEFAULT_PARENT"].read_text())
    assert raw["model"]["parameters"]["attention"]["value"] == "hybrid"
    assert raw["model"]["parameters"]["graph_read_mode"]["value"] == "propagated"
    assert raw["model"]["parameters"]["graph_expander_type"]["value"] == "hamiltonian"
    assert raw["model"]["parameters"]["loss_reduction"]["value"] == "row_mean"
    assert raw["training"]["train_batch_size"]["value"] == 192


@pytest.mark.parametrize("group", [g for g in API["GROUPS"] if g not in ("H3", "G1")])
def test_generated_groups_are_standalone_and_preserve_fixed_parent_contract(tmp_path, group):
    original = yaml.safe_load(PARENT.read_text())
    manifest = API["generate"](PARENT, group, tmp_path / group)
    assert manifest["parent_sha256"] == sha256_file(PARENT)
    for row in manifest["rows"]:
        path = tmp_path / group / row["config"]
        config = yaml.safe_load(path.read_text())
        assert sha256_file(path) == row["sha256"]
        assert config["data"] == original["data"]
        assert config["training"]["train_batch_size"] == original["training"]["train_batch_size"]
        assert config["training"]["max_epochs"] == original["training"]["max_epochs"]
        assert config["model"]["version"] == "v2"
        assert set(config["model"]["parameters"]) == (
            set(original["model"]["parameters"]) | ({"loss_reduction"} if group == "S1" else set())
        )
        for key, value in config["model"]["parameters"].items():
            if key not in row["overrides"]:
                assert value == original["model"]["parameters"][key]
        if group == "H1":
            schedule = config["training"]["scheduler"]["value"]
            assert schedule["min_lr"] == 0.2 * schedule["max_lr"]
            assert schedule["warmup_fraction"] == 0.16
        if group in ("L1", "L2") and row["name"] == "components_000":
            assert config["model"]["parameters"]["lambda1"]["value"] == 0
            assert config["model"]["parameters"]["lambda2"]["value"] == 0
    with pytest.raises(FileExistsError):
        API["generate"](PARENT, group, tmp_path / group)


@pytest.mark.parametrize(
    "mutation", ["unsealed_row", "resealed_confound", "factor_level", "parent"]
)
def test_verifier_rejects_config_drift_and_resealed_confounds(tmp_path, mutation):
    import json

    output = tmp_path / "H1"
    API["generate"](PARENT, "H1", output)
    manifest_path = output / "manifest.json"
    assert API["verify_group"](manifest_path, PARENT)["group"] == "H1"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["rows"][0]
    path = output / row["config"]
    if mutation in ("unsealed_row", "resealed_confound"):
        config = yaml.safe_load(path.read_text())
        config["model"]["parameters"]["dropout"]["value"] = 0.2
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        if mutation == "resealed_confound":
            row["sha256"] = sha256_file(path)
    elif mutation == "factor_level":
        row["overrides"]["width"] = 128
    else:
        manifest["parent_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        API["verify_group"](manifest_path, PARENT)


def test_h3_requires_measured_profiles_and_preserves_learning_rate(tmp_path, monkeypatch):
    import json

    from test_capacity_report import valid_receipt

    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    probes = []
    for batch in (8, 32):
        config = ROOT / f"configs/v2/capacity/m{batch}/gradpert_v2/nadig_jurkat.yaml"
        receipt = tmp_path / f"synthetic-probe-{batch}.json"
        payload = valid_receipt()
        payload["config_sha256"] = sha256_file(config)
        receipt.write_text(json.dumps(payload))
        probes.append(
            {"receipt": str(receipt), "config": str(config), "receipt_sha256": sha256_file(receipt)}
        )
    output = tmp_path / "H3"
    manifest = API["generate"](PARENT, "H3", output, batch_probes=probes)
    assert [r["name"] for r in manifest["rows"]] == ["batch_8", "batch_32"]
    assert API["verify_group"](output / "manifest.json", PARENT)["group"] == "H3"
    for row in manifest["rows"]:
        config = yaml.safe_load((output / row["config"]).read_text())
        assert (
            config["training"]["learning_rate"]
            == yaml.safe_load(PARENT.read_text())["training"]["learning_rate"]
        )
    Path(probes[0]["receipt"]).write_text("changed")
    with pytest.raises(ValueError, match="receipt changed"):
        API["verify_group"](output / "manifest.json", PARENT)


def test_h3_cannot_be_generated_from_unmeasured_defaults(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    with pytest.raises(ValueError, match="measured"):
        API["generate"](PARENT, "H3", tmp_path / "H3")


def test_h3_global_batch_maps_to_per_rank_microbatch():
    rows = dict(API["levels"]("H3", [64, 128], world_size=2))
    assert rows["batch_64"] == {"microbatch": 32, "train_batch_size": 64}
    assert rows["batch_128"] == {"microbatch": 64, "train_batch_size": 128}
    with pytest.raises(ValueError):
        API["levels"]("H3", [63, 128], world_size=2)


def test_superseded_single_rank_initial_groups_are_not_launchable():
    parent = ROOT / "configs/v2/initial_jurkat/parent/gradpert_v2/nadig_jurkat.yaml"
    for group in ("B0", "H1"):
        manifest = ROOT / f"configs/v2/initial_jurkat/{group}/manifest.json"
        with pytest.raises(ValueError, match="superseded"):
            API["verify_group"](manifest, parent)


def test_g1_binds_partition_and_rejects_changed_holdout(tmp_path):
    import json

    from gradpert.training.v2.holdout import make_partition

    path = tmp_path / "partition.json"
    path.write_text(
        json.dumps(make_partition(tuple(f"g{i}" for i in range(2000)), heldout_count=1000, seed=1))
    )
    sealed = {"path": str(path), "sha256": sha256_file(path)}
    output = tmp_path / "G1"
    result = API["generate"](PARENT, "G1", output, holdout=sealed)
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert set(row["overrides"]) == {"expression_holdout_path", "expression_holdout_sha256"}
    assert API["verify_group"](output / "manifest.json", PARENT)["group"] == "G1"
    before = yaml.safe_load(PARENT.read_text())
    after = yaml.safe_load((output / row["config"]).read_text())
    assert after["data"] == before["data"]
    assert after["training"] == before["training"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        API["verify_group"](output / "manifest.json", PARENT)
