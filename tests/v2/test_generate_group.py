import runpy
from pathlib import Path

import pytest
import yaml

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
API = runpy.run_path(str(ROOT / "scripts/v2/generate_group.py"))
PARENT = ROOT / "configs/v2/capacity/m16/gradpert_v2/nadig_jurkat.yaml"


@pytest.mark.parametrize("group", API["GROUPS"])
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
        assert set(config["model"]["parameters"]) == set(original["model"]["parameters"])
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
