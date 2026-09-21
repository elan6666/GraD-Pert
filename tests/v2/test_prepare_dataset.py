import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    spec = importlib.util.spec_from_file_location(
        "prepare_dataset", ROOT / "scripts/v2/prepare_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "dataset",
    ["nadig_jurkat", "nadig_hepg2", "replogle_k562_essential", "replogle_rpe1_essential", "norman"],
)
def test_probe_preserves_exact_dataset_split_and_graph(api, tmp_path, dataset):
    parent = yaml.safe_load(api.template(dataset).read_text())
    path = api.prepare_probe(dataset, 32, tmp_path / dataset)
    raw = yaml.safe_load(path.read_text())
    assert raw["data"] == parent["data"]
    assert raw["evaluation"] == parent["evaluation"]
    for key, value in parent["model"]["parameters"].items():
        if key not in ("microbatch", "world_size"):
            assert raw["model"]["parameters"][key] == value
    assert raw["training"]["train_batch_size"]["value"] == 64
    assert raw["model"]["parameters"]["world_size"]["value"] == 2
    assert raw["model"]["parameters"]["microbatch"]["value"] == 32
    with pytest.raises(FileExistsError):
        api.prepare_probe(dataset, 32, tmp_path / dataset)


def test_real_jurkat_capacity_cannot_authorize_another_dataset(api, tmp_path):
    from gradpert.hashing import sha256_file

    receipt = ROOT / ".byte-os/evidence/v2-capacity/m64-repeat-58da227.json"
    config = ROOT / "configs/v2/capacity/m64/gradpert_v2/nadig_jurkat.yaml"
    with pytest.raises(ValueError, match="dataset"):
        api.prepare_initial("norman", config, receipt, sha256_file(receipt), tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="two-rank"):
        api.prepare_initial(
            "nadig_jurkat", config, receipt, sha256_file(receipt), tmp_path / "single_rank"
        )
