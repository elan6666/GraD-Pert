import runpy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_sweep_accepts_two_rank_profiles_but_rejects_order_and_confounds(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    validate = runpy.run_path(str(ROOT / "scripts/v2/capacity_sweep.py"))["validate_profiles"]
    configs = [
        ROOT / f"configs/v2/capacity/ddp_m{n}/gradpert_v2/nadig_jurkat.yaml" for n in (32, 64)
    ]
    assert validate(configs) == [32, 64]
    with pytest.raises(ValueError, match="increasing"):
        validate(configs[::-1])
    raw = yaml.safe_load(configs[1].read_text())
    raw["model"]["parameters"]["ssl2_koleo"]["value"] = 0
    changed = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    changed.parent.mkdir()
    changed.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="vary only"):
        validate([configs[0], changed])
    with pytest.raises(ValueError, match="two ranks"):
        validate([ROOT / "configs/v2/capacity/m64/gradpert_v2/nadig_jurkat.yaml"])


def test_sweep_accepts_fixed_two_microbatch_accumulation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    validate = runpy.run_path(str(ROOT / "scripts/v2/capacity_sweep.py"))["validate_profiles"]
    default = yaml.safe_load(
        (ROOT / "configs/v2/glm53_flash_jurkat/default/gradpert_v2/nadig_jurkat.yaml").read_text()
    )
    configs = []
    for micro in (32, 40):
        row = yaml.safe_load(yaml.safe_dump(default))
        row["model"]["parameters"]["microbatch"]["value"] = micro
        row["training"]["train_batch_size"]["value"] = 4 * micro
        path = tmp_path / f"m{micro}/gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(row))
        configs.append(path)
    assert validate(configs) == [32, 40]
