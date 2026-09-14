from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.txpert import runner
from gradpert.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def _frozen_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/config-exphormer-mg.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": {},
                "graph": {},
                "datamodule": {
                    "batch_size": 64,
                    "match_cntr": True,
                    "avg_cntr": True,
                    "obsm_key": "raw",
                },
            }
        )
    )
    return path


def test_pending_txpert_restores_official_batch64(tmp_path, monkeypatch) -> None:
    original = load_experiment_config(ROOT / "configs/r50-rerun/txpert_public/nadig_jurkat.yaml")
    override = load_experiment_config(
        ROOT / "configs/r50-r1024-finish/txpert_public/nadig_jurkat.yaml"
    )
    path = _frozen_config(tmp_path)
    official_hash = original.model.parameters["official_config_sha256"].value
    monkeypatch.setattr(runner, "_sha256_file", lambda _: official_hash)
    assert runner._official_config(original, tmp_path)[0] == path
    assert runner._official_config(override, tmp_path)[0] == path
    assert override.training.train_batch_size.value == 64
    assert override.training.train_batch_size.source == "official"
    assert "official_train_batch_override" not in override.model.parameters

    missing_declaration = SimpleNamespace(
        model=SimpleNamespace(
            parameters={
                key: value
                for key, value in override.model.parameters.items()
                if key != "official_train_batch_override"
            }
        ),
        training=override.training,
    )
    assert runner._official_config(missing_declaration, tmp_path)[0] == path

    wrong_batch = SimpleNamespace(
        model=override.model,
        training=SimpleNamespace(train_batch_size=SimpleNamespace(value=512, source="user_locked")),
    )
    with pytest.raises(ValueError, match="frozen official datamodule"):
        runner._official_config(wrong_batch, tmp_path)
