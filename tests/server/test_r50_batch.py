import argparse
import copy
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.execution.system_resources import shared_gpu_budget
from scripts.server.run_r50_selection import command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("batch", [128, 512])
def test_only_training_batch_and_artifact_label_change(batch):
    path = ROOT / f"configs/r50/batch{batch}/gradpert_b2/nadig_jurkat.yaml"
    config = load_experiment_config(path)
    assert config.training.learning_rate.value == 0.001
    assert config.training.train_batch_size.value == batch
    assert config.training.eval_batch_size.value == 256
    assert config.training.max_epochs.value == 50
    assert not config.training.early_stopping
    original = yaml.safe_load((ROOT / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml").read_text())
    observed = yaml.safe_load(path.read_text())
    normalized = copy.deepcopy(observed)
    normalized["training"]["train_batch_size"] = original["training"]["train_batch_size"]
    normalized["artifacts"]["root"] = original["artifacts"]["root"]
    assert normalized == original


def test_memory_cap_independent_of_peer_initialization():
    gib = 1024**3
    assert shared_gpu_budget(32 * gib, 32 * gib, 4 * gib, 0.4) == int(12.8 * gib)
    assert shared_gpu_budget(20 * gib, 32 * gib, 4 * gib, 0.4) == int(12.8 * gib)
    with pytest.raises(RuntimeError):
        shared_gpu_budget(4 * gib, 32 * gib, 4 * gib, 0.4)
    for fraction in (float("nan"), -1, 2):
        with pytest.raises(ValueError):
            shared_gpu_budget(32 * gib, 32 * gib, 4 * gib, fraction)


def test_runner_binds_training_cap(tmp_path):
    args = argparse.Namespace(
        source=ROOT,
        row="batch512",
        root=tmp_path,
        data_root=tmp_path,
        publication=tmp_path / "p",
        publication_sha="a" * 64,
        genept_receipt=tmp_path / "g",
        genept_sha="b" * 64,
        memory_fraction=0.4,
    )
    argv = command(args, "full")
    assert argv[argv.index("--memory-fraction") + 1] == "0.4"
