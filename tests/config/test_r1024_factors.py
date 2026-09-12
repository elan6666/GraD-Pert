from pathlib import Path

import pytest

from gradpert.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
ROWS = ["s1", "s2", "u1", "t1", "t2", "p1", "p2", "c1", "l1", "l2", "l3"]


@pytest.mark.parametrize("row", ROWS)
def test_r1024_resolved_contract(row):
    cfg = load_experiment_config(ROOT / f"configs/r50/r1024_{row}/gradpert_b2/nadig_jurkat.yaml")
    base = load_experiment_config(ROOT / "configs/r50/batch1024/gradpert_b2/nadig_jurkat.yaml")
    assert cfg.training.train_batch_size.value == 1024
    assert cfg.training.max_epochs.value == 50
    assert cfg.training.eval_batch_size == base.training.eval_batch_size
    assert cfg.model.parameters["gene_feature_mode"] == base.model.parameters["gene_feature_mode"]
    expected = {
        "s1": set(),
        "s2": set(),
        "u1": set(),
        "t1": {"teacher_ema_start"},
        "t2": {"teacher_ema_start"},
        "p1": {"projector_hidden_dim"},
        "p2": {"graph_tower_layers"},
        "c1": {"prediction_reduction"},
        "l1": {"local_view_builder"},
        "l2": {"local_node_policy", "essential_gene_list_path", "essential_gene_list_sha256"},
        "l3": {"local_node_policy", "essential_gene_list_path", "essential_gene_list_sha256"},
    }
    different = {
        k
        for k, v in cfg.model.parameters.items()
        if k not in base.model.parameters or v.value != base.model.parameters[k].value
    }
    assert different == expected[row]
