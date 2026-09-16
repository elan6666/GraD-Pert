import json
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.native import NativeArchitectureOptions
from scripts.server.run_r1024_full import NEW_ROWS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("row", NEW_ROWS)
def test_self_contained_factors(row):
    path = ROOT / f"configs/r50/{row}/gradpert_b2/nadig_jurkat.yaml"
    cfg = load_experiment_config(path)
    assert cfg.training.monitor == "val/prediction_loss"
    assert cfg.training.monitor_mode == "min"
    assert cfg.training.max_epochs.value == 50
    assert not cfg.training.early_stopping
    payload = json.loads(path.read_text())
    NativeArchitectureOptions.from_parameters(
        {k: v["value"] for k, v in payload["model"]["parameters"].items()}
    )
    base = yaml.safe_load(
        (ROOT / "configs/r50/r1024_ref_repeat/gradpert_b2/nadig_jurkat.yaml").read_text()
    )
    expected = {
        "r50n_t1_u1": {"teacher_ema_start", "optimizer"},
        "r50n_batch2048": {"train_batch_size"},
        "r50n_k32768": {"prototype_count"},
        "r50n_k8192": {"prototype_count"},
        "r50n_string": {"graph_sources", "graph_encoder_family"},
        "r50n_source_gat": {"graph_encoder_family", "graph_encoder_dropout"},
        "r50n_gat_mlg": {"graph_encoder_family"},
        "r50n_hybrid_bmp": {"graph_encoder_family"},
        "r50n_prediction_only": {
            "condition_consistency_loss_weight",
            "masked_node_loss_weight",
            "spread_loss_weight",
        },
    }[row]
    changed = set()
    for section in ("model", "training"):
        old = base[section]["parameters"] if section == "model" else base[section]
        new = payload[section]["parameters"] if section == "model" else payload[section]
        for key in old:
            if key in {"monitor", "monitor_mode"}:
                continue
            if old[key] != new[key]:
                changed.add(key)
    assert changed == expected
