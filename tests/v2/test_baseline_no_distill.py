"""Check that the pure prediction arm keeps the shared B0 model and recipe."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_relay_method import small_index

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.batch import GraDPertTrainingBatch
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import routes
from gradpert.training.v2.views import assemble_batch

ROOT = Path(__file__).resolve().parents[2] / "configs/v2"
BASELINE = ROOT / "training_speed_20261106/candidate/gradpert_v2/nadig_jurkat.yaml"
CANDIDATE = ROOT / "baseline_no_distill/gradpert_v2/nadig_jurkat.yaml"
SSL = {
    "lambda1",
    "lambda2",
    "ssl1_condition",
    "ssl1_node",
    "ssl1_spread",
    "ssl2_dino",
    "ssl2_ibot",
    "ssl2_koleo",
}


def test_recipe_and_baseline_prediction_path_are_preserved():
    old = load_experiment_config(BASELINE)
    new = load_experiment_config(CANDIDATE)
    changed = SSL | {"distillation_enabled"}
    assert old.model_dump(exclude={"model"}) == new.model_dump(exclude={"model"})
    assert {k: v for k, v in new.model.parameters.items() if k not in changed} == {
        k: v for k, v in old.model.parameters.items() if k not in changed
    }
    before, old_options = V2Options.parse_parameters(old.model.parameters)
    after, new_options = V2Options.parse_parameters(new.model.parameters)
    assert before.has_distillation and not after.has_distillation
    assert after.payload() == {**before.payload(), "distillation_enabled": False}

    torch.set_num_threads(2)
    seeds = torch.zeros(16, 2048)
    torch.manual_seed(1)
    baseline = GraDPertV2(seeds, before).eval()
    torch.manual_seed(1)
    model = GraDPertV2(seeds, after).eval()
    shared = {k: v for k, v in baseline.state_dict().items() if not k.startswith("ssl")}
    assert shared.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(shared[key], value), key
    assert [row for row in routes(baseline) if not row["name"].startswith("ssl")] == routes(model)
    assert not model.distillation_heads()

    from dataclasses import replace

    raw = GraDPertTrainingBatch(
        torch.ones(2, 16),
        torch.zeros(2, 16),
        ("g0", "g1"),
        {"g0": (0,), "g1": (1,)},
        ("t0", "t1"),
        ("c0", "c1"),
    )
    index = small_index(16)
    old_batch = assemble_batch(
        raw, index, replace(old_options, query_count=6), np.random.default_rng(1)
    )
    new_batch = assemble_batch(
        raw, index, replace(new_options, query_count=6), np.random.default_rng(1)
    )
    assert old_batch.graph_views and old_batch.cell_views
    assert not new_batch.graph_views and not new_batch.cell_views
    assert torch.equal(old_batch.query_positions, new_batch.query_positions)
    assert torch.equal(old_batch.graph.ids, new_batch.graph.ids)
    objective = JointObjective(
        model, lambda1=0, lambda2=0, ssl1_weights=(0, 0, 0), ssl2_weights=(0, 0, 0)
    )
    assert objective.teacher is None and not objective.teacher_updates_enabled
    assert not any(
        k.startswith("teacher.") or k.endswith("_center") for k in objective.state_dict()
    )
    with torch.no_grad():
        graph, condition = objective._batch_graph(model, new_batch)
        genes = graph[new_batch.query_positions]
        expected = baseline.encode_response(genes, new_batch.control, condition)["prediction"]
        actual = model.encode_response(genes, new_batch.control, condition)["prediction"]
        assert torch.equal(expected, actual)
        loss, stats = objective(new_batch)
        torch.testing.assert_close(loss, stats["prediction"])
    with pytest.raises(ValueError, match="only the prediction objective"):
        JointObjective(model, lambda1=1)
