import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_components import fixture
from test_evaluation import graph_index

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file
from gradpert.training.batch import GraDPertTrainingBatch
from gradpert.training.v2.holdout import load_partition, make_partition
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.views import assemble_batch


def test_partition_is_reproducible_and_rejects_axis_or_manifest_mutation(tmp_path):
    genes = tuple(f"g{i}" for i in range(10))
    data = make_partition(genes, heldout_count=3, seed=1)
    path = tmp_path / "partition.json"
    path.write_text(json.dumps(data))
    digest = sha256_file(path)
    indices = load_partition(path, digest, genes)
    assert len(indices) == 7
    assert not set(genes[i] for i in indices) & set(data["heldout_gene_ids"])
    with pytest.raises(ValueError, match="axis"):
        load_partition(path, digest, genes[::-1])
    data["training_gene_ids"][0] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_partition(path, digest, genes)
    with pytest.raises(ValueError, match="recipe"):
        load_partition(path, sha256_file(path), genes)


def test_heldout_expression_cannot_change_prediction_or_either_distillation_gradient():
    config = load_experiment_config(
        Path(__file__).resolve().parents[2] / "configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    options = replace(options, query_count=3, local_views=2)
    model, _ = fixture()
    control = torch.randn(2, 4)
    # The heldout gene remains a perturbation identity in the graph, but its
    # expression cannot reach either student/teacher or the prediction target.
    raw = GraDPertTrainingBatch(
        control, control + 1, ("p", "p"), {"p": (3,)}, ("t0", "t1"), ("c0", "c1")
    )
    changed_control, changed_truth = raw.control_expression.clone(), raw.target_expression.clone()
    changed_control[:, 3] = 1e7
    changed_truth[:, 3] = -1e7
    changed = replace(raw, control_expression=changed_control, target_expression=changed_truth)
    reference = None
    for source in (raw, changed):
        batch = assemble_batch(
            source,
            graph_index(),
            options,
            np.random.default_rng(7),
            allowed_expression_ids=np.array([0, 1, 2]),
        )
        objective = JointObjective(copy.deepcopy(model))
        loss, _ = objective(batch)
        loss.backward()
        actual = (loss.detach(), {n: p.grad for n, p in objective.student.named_parameters()})
        if reference is None:
            reference = actual
        else:
            torch.testing.assert_close(actual[0], reference[0])
            for name, gradient in actual[1].items():
                if gradient is None:
                    assert reference[1][name] is None
                else:
                    torch.testing.assert_close(gradient, reference[1][name])
