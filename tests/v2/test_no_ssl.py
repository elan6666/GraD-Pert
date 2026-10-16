"""Prediction-only B0 identity, exact queue scope and resumable update semantics."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.data._io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_no_ssl as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from generate_fgpo_family import configs as anchors  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


def test_d0_changes_only_two_loss_coefficients():
    manifest = family.verify_manifest(ROOT, ROOT / "configs/v2/no_ssl_d0_20261016/manifest.json")
    assert manifest["priority"] == ["D0"]
    assert (manifest["epochs"], manifest["global_batch"], manifest["preflight_updates"]) == (
        4,
        320,
        10,
    )
    actual = family.configs(ROOT)["D0"]
    parent = anchors(ROOT, microbatch=80)["B0"]
    for key in ("lambda1", "lambda2"):
        assert actual["model"]["parameters"][key]["value"] == 0
        actual["model"]["parameters"][key] = parent["model"]["parameters"][key]
    assert actual == parent
    config = load_experiment_config(ROOT / manifest["rows"][0]["config"])
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert options.lambda1 == options.lambda2 == 0
    assert options.microbatch == 80 and options.accumulation == options.world_size == 2
    assert arch.prediction_parameterization == "softplus" and not arch.prototype_weight_norm


@pytest.mark.parametrize("mutation", ["extra_arm", "loss", "batch"])
def test_reject_modified_scope_or_recipe(mutation):
    # Remain source-contained; no tracked fixture files are modified.
    import tempfile

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory) / "family"
        record = family.generate(ROOT, output)
        if mutation == "extra_arm":
            record["rows"].append(copy.deepcopy(record["rows"][0]))
        elif mutation == "loss":
            record["rows"][0]["changes"]["lambda1"] = 1
        else:
            record["global_batch"] = 400
        target = output / "manifest.json"
        atomic_json(target, record)
        with pytest.raises(ValueError):
            family.verify_manifest(ROOT, target)


def test_d0_keeps_last_medium_and_exact_evaluation_bindings(new_queue):
    old, oldout = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / "configs/v2/no_ssl_d0_20261016/manifest.json",
        Path(old["runtime"]),
        oldout.parent / "no-ssl",
        copy.deepcopy(old["shared_gpu_policy"]),
    )
    runner.verify(q)
    assert q["family"] in runner.LAST_ONLY_FAMILIES
    assert list(medium.FAMILY_ARMS[q["family"]]) == ["D0"]
    assert "D0" in medium.AUTHORIZED_ARMS
    assert q["medium_analysis"] == old["medium_analysis"]
    assert len(q["eval_bindings"]) == 2 and "official_dependency" not in q
    row = q["rows"][0]
    for binding in q["eval_bindings"]:
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=128,
            reference_batch=128,
            config=Path(row["plan"]["config"]),
            publication=Path(row["plan"]["publication"]),
            publication_sha256=row["plan"]["publication_sha256"],
            data_root=Path(row["plan"]["data_root"]),
            output=Path(binding["output"]),
            shared_queue=Path(q["output"]) / "queue.json",
            row_name="D0",
        )
        assert validate_eval_binding(q, args, q["source_commit"]) == row
        for change in ({"mode": "wrong"}, {"steps": 128}, {"rows": q["rows"] * 2}):
            with pytest.raises(ValueError):
                validate_eval_binding({**q, **change}, args, q["source_commit"])


def no_ssl_fixture():
    pytest.importorskip("torch")
    from dataclasses import replace

    from test_fgpo_family import new_fixture

    objective, batch = new_fixture("B0")
    objective.lambda1 = objective.lambda2 = 0.0
    # assemble_batch omits both view collections when their coefficients are zero.
    return objective, replace(batch, graph_views=(), cell_views=())


def test_prediction_only_forwards_and_full_state_updates_restore(tmp_path, monkeypatch):
    objective, batch = no_ssl_fixture()
    import numpy as np
    import torch

    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.engine import optimizer_step
    from gradpert.training.v2.optimizer import V2Optimizer

    def inactive(*args, **kwargs):
        raise AssertionError("disabled distillation must not execute")

    monkeypatch.setattr(objective, "graph_loss", inactive)
    monkeypatch.setattr(objective, "cell_loss", inactive)
    optimizer = V2Optimizer(objective.student, 1e-3, 0.01)
    initial = copy.deepcopy(objective.student.state_dict())
    heads = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
    for _ in range(3):
        teacher = {n: p.detach().clone() for n, p in objective.teacher.named_parameters()}
        result = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert result["joint_loss"] == pytest.approx(result["prediction"])
        assert not any(k.startswith(("ssl1_", "ssl2_")) for k in result)
        assert not objective.pending
        for head in heads:
            assert getattr(objective, head + "_center").count_nonzero() == 0
            for n, p in getattr(objective.student, head).named_parameters():
                assert p.grad is None
                torch.testing.assert_close(p, initial[head + "." + n], rtol=0, atol=0)
        for n, p in objective.teacher.named_parameters():
            torch.testing.assert_close(
                p, teacher[n].lerp(dict(objective.student.named_parameters())[n], 0.1)
            )
    assert any(
        not torch.equal(v, initial[n])
        for n, v in objective.student.state_dict().items()
        if not n.startswith(heads)
    )
    path = tmp_path / "last.pt"
    rng = np.random.default_rng(211)
    save_checkpoint(
        path, objective, optimizer, identity={"arm": "D0"}, progress={"step": 3}, generator=rng
    )
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    state = copy.deepcopy(objective.state_dict())
    other, _ = no_ssl_fixture()
    other_optimizer = V2Optimizer(other.student, 1e-3, 0.01)
    load_checkpoint(path, other, other_optimizer, identity={"arm": "D0"}, generator=rng)
    actual = optimizer_step(
        other, other_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for n, v in state.items():
        torch.testing.assert_close(v, other.state_dict()[n], rtol=0, atol=0)


def test_disabled_heads_report_absent_not_collapsed():
    objective, batch = no_ssl_fixture()
    import torch

    from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
    from gradpert.training.v2.process_diagnostics import evaluate_process_diagnostics

    before = copy.deepcopy(objective.state_dict())
    weights = objective.local_reduction_weights(batch)
    assert weights[1].shape == (0, len(batch.control))
    basal = evaluate_loss_diagnostics(objective, batch, bf16=False)
    assert basal["shared_basal_weighted_gradient_l2"]["ssl"] == 0
    assert basal["prediction_ssl_gradient_cosine"] is None
    result = evaluate_process_diagnostics(objective, batch, bf16=False, seed=31)
    assert result["outputs"]["prediction_mse"] >= 0
    assert result["features"]["student.graph.output"]["samples"] > 0
    assert all(v["samples"] == 0 for v in result["distributions"].values())
    for n, v in before.items():
        torch.testing.assert_close(v, objective.state_dict()[n], rtol=0, atol=0)
