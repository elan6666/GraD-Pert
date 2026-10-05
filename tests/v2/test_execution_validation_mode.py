"""Loss-only epochs skip population inference; terminal best/last tests remain."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.data._io import atomic_json
from gradpert.execution import v2 as execution
from gradpert.hashing import sha256_file

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v2/mhc_joint_only_jurkat/three_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml"
)


@pytest.mark.parametrize("mode", ["joint_only", "joint_and_prediction", "disabled"])
def test_formal_orchestration_keeps_test_evaluation_after_optional_validation(
    tmp_path, monkeypatch, mode
):
    from dataclasses import replace

    from gradpert.evaluation import data as evaluation_data
    from gradpert.evaluation import state
    from gradpert.training.v2 import evaluation, exposure, joint_validation, lifecycle, reporting
    from gradpert.training.v2 import runtime as runtime_module

    config = load_experiment_config(CONFIG)
    _, options = V2Options.parse_parameters(config.model.parameters)
    runtime = NS(
        options=replace(options, validation_mode=mode),
        identity={"run_seed": 1},
        steps_per_epoch=1,
        optimizer=NS(steps=1),
        generator=None,
        objective=NS(student="student"),
        index="graph-index",
        allowed_expression_ids=None,
        data=NS(validation_steps_per_epoch=lambda **_: 1),
        validation_batches=lambda **_: [],
        batches=lambda _: [],
    )
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(execution, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(execution, "load_experiment_config", lambda _: config)
    source = NS(commit="a" * 40, payload=lambda: {"commit": "a" * 40})
    monkeypatch.setattr(execution, "inspect_source_identity", lambda *_, **__: source)
    monkeypatch.setattr(execution, "inspect_environment", lambda *_, **__: NS(payload=lambda: {}))
    monkeypatch.setattr(runtime_module, "prepare_runtime", lambda *_, **__: nullcontext(runtime))
    calls = []

    def prepare(**kwargs):
        calls.append(("prepare", "val" if kwargs.get("validation_only") else "test"))

    monkeypatch.setattr(state, "prepare_evaluation_state", prepare)
    monkeypatch.setattr(state, "load_evaluation_state", lambda **_: "reference")

    class Data:
        def __init__(self, **kwargs):
            calls.append(("data", kwargs["split_name"]))
            self.expression_gene_ids = ("a", "b")
            self.control_manifest = NS(draws=["p"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(evaluation_data, "CanonicalEvaluationData", Data)

    def evaluate(*args, **kwargs):
        split = kwargs["expected_split"]
        calls.append(("prediction", split))
        return {
            "split": split,
            "prediction_loss": 0.2,
            "metrics": [],
            "conditions": [],
            "query_recipe": {},
            "control_manifest_sha256": "controls",
            "reference_sha256": "reference",
        }

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(
        joint_validation,
        "evaluate_joint_loss",
        lambda *_, **__: {"joint_loss": 0.1, "batch_count": 1, "components": {}},
    )

    def fit(*args, **kwargs):
        if mode == "disabled":
            assert kwargs["validate"] is None
            validation = {"validation_mode": "disabled", "performed": False}
        else:
            validation = kwargs["validate"]()
            assert validation["validation_mode"] == mode
            assert validation["joint_loss"] == 0.1
        atomic_json(kwargs["root"] / "history.json", [{"validation": validation}])
        calls.append(("fit", "complete"))
        return {
            "epoch": 3,
            "best": None if mode == "disabled" else {"epoch": 2},
            "last": {"epoch": 3},
        }

    monkeypatch.setattr(lifecycle, "fit", fit)
    monkeypatch.setattr(reporting, "export_curves", lambda _: None)
    monkeypatch.setattr(exposure, "checkpoint_expression_groups", lambda *_, **__: ({}, {}))

    def test_selected(*args, **kwargs):
        results = {}
        for role in ("last",) if mode == "disabled" else ("best", "last"):
            kwargs["on_role_start"](role)
            results[role] = kwargs["test"]()
        return results

    monkeypatch.setattr(lifecycle, "test_selected", test_selected)
    plan = {
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "repository_root": str(Path(execution.__file__).resolve().parents[3]),
        "run_root": str(tmp_path / "run"),
        "data_root": str(tmp_path),
        "publication": "synthetic",
        "publication_sha256": "b" * 64,
        "source_commit": source.commit,
        "seed": 1,
        "run_id": "synthetic",
    }
    complete = execution._run_v2(plan)
    assert complete["test_roles"] == (["last"] if mode == "disabled" else ["best", "last"])
    assert calls.count(("prediction", "test")) == (1 if mode == "disabled" else 2)
    assert calls.index(("fit", "complete")) < calls.index(("prepare", "test"))
    for action in ("prepare", "data", "prediction"):
        assert ((action, "val") in calls) is (mode == "joint_and_prediction")
