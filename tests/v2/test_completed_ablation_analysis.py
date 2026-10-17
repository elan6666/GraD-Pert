"""Completed results and matching fixed probes precede interpretation."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/v2/analyze_completed_ablation.py"
spec = importlib.util.spec_from_file_location("analysis", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    manifest = {
        "source": {"commit": "a" * 40, "published_commit": "a" * 40, "dirty": False},
        "run_id": "synthetic",
        "config_sha256": "b" * 64,
    }
    put(root / "run_manifest.json", manifest)
    put(root / "resolved_config.json", {"training": {"max_epochs": {"value": 1}}})
    (root / "fit").mkdir()
    (root / "fit/last.pt").write_bytes(b"synthetic checkpoint")
    checkpoint = {"file": "last.pt", "sha256": module.digest(root / "fit/last.pt")}
    put(
        root / "COMPLETE.json",
        {"identity": manifest, "epoch": 1, "zero_pkl": True, "last": checkpoint},
    )
    diagnostic = {
        "process": {
            "split": "train",
            "selection_use": "none",
            "input_sha256": "c" * 64,
            "pairs": {
                "ssl1_cls": {
                    "cross_entropy": 3.0,
                    "teacher_entropy": 2.0,
                    "teacher_to_student_kl": 1.0,
                }
            },
        }
    }
    for epoch in (0, 1):
        put(root / f"fit/loss-diagnostic-epoch-{epoch:04d}.json", diagnostic)
    put(
        root / "fit/history.json",
        [
            {
                "epoch": 1,
                "optimizer_steps": 2,
                "training": {},
                "training_seconds": 2.0,
                "diagnostic_seconds": 0.1,
                "rank0_peak_allocated_bytes": 0,
                "training_diagnostic": diagnostic,
            }
        ],
    )
    put(
        root / "fit/last-test.json",
        {
            "identity": {
                "training": manifest,
                "checkpoint": checkpoint,
                "role": "last",
                "evaluation": {"source": {"commit": "a" * 40, "dirty": False}},
            },
            "result": {
                "split": "test",
                "metrics": [
                    {
                        "metric_id": "test",
                        "macro_mean": None,
                        "finite_condition_count": 0,
                        "total_condition_count": 1,
                    }
                ],
                "metric_gene_groups": {},
                "prediction_loss": 0.1,
                "expression_exposure": {},
            },
        },
    )
    return root


def collect(run):
    return module.collect(run, name="DP", analysis_identity={"commit": "d" * 40})


def test_read_only_collection_and_undefined_metric(run):
    before = {str(p): module.digest(p) for p in run.rglob("*") if p.is_file()}
    result = collect(run)
    assert result["metrics"][0]["macro_mean"] is None
    assert len(result["snapshots"]) == 2
    assert result["training_source"]["commit"] != result["analysis_identity"]["commit"]
    assert before == {str(p): module.digest(p) for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("all_ssl_disabled", [False, True])
def test_disabled_node_head_is_reported_missing_and_all_figures_render(
    run, tmp_path, all_ssl_disabled
):
    data = collect(run)
    for h in data["history"]:
        h["training"].update(joint_loss=1.0, prediction_mse=0.1)
    for point in data["snapshots"]:
        point["distributions"] = {
            "teacher." + head: dict(
                samples=4,
                mean_sample_entropy=2.0,
                sample_dependence_js=0.5,
                effective_marginal_prototypes=8.0,
                argmax_prototypes_used=3,
                largest_argmax_fraction=0.5,
            )
            for head in module.HEADS
        }
        point["distributions"]["teacher.ssl1_node"] = {"samples": 0}
        point["pairs"].update(
            {
                head: dict(cross_entropy=3.0, teacher_entropy=2.0, teacher_to_student_kl=1.0)
                for head in module.HEADS
                if head != "ssl1_node"
            }
        )
        point["pairs"].pop("ssl1_node", None)
        point["features"] = {
            "student." + feature: dict(
                samples=4, covariance_effective_rank=2.0, relative_centered_variance=0.2
            )
            for feature in module.FEATURES
            if feature != "ssl1_node.bottleneck"
        }
        point["outputs"] = dict(
            prediction_mse=0.1,
            control_copy_mse=0.2,
            mse_gain_over_copy=0.5,
            paired_effect_rms_ratio=0.8,
            negative_prediction_fraction=0.0,
            delta_rms=0.2,
            top_one_percent_loss_fraction=0.1,
        )
        point["graph_gradients"] = dict(
            groups={
                "graph": dict(
                    weighted_gradient_l2=dict(prediction=1.0, ssl1=1.0, ssl2=1.0),
                    cosine=dict(prediction_ssl1=0.2, prediction_ssl2=0.1),
                )
            }
        )
    if all_ssl_disabled:
        for point in data["snapshots"]:
            point["distributions"] = {"teacher." + head: {"samples": 0} for head in module.HEADS}
            point["pairs"] = {}
            for feature in list(point["features"]):
                if "bottleneck" in feature:
                    del point["features"][feature]
            for group in point["graph_gradients"]["groups"].values():
                group["weighted_gradient_l2"].update(ssl1=0.0, ssl2=0.0)
                group["cosine"] = {"prediction_ssl1": None, "prediction_ssl2": None}
    output = tmp_path / "node0-report"
    module.export(data, output)
    text = (output / "report.md").read_text()
    assert "该头未调用" in text and "无有效配对" in text
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.csv"))) == 7
    assert "ssl1_node" not in {
        r["name"] for r in __import__("csv").DictReader((output / "pairs.csv").open())
    }
    if all_ssl_disabled:
        assert "无有效梯度" in text
        saved = json.loads((output / "summary.json").read_text())
        assert (
            saved["snapshots"][-1]["graph_gradients"]["groups"]["graph"]["cosine"][
                "prediction_ssl1"
            ]
            is None
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ("partial", "completed"),
        ("checkpoint", "checkpoint content"),
        ("probe", "input/view"),
        ("ce", "CE decomposition"),
        ("test", "test/checkpoint"),
        ("pkl", "zero PKL"),
    ],
)
def test_bad_evidence_rejected(run, change, match):
    if change == "checkpoint":
        (run / "fit/last.pt").write_bytes(b"changed")
    elif change == "pkl":
        (run / "stale.pkl").write_bytes(b"x")
    else:
        path = (
            run
            / {
                "partial": "COMPLETE.json",
                "probe": "fit/loss-diagnostic-epoch-0000.json",
                "ce": "fit/loss-diagnostic-epoch-0000.json",
                "test": "fit/last-test.json",
            }[change]
        )
        value = module.read(path)
        if change == "partial":
            value["epoch"] = 0
        elif change == "probe":
            value["process"]["input_sha256"] = "e" * 64
        elif change == "ce":
            value["process"]["pairs"]["ssl1_cls"]["teacher_to_student_kl"] = 0.0
        elif change == "test":
            value["identity"]["role"] = "best"
        put(path, value)
    with pytest.raises(ValueError, match=match):
        collect(run)


@pytest.fixture
def staged_run(run):
    import copy

    put(
        run / "resolved_config.json",
        {
            "training": {"max_epochs": {"value": 4}},
            "model": {
                "parameters": {
                    "training_mode": {"value": "ssl_then_mse"},
                    "ssl_pretrain_epochs": {"value": 2},
                }
            },
        },
    )
    complete = module.read(run / "COMPLETE.json")
    complete["epoch"] = 4
    put(run / "COMPLETE.json", complete)
    template = module.read(run / "fit/loss-diagnostic-epoch-0000.json")
    history = []
    for epoch in range(5):
        d = copy.deepcopy(template)
        p = d["process"]
        stage = "ssl_pretrain" if epoch <= 2 else "mse_finetune"
        p.update(
            training_stage=stage,
            input_sha256=("c" if epoch <= 2 else "e") * 64,
            cells=4,
            query_genes=1000,
            seed=9,
            budget={"feature_samples": 128},
        )
        p["distributions"] = {"teacher." + h: dict(samples=0) for h in module.HEADS}
        p["features"] = {
            "student." + f: dict(
                samples=4, covariance_effective_rank=2.0, relative_centered_variance=0.2
            )
            for f in module.FEATURES
        }
        p["graph_gradients"] = dict(
            groups={
                "graph": dict(
                    weighted_gradient_l2=(
                        dict(ssl1=1, ssl2=1) if epoch <= 2 else dict(prediction=1, ssl1=0, ssl2=0)
                    ),
                    cosine={} if epoch <= 2 else dict(prediction_ssl1=None, prediction_ssl2=None),
                )
            }
        )
        p["outputs"] = dict(status="not_applicable:ssl_pretrain_no_prediction_labels")
        if epoch >= 3:
            p["pairs"] = {}
            p["outputs"] = dict(
                prediction_mse=0.1,
                control_copy_mse=0.2,
                mse_gain_over_copy=0.5,
                paired_effect_rms_ratio=0.8,
                negative_prediction_fraction=0.0,
                delta_rms=0.2,
                top_one_percent_loss_fraction=0.1,
            )
        put(run / f"fit/loss-diagnostic-epoch-{epoch:04d}.json", d)
        if epoch:
            history.append(
                dict(
                    epoch=epoch,
                    optimizer_steps=epoch * 2,
                    training_stage=stage,
                    training=dict(joint_loss=3, prediction_mse=0 if epoch <= 2 else 0.1),
                    training_seconds=2,
                    diagnostic_seconds=0.1,
                    rank0_peak_allocated_bytes=0,
                    training_diagnostic=d,
                )
            )
    put(run / "fit/history.json", history)
    return run


def test_staged_probes_keep_separate_identities_and_render_missing_prediction(staged_run, tmp_path):
    data = collect(staged_run)
    assert data["fixed_input_sha256"] is None
    groups = data["fixed_inputs_by_training_stage"]
    assert [g["epochs"] for g in groups] == [[0, 1, 2], [3, 4]]
    assert [g["input_sha256"] for g in groups] == ["c" * 64, "e" * 64]
    x, y = module.trajectory(data["snapshots"], [1, 2, 3, 4, 5])
    assert len(x) == len(y) == 6 and y[3] is None
    output = tmp_path / "staged-report"
    module.export(data, output)
    text = (output / "report.md").read_text()
    assert "跨阶段不是同一输入" in text and "| 1 | 不适用" in text
    assert "epoch 3→4" in text
    assert len(list(output.glob("*.png"))) == 4
    import csv

    gradients = list(csv.DictReader((output / "gradients.csv").open()))
    assert gradients[0]["prediction"] == ""
    assert gradients[0]["prediction_ssl1"] == ""
    assert gradients[-1]["prediction"] == "1"
    assert gradients[-1]["ssl1"] == "0"
    stages = [r["training_stage"] for r in csv.DictReader((output / "training.csv").open())]
    assert stages == ["ssl_pretrain"] * 2 + ["mse_finetune"] * 2


@pytest.mark.parametrize(
    "change,match",
    [
        ("hash", "within training stage"),
        ("stage", "sealed epoch boundary"),
        ("budget", "size/seed/budget"),
    ],
)
def test_stage_support_does_not_hide_identity_drift(staged_run, change, match):
    path = staged_run / "fit/loss-diagnostic-epoch-0004.json"
    d = module.read(path)
    if change == "hash":
        d["process"]["input_sha256"] = "f" * 64
    elif change == "stage":
        d["process"]["training_stage"] = "ssl_pretrain"
    else:
        d["process"]["budget"] = {"feature_samples": 1}
    put(path, d)
    history = module.read(staged_run / "fit/history.json")
    history[-1]["training_diagnostic"] = d
    put(staged_run / "fit/history.json", history)
    with pytest.raises(ValueError, match=match):
        collect(staged_run)
