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


def test_disabled_node_head_is_reported_missing_and_all_figures_render(run, tmp_path):
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
    output = tmp_path / "node0-report"
    module.export(data, output)
    text = (output / "report.md").read_text()
    assert "该头未调用" in text and "无有效配对" in text
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.csv"))) == 7
    assert "ssl1_node" not in {
        r["name"] for r in __import__("csv").DictReader((output / "pairs.csv").open())
    }


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
