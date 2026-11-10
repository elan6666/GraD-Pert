"""Package one completed training, validation and CPU analysis for its existing platform run."""

import json
import math
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from small_softmax_notebook_assets import notebook_assets

eid, source_name, dest_name, platform, expected_updates, base = sys.argv[1:]
workspace = Path(platform)
sys.path.insert(0, str(workspace / "sdk"))
from experiment_workspace import ExperimentPackage  # noqa: E402

source, dest = Path(source_name), Path(dest_name)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def get(route):
    with urlopen(base + route, timeout=20) as response:
        return response.read()


receipt_file = source / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json"
analysis_dir = source / "analysis"
report_file = source / "REPORT_CN.md"
val = read(receipt_file)
analysis = read(analysis_dir / "summary.json")
cpu = read(analysis_dir / "receipt.json")
train_path = source / "TRAIN_COMPLETE.json"
train = read(train_path)
assert val["status"] == "main_independently_accepted_frozen_expression_validation"
assert (
    val["validation_conditions"] == 445
    and val["validation_split"] == "val"
    and val["whole_family_zero_pkl"]
)
assert val["training_sha256"] == digest(train_path) == cpu["input_hashes"]["TRAIN_COMPLETE.json"]
assert val["evaluation_sha256"] in cpu["input_hashes"].values()
assert (
    cpu["status"] == "completed"
    and cpu["model_loads"] == cpu["optimizer_updates"] == 0
    and not cpu["gpu_work"]
)
assert val["run_id"] == train["identity"]["run_id"] == cpu["run_id"] == analysis["run_id"]
assert (
    train["epoch"] == 2
    and val["checkpoint_sha256"] == train["last"]["sha256"] == analysis["checkpoint_sha256"]
)
assert (
    val["training_commit"] == train["identity"]["source"]["commit"]
    and val["config_sha256"] == train["identity"]["config_sha256"]
)
for name, meta in cpu["outputs"].items():
    path = analysis_dir / name
    assert (
        path.is_file() and path.stat().st_size == meta["bytes"] and digest(path) == meta["sha256"]
    )
assert [x["epoch"] for x in val["training_history"]] == [1, 2]
assert len(val["comparison"]) == 9

live = json.loads(get("/api/export?project=grad-pert&id=" + eid))
(e,) = live["experiments"]
plan = read(source / "PLAN.json")
assert plan["run_id"] == val["run_id"] and plan["config_sha256"] == val["config_sha256"]
assert plan["source_commit"] == val["training_commit"]
assert [row["optimizer_steps"] for row in val["training_history"]] == [
    int(expected_updates) // 2,
    int(expected_updates),
]
if not e["runs"]:
    assert e["config"]["status"] == "planned"
    r = ExperimentPackage.run(
        e, val["run_id"], phase="prediction2epoch", phase_name="两轮表达预测训练"
    )
    r.update(source_commit=val["training_commit"], config_sha256=val["config_sha256"])
    r["phases"][0].update(active_losses=["prediction_mse"], checkpoint_role="last")
    raw = plan["resolved_config"]
    values = {
        key: value.get("value", value) if isinstance(value, dict) else value
        for key, value in {**raw["model"]["parameters"], **raw["training"]}.items()
    }
    e["config"]["values"].update(values, planned_updates=int(expected_updates))
    e["config"].update(
        status="verified", raw=raw, source=str(source / "PLAN.json"), sha256=val["config_sha256"]
    )
else:
    (r,) = e["runs"]
assert live["schema_version"] == "1.0" and e["id"] == eid and r["id"] == val["run_id"]
assert r["source_commit"] == val["training_commit"] and r["config_sha256"] == val["config_sha256"]
assert r["statuses"]["training"] in ("running", "completed") and r["statuses"]["analysis"] in (
    "pending",
    "completed",
)
expected_steps = e["config"]["values"]["planned_updates"]
assert expected_steps == int(expected_updates) and expected_steps > 0 and expected_steps % 2 == 0
assert [x["optimizer_steps"] for x in val["training_history"]] == [
    expected_steps // 2,
    expected_steps,
]
pkg = ExperimentPackage("grad-pert", live["project"]["name"], "placeholder")
pkg.package = {
    "schema_version": "1.0",
    "project": {"id": "grad-pert", "name": live["project"]["name"]},
    "experiments": [e],
}
notebook_artifacts = [
    pkg.artifact(path, artifact_id=id_, name=title, kind=kind)
    for id_, (path, _, title, kind) in notebook_assets(source, r["id"]).items()
]
completed = r["statuses"]["analysis"] == "completed"
existing_assets = list(r["artifacts"])
if completed:
    (standard,) = r["analyses"]
    assert standard["id"] == "frozen-expression-and-training"
    assert r["checkpoint_sha256"] == standard["checkpoint_sha256"] == val["checkpoint_sha256"]
    standard["artifacts"] = [
        a for a in standard["artifacts"] if not a["id"].startswith("notebook-")
    ]
    existing_assets.extend(standard["artifacts"])
for a in existing_assets:
    if a.get("file"):
        data = get("/api/asset/" + a["file"])
        assert sha256(data).hexdigest() == a["sha256"]
        pkg.assets[a["file"]] = data
if completed:
    standard["artifacts"].extend(notebook_artifacts)
    e["updated_at"] = datetime.now(timezone.utc).isoformat()
    print(pkg.write(dest))
    sys.exit(0)

(phase,) = r["phases"]
phase["epochs"] = [1, 2]
history = {}
for idx, x in enumerate(val["training_history"]):
    t = x["training"]
    keys = (
        "prediction_mse",
        "control_copy_mse",
        "joint_loss",
        "gradient_norm",
        "gradient_clipped",
        "learning_rate",
    )
    values = {k: t[k] for k in keys}
    values.update(
        training_seconds=x["training_seconds"], peak_allocated_bytes=x["rank0_peak_allocated_bytes"]
    )
    history[x["epoch"]] = dict(
        phase=phase["id"],
        epoch=x["epoch"],
        step=x["optimizer_steps"],
        values=values,
        statuses={},
        source=str(receipt_file) + f"#/training_history/{idx}",
    )
for h in r["history"]:
    previous = history[h["epoch"]]
    aliases = {"rank0_peak_allocated_bytes": "peak_allocated_bytes"}
    assert h["step"] == previous["step"] and all(
        previous["values"][aliases.get(k, k)] == v for k, v in h["values"].items()
    )
    history[h["epoch"]] = h
r["history"] = list(history.values())
stamp = datetime.now(timezone.utc).isoformat()
r.update(
    statuses=dict(training="completed", evaluation="completed", analysis="completed"),
    observed_at=stamp,
    snapshot_note=(
        "两轮训练、445 条件冻结表达验证和 CPU 诊断均已验收。评估范围为 val / expression only。"
    ),
    checkpoint_sha256=val["checkpoint_sha256"],
    evaluation_source_commit=val["eval_commit"],
    analysis_source_commit=cpu["analysis_identity"]["commit"],
    checkpoints=[dict(role="last", epoch=2, sha256=val["checkpoint_sha256"])],
)
e["updated_at"] = stamp
e["config"]["values"].update(
    scientific_complete=False, actual_epochs=2, successful_updates=history[2]["step"]
)
group = {"all_expression": "all", "seen_expression": "seen", "unseen_expression": "unseen"}
for i, m in enumerate(val["comparison"]):
    value = m["candidate"]["macro_mean"]
    n = m["candidate"]["valid"]
    assert 0 <= n <= 445 and (value is None or math.isfinite(value))
    rec = dict(
        id=m["metric"],
        value=value,
        status="completed" if value is not None else "not_applicable",
        direction="max",
        split="val",
        gene_group=group[m["scope"]],
        gene_set="all",
        gene_count=m["genes"],
        aggregation="condition_macro_mean",
        definition_version="gradpert-v2-frozen-expression-v1",
        checkpoint_role="last",
        checkpoint_epoch=2,
        valid_count=n,
        total_count=445,
        source=str(receipt_file) + f"#/comparison/{i}/candidate",
        source_sha256=digest(receipt_file),
        protocol_id="training-speed-jurkat-val445-controls300-expression-v1",
    )
    if value is None:
        rec["reason"] = json.dumps(m["candidate"]["unavailable"], ensure_ascii=False)
    r["metrics"].append(rec)
artifacts = []
files = [
    (report_file, "comparison-report", "冻结验证与训练对比", "report"),
    (train_path, "training-terminal", "训练终态收据", "report"),
    (receipt_file, "validation-terminal", "独立冻结表达验证收据", "report"),
    (analysis_dir / "receipt.json", "cpu-terminal", "CPU 诊断收据", "report"),
    (analysis_dir / "report.md", "cpu-diagnostics", "详细训练诊断", "report"),
]
for filename, title in [
    ("01_losses.png", "训练损失"),
    ("02_prototypes.png", "原型诊断可用性"),
    ("03_representations.png", "表征方差与有效秩"),
    ("04_outputs_gradients.png", "输出偏差与共享梯度"),
    ("features.csv", "表征统计"),
    ("gradients.csv", "梯度明细"),
    ("outputs.csv", "预测输出诊断"),
    ("training.csv", "训练历史"),
    ("metrics.csv", "冻结表达 Pearson"),
]:
    files.append(
        (
            analysis_dir / filename,
            "cpu-" + Path(filename).stem.replace("_", "-"),
            title,
            "image" if filename.endswith(".png") else "table",
        )
    )
for path, id_, title, kind in files:
    artifacts.append(pkg.artifact(path, artifact_id=id_, name=title, kind=kind))
r["artifacts"] += artifacts[:4]
diagnostic_assets = []
for item in artifacts:
    path = Path(item["source"])
    if path.name == "01_losses.png":
        r["artifacts"].append(item)
        continue
    if item["type"] == "image":
        item = {**item, "name": f"{path.stem} · {item['name']}"}
    diagnostic_assets.append(item)
last = analysis["snapshots"][-1]
assert last["epoch"] == 2
features = []
for name, row in last["features"].items():
    features.append(
        {
            **row,
            "name": name,
            "rows": row["samples"],
            "width": row["analyzed_dimensions"],
            "effective_rank": row["covariance_effective_rank"],
            "normalized_rank": None,
            "relative_variance": row["relative_centered_variance"],
        }
    )
assert features
r["analyses"].append(
    dict(
        id="frozen-expression-and-training",
        title="冻结表达验证及训练诊断",
        status="completed",
        scope="445 条件冻结表达验证及初始化和 epoch 1/2 固定训练探针",
        checkpoint_role="last",
        checkpoint_epoch=2,
        source_commit=r["analysis_source_commit"],
        checkpoint_sha256=r["checkpoint_sha256"],
        manifest_sha256=analysis["fixed_input_sha256"],
        summary=dict(
            validation_prediction_mse=val["candidate_mse"],
            frozen_conditions=445,
            final_step=history[2]["step"],
            conditions=None,
            rows=last["cells"],
        ),
        features=features,
        outputs=[dict(epoch=x["epoch"], **x["outputs"]) for x in analysis["snapshots"]],
        report=dict(
            summary="两轮训练和 445 条件冻结表达验证已验收。详见对比报告和固定训练探针图。",
            limitations=["单一种子。诊断来自固定训练探针。", "本轮评估范围为冻结表达验证。"],
        ),
        artifacts=diagnostic_assets + notebook_artifacts,
    )
)
registry = live["project"].get("run_registry", {})
original = next(
    (row for row in reversed(registry.get("run_records", [])) if row["run_id"] == r["id"]), None
)
if original and original["training_status"] != "completed":
    ref = next(
        x for x in registry["baseline_versions"] if x["id"] == original["baseline_version_id"]
    )
    revised = {
        **original,
        "id": original["id"] + "-completed",
        "supersedes": original["id"],
        "training_status": "completed",
        "evidence": original["evidence"]
        + [dict(source=str(train_path), sha256=digest(train_path), fields=["training_status"])],
        "notes": "训练终态及冻结表达验证均已验收。",
    }
    pkg.package["project"]["run_registry"] = dict(
        schema_version="1.0", baseline_versions=[ref], run_records=[original, revised]
    )
print(pkg.write(dest))
