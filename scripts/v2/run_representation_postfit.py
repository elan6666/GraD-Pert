# ruff: noqa: RUF001
"""Validation and frozen training diagnostics; never finalize a test-complete run."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from generate_representation_round1 import ARMS, SCHEMA
from generate_representation_round2 import ARMS as ROUND2_ARMS
from generate_representation_round2 import SCHEMA as ROUND2_SCHEMA
from generate_representation_round3 import ARMS as ROUND3_ARMS
from generate_representation_round3 import SCHEMA as ROUND3_SCHEMA
from generate_representation_round4 import ARMS as ROUND4_ARMS
from generate_representation_round4 import SCHEMA as ROUND4_SCHEMA
from generate_representation_round5 import ARMS as ROUND5_ARMS
from generate_representation_round5 import SCHEMA as ROUND5_SCHEMA
from generate_representation_round6 import ARMS as ROUND6_ARMS
from generate_representation_round6 import SCHEMA as ROUND6_SCHEMA
from generate_representation_round7 import ARMS as ROUND7_ARMS
from generate_representation_round7 import SCHEMA as ROUND7_SCHEMA
from generate_representation_round8 import ARMS as ROUND8_ARMS
from generate_representation_round8 import SCHEMA as ROUND8_SCHEMA

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

FAMILY_ARMS = {
    SCHEMA: ARMS,
    ROUND2_SCHEMA: ROUND2_ARMS,
    ROUND3_SCHEMA: ROUND3_ARMS,
    ROUND4_SCHEMA: ROUND4_ARMS,
    ROUND5_SCHEMA: ROUND5_ARMS,
    ROUND6_SCHEMA: ROUND6_ARMS,
    ROUND7_SCHEMA: ROUND7_ARMS,
    ROUND8_SCHEMA: ROUND8_ARMS,
}


def postfit_command(queue, row, stage):
    if stage not in ("validation", "analysis"):
        raise ValueError("unknown research postfit stage")
    return [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/run_representation_postfit.py"),
        stage,
        "--queue",
        str(Path(queue["output"]) / "queue.json"),
        "--arm",
        row["name"],
    ]


def context(queue, row):
    """Recheck exact training, source, family and last checkpoint at every boundary."""
    from run_br400_group import validate_training_contract, verify

    verify(queue)
    if row["name"] not in FAMILY_ARMS.get(queue["family"], {}) or row not in queue["rows"]:
        raise ValueError("research postfit requires a sealed representation research arm")
    plan = row["plan"]
    root = Path(plan["run_root"])
    if read_json(root / "launch.json") != plan:
        raise ValueError("training launch changed")
    if read_json(root / "resolved_config.json") != plan["resolved_config"]:
        raise ValueError("training resolved config changed")
    training = validate_training_contract(plan)
    if training["last"]["epoch"] != 2:
        raise ValueError("research requires exact two-epoch last")
    if (root / "COMPLETE.json").exists() or (root / "fit/last-test.json").exists():
        raise ValueError("research test must remain deferred until research design lock")
    return training


def validation_path(queue, row):
    return Path(queue["output"]) / f"{row['name']}-validation"


def validate_validation(queue, row):
    training = context(queue, row)
    path = validation_path(queue, row) / "COMPLETE.json"
    receipt = read_json(path)
    plan, evaluated = row["plan"], receipt["plan"]
    expected = {
        "training_run_root": str(Path(plan["run_root"]).resolve()),
        "checkpoint_sha256": training["last"]["sha256"],
        "checkpoint_epoch": 2,
        "checkpoint_role": "last",
        "split": "val",
        "config_sha256": plan["config_sha256"],
        "runtime_sha256": plan["runtime_sha256"],
        "output_root": str(validation_path(queue, row).resolve()),
        "gpu": ["0", "1"],
        "training_identity": training["identity"],
    }
    if (
        any(evaluated.get(k) != v for k, v in expected.items())
        or evaluated["evaluation_source"].get("commit") != queue["source_commit"]
        or receipt.get("zero_pkl") is not True
        or receipt["result"].get("split") != "val"
        or evaluated.get("diagnostic_only")
        or "diagnostic_condition_count" in evaluated
        or read_json(path.parent / "plan.json") != evaluated
        or (path.parent / "FAILURE.json").exists()
    ):
        raise ValueError("postfit validation provenance or full-population scope changed")
    result = receipt["result"]
    groups = result.get("metric_gene_groups", {})
    if set(groups) != {"seen_expression", "unseen_expression"}:
        raise ValueError("validation lacks expression exposure groups")
    from run_unified_group import METRICS

    for metrics in [result["metrics"], *(g["metrics"] for g in groups.values())]:
        if len(metrics) != 6 or {m["metric_id"] for m in metrics} != METRICS:
            raise ValueError("full validation requires 18 grouped metrics")
        for metric in metrics:
            n, total, value = (
                metric["finite_condition_count"],
                metric["total_condition_count"],
                metric["macro_mean"],
            )
            if (
                type(n) is not int
                or type(total) is not int
                or not 0 <= n <= total
                or (n == 0 and value is not None)
                or (n > 0 and (value is None or not math.isfinite(value)))
            ):
                raise ValueError("validation finite counts and NA values differ")
    if not math.isfinite(result["prediction_loss"]):
        raise ValueError("validation prediction loss is nonfinite")
    return {"receipt": str(path), "sha256": sha256_file(path), "split": "val"}


def run_validation(queue, row):
    from gradpert.evaluation.state import prepare_evaluation_state
    from gradpert.execution.v2_checkpoint_eval import (
        execute_evaluation_plan,
        resolve_evaluation_plan,
    )

    training = context(queue, row)
    plan = row["plan"]
    config = plan["resolved_config"]
    prepare_evaluation_state(
        dataset_id=config["dataset_id"],
        protocol_id=config["data"]["protocol_id"],
        data_root=plan["data_root"],
        validation_only=True,
        evaluation_protocol="v2",
    )
    selected = training["last"]
    evaluation = resolve_evaluation_plan(
        argparse.Namespace(
            config=Path(plan["config"]),
            runtime=Path(plan["runtime"]),
            training_run_root=Path(plan["run_root"]),
            checkpoint=Path(plan["run_root"]) / "fit" / selected["file"],
            checkpoint_sha256=selected["sha256"],
            checkpoint_role="last",
            output_root=validation_path(queue, row),
            gpu="0,1",
            split="val",
        )
    )
    evaluation.update(
        cpu_training_state=True,
        cuda_memory_fraction=queue["shared_gpu_policy"]["max_own_memory_fraction"],
    )
    execute_evaluation_plan(evaluation)
    return validate_validation(queue, row)


def output_seal(root):
    return {
        str(p.relative_to(root)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def validate_outputs(root, outputs):
    for name, record in outputs.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != record["sha256"]:
            raise ValueError("research artifact changed or escaped output root")


def validate_panel(queue, row):
    training = context(queue, row)
    root = Path(queue["output"]) / f"{row['name']}-panel"
    receipt = read_json(root / "COMPLETE.json")
    expected = dict(
        status="completed",
        updates=0,
        method_sha=queue["source_commit"],
        run_id=row["plan"]["run_id"],
        checkpoint_sha256=training["last"]["sha256"],
        config_sha256=row["plan"]["config_sha256"],
        state_unchanged=True,
        parent_manifest=queue["medium_analysis"]["parent_manifest"],
        parent_manifest_sha256=queue["medium_analysis"]["parent_manifest_sha256"],
    )
    if any(receipt.get(k) != v for k, v in expected.items()) or (
        receipt["state_before"] != receipt["state_after"] or (root / "FAILURE.json").exists()
    ):
        raise ValueError("fixed-panel identity or model state changed")
    from representation_panel import panel_schedule

    sealed_panel = panel_schedule(
        read_json(Path(expected["parent_manifest"])), seed=20261021, conditions=32, controls=4
    )
    parity = receipt.get("parity", {})
    if (
        read_json(root / "panel.json") != sealed_panel
        or receipt.get("panel_sha256") != sealed_panel["panel_sha256"]
        or receipt.get("panel_rows") != sealed_panel["rows"]
        or receipt.get("checkpoint_endpoint")
        != dict(role="last", epoch=2, training_stage="joint", full_training_epochs=2)
        or parity.get("used_path_passed") is not True
        or parity.get("tolerance") != 2e-6
        or parity.get("cell_chunk_used") is not False
        or any(
            not 0 <= parity.get(k, float("inf")) <= 2e-6
            for k in ("graph_max_abs", "response_max_abs")
        )
    ):
        raise ValueError("fixed-panel population, endpoint or numerical parity differs")
    validate_outputs(root, receipt["outputs"])
    return {
        "receipt": str(root / "COMPLETE.json"),
        "sha256": sha256_file(root / "COMPLETE.json"),
        "panel_sha256": receipt["panel_sha256"],
    }


def panel_command(queue, row):
    plan, spec = row["plan"], queue["medium_analysis"]
    command = [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/run_representation_panel.py"),
    ]
    values = {
        "run-root": plan["run_root"],
        "config": plan["config"],
        "data-root": plan["data_root"],
        "output": str(Path(queue["output"]) / f"{row['name']}-panel"),
        "method-sha": queue["source_commit"],
        "parent-manifest": spec["parent_manifest"],
        "parent-manifest-sha256": spec["parent_manifest_sha256"],
        "resource-queue": str(Path(queue["output"]) / "queue.json"),
        "resource-queue-sha256": sha256_file(Path(queue["output"]) / "queue.json"),
        "checkpoint-role": "last",
        "conditions": "32",
        "controls": "4",
        "seed": "20261021",
    }
    for name, value in values.items():
        command.extend(["--" + name, value])
    return command


def ssl_loss_contributions(history, config):
    """Separate raw component losses from their fixed configured overall weights."""
    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    rows = []
    for entry in history:
        terms = entry.get("training", {})
        row = dict(epoch=entry["epoch"], prediction=terms.get("prediction"))
        for stage, scale, components in (
            (1, options.lambda1, ("condition", "node", "spread")),
            (2, options.lambda2, ("dino", "ibot", "koleo")),
        ):
            weights = [getattr(options, f"ssl{stage}_{name}") for name in components]
            values = [terms.get(f"ssl{stage}_{name}") for name in components]
            raw = (
                None
                if any(w and v is None for w, v in zip(weights, values, strict=True))
                else sum(w * v for w, v in zip(weights, values, strict=True) if w)
            )
            row.update(
                {
                    f"lambda{stage}": scale,
                    f"ssl{stage}_before_overall_weight": raw,
                    f"ssl{stage}_contribution": None if raw is None else scale * raw,
                }
            )
        # Preserve saved means of actual warmup-weighted contributions.
        row.update({k: terms.get(k) for k in ("au_weighted", "cr_weighted", "joint_loss")})
        rows.append(row)
    return rows


def render_medium_report(queue, row, root):
    """Research report reads training/validation artifacts, never last-test.json."""
    import matplotlib
    from cap40_posthoc import csv_write
    from cap40_posthoc_report import flatten, table

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = root / "report"
    report.mkdir(exist_ok=False)
    summary = read_json(root / "analysis/summary.json")
    deep = read_json(root / "analysis/deep.json")
    history = read_json(Path(row["plan"]["run_root"]) / "fit/history.json")
    validation = read_json(validation_path(queue, row) / "COMPLETE.json")["result"]
    tables = {"training": [flatten(r) for r in history]}
    if queue.get("family") in (ROUND7_SCHEMA, ROUND8_SCHEMA):
        tables["training_loss_contributions"] = ssl_loss_contributions(
            history, row["plan"]["config"]
        )
    for name in ("global_features", "global_heads", "metrics", "cls_decomposition", "centers"):
        tables[name] = [{"name": k, **flatten(v)} for k, v in summary.get(name, {}).items()]
    for group in deep:
        for key in ("pairs", "same_gene", "cross_view", "per_cell", "mask_eligibility"):
            tables.setdefault(key, []).extend(
                {"condition_id": group["condition_id"], **r} for r in group.get(key, [])
            )
        loss_scope = group.get("diagnostic_objective_scope", {})
        tables.setdefault("frozen_losses", []).append(
            {
                "condition_id": group["condition_id"],
                "joint": group["joint"],
                **group["loss_terms"],
                **(
                    {
                        **{key: None for key in loss_scope.get("unavailable_terms", {})},
                        **flatten(loss_scope),
                    }
                    if loss_scope
                    else {}
                ),
            }
        )
    tables["gradients"] = [flatten(v) for v in read_json(root / "analysis/gradients.json")]
    grouped = root / "analysis/grouped_views.json"
    if grouped.exists():
        value = read_json(grouped)
        tables.update(grouped_view_pairs=value["pairs"], grouped_view_features=value["features"])
    tables["validation"] = [
        {"gene_group": name, **m}
        for name, metrics in [
            ("all", validation["metrics"]),
            *((k, v["metrics"]) for k, v in validation["metric_gene_groups"].items()),
        ]
        for m in metrics
    ]
    for name, rows in tables.items():
        csv_write(report / (name + ".csv"), rows)
    features = [
        (k, v)
        for k, v in summary["global_features"].items()
        if isinstance(v.get("effective_rank"), (int, float))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("effective_rank", "relative_variance"), strict=True):
        ax.barh([k for k, _ in features], [v[metric] for _, v in features])
        ax.set_title(metric)
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(report / "representation.png", dpi=150)
    plt.close(fig)
    text = [
        f"# {row['name']}：两轮表征研究\n",
        "仅训练侧冻结诊断及训练后验证集评估；测试推迟至研究设计锁定，不作为选型依据。\n",
        f"训练源码：`{queue['source_commit']}`；配置：`{row['plan']['config_sha256']}`。\n",
        "中等诊断沿用原清单；额外固定基因面板独立存储，不替代历史可比统计。\n",
        "![表征有效秩与相对方差](representation.png)\n",
        "有效秩描述差异分散方向，相对方差描述差异能量占比；两者都不是独立质量分数。\n",
        "启用KoLeo时，逐条件小探针缺完整全局异条件群体：KoLeo记NA，joint和SSL2梯度仅含可测项；完整训练loss仍包含KoLeo，见scope字段。\n",
    ]
    if queue.get("family") == ROUND3_SCHEMA:
        text.append(
            "AU作用于两Global的Student Response CLS；完整训练loss逐项记录A、U、"
            "warmup和加权贡献。中等诊断的逐条件小探针缺完整有效批，AU记NA，"
            "该joint不含AU；不得与完整训练joint混同。\n"
        )
    if queue.get("family") == ROUND4_SCHEMA:
        text.append(
            "R4-CR保留A=.05、U=0，并在两Global的Student Response CLS上加入负编码率："
            "CR权重=.00004、epsilon²=.05、warmup=30个完成更新。完整有效批跨所有上下文，"
            "这是受SimDINO启发的改动。逐条件小探针缺完整有效批，A与CR均记NA，"
            "该joint不含这些项；有效秩仍为中心化诊断，不由logdet推导。\n"
        )
    if queue.get("family") == ROUND5_SCHEMA:
        text.append(
            "R5-PROJ保留R4的A=.05和CR=.00004，仅将二者移到现有SSL2 CLS投影头的256维瓶颈，"
            "在L2和原型层之前取值；不增加head、不隔离主干梯度。逐条件小探针中A/CR均为NA。"
            "共同报告投影前后表征；只有瓶颈高秩不构成问题解决。\n"
        )
    if queue.get("family") == ROUND6_SCHEMA:
        text.append(
            "R6-EMA99仅将R5的Teacher EMA终点1改为.99，起点仍.99；"
            "即每次成功优化器更新后恒定.99。A=.05、CR=.00004及瓶颈作用点保持。"
            "复用原对照不重训；Teacher分布变尖或秩上升本身不代表预测效用改善。"
            "逐条件小探针缺完整有效批，A/CR仍为NA。\n"
        )
    if queue.get("family") == ROUND7_SCHEMA:
        text.append(
            "R7-S1D05仅将R6的SSL1总权重lambda1从1改为.05；lambda2仍为user_locked 1，"
            "Teacher EMA保持恒定.99，A=.05、CR=.00004及瓶颈作用点保持。"
            "原始SSL1/SSL2损失与加权贡献分开报告；权重不同的joint不直接排序。"
            "逐条件小探针缺完整有效批，A/CR仍为NA，降低CE或提高秩不构成预测改善。\n"
        )
        text.extend(
            ["## training_loss_contributions\n", table(tables["training_loss_contributions"])]
        )
    if queue.get("family") == ROUND8_SCHEMA:
        text.append(
            "R8-CR0以原R6为科学父臂，仅关闭CR：cr_weight=.00004→0；"
            "lambda1=1、lambda2=user_locked 1、恒定EMA=.99、A=.05与瓶颈作用点均保持。"
            "不重训R6；R7不是单因素父臂。历史中未计算的CR项保留NA，不以0冒充观测。"
            "原始SSL损失与实际加权贡献分列；几何或熵的变化不能替代预测效用。\n"
        )
        text.extend(
            ["## training_loss_contributions\n", table(tables["training_loss_contributions"])]
        )
    for name in ("training", "validation", "global_features", "global_heads", "frozen_losses"):
        text.extend([f"## {name}\n", table(tables[name])])
    text.append("## 完整聚合表\n")
    text.extend(f"- [{name}]({name}.csv)\n" for name in tables)
    text.append("\n逐条件、同基因、同细胞、同条件/异条件与视图对统计见对应表，NA不替换为0。\n")
    (report / "REPORT_CN.md").write_text("\n".join(text))


def validate_research_complete(queue, row):
    from run_fgpo_medium import validate_medium_complete

    context(queue, row)
    root = Path(queue["output"]) / f"{row['name']}-research"
    path = root / "RESEARCH_COMPLETE.json"
    receipt = read_json(path)
    expected = dict(
        schema="representation-research-complete-1",
        status="research_complete",
        scientific_complete=False,
        test_status="deferred_until_design_lock",
        queue_sha256=sha256_file(Path(queue["output"]) / "queue.json"),
        arm=row["name"],
        source_commit=queue["source_commit"],
        validation=validate_validation(queue, row),
        medium=validate_medium_complete(queue, row),
        panel=validate_panel(queue, row),
    )
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("research completion changed or mislabels pending test")
    validate_outputs(root, receipt["outputs"])
    for directory in (
        root,
        Path(row["plan"]["run_root"]),
        validation_path(queue, row),
        Path(queue["output"]) / f"{row['name']}-medium",
        Path(queue["output"]) / f"{row['name']}-panel",
    ):
        if any(directory.rglob("*.pkl")):
            raise ValueError("research completion requires zero PKL")
    return {
        "receipt": str(path),
        "sha256": sha256_file(path),
        "scientific_complete": False,
        "test_status": "deferred_until_design_lock",
    }


def run_analysis(queue, row):
    from run_fgpo_medium import execute, validate_medium_complete

    validation = validate_validation(queue, row)
    queue_path = Path(queue["output"]) / "queue.json"
    execute(queue_path, row["name"])
    medium = validate_medium_complete(queue, row)
    root = Path(queue["output"]) / f"{row['name']}-research"
    root.mkdir(exist_ok=False)
    with (root / "panel.log").open("x") as log:
        subprocess.run(
            panel_command(queue, row),
            cwd=queue["source"],
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                PYTORCH_ALLOC_CONF="expandable_segments:True",
                PYTHONPATH=str(Path(queue["source"]) / "src"),
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    panel = validate_panel(queue, row)
    (root / "REPORT_CN.md").write_text(
        f"# {row['name']} 研究结果索引\n\n"
        f"[中等训练分析与验证](../{row['name']}-medium/report/REPORT_CN.md)\n\n"
        f"[固定面板与逐层响应诊断](../{row['name']}-panel/COMPLETE.json)\n\n"
        "研究阶段完成；独立测试尚未进行，下一轮由主会话依据训练/验证证据重新设计。\n"
    )
    atomic_json(
        root / "RESEARCH_COMPLETE.json",
        dict(
            schema="representation-research-complete-1",
            status="research_complete",
            scientific_complete=False,
            test_status="deferred_until_design_lock",
            queue_sha256=sha256_file(queue_path),
            arm=row["name"],
            source_commit=queue["source_commit"],
            validation=validation,
            medium=medium,
            panel=panel,
            outputs=output_seal(root),
        ),
    )
    return validate_research_complete(queue, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "analysis"))
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=tuple(name for arms in FAMILY_ARMS.values() for name in arms),
        required=True,
    )
    args = parser.parse_args()
    queue = read_json(args.queue)
    if Path(queue["source"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("research worker must use exact queue source")
    if args.queue.resolve() != (Path(queue["output"]) / "queue.json").resolve():
        raise ValueError("sealed queue location changed")
    row = next(r for r in queue["rows"] if r["name"] == args.arm)
    (run_validation if args.stage == "validation" else run_analysis)(queue, row)


if __name__ == "__main__":
    main()
