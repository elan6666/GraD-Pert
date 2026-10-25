"""CPU-only reports from sealed, "
"aggregate training diagnostics; "
"no model loading."""

# Chinese report punctuation is intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

HEADS = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
FEATURES = (
    "identity_map",
    "graph.output",
    "cell.output.genes",
    "cell.output.cls",
    "response.output.genes",
    "response.output.cls",
    "ssl1_cls.bottleneck",
    "ssl1_node.bottleneck",
    "ssl2_cls.bottleneck",
    "ssl2_node.bottleneck",
)
LIMITS = (
    "固定4个训练细胞，eval模式；不是训练全集、测试集或单细胞反事实评估。",
    "每个特征最多128行、前256维，采用遍历前缀；包含相关视图，不能当作独立样本。",
    "所有层/头原始聚合统计均导出CSV；图示只选取主要模块。",
    "头分布统计的前缀与实际CE配对遮蔽行可能不同，不能将两套Teacher熵混用。",
    "当前探针不含未见表达查询的表征分组、CKA、UMAP或顺序/control干预；缺失明确保留。",
    "低秩、同一argmax或负梯度余弦只是诊断信号，不单独证明坍塌或辅助任务无效。",
    "本工具只读取JSON并绘图，不加载pt/模型，不更新optimizer、EMA、center或随机数。",
)


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def scalar_rows(epoch: int, values: dict) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "name": name,
            **{
                k: v
                for k, v in stats.items()
                if v is None or isinstance(v, (str, int, float, bool))
            },
        }
        for name, stats in values.items()
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def probe_groups(config: dict, snapshots: list[dict]) -> list[dict]:
    """One immutable probe per declared phase, never pretend two recipes match."""
    parameters = config.get("model", {}).get("parameters", {})
    staged = parameters.get("training_mode", {}).get("value") == "ssl_then_mse"
    if not staged:
        if len({p["input_sha256"] for p in snapshots}) != 1:
            raise ValueError("fixed probe input/view identity changed across epochs")
        return []
    boundary = parameters["ssl_pretrain_epochs"]["value"]
    if not isinstance(boundary, int) or not 0 < boundary < snapshots[-1]["epoch"]:
        raise ValueError("invalid sealed pretraining boundary")
    groups = {}
    signature = ("cells", "query_genes", "seed", "budget")
    expected_signature = {k: snapshots[0][k] for k in signature}
    for p in snapshots:
        stage = "ssl_pretrain" if p["epoch"] <= boundary else "mse_finetune"
        if p.get("training_stage") != stage:
            raise ValueError("probe training stage differs from sealed epoch boundary")
        if {k: p[k] for k in signature} != expected_signature:
            raise ValueError("fixed probe size/seed/budget changed across stages")
        group = groups.setdefault(
            stage, dict(stage=stage, epochs=[], input_sha256=p["input_sha256"])
        )
        if group["input_sha256"] != p["input_sha256"]:
            raise ValueError("fixed probe input/view identity changed within training stage")
        group["epochs"].append(p["epoch"])
    return list(groups.values())


def trajectory(points: list[dict], values: list) -> tuple[list, list]:
    """Break curves at a declared phase boundary, without inventing samples."""
    x, y = [], []
    for i, (point, value) in enumerate(zip(points, values, strict=True)):
        if i and point.get("training_stage") != points[i - 1].get("training_stage"):
            x.append(float("nan"))
            y.append(None)
        x.append(point["epoch"])
        y.append(value)
    return x, y


def diagnostic_path(root: Path, epoch: int, manifest: dict, history: list) -> tuple[Path, list]:
    """Resolve inherited snapshots only through an explicit verified recovery lineage."""
    path = root / f"fit/loss-diagnostic-epoch-{epoch:04d}.json"
    if path.exists():
        return path, []
    receipt_path = root / "RECOVERY_MIGRATION.json"
    recovery = read(receipt_path)
    inherited = recovery.get("inherited_epochs", [])
    parent = Path(recovery["parent_run_root"]).resolve()
    prior = read(parent / "run_manifest.json")
    prior_history = read(parent / "fit/history.json")
    if (
        recovery.get("status") != "verified"
        or recovery.get("run_id") != manifest["run_id"]
        or recovery.get("parent_run_id") != prior["run_id"]
        or recovery.get("source_commit") != manifest["source"]["commit"]
        or recovery.get("config_sha256") != manifest["config_sha256"]
        or any(prior[k] != manifest[k] for k in ("source", "data", "config_sha256"))
        or not inherited
        or inherited != list(range(1, len(inherited) + 1))
        or (epoch != 0 and epoch not in inherited)
        or history[: len(inherited)] != prior_history[: len(inherited)]
        or parent == root.resolve()
    ):
        raise ValueError("inherited diagnostic recovery identity or epoch differs")
    return parent / path.relative_to(root), [
        receipt_path,
        parent / "run_manifest.json",
        parent / "fit/history.json",
    ]


def collect(
    root: Path, *, name: str, analysis_identity: dict, checkpoint_role: str = "last"
) -> dict:
    """Reject partial runs and "
    "drift; expose scalar aggregates, "
    "never cell matrices."""
    manifest = read(root / "run_manifest.json")
    complete = read(root / "COMPLETE.json")
    config = read(root / "resolved_config.json")
    history = read(root / "fit/history.json")
    test = read(root / "fit/last-test.json")
    source = manifest["source"]
    if source["dirty"] or source["commit"] != source["published_commit"]:
        raise ValueError("training source must be clean and published")
    epochs = int(config["training"]["max_epochs"]["value"])
    if complete["identity"] != manifest or complete["epoch"] != epochs or not complete["zero_pkl"]:
        raise ValueError("exact completed run required")
    if [h["epoch"] for h in history] != list(range(1, epochs + 1)):
        raise ValueError("all committed epochs required")
    if (
        test["identity"]["training"] != manifest
        or test["identity"]["checkpoint"] != complete["last"]
        or test["identity"]["role"] != "last"
        or test["result"]["split"] != "test"
    ):
        raise ValueError("test/checkpoint identity mismatch")
    checkpoint = (root / "fit" / complete["last"]["file"]).resolve()
    if (
        not checkpoint.is_relative_to((root / "fit").resolve())
        or digest(checkpoint) != complete["last"]["sha256"]
    ):
        raise ValueError("checkpoint content changed")
    if any(root.rglob("*.pkl")):
        raise ValueError("completed run must have zero PKL")
    paths = [
        root / "run_manifest.json",
        root / "COMPLETE.json",
        root / "resolved_config.json",
        root / "fit/history.json",
        root / "fit/last-test.json",
    ]
    snapshots = []
    for epoch in range(epochs + 1):
        path, origin = diagnostic_path(root, epoch, manifest, history)
        paths.extend(p for p in origin if p not in paths)
        d = read(path)
        p = d["process"]
        if p["split"] != "train" or p["selection_use"] != "none":
            raise ValueError("training-only diagnostic required")
        for pair in p["pairs"].values():
            if not math.isclose(
                pair["cross_entropy"],
                pair["teacher_entropy"] + pair["teacher_to_student_kl"],
                abs_tol=1e-8,
                rel_tol=1e-8,
            ):
                raise ValueError("CE decomposition inconsistent")
        if epoch and d != history[epoch - 1]["training_diagnostic"]:
            raise ValueError("history/diagnostic mismatch")
        snapshots.append(
            {
                "epoch": epoch,
                **p,
                "basal_gradients": d.get("shared_basal_weighted_gradient_l2"),
                "basal_prediction_ssl_cosine": d.get("prediction_ssl_gradient_cosine"),
            }
        )
        paths.append(path)
    groups = probe_groups(config, snapshots)
    endpoint_epoch = epochs
    endpoint_stage = snapshots[-1].get("training_stage", "joint")
    selected = complete["last"]
    if checkpoint_role == "pretrain":
        from checkpoint_endpoint import select_checkpoint

        selected, endpoint = select_checkpoint(root, checkpoint_role)
        endpoint_epoch, endpoint_stage = endpoint["epoch"], endpoint["training_stage"]
        # Validate the entire completed run before selecting an earlier endpoint.
        history = [h for h in history if h["epoch"] <= endpoint_epoch]
        snapshots = [p for p in snapshots if p["epoch"] <= endpoint_epoch]
        groups = [g for g in groups if g["stage"] == endpoint_stage]
        paths.append(root / "fit/epoch_state.json")
    elif checkpoint_role != "last":
        raise ValueError("unsupported diagnostic checkpoint role")
    metrics = []
    result = test["result"]
    for group, values in {
        "all": {"metrics": result["metrics"]},
        **result["metric_gene_groups"],
    }.items():
        for m in values["metrics"]:
            metrics.append(
                {
                    "group": group,
                    **{
                        k: m[k]
                        for k in (
                            "metric_id",
                            "macro_mean",
                            "finite_condition_count",
                            "total_condition_count",
                        )
                    },
                }
            )
    return {
        "schema": "completed-ablation-analysis-1",
        "name": name,
        "run_id": manifest["run_id"],
        "run_root": str(root),
        "training_source": source,
        "evaluation_source": test["identity"]["evaluation"]["source"]
        if checkpoint_role == "last"
        else None,
        "analysis_identity": analysis_identity,
        "checkpoint_sha256": selected["sha256"],
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": endpoint_epoch,
        "training_stage": endpoint_stage,
        "full_training_epochs": epochs,
        "test_performed_for_endpoint": checkpoint_role == "last",
        "config_sha256": manifest["config_sha256"],
        "resolved_config_sha256": digest(root / "resolved_config.json"),
        "inputs": {
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest(p) for p in paths
        },
        "fixed_input_sha256": snapshots[0]["input_sha256"] if not groups else None,
        "fixed_inputs_by_training_stage": groups,
        "cross_stage_probe_comparison": "different input/view recipes; compare within each phase"
        if groups
        else "one fixed recipe",
        "epochs": endpoint_epoch,
        "history": [
            {
                k: h[k]
                for k in (
                    "epoch",
                    "optimizer_steps",
                    "training",
                    "training_seconds",
                    "diagnostic_seconds",
                    "rank0_peak_allocated_bytes",
                    "training_stage",
                )
                if k in h
            }
            for h in history
        ],
        "snapshots": snapshots,
        "metrics": metrics if checkpoint_role == "last" else [],
        "test_prediction_mse": result["prediction_loss"] if checkpoint_role == "last" else None,
        "expression_exposure": result["expression_exposure"],
        "limitations": LIMITS,
    }


def graph_gradient_groups(snapshot: dict) -> dict:
    """Frozen backbone diagnostics truthfully have no trainable graph groups."""
    return snapshot.get("graph_gradients", {}).get("groups", {})


def graph_gradient_status(snapshot: dict) -> str:
    if graph_gradient_groups(snapshot):
        return "observed"
    return "unavailable:" + snapshot.get("graph_gradients", {}).get("reason", "not_recorded")


def graph_gradient_rows(snapshots: list[dict]) -> list[dict]:
    rows = []
    for point in snapshots:
        base = {"epoch": point["epoch"], "training_stage": point.get("training_stage", "joint")}
        groups = graph_gradient_groups(point)
        if not groups:
            rows.append({**base, "group": None, "status": graph_gradient_status(point)})
        for name, value in groups.items():
            rows.append({**base, "group": name, **value["weighted_gradient_l2"], **value["cosine"]})
    return rows


def report(data: dict) -> str:
    first, last = data["snapshots"][0], data["snapshots"][-1]
    groups = data.get("fixed_inputs_by_training_stage", [])
    if groups:
        same_phase = [p for p in data["snapshots"] if p["training_stage"] == last["training_stage"]]
        first = same_phase[0]
    active = [p for p in data["snapshots"] if p["pairs"]]
    head_first, head_last = (active[0], active[-1]) if groups and active else (first, last)
    phase_notes = (
        [
            "阶段探针分别固定：SSL预训练不读扰动真值并含SSL视图；MSE微调读真值且不构造SSL视图。",
            "跨阶段不是同一输入/视图，图中断开连线；表征在最后有效阶段内比较，蒸馏表只比较有效SSL阶段。",
            "分阶段输入hash及epoch清单见summary.json；禁止把阶段切换当作同探针的表征突变。",
            "预训练Joint仅含SSL，微调Joint仅含MSE；两阶段数值不直接比较。",
            "",
        ]
        if groups
        else []
    )
    lines = [
        f"# {data['name']}：训练过程与最终评估分析",
        "",
        f"运行 `{data['run_id']}`；分析第{data['epochs']}轮 "
        f"{data.get('checkpoint_role', 'last')} checkpoint。",
        f"训练SHA `{data['training_source']['commit']}"
        f"`；评估SHA `{(data['evaluation_source'] or {}).get('commit', 'N/A:本端点不做正式测试')}"
        f"`；"
        f"分析SHA `{data['analysis_identity']['commit']}`。",
        f"checkpoint SHA256 `{data['checkpoint_sha256']}"
        f"`。完整身份及每个输入hash见 "
        f"[summary.json](summary.json)。",
        "",
        *phase_notes,
        *(
            [
                "R11仅原prediction MLP额外适配；Teacher、centers与主干冻结。"
                "SSL/A/U/CR/EMA停用，零项不解释为改善；冻结Teacher几何仅引用已验证R9。"
            ]
            if last.get("training_stage") == "frozen_readout"
            else []
        ),
        *(
            [
                "R12使用原RI架构从头训练，soft-rank/cos作用于完整320唯一输入的Global Response CLS；"
                "小探针不具备完整总体，新项NA；实际训练原始与加权贡献见training.csv。"
                "R3-RI为历史跨版本主参照，RS-B0/RS-RI为背景；单seed不保证轨迹相同。"
            ]
            if any("spectral_weighted" in h["training"] for h in data["history"])
            else []
        ),
        "## 训练损失和耗时",
        "",
        "| Epoch | 预测MSE | Joint | 训练分钟 | 诊断秒 |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in data["history"]:
        mse = (
            "不适用"
            if h.get("training_stage") == "ssl_pretrain"
            else f"{h['training']['prediction_mse']:.6f}"
        )
        lines.append(
            f"| {h['epoch']} | {mse}"
            f" |"
            f" {h['training']['joint_loss']:.4f}"
            f" | {h['training_seconds'] / 60:.2f}"
            f" "
            f"| {h['diagnostic_seconds']:.2f} |"
        )
    lines += [
        "",
        "Joint逐轮值为训练更新的平均；下文为固定训练探针，不能互换。",
        "",
        "![训练损失与CE分解](01_losses.png)",
        "",
        "## 四个蒸馏头",
        "",
        "以下是最后有效蒸馏阶段的Teacher分布前缀统计：JS=边际熵−样本平均熵。MSE阶段未调用Teacher不是熵为零。",
        "",
        "| 头 | 样本熵 | JS：初始化→最终 | 有效边际原型 | argmax使用数/样本数 | 最大argmax占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for head in HEADS:
        a, b = (
            head_first["distributions"].get("teacher." + head, {}),
            head_last["distributions"].get("teacher." + head, {}),
        )
        if not a.get("samples") or not b.get("samples"):
            lines.append(f"| {head} | 未采集：该头未调用 | — | — | — | — |")
            continue
        lines.append(
            f"| {head} | {b['mean_sample_entropy']:.4f}"
            f" | {a['sample_dependence_js']:.4f}"
            f"→{b['sample_dependence_js']:.4f}"
            f" "
            f"| {b['effective_marginal_prototypes']:.1f}"
            f" | {b['argmax_prototypes_used']}"
            f"/{b['samples']} "
            f"| {b['largest_argmax_fraction']:.1%} |"
        )
    lines += [
        "",
        "实际CE配对另行分解：CE=H(Teacher)+KL(Teacher‖Student)。它与上表采样行可能不同。",
        "",
        "| 头 | CE：初始化→最终 | Teacher熵：初始化→最终 | KL：初始化→最终 |",
        "|---|---:|---:|---:|",
    ]
    for head in HEADS:
        a, b = head_first["pairs"].get(head), head_last["pairs"].get(head)
        if a is None or b is None:
            lines.append(f"| {head} | 未采集：无有效配对 | — | — |")
            continue
        lines.append(
            f"| {head} | {a['cross_entropy']:.4f}"
            f"→{b['cross_entropy']:.4f} "
            f"| {a['teacher_entropy']:.4f}"
            f"→{b['teacher_entropy']:.4f}"
            f" "
            f"| {a['teacher_to_student_kl']:.4f}"
            f"→{b['teacher_to_student_kl']:.4f}"
            f" "
            f"|"
        )
    lines += [
        "",
        "![Teacher目标分布](02_prototypes.png)",
        "",
        "## 基因token、CLS及投影瓶颈",
        "",
        "有效秩为 exp(协方差特征值归一化后的熵)；相对方差为中心化方差/未中心化二阶矩。"
        "高秩不直接等于生物学有效。",
        f"本表端点为epoch {first['epoch']}→{last['epoch']}；全部阶段统计均保留在CSV与分段曲线。",
        "",
        "| Student位置 | 有效秩：初始化→最终 | 相对方差：初始化→最终 | 最终采样行数 |",
        "|---|---:|---:|---:|",
    ]
    for feature in FEATURES:
        a, b = (
            first["features"].get("student." + feature),
            last["features"].get("student." + feature),
        )
        if not a or not b or not a.get("samples") or not b.get("samples"):
            lines.append(f"| {feature} | 未采集：未调用 | — | 0 |")
            continue
        lines.append(
            f"| {feature} | {a['covariance_effective_rank']:.2f}"
            f"→{b['covariance_effective_rank']:.2f}"
            f" "
            f"| {a['relative_centered_variance']:.5g}"
            f"→{b['relative_centered_variance']:.5g}"
            f" "
            f"| {b['samples']} |"
        )
    lines += [
        "",
        "所有Student/Teacher逐层统计见features.csv，"
        "CLS样本包含重复视图；不能与128个gene行直接比秩上限。",
        "",
        "![表征轨迹](03_representations.png)",
        "",
        "## 输出行为和梯度关系",
        "",
    ]
    o = last["outputs"]
    lines += (
        [
            "此端点尚未训练预测头，预测MSE、输出行为及预测梯度不适用；没有把它们填为0。",
            "完整中等诊断若只读调用未训练读出，会单独标为diagnostic-only。",
        ]
        if data.get("checkpoint_role") == "pretrain"
        else [
            f"固定训练探针：预测MSE **{o['prediction_mse']:.6f}"
            f"**，复制control的MSE "
            f"**{o['control_copy_mse']:.6f}"
            f"**，相对降低 **{o['mse_gain_over_copy']:.1%}"
            f"**。",
            f"预测Δ RMS **{o['delta_rms']:.4f}"
            f"**，随机配对真实效应RMS比 **{o['paired_effect_rms_ratio']:.4f}"
            f"**；"
            f"负预测占比 **{o['negative_prediction_fraction']:.1%}"
            f"**；误差最大的1%元素贡献 "
            f"**{o['top_one_percent_loss_fraction']:.1%}**。",
            "这些查询只来自训练可见表达基因；该效应比不是生物学反事实幅度校准，也不描述未见表达组。",
            "",
        ]
    )
    lines += [
        "| 共享参数组 | ∥预测梯度∥ | ∥SSL1梯度∥ | ∥SSL2梯度∥ | cos(预测,SSL1) | cos(预测,SSL2) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not graph_gradient_groups(last):
        lines.append("图梯度：" + graph_gradient_status(last) + "；冻结主干，不计算SSL梯度或余弦。")
    for name, g in graph_gradient_groups(last).items():
        n, c = g["weighted_gradient_l2"], g["cosine"]
        # A disabled/zero-gradient objective has no cosine, not a zero cosine.
        cosines = {
            key: "不适用（无有效梯度）" if c.get(key) is None else f"{c[key]:.4f}"
            for key in ("prediction_ssl1", "prediction_ssl2")
        }
        lines.append(
            f"| {name} | "
            + ("不适用" if data.get("checkpoint_role") == "pretrain" else f"{n['prediction']:.4g}")
            + f" | {n['ssl1']:.4g} | {n['ssl2']:.4g}"
            f" "
            f"| {cosines['prediction_ssl1']}"
            f" | {cosines['prediction_ssl2']}"
            f" "
            f"|"
        )
    lines += [
        "",
        "梯度已含实际loss权重；大梯度或负余弦需要结合跨轮轨迹，不能单次判定辅助目标无效。",
        "",
        "![输出与图梯度](04_outputs_gradients.png)",
        "",
        "## 最终测试：全部、表达见过、表达未见过",
        "",
        "| 分组 | 指标 | Pearson | 有效/总条件 |",
        "|---|---|---:|---:|",
    ]
    for m in data["metrics"]:
        v = "不可定义" if m["macro_mean"] is None else f"{m['macro_mean']:.6f}"
        lines.append(
            f"| {m['group']} | {m['metric_id']}"
            f" | {v} | {m['finite_condition_count']}"
            f"/{m['total_condition_count']}"
            f" "
            f"|"
        )
    if not data.get("test_performed_for_endpoint", True):
        lines.append("本端点预测头尚未训练，不复用最终checkpoint的测试结果。")
    lines += [
        "",
        (
            f"最终测试MSE **{data['test_prediction_mse']:.6f}"
            "**；与上述训练探针MSE属于不同数据/采样口径。"
            if data.get("test_performed_for_endpoint", True)
            else "正式测试MSE：N/A；只有最终微调端点做正式测试。"
        ),
        "",
        "## 自动诊断线索与边界",
        "",
    ]
    for head in HEADS:
        d = last["distributions"].get("teacher." + head, {})
        if not d.get("samples"):
            continue
        if d["largest_argmax_fraction"] >= 0.95:
            share = d["largest_argmax_fraction"]
            lines.append(
                f"- {head}：{d['samples']}个前缀样本的argmax高度集中（{share:.1%}"
                f"），"
                f"JS={d['sample_dependence_js']:.5g}"
                f"，有效边际原型={d['effective_marginal_prototypes']:.1f}"
                f"。"
                f"这是目标差异不足的检查线索；不是单原型坍塌的充分证明。"
            )
    graph = last["features"]["student.graph.output"]["covariance_effective_rank"]
    response = last["features"]["student.response.output.genes"]["covariance_effective_rank"]
    lines.append(
        f"- 图输出有效秩{graph:.2f}，Response基因输出有效秩{response:.2f}"
        f"：检查信息是否在下游汇总时集中到少数方向，不能仅由秩判断预测能力。"
    )
    lines += [
        "- 单seed、固定小探针；与其他实验比较表征前先比较fixed_input_sha256、采样预算和统计规则。",
        "",
    ]
    lines.extend("- " + v for v in data["limitations"])
    lines += [
        "",
        "## 文件清单",
        "",
        "summary.json；training.csv；"
        "heads.csv；pairs.csv；features.csv；"
        "gradients.csv；outputs.csv；"
        "metrics.csv；本报告；四张PNG；receipt.json。"
        "原始聚合来源及每个输入SHA均在summary。",
    ]
    return "\n".join(lines) + "\n"


def plots(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
    points = data["snapshots"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name)
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(
        *trajectory(data["history"], [h["training"]["joint_loss"] for h in data["history"]]),
        "o-",
        label="Train joint",
    )
    ax[0].set_ylabel("Joint loss")
    a = ax[0].twinx()
    a.plot(
        [h["epoch"] for h in data["history"]],
        [
            None if h.get("training_stage") == "ssl_pretrain" else h["training"]["prediction_mse"]
            for h in data["history"]
        ],
        "s--",
        color="orange",
        label="Train MSE",
    )
    a.set_ylabel("Prediction MSE")
    ax[0].legend(loc="upper right")
    a.legend(loc="lower left")
    ax[0].set_title("Epoch training means (different axes)")
    for head in HEADS:
        if any(head in p["pairs"] for p in points):
            ax[1].plot(
                *trajectory(
                    points, [p["pairs"].get(head, {}).get("cross_entropy") for p in points]
                ),
                "o-",
                label=head,
            )
    ax[1].set_title("Fixed-probe CE pair means")
    ax[1].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.2)
    save(fig, "01_losses.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for head in HEADS:
        for a, key in zip(ax, ("mean_sample_entropy", "sample_dependence_js"), strict=True):
            values = [p["distributions"].get("teacher." + head, {}).get(key) for p in points]
            if any(v is not None for v in values):
                a.plot(*trajectory(points, values), "o-", label=head)
    ax[0].set_title("Teacher entropy (head-prefix rows)")
    ax[1].set_title("Teacher sample dependence JS")
    for a in ax:
        a.set_xlabel("Epoch")
        a.legend(fontsize=7)
        a.grid(alpha=0.2)
    save(fig, "02_prototypes.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for feature in FEATURES[:6]:
        for a, key in zip(
            ax, ("covariance_effective_rank", "relative_centered_variance"), strict=True
        ):
            a.plot(
                *trajectory(
                    points, [p["features"].get("student." + feature, {}).get(key) for p in points]
                ),
                "o-",
                label=feature,
            )
    ax[0].set_title("Student effective covariance rank")
    ax[1].set_title("Student relative centered variance")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("Epoch")
        a.legend(fontsize=6)
        a.grid(alpha=0.2)
    save(fig, "03_representations.png")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for key in ("prediction_mse", "control_copy_mse"):
        ax[0, 0].plot(*trajectory(points, [p["outputs"].get(key) for p in points]), "o-", label=key)
    for key in ("mse_gain_over_copy", "paired_effect_rms_ratio", "negative_prediction_fraction"):
        ax[0, 1].plot(*trajectory(points, [p["outputs"].get(key) for p in points]), "o-", label=key)
    for loss in ("prediction", "ssl1", "ssl2"):
        ax[1, 0].plot(
            *trajectory(
                points,
                [
                    graph_gradient_groups(p)
                    .get("graph", {})
                    .get("weighted_gradient_l2", {})
                    .get(loss)
                    for p in points
                ],
            ),
            "o-",
            label=loss,
        )
    for key in ("prediction_ssl1", "prediction_ssl2"):
        ax[1, 1].plot(
            *trajectory(
                points,
                [
                    graph_gradient_groups(p).get("graph", {}).get("cosine", {}).get(key)
                    for p in points
                ],
            ),
            "o-",
            label=key,
        )
    ax[1, 1].axhline(0, color="grey", lw=0.7)
    for a, title in zip(
        ax.flat,
        (
            "Fixed-probe MSE",
            "Output ratios (random pairing)",
            "Weighted shared-graph gradient norms",
            "Prediction/SSL gradient cosines",
        ),
        strict=True,
    ):
        a.set_title(title)
        a.set_xlabel("Epoch")
        a.legend(fontsize=7)
        a.grid(alpha=0.2)
    save(fig, "04_outputs_gradients.png")


def export(data: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", data)
    for key, filename in (
        ("features", "features.csv"),
        ("distributions", "heads.csv"),
        ("pairs", "pairs.csv"),
    ):
        write_csv(
            output / filename,
            [
                {"training_stage": p.get("training_stage", "joint"), **row}
                for p in data["snapshots"]
                for row in scalar_rows(p["epoch"], p[key])
            ],
        )
    write_csv(
        output / "outputs.csv",
        [
            {
                "epoch": p["epoch"],
                "training_stage": p.get("training_stage", "joint"),
                **p["outputs"],
            }
            for p in data["snapshots"]
        ],
    )
    write_csv(output / "gradients.csv", graph_gradient_rows(data["snapshots"]))
    write_csv(
        output / "training.csv",
        [
            {
                "epoch": h["epoch"],
                "training_stage": h.get("training_stage", "joint"),
                "training_seconds": h["training_seconds"],
                "diagnostic_seconds": h["diagnostic_seconds"],
                **h["training"],
            }
            for h in data["history"]
        ],
    )
    write_csv(output / "metrics.csv", data["metrics"])
    plots(data, output)
    (output / "report.md").write_text(report(data))
    write_json(
        output / "receipt.json",
        {
            "schema": "completed-ablation-analysis-receipt-1",
            "status": "completed",
            "run_id": data["run_id"],
            "analysis_identity": data["analysis_identity"],
            "input_hashes": data["inputs"],
            "outputs": {
                p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted(output.iterdir())
            },
            "model_loads": 0,
            "optimizer_updates": 0,
            "gpu_work": False,
            "scope": "read_only_saved_aggregate_postprocessing",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint-role", choices=("last", "pretrain"), default="last")
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if (
        not root.is_relative_to(Path("/data/yilangliu"))
        or not output.is_relative_to(Path("/data/yilangliu"))
        or output.is_relative_to(root)
    ):
        parser.error("scientific inputs and fresh independent analysis output must stay on server")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        parser.error("analysis source must be clean")
    identity = {
        "commit": commit,
        "script_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "training_method_unchanged": True,
    }
    data = collect(
        root, name=args.name, analysis_identity=identity, checkpoint_role=args.checkpoint_role
    )
    export(data, output)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "receipt_sha256": digest(output / "receipt.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
