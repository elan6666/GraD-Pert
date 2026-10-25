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
from generate_representation_round9 import ARMS as ROUND9_ARMS
from generate_representation_round9 import ENDPOINTS as ROUND9_ENDPOINTS
from generate_representation_round9 import SCHEMA as ROUND9_SCHEMA
from generate_representation_round11 import ARMS as ROUND11_ARMS
from generate_representation_round11 import SCHEMA as ROUND11_SCHEMA
from generate_representation_round12 import ARMS as ROUND12_ARMS
from generate_representation_round12 import SCHEMA as ROUND12_SCHEMA

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

FAMILY_ARMS = {
    ROUND9_SCHEMA: ROUND9_ARMS,
    ROUND11_SCHEMA: ROUND11_ARMS,
    ROUND12_SCHEMA: ROUND12_ARMS,
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
    from run_br400_group import validate_frozen_readout_parent, validate_training_contract, verify

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
    if queue.get("family") == ROUND11_SCHEMA:
        proof = read_json(root / "fit/frozen_readout_proof.json")
        identity = read_json(root / "run_manifest.json")
        parent = identity.get("frozen_readout", {})
        validate_frozen_readout_parent(parent)
        if (
            proof.get("frozen_equal") is not True
            or proof.get("frozen_gradients_none") is not True
            or proof.get("frozen_sha256") != parent.get("frozen_sha256")
            or proof.get("teacher_updates") != 0
            or proof.get("center_updates") != 0
            or proof.get("trainable_parameters") != 131585
        ):
            raise ValueError("R11 final frozen-state proof differs from loaded R9 parent")
    if training["last"]["epoch"] != (3 if queue.get("family") == ROUND9_SCHEMA else 2):
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
        "checkpoint_epoch": 3 if queue.get("family") == ROUND9_SCHEMA else 2,
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
                or (
                    queue.get("family") in (ROUND9_SCHEMA, ROUND11_SCHEMA, ROUND12_SCHEMA)
                    and total != 445
                )
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


def validate_staged_joint_teacher_signal(root, receipt, panel):
    """R9 requires the sealed native Teacher signal observation at each endpoint."""
    artifact = root / "teacher_signal.json"
    record = receipt.get("outputs", {}).get("teacher_signal.json")
    if (
        receipt.get("diagnostic_mode") != "teacher_signal_audit"
        or receipt.get("teacher_signal_input_mode") != "native_student_masked_teacher_clean"
        or not isinstance(record, dict)
        or not artifact.is_file()
        or record.get("sha256") != sha256_file(artifact)
        or record.get("bytes") != artifact.stat().st_size
    ):
        raise ValueError("R9 needs a sealed native Teacher signal artifact")
    signal = read_json(artifact)
    expected = dict(
        schema="teacher-signal-localization-1",
        status="completed",
        updates=0,
        encoder_recompute=0,
        tolerance=2e-6,
        active_head="ssl2_cls",
    )
    if (
        any(signal.get(k) != v for k, v in expected.items())
        or panel.get("conditions") != 32
        or panel.get("rows") != 128
        or len(panel["schedule"]) != 32
        or any(len(group["rows"]) != 4 for group in panel["schedule"])
        or any(
            not signal.get(k)
            for k in ("native_pairs", "pairs", "feature_pairs", "context_features", "sensitivity")
        )
    ):
        raise ValueError("R9 Teacher signal schema or frozen32x4 scope is incomplete")
    captures = signal.get("captures", [])
    native_pairs = {(path, view) for path in ("SS", "TT") for view in ("global0", "global1")}
    if len(captures) != len(panel["schedule"]):
        raise ValueError("R9 Teacher signal capture count differs from the sealed panel")
    for capture, group in zip(captures, panel["schedule"], strict=True):
        parity = capture.get("parity", [])
        if (
            capture.get("rows") != len(group["rows"])
            or capture.get("captured_globals_per_role") != 2
            or capture.get("head_only_replays") != 8
            or capture.get("encoder_recompute") != 0
            or capture.get("native_head_calls") != {"student": 4, "teacher": 2}
            or len(parity) != 4
            or {(p.get("path"), p.get("view")) for p in parity} != native_pairs
            or any(
                not 0 <= p.get(k, float("inf")) <= 2e-6
                for p in parity
                for k in ("bottleneck_maxabs", "logits_maxabs")
            )
        ):
            raise ValueError("R9 Teacher signal capture or native replay parity is incomplete")
    paths = signal.get("paths", [])
    expected_paths = {
        (path, view, stage)
        for path in ("SS", "ST", "TS", "TT")
        for view in ("global0", "global1")
        for stage in ("input", "bottleneck_pre_l2", "bottleneck_unit", "raw_logits")
    }
    if (
        len(paths) != len(expected_paths)
        or {(p.get("path"), p.get("view"), p.get("stage")) for p in paths} != expected_paths
        or any(p.get("conditions") != 32 or p.get("rows") != 128 for p in paths)
    ):
        raise ValueError("R9 Teacher signal paths omit the frozen panel population")


def validate_panel(queue, row, *, checkpoint_role=None, root=None):
    if root is not None:
        if queue.get("family") != ROUND11_SCHEMA or checkpoint_role not in (None, "last"):
            raise ValueError("panel root override is scoped to R11 last-checkpoint repair")
        from run_readout_analysis_repair import validate_output

        root = validate_output(queue, root)
    training = context(queue, row)
    if queue.get("family") == ROUND9_SCHEMA and checkpoint_role is None:
        endpoints = {
            label: validate_panel(queue, row, checkpoint_role=spec["role"])
            for label, spec in ROUND9_ENDPOINTS.items()
        }
        if len({value["panel_sha256"] for value in endpoints.values()}) != 1:
            raise ValueError("R9 endpoints require identical fixed panels")
        return {"endpoints": endpoints}
    from checkpoint_endpoint import select_checkpoint

    role = checkpoint_role or "last"
    if queue.get("family") == ROUND9_SCHEMA:
        cp, endpoint = select_checkpoint(Path(row["plan"]["run_root"]), role)
    else:
        if role != "last":
            raise ValueError("additional panel endpoints require the sealed R9 family")
        cp = training["last"]
        endpoint = dict(
            role="last",
            epoch=2,
            training_stage="frozen_readout" if queue.get("family") == ROUND11_SCHEMA else "joint",
            full_training_epochs=2,
        )
    root = Path(root) if root is not None else Path(queue["output"]) / f"{row['name']}-panel"
    if queue.get("family") == ROUND9_SCHEMA:
        root = root / role
    receipt = read_json(root / "COMPLETE.json")
    expected = dict(
        status="completed",
        updates=0,
        method_sha=queue["source_commit"],
        run_id=row["plan"]["run_id"],
        checkpoint_sha256=cp["sha256"],
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
        or receipt.get("checkpoint_endpoint") != endpoint
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
    if queue.get("family") == ROUND9_SCHEMA:
        validate_staged_joint_teacher_signal(root, receipt, sealed_panel)
    return {
        "receipt": str(root / "COMPLETE.json"),
        "sha256": sha256_file(root / "COMPLETE.json"),
        "panel_sha256": receipt["panel_sha256"],
    }


def panel_command(queue, row, *, checkpoint_role="last"):
    plan, spec = row["plan"], queue["medium_analysis"]
    command = [
        sys.executable,
        str(Path(queue["source"]) / "scripts/v2/run_representation_panel.py"),
    ]
    values = {
        "run-root": plan["run_root"],
        "config": plan["config"],
        "data-root": plan["data_root"],
        "output": str(Path(queue["output"]) / f"{row['name']}-panel" / checkpoint_role)
        if queue.get("family") == ROUND9_SCHEMA
        else str(Path(queue["output"]) / f"{row['name']}-panel"),
        "method-sha": queue["source_commit"],
        "parent-manifest": spec["parent_manifest"],
        "parent-manifest-sha256": spec["parent_manifest_sha256"],
        "resource-queue": str(Path(queue["output"]) / "queue.json"),
        "resource-queue-sha256": sha256_file(Path(queue["output"]) / "queue.json"),
        "checkpoint-role": checkpoint_role,
        "conditions": "32",
        "controls": "4",
        "seed": "20261021",
    }
    for name, value in values.items():
        command.extend(["--" + name, value])
    if queue.get("family") == ROUND9_SCHEMA:
        command.append("--teacher-signal-audit")
    return command


def ssl_loss_contributions(history, config):
    """Separate raw component losses from their fixed configured overall weights."""
    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options

    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    rows = []
    for entry in history:
        terms = entry.get("training", {})
        row = dict(
            epoch=entry["epoch"],
            prediction=terms.get("prediction_mse", terms.get("prediction")),
            training_stage=entry.get(
                "training_stage", entry.get("staged_training", {}).get("phase")
            ),
        )
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
        row.update(
            {
                k: terms.get(k)
                for k in (
                    "au_weighted",
                    "cr_weighted",
                    "spectral_rank",
                    "spectral_cosine",
                    "spectral_rank_weighted",
                    "spectral_cosine_weighted",
                    "spectral_weighted",
                    "spectral_warmup",
                    "joint_loss",
                )
            }
        )
        if options.training_mode == "frozen_readout":
            row.update(
                ssl_status="inactive",
                ssl1_before_overall_weight=None,
                ssl2_before_overall_weight=None,
                ssl1_contribution=None,
                ssl2_contribution=None,
                au_weighted=None,
                cr_weighted=None,
            )
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
    if queue.get("family") in (
        ROUND7_SCHEMA,
        ROUND8_SCHEMA,
        ROUND9_SCHEMA,
        ROUND11_SCHEMA,
        ROUND12_SCHEMA,
    ):
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
    if queue.get("family") == ROUND12_SCHEMA:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        contributions = tables["training_loss_contributions"]
        for key in ("spectral_rank", "spectral_cosine"):
            axes[0].plot(
                [r["epoch"] for r in contributions],
                [r.get(key) for r in contributions],
                "o-",
                label=key,
            )
        for key in ("spectral_rank_weighted", "spectral_cosine_weighted", "spectral_weighted"):
            axes[1].plot(
                [r["epoch"] for r in contributions],
                [r.get(key) for r in contributions],
                "o-",
                label=key,
            )
        for ax, title in zip(
            axes,
            ("Raw complete-population losses", "Actual warmup-weighted contributions"),
            strict=True,
        ):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(report / "spectral_losses.png", dpi=150)
        plt.close(fig)
    text = [
        f"# {row['name']}：冻结端点表征研究\n",
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
    if queue.get("family") == ROUND12_SCHEMA:
        text.append(
            "![完整总体原始及实际加权谱/余弦损失](spectral_losses.png)\n"
            "R12恢复原软秩+正对余弦设计：BAL target=.20/rank=.02/cos=.10，"
            "AGG target=.35/rank=.04/cos=.20；tau=.25，warmup30 completed更新。"
            "原RI架构从头2 epoch/300步，EMA .99→1，AU/CR/KoLeo关闭。"
            "原始SSL/MSE与实际warmup加权rank/cos贡献分列；小探针缺完整320总体，"
            "新项保留NA。主对照复用历史R3-RI，RS-B0/RS-RI仅背景；"
            "跨版本单seed比较不保证同轨迹，无baseline重训，最终val18+MSE，test延期。\n"
        )
        text.extend(
            ["## training_loss_contributions\n", table(tables["training_loss_contributions"])]
        )
    if queue.get("family") == ROUND11_SCHEMA:
        text.append(
            "R11从R9 epoch3完整warm-start，仅原prediction MLP训练300步；"
            "Teacher、centers与backbone保持冻结。父450总步/300MSE加子300 head MSE，"
            "总750/600；额外预算不匹配。SSL/A/U/CR/EMA停用，NA不表示损失改善。"
            "仅最终val18；测试推迟。Teacher几何仅在冻结hash证明后引用R9，"
            "不声称新SSL学习。\n"
        )
    if queue.get("family") == ROUND9_SCHEMA:
        text.append(
            "R9先150步无MSE，再300步持续联合；Teacher、centers和moments连续。总450步比R6多50%，监督仍300步；不是预算匹配。前缀readout未经训练，仅作诊断，不解释为预测效用；只有最后epoch3验证，测试仍推迟。\n"
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
    roles = tuple(ROUND9_ENDPOINTS) if queue.get("family") == ROUND9_SCHEMA else ("last",)
    for role in roles:
        log_name = f"panel-{role}.log" if queue.get("family") == ROUND9_SCHEMA else "panel.log"
        with (root / log_name).open("x") as log:
            subprocess.run(
                panel_command(queue, row, checkpoint_role=role)
                if queue.get("family") == ROUND9_SCHEMA
                else panel_command(queue, row),
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
        + (
            "".join(
                f"[{label}中等分析](../{row['name']}-medium/{label}/report/REPORT_CN.md)\n\n"
                for label in ROUND9_ENDPOINTS
            )
            if queue.get("family") == ROUND9_SCHEMA
            else f"[中等训练分析与验证](../{row['name']}-medium/report/REPORT_CN.md)\n\n"
        )
        + (
            "".join(
                f"[{label}固定面板](../{row['name']}-panel/{label}/COMPLETE.json)\n\n"
                for label in ROUND9_ENDPOINTS
            )
            if queue.get("family") == ROUND9_SCHEMA
            else f"[固定面板与逐层响应诊断](../{row['name']}-panel/COMPLETE.json)\n\n"
        )
        + "研究阶段完成；独立测试尚未进行，下一轮由主会话依据训练/验证证据重新设计。\n"
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
