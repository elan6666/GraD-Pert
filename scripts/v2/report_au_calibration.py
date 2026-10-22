"""Export verified AU calibration aggregates, without importing a model or torch.

Example: python scripts/v2/report_au_calibration.py --init INIT_DIR
  --ri-last RI_DIR --output NEW_REPORT_DIR --alignment-weight 0.1
  --uniformity-weight 0.1 --strong-multiplier 2
Both inputs must contain all eight completed, hash-indexed batch receipts.
"""
# Chinese prose and Markdown table rows intentionally remain on one source line.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

LOSSES = ("A", "U", "MSE", "joint")
GROUPS = ("response", "graph")
VIEWS = ("global0", "global1")
PAIRS = tuple(itertools.combinations(LOSSES, 2))
FROZEN = {
    "student",
    "teacher",
    "centers",
    "optimizer",
    "gradients",
    "numpy_rng",
    "python_rng",
    "torch_rng",
    "pending_statistics",
    "auxiliary_statistics",
    "module_modes",
}


def sha_json(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def finite(value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric aggregate")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError("aggregate outside its defined range")
    return value


def digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path, source_index):
    raw = path.read_bytes()

    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    value = json.loads(raw, parse_constant=reject)
    source_index[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return value


def validate_census(value):
    require(value["input_count"] == 320, "census must contain the full population320")
    n = finite(value["unique_count"], minimum=2, maximum=320)
    require(n + value["duplicate_count"] == 320, "inconsistent dedup census")
    pairs = finite(value["pair_count"], minimum=1)
    contexts = value["contexts"]
    require(len(contexts) == value["context_count"], "inconsistent context count")
    require(len({r["context_id"] for r in contexts}) == len(contexts), "duplicate context rows")
    require(sum(r["input_count"] for r in contexts) == n, "inconsistent context population")
    require(sum(r["pair_count"] for r in contexts) == pairs, "inconsistent context pair count")
    for row in contexts:
        finite(row["pair_count"], minimum=0)
        require(
            math.isclose(row["pair_weight"], row["pair_count"] / pairs, abs_tol=1e-12),
            "context pair weight differs from census",
        )
    require(digest(value["ordered_identity_sha256"]), "missing ordered population identity")


def validate_record(record, index, plan_hash, census):
    require(
        record["plan_sha256"] == plan_hash and record["batch_index"] == index,
        "batch belongs to another plan or position",
    )
    require(record["census"] == census, "batch census differs from sealed metadata")
    require(
        record["updates"] == 0 and record["state_unchanged"] is True,
        "batch is not a no-update observation",
    )
    require(
        record["unique_count"] == census["unique_count"]
        and record["pair_count"] == census["pair_count"],
        "AU population differs from census",
    )
    require(record["replay_tolerance"] == 2e-6, "replay tolerance has been changed")
    finite(record["replay_max_abs_error"], minimum=0, maximum=2e-6)
    require(
        set(record["frozen_state"]) >= FROZEN
        and all(digest(record["frozen_state"][k]) for k in FROZEN),
        "missing frozen state fingerprints",
    )
    require(bool(record["environment"]), "missing observed runtime identity")
    require(set(record["losses"]) == set(LOSSES), "missing calibrated losses")
    for loss in LOSSES:
        finite(record["losses"][loss])
    require(set(record["gradients"]) == set(GROUPS), "missing gradient parameter group")
    for group in GROUPS:
        gradients = record["gradients"][group]
        require(set(gradients["norms"]) == set(LOSSES), "missing gradient norms")
        finite(gradients["parameter_count"], minimum=1)
        for loss in LOSSES:
            finite(gradients["norms"][loss], minimum=0)
        require(
            set(gradients["pairs"]) == {a + "__" + b for a, b in PAIRS}, "missing gradient angles"
        )
        for a, b in PAIRS:
            pair = gradients["pairs"][a + "__" + b]
            if gradients["norms"][a] * gradients["norms"][b] == 0:
                require(
                    pair["cosine"] is None and pair["angle_degrees"] is None,
                    "zero gradient must have undefined angles",
                )
            else:
                cosine = finite(pair["cosine"], minimum=-1, maximum=1)
                angle = finite(pair["angle_degrees"], minimum=0, maximum=180)
                require(
                    math.isclose(angle, math.degrees(math.acos(cosine)), abs_tol=1e-6),
                    "gradient angle is inconsistent with cosine",
                )
    require(set(record["spectrum"]) == set(VIEWS), "missing Global spectrum")
    for view in VIEWS:
        spectrum = record["spectrum"][view]
        require(spectrum["sample_count"] == census["unique_count"], "spectrum population differs")
        upper = min(spectrum["width"], spectrum["sample_count"] - 1)
        rank = finite(spectrum["effective_rank"], minimum=0, maximum=upper + 1e-7)
        finite(spectrum["normalized_rank"], minimum=0, maximum=1 + 1e-7)
        require(
            math.isclose(spectrum["normalized_rank"], rank / upper, abs_tol=1e-8),
            "normalized rank has a different denominator",
        )
        finite(spectrum["relative_variance"], minimum=0, maximum=1 + 1e-7)
    elapsed = finite(record["elapsed_seconds"], minimum=1e-12)
    require(
        math.isclose(record["cells_per_second"], 320 / elapsed, rel_tol=1e-8),
        "throughput definition differs",
    )
    finite(record["peak_allocated_bytes"], minimum=0)
    finite(record["peak_reserved_bytes"], minimum=record["peak_allocated_bytes"])


def load_endpoint(root, name, source_index):
    root = root.resolve(strict=True)
    plan = _read(root / "PLAN.json", source_index)
    census = _read(root / "CENSUS.json", source_index)
    complete = _read(root / "COMPLETE.json", source_index)
    index = _read(root / "BATCH_INDEX.json", source_index)
    plan_hash = sha_json(plan)
    require(
        plan["schema"] == "gradpert-au-calibration-1" and plan["mode"] == "gradients",
        "unsupported calibration plan",
    )
    require(
        plan["batches"] == 8
        and plan["effective_population"] == 320
        and plan["microbatch"] == 80
        and plan["updates"] == 0
        and plan["split"] == "train_only"
        and plan["topology"] == "single_gpu_diagnostic_not_ddp_resume",
        "calibration contract differs",
    )
    require(
        plan["source"]["dirty"] is False and plan["source"]["formal_eligible"] is True,
        "calibration source was not clean and published",
    )
    require(
        census["plan_sha256"] == plan_hash
        and census["complete"] is True
        and census["model_forward"] is False
        and census["updates"] == 0
        and len(census["batches"]) == 8,
        "census is partial or mismatched",
    )
    require(
        complete["plan_sha256"] == plan_hash
        and complete["calibration_complete"] is True
        and complete["scientific_complete"] is False
        and complete["updates"] == 0
        and complete["batches_completed"] == 8
        and complete["expected"] == 8,
        "eight complete no-update observations are required",
    )
    names = {f"batch-{i:02d}.json" for i in range(8)}
    require(
        index["plan_sha256"] == plan_hash
        and set(index["receipts"]) == names
        and complete["batch_receipts"] == index["receipts"],
        "completion and checksum index differ",
    )
    require(
        {p.name for p in root.glob("batch-*.json")} == names, "missing or unexpected batch files"
    )
    records = []
    for number in range(8):
        file = root / f"batch-{number:02d}.json"
        record = _read(file, source_index)
        require(source_index[str(file)] == index["receipts"][file.name], "batch checksum mismatch")
        validate_census(census["batches"][number])
        validate_record(record, number, plan_hash, census["batches"][number])
        records.append(record)
    for key in ("student", "teacher", "centers", "optimizer"):
        require(
            len({r["frozen_state"][key] for r in records}) == 1,
            "frozen model state changed between calibration populations",
        )
    return dict(name=name, plan=plan, census=census["batches"], records=records)


def compose_gradients(gradients, alignment_weight, uniformity_weight):
    """Reconstruct exact linear-combination geometry from saved norms and cosines."""
    norms = gradients["norms"]

    def dot(a, b):
        if a == b:
            return norms[a] ** 2
        if norms[a] * norms[b] == 0:
            return 0.0
        key = a + "__" + b
        pair = gradients["pairs"].get(key, gradients["pairs"].get(b + "__" + a))
        return norms[a] * norms[b] * pair["cosine"]

    output = {}
    for name, coefficients in (
        ("regularizer", {"A": alignment_weight, "U": uniformity_weight}),
        ("total", {"A": alignment_weight, "U": uniformity_weight, "joint": 1.0}),
    ):
        terms = [
            ca * cb * dot(a, b) for a, ca in coefficients.items() for b, cb in coefficients.items()
        ]
        squared = sum(terms)
        require(
            squared >= -1e-10 * max(1.0, sum(map(abs, terms))),
            "saved gradient Gram matrix is inconsistent",
        )
        norm = math.sqrt(max(0.0, squared))
        output[name + "_norm"] = norm
        for reference in ("MSE", "joint"):
            cross = sum(c * dot(loss, reference) for loss, c in coefficients.items())
            cosine = (
                max(-1.0, min(1.0, cross / (norm * norms[reference])))
                if norm * norms[reference]
                else None
            )
            output[f"{name}__{reference}_cosine"] = cosine
            output[f"{name}__{reference}_angle_degrees"] = (
                math.degrees(math.acos(cosine)) if cosine is not None else None
            )
    return output


def flatten(endpoints, weights):
    rows = []
    for endpoint in endpoints:
        for record in endpoint["records"]:
            row = dict(
                endpoint=endpoint["name"],
                batch_index=record["batch_index"],
                training_source_commit=(endpoint["plan"].get("training_identity") or {})
                .get("source", {})
                .get("commit"),
                calibration_source_commit=endpoint["plan"]["source"]["commit"],
                checkpoint_sha256=endpoint["plan"].get("checkpoint_sha256"),
                ordered_identity_sha256=record["census"]["ordered_identity_sha256"],
            )
            row.update({"loss_" + k: v for k, v in record["losses"].items()})
            for key in (
                "input_count",
                "unique_count",
                "duplicate_count",
                "pair_count",
                "context_count",
            ):
                row[key] = record["census"][key]
            row["contexts_json"] = json.dumps(record["census"]["contexts"], ensure_ascii=False)
            for key in (
                "elapsed_seconds",
                "cells_per_second",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "replay_max_abs_error",
            ):
                row[key] = record[key]
            for group in GROUPS:
                gradients = record["gradients"][group]
                row[group + "_parameter_count"] = gradients["parameter_count"]
                row.update({f"{group}_{k}_norm": v for k, v in gradients["norms"].items()})
                for pair, values in gradients["pairs"].items():
                    row.update({f"{group}_{pair}_{key}": value for key, value in values.items()})
                for variant, (a, u) in weights.items():
                    row.update(
                        {
                            f"{group}_{variant}_{key}": value
                            for key, value in compose_gradients(gradients, a, u).items()
                        }
                    )
            for view in VIEWS:
                row.update({f"{view}_{k}": v for k, v in record["spectrum"][view].items()})
            rows.append(row)
    return rows


def figures(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"init": "#d76f57", "ri-last": "#4276b8"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, group in zip(axes, GROUPS, strict=True):
        for endpoint, offset in (("init", -0.12), ("ri-last", 0.12)):
            for i, label in enumerate(LOSSES):
                values = [r[f"{group}_{label}_norm"] for r in rows if r["endpoint"] == endpoint]
                ax.scatter([i + offset] * 8, values, s=19, alpha=0.5, color=colors[endpoint])
                ax.scatter(
                    i + offset,
                    statistics.mean(values),
                    s=70,
                    marker="_",
                    color=colors[endpoint],
                    label=endpoint if i == 0 else None,
                )
        ax.set(
            title=group + " parameter gradients",
            xticks=range(4),
            xticklabels=LOSSES,
            ylabel="Gradient L2 norm (8 fixed batches)",
        )
        norms = [r[f"{group}_{loss}_norm"] for r in rows for loss in LOSSES]
        positive = [value for value in norms if value > 0]
        if len(positive) == len(norms):
            ax.set_yscale("log")
            ax.set_ylim(min(positive) * 0.5, max(positive) * 2)
        else:
            ax.set_yscale("symlog", linthresh=min(positive, default=1.0) * 0.5)
            ax.set_ylim(0, max(positive, default=1.0) * 2)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.savefig(output / "gradient_norms.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("effective_rank", "normalized_rank", "relative_variance"),
        ("Effective rank", "Normalized effective rank", "Relative variance"),
        strict=True,
    ):
        for endpoint, offset in (("init", -0.12), ("ri-last", 0.12)):
            for i, view in enumerate(VIEWS):
                values = [r[f"{view}_{metric}"] for r in rows if r["endpoint"] == endpoint]
                ax.scatter([i + offset] * 8, values, s=22, alpha=0.5, color=colors[endpoint])
                ax.scatter(
                    i + offset,
                    statistics.mean(values),
                    s=80,
                    marker="_",
                    color=colors[endpoint],
                    label=endpoint if i == 0 else None,
                )
        ax.set(title=title, xticks=(0, 1), xticklabels=("Global 1", "Global 2"))
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.savefig(output / "representation_geometry.png", dpi=180)
    plt.close(fig)


def mean_sd(rows, key, endpoint):
    values = [row[key] for row in rows if row["endpoint"] == endpoint and row[key] is not None]
    return "NA" if not values else f"{statistics.mean(values):.6g} ± {statistics.stdev(values):.4g}"


def report(endpoints, rows, weights):
    lines = [
        "# AU 无更新校准：初始化与 RI 末轮",
        "",
        "本报告核对初始化与 RI-last 两端各 8 个完整固定训练批（每批 320 条，4×80 microbatch）。仅汇总保存的标量与指纹，不加载模型、不执行前向、不改变历史实验。完整校准表示观测证据齐备，**不代表表征问题已经解决**。",
        "",
        "## 统计口径",
        "",
        "- 采用训练模式 dropout 和私有固定随机种子；两端使用相同的 8 份有序训练人口及视图采样规则。",
        "- 每端是单 GPU 独立无更新诊断，不是双卡训练恢复或训练路径等价验证。",
        "- AU 使用两路 Student Global Response CLS，经原有末端 RMSNorm 后取出；仅 AU 辅助运算做 L2 归一化。谱与相对方差统计使用原始 CLS。",
        "- 先按 `(control, perturbation)` 全批去重；U 仅比较同 cell_type::batch、不同扰动的无序对，按有效配对数加权。",
        "- 以下均值±标准差来自 8 批观测。它不是独立生物学样本的置信区间，也不是跨批拼接后重新计算的谱或梯度范数。",
        "- 这些训练模式、固定人口的秩不能直接等同于 medium 分析的评估模式秩；耗时包含两次前向、四类梯度及状态指纹，不能等同于训练吞吐。",
        "",
        "## 来源与校验",
        "",
        "|端点|运行 ID|校准源码 SHA|训练源码 SHA|checkpoint SHA256|",
        "|---|---|---|---|---|",
    ]
    for endpoint in endpoints:
        plan = endpoint["plan"]
        training = (
            (plan.get("training_identity") or {}).get("source", {}).get("commit", "不适用：初始化")
        )
        lines.append(
            f"|{endpoint['name']}|{plan['run_id']}|{plan['source']['commit']}|{training}|{plan.get('checkpoint_sha256') or '不适用'}|"
        )
    lines += [
        "",
        "所有 16 批的文件哈希与两份完成收据、索引一致；updates=0、state_unchanged=true、重放最大误差≤2e-6。逐文件 SHA256 见 [SOURCE_INDEX.json](SOURCE_INDEX.json)。",
        "",
        "## 损失与成本",
        "",
        "|指标|初始化|RI-last|",
        "|---|---:|---:|",
    ]
    for key in [
        *("loss_" + k for k in LOSSES),
        "elapsed_seconds",
        "cells_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ]:
        lines.append(f"|{key}|{mean_sd(rows, key, 'init')}|{mean_sd(rows, key, 'ri-last')}|")
    lines += [
        "",
        "A 衡量同一样本两路 Global 的不一致；越低表示更对齐。U 是负平方距离的 log-mean-exp；越负通常表示所选负对更分散。单独降低任何一项都不能证明生物学区分度改善。",
        "",
        "## 梯度强度与方向",
        "",
        "![梯度范数](gradient_norms.png)",
        "",
        "圆点为各批梯度范数，横线为八批算术均值。纵轴在全部范数为正时采用 log；存在零值时采用保留零值的 symlog。Response 与 Graph 的参数规模不同，不宜只比较绝对范数。",
        "",
        "|参数组／目标|初始化范数|RI-last 范数|",
        "|---|---:|---:|",
    ]
    for group in GROUPS:
        for loss in LOSSES:
            key = f"{group}_{loss}_norm"
            lines.append(
                f"|{group} / {loss}|{mean_sd(rows, key, 'init')}|{mean_sd(rows, key, 'ri-last')}|"
            )
    lines += ["", "|参数组／梯度对|初始化夹角（度）|RI-last 夹角（度）|", "|---|---:|---:|"]
    for group in GROUPS:
        for a, b in PAIRS:
            key = f"{group}_{a}__{b}_angle_degrees"
            lines.append(
                f"|{group} / {a}–{b}|{mean_sd(rows, key, 'init')}|{mean_sd(rows, key, 'ri-last')}|"
            )
    lines += [
        "",
        "夹角小于90°表示局部梯度方向较一致，大于90°表示局部冲突；它不等于一次实际更新后的损失变化。任一梯度为零时夹角为 NA。",
        "",
        "## 表征几何",
        "",
        "![有效秩与相对方差](representation_geometry.png)",
        "",
        "每个圆点对应一批；两路 Global 分别统计。有效秩描述中心化差异的谱分散度；归一化秩除以 min(维数,N−1)；相对方差描述中心化变化相对于总能量的占比。高值也可能来自噪声，需要与任务价值及稳定性共同解释。",
        "",
        "|Global／指标|初始化|RI-last|",
        "|---|---:|---:|",
    ]
    for view in VIEWS:
        for metric in ("effective_rank", "normalized_rank", "relative_variance"):
            key = f"{view}_{metric}"
            lines.append(
                f"|{view} / {metric}|{mean_sd(rows, key, 'init')}|{mean_sd(rows, key, 'ri-last')}|"
            )
    lines += [
        "",
        "## 有效人口与配对",
        "",
        "下表两端完全一致；不同批之间可能重复细胞或条件，不能把 8×320 视作 2,560 个独立细胞。",
        "",
        "|批次|输入行|去重后|重复行|有效无序负对|context数|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, census in enumerate(endpoints[0]["census"]):
        lines.append(
            f"|{i}|{census['input_count']}|{census['unique_count']}|{census['duplicate_count']}|{census['pair_count']}|{census['context_count']}|"
        )
    if weights:
        lines += [
            "",
            "## 指定系数的梯度组合推导",
            "",
            "这部分只对已保存梯度几何做线性组合，不选择系数、不模拟参数更新。设辅助梯度 gᵣ=a·gA+u·gU，总梯度 gₜ=g_joint+gᵣ：",
            "",
            "$$",
            r"\|g_r\|^2=a^2\|g_A\|^2+u^2\|g_U\|^2+2au\|g_A\|\|g_U\|\cos(g_A,g_U).",
            "$$",
            "",
            "与 MSE／joint 的内积按相同线性关系求和，再除以双方范数得到余弦和夹角。结果在保存的浮点标量精度下精确重建；使用完整指定系数，不包含 warmup 缩放。",
            "",
            "|版本|a|u|",
            "|---|---:|---:|",
        ]
        for variant, (a, u) in weights.items():
            lines.append(f"|{variant}|{a:g}|{u:g}|")
        lines += ["", "|参数组／版本／指标|初始化|RI-last|", "|---|---:|---:|"]
        for group in GROUPS:
            for variant in weights:
                for metric in (
                    "regularizer_norm",
                    "total_norm",
                    "regularizer__MSE_angle_degrees",
                    "regularizer__joint_angle_degrees",
                    "total__MSE_angle_degrees",
                    "total__joint_angle_degrees",
                ):
                    key = f"{group}_{variant}_{metric}"
                    lines.append(
                        f"|{group} / {variant} / {metric}|{mean_sd(rows, key, 'init')}|{mean_sd(rows, key, 'ri-last')}|"
                    )
    lines += [
        "",
        "## 完整分析数据",
        "",
        "[16批全部标量、梯度夹角、context权重与成本 CSV](calibration_batches.csv) · [来源文件与输出图表索引](SOURCE_INDEX.json)",
        "",
        "本报告提供下一轮系数与机制判断的校准证据。是否改善表征，仍需后续训练结果、相同 medium 清单的稳定性／区分度分析以及预测指标共同验证。",
        "",
    ]
    return "\n".join(lines)


def export_report(
    init, ri_last, output, *, alignment_weight=None, uniformity_weight=None, strong_multiplier=2.0
):
    require(
        (alignment_weight is None) == (uniformity_weight is None),
        "supply both AU coefficients or neither",
    )
    weights = {}
    if alignment_weight is not None:
        a, u = finite(alignment_weight, minimum=0), finite(uniformity_weight, minimum=0)
        multiplier = finite(strong_multiplier, minimum=1)
        weights = {"balanced": (a, u), "strong": (a * multiplier, u * multiplier)}
    require(init.resolve() != ri_last.resolve(), "two distinct endpoints required")
    require(not output.exists(), "report output must be new; preserve historical reports")
    sources = {}
    endpoints = [load_endpoint(init, "init", sources), load_endpoint(ri_last, "ri-last", sources)]
    initial, final = (e["plan"] for e in endpoints)
    require(
        initial.get("checkpoint_sha256") is None and initial.get("training_identity") is None,
        "initial endpoint unexpectedly loads a checkpoint",
    )
    require(
        digest(final.get("checkpoint_sha256")) and bool(final.get("training_identity")),
        "RI endpoint lacks checkpoint identity",
    )
    for key in ("config_sha256", "seed", "rng_seed", "temperature", "source"):
        require(initial[key] == final[key], "endpoint calibration semantics differ: " + key)
    require(
        endpoints[0]["census"] == endpoints[1]["census"],
        "the ordered eight training populations differ",
    )
    rows = flatten(endpoints, weights)
    # All source checks precede writes; report construction never changes input receipts.
    for path, checksum in sources.items():
        require(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum,
            "source changed during export",
        )
    output.mkdir(parents=True)
    with (output / "calibration_batches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(rows, output)
    (output / "REPORT_CN.md").write_text(report(endpoints, rows, weights))
    index = dict(
        schema="gradpert-au-calibration-report-1",
        input_files=sources,
        calibration_observations=16,
        model_forward=False,
        updates=0,
        scientific_complete=False,
        coefficient_source="explicit_arguments" if weights else None,
        coefficients=weights,
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
    )
    (output / "SOURCE_INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("init", "ri-last", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--alignment-weight", type=float)
    parser.add_argument("--uniformity-weight", type=float)
    parser.add_argument("--strong-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    result = export_report(
        args.init,
        args.ri_last,
        args.output,
        alignment_weight=args.alignment_weight,
        uniformity_weight=args.uniformity_weight,
        strong_multiplier=args.strong_multiplier,
    )
    print(
        json.dumps(
            dict(
                report=str(args.output / "REPORT_CN.md"),
                observations=result["calibration_observations"],
                scientific_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
