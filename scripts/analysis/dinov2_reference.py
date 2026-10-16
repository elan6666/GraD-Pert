"""Read-only official DINOv2 features; not a GraD-Pert model or SSL replica.

Imports the pinned official backbone. Reuses GraD-Pert's diagnostic statistic
definitions, but keeps image, class and patch sample units explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))
from cap40_posthoc_accel import sample_moments
from cap40_posthoc_stats import Moments, seeded_int

UPSTREAM = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
LAYERS = (0, 2, 5, 8, 11)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def balanced(records, count, seed):
    """Hash-ranked round-robin classes; independent of input listing order."""
    groups = {}
    for row in records:
        groups.setdefault(row["class_id"], []).append(row)
    if len({r["image_id"] for r in records}) != len(records):
        raise ValueError("duplicate image identity")
    for rows in groups.values():
        rows.sort(key=lambda r: (seeded_int(r["image_id"], seed), r["image_id"]))
    result = []
    for rank in range(max(map(len, groups.values()), default=0)):
        for label in sorted(groups):
            if rank < len(groups[label]):
                result.append(groups[label][rank])
                if len(result) == count:
                    return result
    raise ValueError("not enough images for requested manifest")


def view_image(image, image_id, view, seed):
    """Deterministic geometry only: analysis crops, not official train replay."""
    image = image.convert("RGB")
    w, h = image.size
    if view == "canonical":
        scale = 256 / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        return ImageOps.crop(
            image,
            (
                (image.width - 224) // 2,
                (image.height - 224) // 2,
                (image.width - 224 + 1) // 2,
                (image.height - 224 + 1) // 2,
            ),
        ), None
    rng = np.random.default_rng(seeded_int([image_id, view], seed))
    low, high, size = (0.60, 0.90, 224) if view.startswith("G") else (0.25, 0.50, 98)
    box = None
    for _ in range(10):
        area = w * h * rng.uniform(low, high)
        ratio = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
        cw, ch = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if 0 < cw <= w and 0 < ch <= h:
            left, top = int(rng.integers(w - cw + 1)), int(rng.integers(h - ch + 1))
            box = (left, top, left + cw, top + ch)
            break
    if box is None:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        box = (left, top, left + side, top + side)
    return image.crop(box).resize((size, size), Image.Resampling.BICUBIC), box


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-30)


def paired_cosine(a, b):
    return (unit(a) * unit(b)).sum(-1)


def pair_summary(values, labels):
    """Exact ordered-pair mean; no self-pairs and no patch pseudoreplication."""
    x = unit(values)
    same_sum = same_n = 0
    for label in sorted(set(labels)):
        z = x[np.asarray(labels) == label]
        same_sum += float(z.sum(0) @ z.sum(0) - (z * z).sum())
        same_n += len(z) * (len(z) - 1)
    total_sum = float(x.sum(0) @ x.sum(0) - (x * x).sum())
    total_n = len(x) * (len(x) - 1)
    return [
        {
            "pair_type": "different_image_same_class",
            "ordered_pairs": same_n,
            "mean_cosine": same_sum / same_n if same_n else None,
        },
        {
            "pair_type": "different_class",
            "ordered_pairs": total_n - same_n,
            "mean_cosine": (total_sum - same_sum) / (total_n - same_n)
            if total_n > same_n
            else None,
        },
    ]


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "clean": not git("status", "--porcelain")}


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "CPU analysis requires explicit CUDA_VISIBLE_DEVICES='' to protect active jobs"
        )
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=args.threads)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    native = git_identity(Path(__file__).resolve().parents[2])
    official = git_identity(args.official)
    if not native["clean"] or official != {"commit": UPSTREAM, "clean": True}:
        raise RuntimeError("published clean native and pinned official checkout required")
    if sha256(args.weights) != WEIGHTS:
        raise RuntimeError("unexpected official checkpoint hash")
    records = [
        {
            "image_id": p.relative_to(args.dataset).as_posix(),
            "class_id": p.parent.name,
            "path": str(p),
            "sha256": sha256(p),
        }
        for p in sorted((args.dataset / "val").glob("*/*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    chosen = balanced(records, args.images, args.seed)
    deep = balanced(chosen, min(args.deep_images, args.images), args.seed + 1)
    deep_ids = {r["image_id"] for r in deep}
    manifest = {
        "schema": "dinov2-image-manifest-v1",
        "seed": args.seed,
        "split": "Imagenette-320-validation",
        "images": chosen,
        "deep_image_ids": sorted(deep_ids),
        "selection_uses_features": False,
        "patches_per_deep_image": args.patches,
        "canonical": "Resize shortest256 bicubic, center224, ImageNet normalization",
        "views": {
            "G1": [0.60, 0.90, 224],
            "G2": [0.60, 0.90, 224],
            "L1": [0.25, 0.50, 98],
            "L2": [0.25, 0.50, 98],
        },
        "view_limit": "area fractions of pixels are not gene fractions; geometry-only analysis",
    }
    write_json(out / "manifest.json", manifest)
    if (args.dataset / "COMPLETE.json").exists():
        write_json(
            out / "dataset_receipt.json", json.loads((args.dataset / "COMPLETE.json").read_text())
        )
    sys.path.insert(0, str(args.official))
    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False).eval().requires_grad_(False)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    before = state_digest(model)
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 224, 224)
        ref = model.forward_features(probe)
        layer = model.get_intermediate_layers(probe, n=[11], return_class_token=True)[0]
        parity = max(
            float((ref["x_norm_clstoken"] - layer[1]).abs().max()),
            float((ref["x_norm_patchtokens"] - layer[0]).abs().max()),
        )
    if parity > 1e-6:
        raise RuntimeError("official intermediate-layer accessor parity failed")
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def tensor(row, view):
        with Image.open(row["path"]) as image:
            crop, box = view_image(image, row["image_id"], view, args.seed)
            array = (np.asarray(crop, dtype=np.float32) / 255 - mean) / std
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), box

    moments = {
        (i, kind, norm): Moments(384)
        for i in LAYERS
        for kind in ["CLS", "patch_mixed_deep"]
        for norm in ["raw", "final_LN_applied"]
    }
    all_cls, final_pooled, image_stats, class_ids = [], [], [], []
    load_seconds = forward_seconds = statistics_seconds = 0.0
    for start in range(0, len(chosen), args.batch):
        rows = chosen[start : start + args.batch]
        t = time.perf_counter()
        batch = torch.stack([tensor(row, "canonical")[0] for row in rows])
        load_seconds += time.perf_counter() - t
        t = time.perf_counter()
        with torch.inference_mode():
            layers = model.get_intermediate_layers(
                batch, n=LAYERS, return_class_token=True, norm=False
            )
            normalized = [(model.norm(p), model.norm(c)) for p, c in layers]
        forward_seconds += time.perf_counter() - t
        t = time.perf_counter()
        for i, raw, normed in zip(LAYERS, layers, normalized, strict=True):
            for name, pair in [("raw", raw), ("final_LN_applied", normed)]:
                patch, cls = [v.numpy() for v in pair]
                moments[i, "CLS", name].add(cls)
                for j, row in enumerate(rows):
                    if row["image_id"] not in deep_ids:
                        continue
                    rng = np.random.default_rng(seeded_int([row["image_id"], "patch"], args.seed))
                    ids = rng.choice(len(patch[j]), min(args.patches, len(patch[j])), replace=False)
                    panel = patch[j, ids]
                    moments[i, "patch_mixed_deep", name].add(panel)
                    if name == "final_LN_applied":
                        stat = sample_moments(panel)
                        image_stats.append(
                            {
                                "image_id": row["image_id"],
                                "class_id": row["class_id"],
                                "layer": i + 1,
                                **stat,
                            }
                        )
        all_cls.extend(normalized[-1][1].numpy())
        final_pooled.extend(normalized[-1][0].mean(1).numpy())
        class_ids.extend(row["class_id"] for row in rows)
        statistics_seconds += time.perf_counter() - t
        write_json(
            out / "progress.json",
            {
                "phase": "canonical",
                "images_completed": start + len(rows),
                "images_total": len(chosen),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    features = np.asarray(all_cls)
    patch_means = np.asarray(final_pooled)
    # Matrices stay on the server. Only summaries/plots are transfer candidates.
    np.savez(
        out / "server_only_features.npz",
        cls=features,
        patch_mean=patch_means,
        labels=np.asarray(class_ids),
    )
    layer_stats = []
    for (i, kind, norm), moment in moments.items():
        result = moment.result()
        result["normalized_rank"] = result["effective_rank"] / result["rank_ceiling"]
        layer_stats.append({"layer": i + 1, "feature": kind, "normalization": norm, **result})
    groups = []
    for label in sorted(set(class_ids)):
        sel = np.asarray(class_ids) == label
        for kind, array in [("CLS", features), ("patch_mean", patch_means)]:
            groups.append({"class_id": label, "feature": kind, **sample_moments(array[sel])})
    pairs = pair_summary(features, class_ids)
    view_rows, crop_rows = [], []
    with torch.inference_mode():
        for start in range(0, len(deep), args.batch):
            rows = deep[start : start + args.batch]
            views = {}
            for view in ["G1", "G2", "L1", "L2"]:
                ts = [tensor(r, view) for r in rows]
                t = time.perf_counter()
                views[view] = model.forward_features(torch.stack([x[0] for x in ts]))[
                    "x_norm_clstoken"
                ].numpy()
                forward_seconds += time.perf_counter() - t
                crop_rows.extend(
                    {"image_id": row["image_id"], "view": view, "box": box}
                    for row, (_, box) in zip(rows, ts, strict=True)
                )
            for a, b in [("G1", "G2"), ("G1", "L1"), ("G1", "L2"), ("G2", "L1"), ("G2", "L2")]:
                cos = paired_cosine(views[a], views[b])
                view_rows.extend(
                    {
                        "image_id": r["image_id"],
                        "class_id": r["class_id"],
                        "pair_type": f"same_image_{a}_{b}",
                        "cosine": float(c),
                    }
                    for r, c in zip(rows, cos, strict=True)
                )
            write_json(
                out / "progress.json",
                {
                    "phase": "views",
                    "deep_images_completed": start + len(rows),
                    "deep_images_total": len(deep),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("read-only state/gradient contract failed")
    for pair in sorted({r["pair_type"] for r in view_rows}):
        values = [r["cosine"] for r in view_rows if r["pair_type"] == pair]
        pairs.append(
            {"pair_type": pair, "ordered_pairs": len(values), "mean_cosine": float(np.mean(values))}
        )
    csv_write(out / "layer_statistics.csv", layer_stats)
    csv_write(out / "class_statistics.csv", groups)
    csv_write(out / "within_image_patch_statistics.csv", image_stats)
    csv_write(out / "cosine_groups.csv", pairs)
    csv_write(out / "same_image_views.csv", view_rows)
    write_json(out / "crop_manifest.json", crop_rows)
    summary = {
        "schema": "dinov2-reference-medium-v1",
        "native_source": native,
        "official_source": official,
        "weights_sha256": WEIGHTS,
        "manifest_sha256": sha256(out / "manifest.json"),
        "dataset_receipt_sha256": sha256(args.dataset / "COMPLETE.json")
        if (args.dataset / "COMPLETE.json").exists()
        else None,
        "images": len(chosen),
        "classes": len(set(class_ids)),
        "deep_images": len(deep),
        "patches_per_image": args.patches,
        "width": 384,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": "cpu",
        "threads": args.threads,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "optimizer_updates": 0,
        "gradients": 0,
        "state_unchanged": True,
        "state_sha256": after,
        "accessor_parity_max_abs": parity,
        "layer_statistics": layer_stats,
        "cosine_groups": pairs,
        "prototype_statistics": {
            "status": "not_available",
            "reason": (
                "official public checkpoint is backbone-only; no original SSL head/teacher/center"
            ),
        },
        "timings": {
            "canonical_load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "canonical_statistics_seconds": statistics_seconds,
            "analysis_wall_seconds": time.perf_counter() - started,
        },
    }
    write_json(out / "summary.json", summary)
    render(out, summary, image_stats)
    files = [
        {
            "path": p.name,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "transferable": p.suffix in {".json", ".csv", ".md", ".png"},
        }
        for p in sorted(out.iterdir())
        if p.is_file()
    ]
    write_json(
        out / "COMPLETE.json",
        {
            "success": True,
            "summary_sha256": sha256(out / "summary.json"),
            "state_unchanged": True,
            "optimizer_updates": 0,
            "files": files,
        },
    )


def render(out, summary, image_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["layer_statistics"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for kind in ["CLS", "patch_mixed_deep"]:
        for norm, style in [("raw", "--"), ("final_LN_applied", "-")]:
            sub = [r for r in rows if r["feature"] == kind and r["normalization"] == norm]
            for ax, field in zip(
                axes, ["effective_rank", "normalized_rank", "relative_variance"], strict=True
            ):
                ax.plot(
                    [r["layer"] for r in sub],
                    [r[field] for r in sub],
                    style,
                    marker="o",
                    label=f"{kind}/{norm}",
                )
                ax.set(xlabel="Transformer layer", ylabel=field)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "layer_statistics.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [[r["effective_rank"] for r in image_stats if r["layer"] == i + 1] for i in LAYERS],
        tick_labels=[i + 1 for i in LAYERS],
        showfliers=False,
    )
    ax.set(xlabel="Transformer layer", ylabel="Within-image patch effective rank")
    fig.tight_layout()
    fig.savefig(out / "within_image_patch_rank.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    pairs = summary["cosine_groups"]
    ax.bar(range(len(pairs)), [r["mean_cosine"] for r in pairs])
    ax.set_xticks(
        range(len(pairs)),
        [r["pair_type"].replace("_", " ") for r in pairs],
        rotation=25,
        ha="right",
        fontsize=7,
    )
    ax.set(ylabel="Mean cosine", ylim=(-1, 1))
    fig.tight_layout()
    fig.savefig(out / "cosine_groups.png", dpi=170)
    plt.close(fig)
    final = [r for r in rows if r["layer"] == 12 and r["normalization"] == "final_LN_applied"]
    table = "\n".join(
        f"|{r['feature']}|{r['rows']}|{r['effective_rank']:.3f}|{r['normalized_rank']:.4f}|{r['relative_variance']:.4f}|"
        for r in final
    )
    from string import Template

    template = Path(__file__).parent / "templates/dinov2_reference_cn.md"
    report = Template(template.read_text()).substitute(
        images=summary["images"],
        classes=summary["classes"],
        deep_images=summary["deep_images"],
        threads=summary["threads"],
        minutes=f"{summary['timings']['analysis_wall_seconds'] / 60:.2f}",
        table=table,
        upstream=UPSTREAM,
        native=summary["native_source"]["commit"],
        weights=WEIGHTS,
        manifest=summary["manifest_sha256"],
        dataset_receipt=summary["dataset_receipt_sha256"],
        parity=summary["accessor_parity_max_abs"],
    )
    (out / "REPORT_CN.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1335)
    parser.add_argument("--deep-images", type=int, default=506)
    parser.add_argument("--patches", type=int, choices=[128], default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.images, args.deep_images, args.batch, args.threads) < 1 or args.threads > 4:
        parser.error("positive counts and at most four CPU threads required")
    run(args)


if __name__ == "__main__":
    main()
