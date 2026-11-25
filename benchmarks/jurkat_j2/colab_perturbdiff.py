"""Run frozen Jurkat PerturbDiff on a temporary GPU using only inference inputs."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json


def prepare(output, origin):
    """Export deduplicated ordered controls, checkpoint, and released prior only."""
    import yaml
    from scipy import sparse

    from benchmarks.crosscell.official_fit_probe import COMMITS
    from benchmarks.jurkat_j2.perturbdiff import ASSETS, CONFIG, UPSTREAM, j2_prior, target
    from benchmarks.jurkat_j2.study import COHORT_SHA, cohort, context, fit_data

    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(expression_genes=5000, max_wall_seconds=10800)
    original = read_json(origin / "result.json")
    conditions_expected = 592
    checkpoint = origin / "checkpoint.pt"
    if (
        original["status"] != "evaluated"
        or not original["formal_eligible"]
        or original["checkpoint_sha256"] != sha256_file(checkpoint)
        or original["test_updates"] != 0
        or original["official_commit"] != COMMITS["perturbdiff"]
        or original["cohort_sha256"] != COHORT_SHA
        or original["config_sha256"] != sha256_json(cfg)
    ):
        raise ValueError("the accepted PerturbDiff training receipt changed")
    prepared = fit_data(cfg["seed"])
    fit_names = {target(n) for n in (*prepared.train_conditions, *prepared.val_conditions)}
    row_context = dict(zip(prepared.adata.obs_names, prepared.row_context_ids, strict=True))
    fit_batches = {row_context[row] for row in prepared.control_row_ids}
    with context() as data:
        draws = data.control_manifest.draws
        conditions = [draw.condition_id for draw in draws]
        batch_by_row = dict(zip(data.row_ids, data.context_ids, strict=True))
        unique_rows = list(dict.fromkeys(row for draw in draws for row in draw.ordered_row_ids))
        row_position = {row: i for i, row in enumerate(unique_rows)}
        if (
            len(conditions) != conditions_expected
            or data.control_manifest_file_sha256 != cohort()["evaluation_controls_sha256"]
            or sha256_json(list(data.expression_gene_ids))
            != cohort()["expression_gene_order_sha256"]
            or any(len(draw.ordered_row_ids) != 300 for draw in draws)
        ):
            raise ValueError("the frozen J2 controls or complete gene axis changed")
        batches = [batch_by_row[row] for row in unique_rows]
        if not set(batches) <= fit_batches:
            raise ValueError("unexpected unseen batch in the portable inference inputs")
        draw_indices = np.asarray(
            [[row_position[row] for row in draw.ordered_row_ids] for draw in draws],
            dtype=np.int32,
        )
        blocks = []
        for i in range(0, len(unique_rows), 256):
            rows = tuple(unique_rows[i : i + 256])
            values = data.load_control_rows(rows)
            if values.ordered_row_ids != rows:
                raise ValueError("the canonical control loader did not preserve row order")
            blocks.append(sparse.csr_matrix(values.expression))
        controls = sparse.vstack(blocks, format="csr")
        paired_sha = data.control_manifest_file_sha256
    test_names = {target(n) for n in conditions}
    vocab = {
        "pert": {g: i for i, g in enumerate(sorted(fit_names | test_names | {"non-targeting"}))},
        "cell_line": {"jurkat": 0},
        "batch": {name: i for i, name in enumerate(sorted(fit_batches))},
    }
    output.mkdir()
    missing_fit, missing_test = j2_prior(fit_names, test_names, output / "gene-prior.pkl")
    if [missing_fit, missing_test] != [
        original["missing_fit_targets"],
        original["missing_test_targets"],
    ]:
        raise ValueError("the frozen prior coverage changed")
    sparse.save_npz(output / "input-controls.npz", controls)
    np.savez_compressed(
        output / "draws.npz", condition_ids=conditions, draw_indices=draw_indices, batches=batches
    )
    shutil.copyfile(checkpoint, output / "checkpoint.pt")
    shutil.copyfile(ASSETS / "celltype.pkl", output / "celltype.pkl")
    from subprocess import run

    run(
        [
            "git",
            "-C",
            str(UPSTREAM),
            "archive",
            "--format=tar",
            "-o",
            str(output / "official.tar"),
            original["official_commit"],
            "src",
            "configs",
        ],
        check=True,
    )
    files = {file.name: sha256_file(file) for file in output.iterdir()}
    atomic_json(
        output / "manifest.json",
        {
            "files": files,
            "config": cfg,
            "vocabulary": vocab,
            "conditions": len(conditions),
            "unique_control_rows": len(unique_rows),
            "cohort_sha256": COHORT_SHA,
            "paired_rows_sha256": paired_sha,
            "test_updates": 0,
            "official_commit": original["official_commit"],
            "original_receipt_sha256": sha256_file(origin / "result.json"),
        },
    )


def infer(inputs, output, *, stop=None):
    """Emit a condition-mean prediction and its receipts. No truth is loaded."""
    import subprocess
    import tarfile

    import torch
    from scipy import sparse

    from benchmarks.crosscell.official_fit_probe import build_perturbdiff
    from benchmarks.crosscell.perturbdiff_c2 import predict

    manifest = read_json(inputs / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(inputs / name) != expected:
            raise ValueError(f"portable inference checksum mismatch: {name}")
    if torch.cuda.device_count() != 1:
        raise ValueError("portable inference requires a single GPU")
    output.mkdir(exist_ok=True)
    upstream = output / "official"
    upstream.mkdir(exist_ok=True)
    with tarfile.open(inputs / "official.tar") as package:
        package.extractall(upstream, filter="data")
    sys.path.insert(0, str(upstream))
    cfg, vocab = manifest["config"], manifest["vocabulary"]
    model, *_ = build_perturbdiff(
        upstream,
        inputs / "gene-prior.pkl",
        vocabulary=vocab,
        genes=5000,
        cell_prior=inputs / "celltype.pkl",
        max_steps=cfg["max_updates"],
    )
    model.load_state_dict(
        torch.load(inputs / "checkpoint.pt", map_location="cpu", weights_only=True)
    )
    model.eval().requires_grad_(False)
    controls = sparse.load_npz(inputs / "input-controls.npz")
    with np.load(inputs / "draws.npz", allow_pickle=False) as binding:
        conditions = binding["condition_ids"]
        rows = binding["draw_indices"]
        batches = binding["batches"]
    if rows.shape != (manifest["conditions"], 300) or controls.shape[1] != 5000:
        raise ValueError("portable frozen input population changed")
    count = len(conditions) if stop is None else min(stop, len(conditions))
    started = time.monotonic()
    means = []
    for i, (name, indices) in enumerate(zip(conditions[:count], rows[:count], strict=True), 1):
        name = str(name)
        torch.manual_seed(int(sha256_json([cfg["evaluation_seed"], name])[:8], 16))
        values = predict(
            model,
            vocab,
            cfg,
            name.removesuffix("+ctrl"),
            "jurkat",
            controls[indices].toarray(),
            batches[indices],
        )
        if values.shape != (300, 5000) or not np.isfinite(values).all():
            raise ValueError("invalid portable PerturbDiff predictions")
        means.append(values.mean(0, dtype=np.float64))
        if i == 1 or i % 10 == 0 or i == count:
            print(f"{i}/{count} conditions, {time.monotonic() - started:.1f}s", flush=True)
    predictions = output / "condition-means.npz"
    np.savez_compressed(predictions, conditions=conditions[:count], means=np.asarray(means))
    atomic_json(
        output / "inference.json",
        {
            "manifest_sha256": sha256_file(inputs / "manifest.json"),
            "predictions_sha256": sha256_file(predictions),
            "checkpoint_sha256": manifest["files"]["checkpoint.pt"],
            "conditions": count,
            "test_controls_per_condition": 300,
            "test_updates": 0,
            "test_truth_accessed": False,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "elapsed_seconds": time.monotonic() - started,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
        },
    )


def score(input_manifest, inference, output, origin):
    """Rejoin exact condition means with the original truth on the study server."""
    from benchmarks.common.full_gene_evaluation import replay_fields
    from benchmarks.jurkat_j2.study import COHORT_SHA, EXPECTED_CONDITIONS, evaluate

    manifest = read_json(input_manifest)
    receipt = read_json(inference / "inference.json")
    original = read_json(origin / "result.json")
    predictions = inference / "condition-means.npz"
    if (
        receipt["conditions"] != EXPECTED_CONDITIONS
        or receipt["test_controls_per_condition"] != 300
        or receipt["test_truth_accessed"]
        or receipt["test_updates"] != 0
        or receipt["manifest_sha256"] != sha256_file(input_manifest)
        or receipt["predictions_sha256"] != sha256_file(predictions)
        or manifest["cohort_sha256"] != COHORT_SHA
        or manifest["original_receipt_sha256"] != sha256_file(origin / "result.json")
        or receipt["checkpoint_sha256"] != original["checkpoint_sha256"]
        or original["checkpoint_sha256"] != sha256_file(origin / "checkpoint.pt")
    ):
        raise ValueError("portable run failed frozen inference provenance verification")
    with np.load(predictions, allow_pickle=False) as archive:
        names, means = archive["conditions"], archive["means"]
    if means.shape != (EXPECTED_CONDITIONS, 5000) or not np.isfinite(means).all():
        raise ValueError("invalid condition means")
    lookup = dict(zip(names, means, strict=True))
    if len(lookup) != EXPECTED_CONDITIONS:
        raise ValueError("a portable condition is duplicated")
    output.mkdir()
    result = evaluate(
        lambda name, controls, _ids, _axis: np.broadcast_to(lookup[name], controls.shape),
        output,
        journal_identity=receipt["predictions_sha256"],
        full_axis=True,
    )
    if result["paired_rows_sha256"] != manifest["paired_rows_sha256"]:
        raise ValueError("portable fold and scoring use different controls")
    import subprocess

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    atomic_json(
        output / "result.json",
        {
            **original,
            **result,
            **replay_fields(origin, {"run_id": output.name, "source_commit": sha}),
            "score_panel_sha256": None,
            "inference_provider": "Google Colab",
            "inference": receipt,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action, positional in (
        ("prepare", ("output", "origin")),
        ("infer", ("inputs", "output")),
        ("score", ("input_manifest", "inference", "output", "origin")),
    ):
        command = commands.add_parser(action)
        for name in positional:
            command.add_argument(name, type=Path)
        if action == "infer":
            command.add_argument("--stop", type=int)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "infer": infer, "score": score}[command](**args)


if __name__ == "__main__":
    main()
