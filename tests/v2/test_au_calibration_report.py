"""Aggregate report acceptance and geometry need no model or training runtime."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/v2/report_au_calibration.py"
SPEC = importlib.util.spec_from_file_location("au_report", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def gradients(scale=1):
    vectors = dict(A=(1, 0), U=(0, 2), MSE=(1, 1), joint=(2, 1))
    norms = {k: math.hypot(*v) * scale for k, v in vectors.items()}
    pairs = {}
    for a, b in reporter.PAIRS:
        cosine = (
            sum(x * y for x, y in zip(vectors[a], vectors[b], strict=True))
            * scale**2
            / (norms[a] * norms[b])
        )
        pairs[a + "__" + b] = dict(cosine=cosine, angle_degrees=math.degrees(math.acos(cosine)))
    return dict(norms=norms, pairs=pairs, parameter_count=6)


def endpoint(root, checkpoint=False):
    root.mkdir()
    plan = dict(
        schema="gradpert-au-calibration-1",
        mode="gradients",
        batches=8,
        effective_population=320,
        microbatch=80,
        updates=0,
        split="train_only",
        topology="single_gpu_diagnostic_not_ddp_resume",
        source=dict(commit="a" * 40, dirty=False, formal_eligible=True),
        config_sha256="b" * 64,
        seed=1,
        rng_seed=41239,
        temperature=2,
        run_id="ri-run" if checkpoint else "init-run",
        checkpoint_sha256="c" * 64 if checkpoint else None,
        training_identity=dict(source=dict(commit="d" * 40)) if checkpoint else None,
    )
    ph = reporter.sha_json(plan)
    census = [
        dict(
            input_count=320,
            unique_count=320,
            duplicate_count=0,
            pair_count=25600,
            context_count=1,
            contexts=[
                dict(
                    context_id="cell::batch",
                    input_count=320,
                    condition_count=2,
                    pair_count=25600,
                    pair_weight=1,
                )
            ],
            ordered_identity_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
        )
        for i in range(8)
    ]
    write(root / "PLAN.json", plan)
    write(
        root / "CENSUS.json",
        dict(plan_sha256=ph, complete=True, model_forward=False, updates=0, batches=census),
    )
    checksums = {}
    for i in range(8):
        spectrum = dict(
            effective_rank=10,
            normalized_rank=10 / 256,
            relative_variance=0.1,
            covariance_trace=10,
            mean_squared_norm=90,
            sample_count=320,
            width=256,
        )
        record = dict(
            plan_sha256=ph,
            batch_index=i,
            census=census[i],
            updates=0,
            state_unchanged=True,
            unique_count=320,
            pair_count=25600,
            replay_tolerance=2e-6,
            replay_max_abs_error=1e-7,
            frozen_state={k: "a" * 64 for k in reporter.FROZEN},
            environment=dict(torch_version="synthetic"),
            losses=dict(A=0.2, U=-0.5, MSE=1.2, joint=30),
            gradients={g: gradients(2 if checkpoint else 1) for g in reporter.GROUPS},
            spectrum={v: spectrum for v in reporter.VIEWS},
            elapsed_seconds=4,
            cells_per_second=80,
            peak_allocated_bytes=1024,
            peak_reserved_bytes=2048,
        )
        file = root / f"batch-{i:02d}.json"
        write(file, record)
        checksums[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    write(root / "BATCH_INDEX.json", dict(plan_sha256=ph, receipts=checksums))
    write(
        root / "COMPLETE.json",
        dict(
            plan_sha256=ph,
            calibration_complete=True,
            scientific_complete=False,
            updates=0,
            batches_completed=8,
            expected=8,
            batch_receipts=checksums,
        ),
    )


def test_linear_combination_matches_analytic_vectors_and_handles_zero():
    actual = reporter.compose_gradients(gradients(), 0.5, 0.25)
    # reg=(0.5,0.5), total=(2.5,1.5), mse=(1,1), joint=(2,1).
    assert actual["regularizer_norm"] == pytest.approx(math.sqrt(0.5))
    assert actual["regularizer__MSE_cosine"] == pytest.approx(1)
    assert actual["total_norm"] == pytest.approx(math.sqrt(8.5))
    assert actual["total__MSE_cosine"] == pytest.approx(4 / math.sqrt(17))
    assert actual["total__joint_cosine"] == pytest.approx(6.5 / math.sqrt(42.5))
    zero = reporter.compose_gradients(gradients(), 0, 0)
    assert zero["regularizer_norm"] == 0 and zero["regularizer__MSE_cosine"] is None
    assert zero["total_norm"] == pytest.approx(math.sqrt(5))


def test_complete_report_exports_sixteen_rows_two_figures_and_all_source_hashes(tmp_path):
    init, last, output = tmp_path / "init", tmp_path / "ri-last", tmp_path / "report"
    endpoint(init)
    endpoint(last, True)
    result = reporter.export_report(
        init, last, output, alignment_weight=0.5, uniformity_weight=0.25
    )
    rows = list(csv.DictReader((output / "calibration_batches.csv").open()))
    assert len(rows) == 16 and len(result["input_files"]) == 24
    assert result["scientific_complete"] is False and result["model_forward"] is False
    assert result["coefficients"]["strong"] == (1, 0.5)
    assert "response_strong_total__joint_angle_degrees" in rows[0]
    assert len(list(output.glob("*.png"))) == 2
    for name, digest in result["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    text = (output / "REPORT_CN.md").read_text()
    assert "不能直接等同于 medium" in text and "不能等同于训练吞吐" in text
    with pytest.raises(ValueError, match="preserve historical"):
        reporter.export_report(init, last, output)


def refresh_hashes(root):
    index = json.loads((root / "BATCH_INDEX.json").read_text())
    for name in index["receipts"]:
        index["receipts"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    complete = json.loads((root / "COMPLETE.json").read_text())
    complete["batch_receipts"] = index["receipts"]
    write(root / "BATCH_INDEX.json", index)
    write(root / "COMPLETE.json", complete)


@pytest.mark.parametrize("mutation", ["partial", "hash", "state", "replay", "population", "angle"])
def test_report_rejects_partial_tampered_or_incomparable_evidence(tmp_path, mutation):
    init, last, output = tmp_path / "init", tmp_path / "ri-last", tmp_path / "report"
    endpoint(init)
    endpoint(last, True)
    if mutation == "partial":
        complete = json.loads((last / "COMPLETE.json").read_text())
        complete["batches_completed"] = 1
        write(last / "COMPLETE.json", complete)
    else:
        file = last / "batch-00.json"
        record = json.loads(file.read_text())
        if mutation == "hash":
            record["losses"]["A"] = 0.3
        elif mutation == "state":
            record["state_unchanged"] = False
        elif mutation == "replay":
            record["replay_max_abs_error"] = 3e-6
        elif mutation == "angle":
            record["gradients"]["graph"]["pairs"]["A__U"]["angle_degrees"] = 45
        else:
            record["census"]["ordered_identity_sha256"] = "e" * 64
            census = json.loads((last / "CENSUS.json").read_text())
            census["batches"][0] = record["census"]
            write(last / "CENSUS.json", census)
        write(file, record)
        if mutation != "hash":
            refresh_hashes(last)
    with pytest.raises(ValueError):
        reporter.export_report(init, last, output)
    assert not output.exists()
