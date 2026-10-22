"""R3 cannot dispatch without complete calibrated, proportional AU strengths."""

import contextlib
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round3 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as research  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from test_au_calibration_report import endpoint as calibration_endpoint  # noqa: E402
from test_br400_group import eval_receipt  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402
from test_fgpo_group import training_gate  # noqa: E402


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    for relative in (first.BASELINE, family.CALIBRATION_CONFIG):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    decision_path = source / "decision-input/decision.json"
    endpoints = {}
    for endpoint in ("init", "ri_last"):
        root = decision_path.parent / "calibration" / endpoint
        root.parent.mkdir(parents=True, exist_ok=True)
        calibration_endpoint(root, checkpoint=endpoint == "ri_last")
        plan = read_json(root / "PLAN.json")
        plan.update(
            seed=1,
            rng_seed=700,
            config_sha256=sha256_file(source / family.CALIBRATION_CONFIG),
            source=dict(
                commit=family.CALIBRATION_SOURCE,
                published_commit=family.CALIBRATION_SOURCE,
                dirty=False,
                formal_eligible=True,
            ),
        )
        ph = sha256_json(plan)
        census = read_json(root / "CENSUS.json")
        census["plan_sha256"] = ph
        atomic_json(root / "PLAN.json", plan)
        atomic_json(root / "CENSUS.json", census)
        hashes = {}
        for i in range(8):
            file = root / f"batch-{i:02d}.json"
            record = read_json(file)
            record["plan_sha256"] = ph
            atomic_json(file, record)
            hashes[file.name] = sha256_file(file)
        atomic_json(root / "BATCH_INDEX.json", dict(plan_sha256=ph, receipts=hashes))
        atomic_json(
            root / "COMPLETE.json",
            dict(
                calibration_complete=True,
                scientific_complete=False,
                updates=0,
                plan_sha256=ph,
                batches_completed=8,
                expected=8,
                batch_receipts=hashes,
            ),
        )
        endpoints[endpoint] = dict(
            root=f"calibration/{endpoint}", complete_sha256=sha256_file(root / "COMPLETE.json")
        )
    atomic_json(
        decision_path,
        dict(
            schema=family.DECISION_SCHEMA,
            calibration_source_commit=family.CALIBRATION_SOURCE,
            balanced=dict(alignment_weight=0.1, uniformity_weight=0.02),
            strong=dict(alignment_weight=0.3, uniformity_weight=0.06),
            strength_multiplier=3,
            rationale="synthetic fixture coefficients; not an experiment decision",
            calibration=endpoints,
        ),
    )
    return source, decision_path, source / "configs/v2/round3"


def test_round3_preserves_ri_and_seals_original_decision_and_both_endpoints(bundle):
    source, decision, output = bundle
    original = decision.read_bytes()
    manifest = family.generate(source, output, decision)
    assert (output / "decision.json").read_bytes() == original
    assert manifest["coefficient_decision"]["sha256"] == sha256_file(decision)
    assert manifest["epochs"] == 2 and manifest["global_batch"] == 320
    assert manifest["test_timing"] == "after_research_design_lock"
    values = family.configs(source, family.load_decision(source, decision))
    assert values["R3-RI"] == first.configs(source)["RS-RI"]
    for arm in family.ARMS[1:]:
        altered = copy.deepcopy(values[arm])
        for key in family.changes(family.load_decision(source, decision))[arm]:
            altered["model"]["parameters"].pop(key)
        assert altered == values["R3-RI"]
    # Verification is self-contained after the original input bundle disappears.
    shutil.rmtree(decision.parent)
    assert family.verify_manifest(source, output / "manifest.json") == manifest
    assert not any(p.suffix in {".pt", ".pkl", ".npz"} for p in output.rglob("*"))


@pytest.mark.parametrize(
    "mutation", ["zero", "nan", "ratio", "source", "partial", "batch", "external"]
)
def test_decision_rejects_unsafe_or_incomplete_calibration(bundle, mutation):
    source, decision, output = bundle
    value = read_json(decision)
    if mutation == "zero":
        value["balanced"]["alignment_weight"] = 0
    elif mutation == "nan":
        value["strong"]["uniformity_weight"] = float("nan")
    elif mutation == "ratio":
        value["strong"]["uniformity_weight"] *= 2
    elif mutation == "source":
        value["calibration_source_commit"] = "c" * 40
    elif mutation == "external":
        value["calibration"]["init"]["root"] = "../calibration/init"
    else:
        root = decision.parent / "calibration/init"
        if mutation == "batch":
            (root / "batch-07.json").write_text("{}")
        else:
            complete = read_json(root / "COMPLETE.json")
            complete["batches_completed"] = 1
            atomic_json(root / "COMPLETE.json", complete)
            value["calibration"]["init"]["complete_sha256"] = sha256_file(root / "COMPLETE.json")
    decision.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        family.generate(source, output, decision)
    assert not output.exists()


def reseal_batch(decision_path, endpoint, batch_index, mutate):
    """Make hashes consistent so rejection must come from evidence semantics."""
    decision = read_json(decision_path)
    root = decision_path.parent / decision["calibration"][endpoint]["root"]
    file = root / f"batch-{batch_index:02d}.json"
    record = read_json(file)
    mutate(record)
    atomic_json(file, record)
    index = read_json(root / "BATCH_INDEX.json")
    index["receipts"][file.name] = sha256_file(file)
    atomic_json(root / "BATCH_INDEX.json", index)
    complete = read_json(root / "COMPLETE.json")
    complete["batch_receipts"] = index["receipts"]
    atomic_json(root / "COMPLETE.json", complete)
    decision["calibration"][endpoint]["complete_sha256"] = sha256_file(root / "COMPLETE.json")
    atomic_json(decision_path, decision)


@pytest.mark.parametrize("endpoint", ["init", "ri_last"])
@pytest.mark.parametrize("mutation", ["missing", "invalid"])
@pytest.mark.parametrize(
    "key",
    [
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
    ],
)
def test_decision_rejects_rehashed_missing_or_invalid_frozen_fingerprint(
    bundle, endpoint, mutation, key
):
    source, decision, output = bundle

    def mutate(record):
        if mutation == "missing":
            record["frozen_state"].pop(key)
        else:
            record["frozen_state"][key] = "x" * 64

    reseal_batch(decision, endpoint, 7, mutate)
    with pytest.raises(ValueError, match="frozen state fingerprints"):
        family.generate(source, output, decision)
    assert not output.exists()


@pytest.mark.parametrize("endpoint", ["init", "ri_last"])
@pytest.mark.parametrize("key", ["student", "teacher", "centers", "optimizer"])
def test_decision_rejects_rehashed_frozen_state_drift_between_batches(bundle, endpoint, key):
    source, decision, output = bundle
    reseal_batch(decision, endpoint, 7, lambda r: r["frozen_state"].update({key: "e" * 64}))
    with pytest.raises(ValueError, match="state changed between calibration populations"):
        family.generate(source, output, decision)
    assert not output.exists()


@pytest.mark.parametrize("endpoint", ["init", "ri_last"])
@pytest.mark.parametrize("key", ["gradients", "spectrum"])
def test_decision_rejects_rehashed_empty_calibration_statistics(bundle, endpoint, key):
    source, decision, output = bundle
    reseal_batch(decision, endpoint, 7, lambda r: r.update({key: {}}))
    with pytest.raises(ValueError, match=r"missing (gradient parameter group|Global spectrum)"):
        family.generate(source, output, decision)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["coefficient", "budget", "test", "decision_bytes", "extra_arm"]
)
def test_family_rejects_rehashed_protocol_changes(bundle, mutation):
    source, decision, output = bundle
    manifest = family.generate(source, output, decision)
    if mutation == "decision_bytes":
        with (output / "decision.json").open("a") as file:
            file.write("\n")
    elif mutation == "test":
        manifest["research_iteration"]["test_results_for_selection"] = True
    elif mutation == "extra_arm":
        manifest["rows"].append(manifest["rows"][0])
    else:
        row = manifest["rows"][1]
        path = source / row["config"]
        value = yaml.safe_load(path.read_text())
        if mutation == "budget":
            value["training"]["max_epochs"]["value"] = 4
        else:
            value["model"]["parameters"]["au_alignment_weight"]["value"] = 1
        path.write_text(yaml.safe_dump(value))
        row["sha256"] = sha256_file(path)
    atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        family.verify_manifest(source, output / "manifest.json")


@pytest.fixture
def round3_queue(bundle, new_queue, monkeypatch):
    source, decision, family_root = bundle
    family.generate(source, family_root, decision)
    old, output = new_queue
    runtime = Path(old["runtime"])
    atomic_json(runtime, {**read_json(runtime), "repository_root": str(source)})
    counter = 0

    def resolve(args):
        nonlocal counter
        counter += 1
        plan = copy.deepcopy(old["rows"][0]["plan"])
        run_id = f"r3-{counter}"
        plan.update(
            config=str(args.config),
            config_sha256=sha256_file(args.config),
            resolved_config=load_experiment_config(args.config).model_dump(mode="json"),
            runtime_sha256=sha256_file(runtime),
            repository_root=str(source),
            run_id=run_id,
            run_root=str(Path(plan["run_root"]).parent / run_id),
            test_root=str(Path(plan["test_root"]).parent / (run_id + "-test")),
        )
        return plan

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    monkeypatch.setattr(runner, "__file__", str(source / "scripts/v2/run_br400_group.py"))
    queue = runner.prepare(
        source,
        family_root / "manifest.json",
        runtime,
        output.parent / "round3",
        old["shared_gpu_policy"],
    )
    return queue, Path(queue["output"])


def test_round3_native_gates_and_separate_training_source(round3_queue):
    queue, output = round3_queue
    assert queue["source_commit"] != family.CALIBRATION_SOURCE
    assert list(research.FAMILY_ARMS[queue["family"]]) == list(family.ARMS)
    assert list(medium.FAMILY_ARMS[queue["family"]]) == list(family.ARMS)
    assert queue["steps"] == 10
    for row in queue["rows"]:
        plan = row["plan"]
        assert runner.expected_epochs(plan) == 2
        atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(queue, row))
        runner.validate_training_gate(
            queue, row, runner.gate_entry(output / f"{row['name']}-preflight/receipt.json")
        )
        for binding in (b for b in queue["eval_bindings"] if b["row_name"] == row["name"]):
            args = SimpleNamespace(
                gpu=binding["gpu"],
                cell_batch=128,
                reference_batch=128,
                config=Path(plan["config"]),
                publication=Path(plan["publication"]),
                publication_sha256=plan["publication_sha256"],
                data_root=Path(plan["data_root"]),
                output=Path(binding["output"]),
                shared_queue=output / "queue.json",
                row_name=row["name"],
            )
            assert validate_eval_binding(queue, args, queue["source_commit"]) == row


def test_round3_controller_runs_val_and_diagnostics_then_returns_for_research(
    round3_queue, monkeypatch
):
    queue, output = round3_queue
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "snapshot", lambda pid: dict(processes=[], gpus={}))
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    monkeypatch.setattr(runner, "lock", lambda path: contextlib.nullcontext())
    calls = []

    def stage(q, row, name, command, gpu, state, guard):
        calls.append((row["name"], name))
        assert name not in ("last", "best-last")
        if name == "preflight":
            atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(q, row))
        elif name.startswith("eval-capacity"):
            binding = next(
                b for b in q["eval_bindings"] if b["row_name"] == row["name"] and b["gpu"] == gpu
            )
            receipt = eval_receipt(q, row, binding)
            receipt["cell_batch"] = 128
            atomic_json(Path(binding["output"]) / "receipt.json", receipt)

    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda _: atomic_json(
            output / "MAIN_ACCEPTANCE.json",
            {**read_json(output / "PREFLIGHT_COMPLETE.json"), "approved": True},
        ),
    )
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: {})
    monkeypatch.setattr(runner, "validate_complete", lambda p: pytest.fail("test finalizer used"))
    monkeypatch.setattr(research, "validate_research_complete", lambda *a: {"research": True})
    runner.execute(queue, output)
    assert calls[9:] == [
        (a, s) for a in family.ARMS for s in ("fit", "postfit-validation", "research-analysis")
    ]
    assert read_json(output / "RESEARCH_COMPLETE.json")["scientific_complete"] is False
    assert not (output / "COMPLETE.json").exists()


def test_au_partial_analysis_exports_both_terms_as_na(tmp_path):
    import csv

    root, run = tmp_path / "medium", tmp_path / "run"
    row = dict(name="R3-AUB", plan=dict(run_root=str(run), config_sha256="c" * 64))
    queue = dict(output=str(tmp_path), source_commit="a" * 40, family=family.SCHEMA)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    unavailable = {
        key: "requires_complete_global_population" for key in ("au_alignment", "au_uniformity")
    }
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p1",
                joint=2,
                loss_terms=dict(prediction=1, ssl2_dino=1),
                diagnostic_objective_scope=dict(
                    unavailable_terms=unavailable,
                    joint_scope="prediction_and_ssl_excluding_unavailable_representation_terms",
                ),
            )
        ],
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(run / "fit/history.json", [dict(epoch=1), dict(epoch=2)])
    atomic_json(
        research.validation_path(queue, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    research.render_medium_report(queue, row, root)
    rows = list(csv.DictReader((root / "report/frozen_losses.csv").open()))
    for key in unavailable:
        assert rows[0][key] == ""
        assert rows[0]["unavailable_terms." + key] == unavailable[key]
