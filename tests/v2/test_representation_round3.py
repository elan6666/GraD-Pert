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
def round3_queue(bundle, new_queue, monkeypatch, request):
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
        only_arms=getattr(request, "param", None),
    )
    return queue, Path(queue["output"])


@pytest.mark.parametrize("round3_queue", [None, ["R3-AUB", "R3-AUS"]], indirect=True)
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


@pytest.mark.parametrize("round3_queue", [None, ["R3-AUB", "R3-AUS"]], indirect=True)
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
    arms = [row["name"] for row in queue["rows"]]
    selected = runner.selected_matrix(
        runner.verify_manifest(Path(queue["source"]), Path(queue["manifest"])),
        queue.get("only_arms"),
    )
    assert calls[: len(arms) * 3] == [
        (a, s)
        for a in selected["preflight_priority"]
        for s in ("preflight", "eval-capacity-g0", "eval-capacity-g1")
    ]
    assert calls[len(arms) * 3 :] == [
        (a, s) for a in arms for s in ("fit", "postfit-validation", "research-analysis")
    ]
    assert read_json(output / "RESEARCH_COMPLETE.json")["scientific_complete"] is False
    assert not (output / "COMPLETE.json").exists()


def test_round3_continuation_uses_fresh_queue_and_ids_preserving_original_ri(round3_queue):
    original, old_output = round3_queue
    original_bytes = (old_output / "queue.json").read_bytes()
    atomic_json(old_output / "state.json", dict(status="failed"))
    atomic_json(old_output / "FAILURE.json", dict(error="original analysis failure"))
    ri_last = Path(original["rows"][0]["plan"]["run_root"]) / "fit/last.pt"
    ri_last.parent.mkdir(parents=True)
    ri_last.write_bytes(b"preserved original RI checkpoint")
    checkpoint_hash = sha256_file(ri_last)
    queue = runner.prepare(
        Path(original["source"]),
        Path(original["manifest"]),
        Path(original["runtime"]),
        old_output.parent / "aub-aus-continuation",
        original["shared_gpu_policy"],
        only_arms=["R3-AUB", "R3-AUS"],
    )
    runner.verify(queue)
    assert [r["name"] for r in queue["rows"]] == ["R3-AUB", "R3-AUS"]
    assert len(queue["eval_bindings"]) == 4 and queue["steps"] == 10
    assert queue["medium_analysis"] == original["medium_analysis"]
    for key in ("run_id", "run_root", "test_root"):
        assert {r["plan"][key] for r in original["rows"]}.isdisjoint(
            r["plan"][key] for r in queue["rows"]
        )
    assert (old_output / "queue.json").read_bytes() == original_bytes
    assert read_json(old_output / "state.json") == dict(status="failed")
    assert read_json(old_output / "FAILURE.json") == dict(error="original analysis failure")
    assert sha256_file(ri_last) == checkpoint_hash
    with pytest.raises(ValueError, match="fresh server queue root"):
        runner.prepare(
            Path(original["source"]),
            Path(original["manifest"]),
            Path(original["runtime"]),
            old_output,
            original["shared_gpu_policy"],
            only_arms=["R3-AUB", "R3-AUS"],
        )


@pytest.mark.parametrize(
    "arms",
    [
        [],
        ["R3-RI"],
        ["R3-AUB"],
        ["R3-AUS"],
        ["R3-AUS", "R3-AUB"],
        ["R3-AUB", "R3-AUB", "R3-AUS"],
        ["R3-RI", "R3-AUB", "R3-AUS"],
        ["R3-AUB", "unknown"],
        ["S12-L4"],
    ],
)
def test_round3_continuation_rejects_any_other_subset(arms):
    with pytest.raises(ValueError, match=r"R3-AUB.*R3-AUS"):
        runner.selected_matrix(dict(schema=family.SCHEMA), arms)


@pytest.mark.parametrize("round3_queue", [["R3-AUB", "R3-AUS"]], indirect=True)
@pytest.mark.parametrize("bad", ["selection", "mixed_subset", "order", "row", "eval_binding"])
def test_round3_continuation_rejects_resealed_queue_scope_tampering(round3_queue, bad):
    queue, output = round3_queue
    queue = copy.deepcopy(queue)
    row, binding = queue["rows"][0], queue["eval_bindings"][0]
    args = SimpleNamespace(
        gpu=binding["gpu"],
        cell_batch=128,
        reference_batch=128,
        config=Path(row["plan"]["config"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        data_root=Path(row["plan"]["data_root"]),
        output=Path(binding["output"]),
        shared_queue=output / "queue.json",
        row_name=row["name"],
    )
    if bad == "selection":
        queue["only_arms"] = ["R3-RI"]
    elif bad == "mixed_subset":
        queue["only_arms"] = ["R3-AUB", "S12-L4"]
    elif bad == "order":
        queue["rows"].reverse()
    elif bad == "row":
        queue["rows"][0]["name"] = "R3-RI"
    else:
        queue["eval_bindings"][0]["reference_batch"] = 64
    atomic_json(output / "queue.json", queue)
    with pytest.raises(ValueError):
        runner.verify(queue)
    with pytest.raises(ValueError):
        validate_eval_binding(queue, args, queue["source_commit"])


def test_round3_continuation_cli_accepts_exact_pair(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    atomic_json(policy, {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_br400_group.py",
            "--prepare",
            "--source",
            str(tmp_path / "source"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--runtime",
            str(tmp_path / "runtime.json"),
            "--output",
            str(tmp_path / "fresh"),
            "--shared-policy",
            str(policy),
            "--only-arms",
            "R3-AUB",
            "R3-AUS",
        ],
    )
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *a, **kw: calls.append((a, kw)))
    runner.main()
    assert len(calls) == 1 and calls[0][1] == dict(only_arms=["R3-AUB", "R3-AUS"])


@pytest.mark.parametrize("round3_queue", [["R3-AUB", "R3-AUS"]], indirect=True)
@pytest.mark.parametrize("substitution", ["ri_under_aub", "aub_aus_swapped"])
def test_round3_eval_rejects_rehashed_real_config_substitution(round3_queue, substitution):
    queue, output = round3_queue
    queue = copy.deepcopy(queue)
    source = Path(queue["source"])
    full = runner.verify_manifest(source, Path(queue["manifest"]))
    replacements = (
        [full["rows"][0]] if substitution == "ri_under_aub" else list(reversed(full["rows"][1:]))
    )
    for row, spec in zip(queue["rows"], replacements, strict=False):
        config = source / spec["config"]
        row["plan"].update(
            config=str(config),
            config_sha256=sha256_file(config),
            resolved_config=load_experiment_config(config).model_dump(mode="json"),
        )
    atomic_json(output / "queue.json", queue)
    for row in queue["rows"][: len(replacements)]:
        binding = next(b for b in queue["eval_bindings"] if b["row_name"] == row["name"])
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=128,
            reference_batch=128,
            config=Path(row["plan"]["config"]),
            publication=Path(row["plan"]["publication"]),
            publication_sha256=row["plan"]["publication_sha256"],
            data_root=Path(row["plan"]["data_root"]),
            output=Path(binding["output"]),
            shared_queue=output / "queue.json",
            row_name=row["name"],
        )
        with pytest.raises(ValueError, match="manifest configuration"):
            validate_eval_binding(queue, args, queue["source_commit"])


@pytest.mark.parametrize("bad", [None, "checkpoint", "history", "terminal", "failure"])
def test_round3_medium_requires_real_training_terminal_without_test(round3_queue, monkeypatch, bad):
    from gradpert.execution.v2_training_stage import seal_training_stage

    queue, output = round3_queue
    row = queue["rows"][0]
    plan = row["plan"]
    root = Path(plan["run_root"])
    identity = dict(
        run_id=plan["run_id"],
        config_sha256=plan["config_sha256"],
        source=dict(
            commit=queue["source_commit"], published_commit=queue["source_commit"], dirty=False
        ),
    )
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "run_manifest.json", identity)
    checkpoint = root / "fit/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic last checkpoint")
    atomic_json(
        root / "fit/epoch_state.json",
        dict(
            identity=identity,
            epoch=2,
            budget=[2, 1],
            best=None,
            last=dict(file=checkpoint.name, epoch=2, sha256=sha256_file(checkpoint)),
        ),
    )
    atomic_json(
        root / "fit/history.json",
        [
            dict(epoch=i, validation=dict(validation_mode="disabled", performed=False))
            for i in (1, 2)
        ],
    )
    seal_training_stage(root, identity, 2)
    if bad == "checkpoint":
        checkpoint.write_bytes(b"changed")
    elif bad == "history":
        atomic_json(root / "fit/history.json", [dict(epoch=1)])
    elif bad == "terminal":
        (root / "TRAIN_COMPLETE.json").unlink()
    elif bad == "failure":
        atomic_json(root / "FAILURE.json", {})
    monkeypatch.setattr(
        medium, "__file__", str(Path(queue["source"]) / "scripts/v2/run_fgpo_medium.py")
    )

    def reached_checkpoint(*args):
        raise RuntimeError("passed training terminal gate")

    monkeypatch.setattr(medium, "select_checkpoint", reached_checkpoint)
    if bad is None:
        with pytest.raises(RuntimeError, match="passed training terminal gate"):
            medium.execute(output / "queue.json", row["name"])
    else:
        error, message = {
            "checkpoint": (ValueError, "checkpoint escaped or changed"),
            "history": (ValueError, "complete sealed epoch budget"),
            "terminal": (FileNotFoundError, "TRAIN_COMPLETE.json"),
            "failure": (ValueError, "failed training cannot advance"),
        }[bad]
        with pytest.raises(error, match=message):
            medium.execute(output / "queue.json", row["name"])
        assert not (output / f"{row['name']}-medium").exists()
    assert not (root / "COMPLETE.json").exists()


def test_nonresearch_medium_still_requires_test_completion(new_queue):
    queue, output = new_queue
    with pytest.raises(ValueError, match="scientific run has no successful native terminal"):
        medium.execute(output / "queue.json", queue["rows"][0]["name"])
    assert not (output / f"{queue['rows'][0]['name']}-medium").exists()


@pytest.mark.parametrize(
    "bad",
    [None, "dirty", "numerical", "controller", "queue_output", "run_output", "other_arm", "family"],
)
@pytest.mark.parametrize("repair_arm", ["R3-RI", "R3-AUB", "R3-AUS"])
def test_round3_ri_repair_freezes_source_and_preserves_original_outputs(
    tmp_path, monkeypatch, bad, repair_arm
):
    monkeypatch.setattr(medium, "SERVER_ROOT", tmp_path)
    queue = dict(
        family=family.SCHEMA,
        source_commit="a" * 40,
        output=str(tmp_path / "original-queue"),
        rows=[dict(plan=dict(run_root=str(tmp_path / "original-run")))],
    )
    before = copy.deepcopy(queue)

    def git(command, **kwargs):
        if "status" in command:
            return " M scripts/v2/run_fgpo_medium.py" if bad == "dirty" else ""
        assert command[command.index("--name-only") + 1 : command.index("--")] == [
            queue["source_commit"],
            "HEAD",
        ]
        assert "src" in command and "scripts/v2/run_br400_group.py" in command
        assert "scripts/v2/cap40_posthoc*.py" in command
        # The dispatch-only worker repair is outside the numerical source diff.
        assert "scripts/v2/run_fgpo_medium.py" not in command
        if bad == "numerical":
            return "src/gradpert/training/v2/objective.py\n"
        return "scripts/v2/run_br400_group.py\n" if bad == "controller" else ""

    monkeypatch.setattr(medium.subprocess, "check_output", git)
    output = {
        "queue_output": tmp_path / "original-queue/repair",
        "run_output": tmp_path / "original-run/repair",
    }.get(bad, tmp_path / "independent-repair")
    arm = "RS-RI" if bad == "other_arm" else repair_arm
    if bad == "family":
        queue["family"] = "unsupported"
        before = copy.deepcopy(queue)
    if bad is None:
        assert (
            medium.validate_report_repair(queue, arm, output, tmp_path / "repair-source") == output
        )
    else:
        with pytest.raises(ValueError):
            medium.validate_report_repair(queue, arm, output, tmp_path / "repair-source")
    assert queue == before


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


def au_binding_case(arm):
    from gradpert.config.v2 import V2Options

    config = ROOT / f"configs/v2/representation_round3_20261023/{arm}/gradpert_v2/nadig_jurkat.yaml"
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    protocol = dict(
        alignment_weight=options.au_alignment_weight,
        uniformity_weight=options.au_uniformity_weight,
        temperature=2.0,
        warmup_completed_optimizer_updates=30,
        representation="student_response_cls_after_final_rmsnorm_before_projector",
        population_scope="complete_effective_batch_all_microbatches_and_ranks",
        deduplication="control_row_id_and_condition_id_before_rank_partition",
        alignment="same_control_and_condition_two_global_views_both_student_gradients",
        uniformity_pairs="unordered_same_cell_type_batch_context_different_condition",
        reduction="log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
        partial_diagnostics="unavailable_requires_complete_global_population",
    )
    parent = dict(
        data=dict(canonical_sha256="a" * 64, loss_protocol=dict(reduction="row_mean")),
        schedule=[
            dict(
                index=0,
                condition_id="p",
                rows=[dict(control_row_id="c", truth_row_id="t")],
                deep_rows=[],
                gradient_rows=[],
                queries=[1],
                view_seed=3,
            )
        ],
    )
    identity = dict(run_id=arm, data=copy.deepcopy(parent["data"]))
    identity["data"]["loss_protocol"]["alignment_uniformity"] = protocol
    kwargs = dict(arm=arm, config=config, parent_path="original.json", parent_sha256="b" * 64)
    return parent, identity, kwargs


@pytest.mark.parametrize("arm", ["R3-AUB", "R3-AUS"])
def test_au_parent_binding_preserves_population_and_rejects_drift(arm):
    parent, identity, kwargs = au_binding_case(arm)
    original = copy.deepcopy(parent)
    protocol = identity["data"]["loss_protocol"]["alignment_uniformity"]
    bound = medium.au_parent_binding(parent, identity, **kwargs)
    assert bound["data"] == identity["data"]
    assert bound["schedule"] == parent["schedule"]
    binding = bound["training_protocol_binding"]
    assert binding["training_value"] == protocol
    assert binding["original_parent_sha256"] == "b" * 64
    assert binding["sampling_changed"] is False
    for drift in ("data", "protocol", "weight", "semantics", "extra", "missing"):
        changed = copy.deepcopy(identity)
        au = changed["data"]["loss_protocol"]["alignment_uniformity"]
        if drift == "data":
            changed["data"]["canonical_sha256"] = "c" * 64
        elif drift == "protocol":
            changed["data"]["loss_protocol"]["reduction"] = "other"
        elif drift == "weight":
            au["alignment_weight"] *= 2
        elif drift == "semantics":
            au["population_scope"] = "microbatch"
        elif drift == "extra":
            au["unknown"] = True
        else:
            del au["deduplication"]
        with pytest.raises(ValueError):
            medium.au_parent_binding(parent, changed, **kwargs)
    with pytest.raises(ValueError):
        medium.au_parent_binding(parent, identity, **dict(kwargs, arm="R3-RI"))
    with pytest.raises(ValueError):
        medium.au_parent_binding(bound, identity, **kwargs)
    bound["schedule"][0]["queries"].append(2)
    assert parent == original


@pytest.mark.parametrize("round3_queue", [["R3-AUB", "R3-AUS"]], indirect=True)
@pytest.mark.parametrize("population_drift", [False, True])
def test_au_medium_execute_seals_bound_parent_with_frozen_source(
    round3_queue, monkeypatch, population_drift
):
    queue, output = round3_queue
    arm = "R3-AUB"
    row = next(r for r in queue["rows"] if r["name"] == arm)
    plan = row["plan"]
    parent, identity, binding_kwargs = au_binding_case(arm)
    plan["config"] = str(binding_kwargs["config"])
    queue_path = output / "queue.json"
    atomic_json(queue_path, queue)
    identity["run_id"] = plan["run_id"]
    atomic_json(Path(plan["run_root"]) / "run_manifest.json", identity)
    original = copy.deepcopy(parent)
    repair = output.parent / "fresh-au-report-repair"
    checkpoint = dict(sha256="d" * 64)
    monkeypatch.setattr(runner, "verify", lambda q: None)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: None)
    monkeypatch.setattr(medium, "validate_report_repair", lambda *a: repair)
    monkeypatch.setattr(medium, "validate_analysis_spec", lambda spec: parent)
    monkeypatch.setattr(medium, "select_checkpoint", lambda *a: (checkpoint, {}))

    def run(command, **kwargs):
        assert "seal-medium" in command
        assert "--reuse-medium-schedule" in command
        assert command[command.index("--method-sha") + 1] == queue["source_commit"]
        assert kwargs["env"]["PYTHONPATH"] == str(Path(queue["source"]) / "src")
        assert command[command.index("--run-root") + 1] == plan["run_root"]
        path = Path(command[command.index("--parent-manifest") + 1])
        assert path.parent == repair
        assert sha256_file(path) == command[command.index("--parent-manifest-sha256") + 1]
        bound = read_json(path)
        assert bound["data"] == identity["data"]
        assert bound["schedule"] == original["schedule"]
        assert (
            bound["training_protocol_binding"]["original_parent_sha256"]
            == queue["medium_analysis"]["parent_manifest_sha256"]
        )
        if population_drift:
            bound["schedule"][0]["rows"][0]["truth_row_id"] = "wrong-truth-row"
            bound.update(
                method_sha=queue["source_commit"], view_recipe_config_sha256=plan["config_sha256"]
            )
            atomic_json(Path(command[command.index("--output") + 1]) / "manifest.json", bound)
        else:
            raise RuntimeError("verified real AU seal dispatch")

    monkeypatch.setattr(medium.subprocess, "run", run)
    error, message = (
        (ValueError, "resealing changed the common diagnostic population")
        if population_drift
        else (RuntimeError, "verified real AU seal dispatch")
    )
    with pytest.raises(error, match=message):
        medium.execute(output / "queue.json", arm, repair_output=repair)
    assert parent == original
    assert read_json(repair / "FAILURE.json")["status"] == "failed"
    assert not (output / f"{arm}-medium").exists()


@pytest.mark.parametrize(
    "field,value", [("au_alignment_weight", 0.1), ("au_temperature", 3.0), ("au_warmup_steps", 31)]
)
def test_au_binding_rejects_changed_frozen_config(tmp_path, field, value):
    parent, identity, kwargs = au_binding_case("R3-AUB")
    config = yaml.safe_load(kwargs["config"].read_text())
    config["model"]["parameters"][field]["value"] = value
    target = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    target.parent.mkdir()
    target.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="frozen arm"):
        medium.au_parent_binding(parent, identity, **dict(kwargs, config=target))
