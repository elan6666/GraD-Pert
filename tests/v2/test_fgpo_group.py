"""New-family gates, identical diagnostic population and postfit lease ownership."""

import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

import generate_fgpo_family as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from test_br400_group import eval_receipt, training_receipt  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402


def test_common_capacity_fallback_changes_only_the_declared_batch():
    old = family.verify_manifest(ROOT, ROOT / "configs/v2/fgpo_node0_20261015/manifest.json")
    new = family.verify_manifest(ROOT, ROOT / "configs/v2/fgpo_node0_20261016_b320/manifest.json")
    assert new["global_batch"] == 320 and new["common_microbatch"] == 80
    assert new["priority"] == old["priority"]
    for before, after in zip(old["rows"], new["rows"], strict=True):
        import yaml

        a = yaml.safe_load((ROOT / before["config"]).read_text())
        b = yaml.safe_load((ROOT / after["config"]).read_text())
        assert a["model"]["parameters"]["microbatch"]["value"] == 100
        assert b["model"]["parameters"]["microbatch"]["value"] == 80
        assert a["training"]["train_batch_size"]["value"] == 400
        assert b["training"]["train_batch_size"]["value"] == 320
        a["model"]["parameters"]["microbatch"]["value"] = 80
        a["training"]["train_batch_size"]["value"] = 320
        assert a == b


def sampling_fixture(tmp_path):
    deep = {0, 1, *range(7, 133)}
    grad = {0, *range(7, 38)}
    schedule = []
    for i in range(1335):
        count = 1 if i < 7 else 2 if i == 1334 else 4
        rows = [
            {
                "truth_row_id": f"t{i}-{j}",
                "control_row_id": f"c{i}-{j}",
                "condition_id": f"p{i}",
                "context_id": "T::1",
            }
            for j in range(count)
        ]
        schedule.append(
            dict(
                index=i,
                condition_id=f"p{i}",
                rows=rows,
                deep_rows=rows if i in deep else [],
                gradient_rows=rows if i in grad else [],
                queries=[1, 3],
                view_seed=i,
                recipe_file=None,
                recipe_sha256=None,
            )
        )
    path = tmp_path / "medium.json"
    atomic_json(
        path,
        dict(analysis_scope="medium_cap40_stratified", seed=42, schedule=schedule, **medium.COUNTS),
    )
    performance = tmp_path / "performance.json"
    atomic_json(
        performance,
        dict(
            status="completed",
            same_sample_scope=True,
            deterministic_algorithms=True,
            model_updates=0,
            full_parity=dict(passed=True, atol=1e-8, max_abs=2e-11),
            full_fast_seconds=90.0,
            full_reference_seconds=100.0,
        ),
    )
    return dict(
        parent_manifest=str(path),
        parent_manifest_sha256=sha256_file(path),
        performance_receipt=str(performance),
        performance_receipt_sha256=sha256_file(performance),
    )


@pytest.fixture
def new_queue(prepared, monkeypatch):
    old, oldout = prepared
    monkeypatch.setattr(medium, "SERVER_ROOT", oldout.parent)
    spec = sampling_fixture(oldout.parent)
    runtime = Path(old["runtime"])
    atomic_json(runtime, {**read_json(runtime), "medium_analysis": spec})
    policy = copy.deepcopy(old["shared_gpu_policy"])
    policy.update(
        max_own_memory_fraction=0.9,
        additional_command_basenames=["run_totalvi_source_validation_development.py"],
        authorization_reference="2026-10-12-user-shared-memory90-coexistence",
    )
    output = oldout.parent / "new-family"
    q = runner.prepare(
        ROOT, ROOT / "configs/v2/fgpo_node0_20261015/manifest.json", runtime, output, policy
    )
    return q, output


def test_exact_new_family_and_no_official_dependency_or_old_br(new_queue):
    q, _ = new_queue
    runner.verify(q)
    assert [r["name"] for r in q["rows"]] == list(family.ARMS)
    assert q["family"] in runner.LAST_ONLY_FAMILIES
    assert "official_dependency" not in q
    assert len(q["eval_bindings"]) == 20
    assert all(b["cell_batch"] == 128 for b in q["eval_bindings"])


@pytest.mark.parametrize("change", ["hash", "gain", "scope", "tolerance", "state", "rows"])
def test_analysis_cannot_expand_population_or_bypass_performance_acceptance(
    tmp_path, monkeypatch, change
):
    monkeypatch.setattr(medium, "SERVER_ROOT", tmp_path)
    spec = sampling_fixture(tmp_path)
    medium.validate_analysis_spec(spec)
    path = Path(spec["parent_manifest"] if change == "rows" else spec["performance_receipt"])
    record = read_json(path)
    if change == "gain":
        record["full_fast_seconds"] = record["full_reference_seconds"]
    elif change == "scope":
        record["same_sample_scope"] = False
    elif change == "tolerance":
        record["full_parity"]["atol"] = 1e-6
    elif change == "state":
        record["model_updates"] = 1
    elif change == "rows":
        record["schedule"][0]["rows"] = []
    else:
        record["status"] = "running"
    atomic_json(path, record)
    if change != "hash":
        key = "parent_manifest" if change == "rows" else "performance_receipt"
        spec[key + "_sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        medium.validate_analysis_spec(spec)


def training_gate(q, row):
    r = training_receipt(q, row)
    r.pop("joint_validation_check")
    r.update(joint_validation_check_requested=False, evaluation_cell_batch=128)
    diagnostic = {
        "process": dict(
            schema="training-process-diagnostics-1",
            split="train",
            selection_use="none",
            distributions={"teacher.ssl1_node": {"samples": 0}},
            features={"student.graph.output": {"samples": 4}},
            outputs={"prediction_mse": 1},
        )
    }
    r.update(initial_loss_diagnostic=diagnostic, updated_loss_diagnostic=diagnostic)
    return r


def test_new_arm_actual_binding_and_required_process_probe(new_queue):
    q, output = new_queue
    row = q["rows"][0]
    receipt = training_gate(q, row)
    path = output / "B0-preflight/receipt.json"
    atomic_json(path, receipt)
    runner.validate_training_gate(q, row, runner.gate_entry(path))
    receipt.pop("updated_loss_diagnostic")
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="process probes"):
        runner.validate_training_gate(q, row, runner.gate_entry(path))
    binding = q["eval_bindings"][0]
    args = SimpleNamespace(
        gpu="0",
        cell_batch=128,
        reference_batch=128,
        config=Path(row["plan"]["config"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        data_root=Path(row["plan"]["data_root"]),
        output=Path(binding["output"]),
        shared_queue=output / "queue.json",
        row_name="B0",
    )
    assert validate_eval_binding(q, args, q["source_commit"]) == row
    malformed = copy.deepcopy(q)
    malformed["rows"].pop()
    with pytest.raises(ValueError, match="ten-arm"):
        validate_eval_binding(malformed, args, q["source_commit"])


@pytest.mark.parametrize("fail", [None, "last", "medium-analysis"])
@pytest.mark.parametrize("remaining_only", [False, True])
def test_all_gates_before_fit_and_last_then_medium_before_next_arm(
    new_queue, monkeypatch, fail, remaining_only
):
    q, output = new_queue
    if remaining_only:
        q = runner.prepare(
            ROOT,
            ROOT / "configs/v2/dp2_remaining_20261017/manifest.json",
            Path(q["runtime"]),
            output.parent / "dp2-only",
            copy.deepcopy(q["shared_gpu_policy"]),
        )
        output = Path(q["output"])
        runner.verify(q)
    arms = [row["name"] for row in q["rows"]]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "snapshot", lambda pid: dict(processes=[], gpus={}))
    monkeypatch.setattr(runner, "admit", lambda *args: None)
    locks = set()

    @contextlib.contextmanager
    def lock(path):
        name = path.name
        assert name not in locks
        locks.add(name)
        try:
            yield
        finally:
            locks.remove(name)

    monkeypatch.setattr(runner, "lock", lock)
    calls = []

    def stage(queue, row, name, command, gpu, state, guard):
        calls.append((row["name"], name))
        held = {s for s in locks if s.startswith("v2-gpu-")}
        assert len(held) == (0 if name == "medium-analysis" else 2)
        if row["name"] == arms[0] and name == fail:
            raise RuntimeError("retained failure")
        if name == "preflight":
            assert "--joint-validation-check" not in command
            atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(q, row))
        elif name.startswith("eval-capacity"):
            binding = next(
                b for b in q["eval_bindings"] if b["row_name"] == row["name"] and b["gpu"] == gpu
            )
            r = eval_receipt(q, row, binding)
            r["cell_batch"] = 128
            atomic_json(Path(binding["output"]) / "receipt.json", r)

    def acceptance(_seconds):
        ready = read_json(output / "PREFLIGHT_COMPLETE.json")
        atomic_json(output / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})

    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner.time, "sleep", acceptance)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: {})
    monkeypatch.setattr(runner, "validate_complete", lambda p: {"last": True})
    monkeypatch.setattr(medium, "validate_medium_complete", lambda *a: {"medium": True})
    if fail:
        with pytest.raises(RuntimeError, match="retained failure"):
            runner.execute(q, output)
        assert all((name, "fit") not in calls for name in arms[1:])
        assert (output / "FAILURE.json").exists()
    else:
        runner.execute(q, output)
        assert calls[len(arms) * 3 :] == [
            (arm, phase) for arm in arms for phase in ("fit", "last", "medium-analysis")
        ]
        complete = read_json(output / "COMPLETE.json")
        assert set(complete["medium_analysis"]) == set(arms)
        assert read_json(output / "PREFLIGHT_COMPLETE.json")["formal_order"] == arms
    if remaining_only:
        assert {arm for arm, _ in calls} == {"DP2"}
    assert not locks


def test_prioritized_dino_pair_reuses_last_and_medium_gates(new_queue):
    from generate_dino_pair import SCHEMA as DINO_SCHEMA

    old, oldout = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / "configs/v2/dino_pair_20261016/manifest.json",
        Path(old["runtime"]),
        oldout.parent / "dino-pair",
        copy.deepcopy(old["shared_gpu_policy"]),
    )
    runner.verify(q)
    assert q["family"] == DINO_SCHEMA
    assert [r["name"] for r in q["rows"]] == ["B0-DP", "DP2"]
    assert DINO_SCHEMA in runner.LAST_ONLY_FAMILIES
    assert medium.FAMILY_ARMS[DINO_SCHEMA].keys() == {"B0-DP", "DP2"}
    assert len(q["eval_bindings"]) == 4
    assert q["medium_analysis"] == old["medium_analysis"]
    assert "official_dependency" not in q
    for binding in q["eval_bindings"]:
        row = next(r for r in q["rows"] if r["name"] == binding["row_name"])
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=binding["cell_batch"],
            reference_batch=128,
            config=Path(row["plan"]["config"]),
            publication=Path(row["plan"]["publication"]),
            publication_sha256=row["plan"]["publication_sha256"],
            data_root=Path(row["plan"]["data_root"]),
            output=Path(binding["output"]),
            shared_queue=Path(q["output"]) / "queue.json",
            row_name=binding["row_name"],
        )
        assert validate_eval_binding(q, args, q["source_commit"]) == row
    for change in (
        {"mode": "unapproved"},
        {"steps": 128},
        {"rows": list(reversed(q["rows"]))},
        {"rows": q["rows"][:1]},
    ):
        malformed = {**q, **change}
        with pytest.raises(ValueError, match="two-arm"):
            validate_eval_binding(malformed, args, q["source_commit"])


def test_dp2_continuation_exact_eval_and_medium_scope(new_queue):
    from generate_dino_pair import REMAINING_SCHEMA

    old, oldout = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / "configs/v2/dp2_remaining_20261017/manifest.json",
        Path(old["runtime"]),
        oldout.parent / "dp2-eval-scope",
        copy.deepcopy(old["shared_gpu_policy"]),
    )
    assert medium.FAMILY_ARMS[REMAINING_SCHEMA].keys() == {"DP2"}
    assert len(q["eval_bindings"]) == 2
    for binding in q["eval_bindings"]:
        plan = q["rows"][0]["plan"]
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=128,
            reference_batch=128,
            config=Path(plan["config"]),
            publication=Path(plan["publication"]),
            publication_sha256=plan["publication_sha256"],
            data_root=Path(plan["data_root"]),
            output=Path(binding["output"]),
            shared_queue=Path(q["output"]) / "queue.json",
            row_name="DP2",
        )
        assert validate_eval_binding(q, args, q["source_commit"]) == q["rows"][0]
        for change in ({"steps": 128}, {"mode": "other"}, {"rows": q["rows"] * 2}):
            with pytest.raises(ValueError, match="singleton"):
                validate_eval_binding({**q, **change}, args, q["source_commit"])


@pytest.mark.parametrize(
    "name,directory",
    [
        ("PT1", "pt1_staged_20261017"),
        ("PT2", "pt2_native_io_20261017"),
    ],
)
def test_staged_singleton_scope_and_phase_specific_read_only_gates(new_queue, name, directory):
    import generate_pt1
    import generate_pt2

    schema = generate_pt1.SCHEMA if name == "PT1" else generate_pt2.SCHEMA

    old, oldout = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / f"configs/v2/{directory}/manifest.json",
        Path(old["runtime"]),
        oldout.parent / (name.lower() + "-scope"),
        copy.deepcopy(old["shared_gpu_policy"]),
    )
    runner.verify(q)
    assert [r["name"] for r in q["rows"]] == [name]
    assert schema in runner.LAST_ONLY_FAMILIES and list(medium.FAMILY_ARMS[schema]) == [name]
    row = q["rows"][0]
    receipt = training_receipt(q, row)
    receipt.pop("joint_validation_check")
    receipt.update(
        joint_validation_check_requested=False,
        staged_training_gate={"passed": True},
        evaluation_cell_batch=128,
    )
    for key, phase in (
        ("initial_loss_diagnostic", "ssl_pretrain"),
        ("updated_loss_diagnostic", "mse_finetune"),
    ):
        receipt[key] = {
            "process": {
                "split": "train",
                "selection_use": "none",
                "schema": "training-process-diagnostics-1",
                "training_stage": phase,
                "distributions": {"head": {}} if phase == "ssl_pretrain" else {},
                "features": {"f": {}},
                "outputs": {"status": "no_labels"}
                if phase == "ssl_pretrain"
                else {"prediction": {}},
                "components": {"ssl2_dino": 1} if phase == "ssl_pretrain" else {"prediction": 1},
            }
        }
    path = Path(q["output"]) / f"{name}-preflight/receipt.json"
    atomic_json(path, receipt)
    entry = runner.gate_entry(path)
    runner.validate_training_gate(q, row, entry)
    receipt["staged_training_gate"]["passed"] = False
    atomic_json(path, receipt)
    entry["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="actual SSL/MSE"):
        runner.validate_training_gate(q, row, entry)
