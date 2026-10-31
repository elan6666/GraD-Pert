"""Precision queue admission contracts; synthetic receipts are never native pass evidence."""

import contextlib
import copy
from pathlib import Path

import pytest
import run_br400_group as runner

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/v2/precision_ri_20261031/manifest.json"


def arm_row(name):
    config = ROOT / f"configs/v2/precision_ri_20261031/{name}/gradpert_v2/nadig_jurkat.yaml"
    return dict(
        name=name,
        plan=dict(
            repository_root=str(ROOT),
            config=str(config),
            postfit_policy="deferred",
            resolved_config=load_experiment_config(config).model_dump(mode="json"),
        ),
    )


def receipt_for(row):
    # Explicit synthetic signatures independent of coverage helper; they model the frozen RI layout.
    name = row["name"]
    if name == "PREC-SSL32":
        heads = [
            f"{side}.{head}.{leaf}"
            for side in ("student", "teacher")
            for head in ("ssl1_cls", "ssl2_cls", "ssl2_node")
            for leaf in ("features.0", "features.2", "prototypes")
        ]
    else:
        heads = ["student.prediction.0", "student.prediction.2"] if name == "PREC-PRED32" else []
    modules = [
        *(f"graph.layers.{i}.ffn" for i in range(4)),
        *(f"cell.layers.{i}.sublayer" for i in (1, 3, 5)),
        *(f"response.ffn_layers.{i}.sublayer" for i in range(3)),
    ]
    gemms = []
    if name == "PREC-FFN8":
        for phase in ("before_restore", "after_restore"):
            for side in ("student", "teacher"):
                for module in modules:
                    for projection in ("up", "down"):
                        for op in (
                            ("forward", "dgrad", "wgrad") if side == "student" else ("forward",)
                        ):
                            gemms.append(
                                dict(
                                    phase=phase,
                                    path=f"{side}.{module}.{projection}.weight",
                                    operation=op,
                                    a="torch.float8_e4m3fn"
                                    if op == "forward"
                                    else "torch.float8_e5m2",
                                    b="torch.float8_e4m3fn",
                                    output="torch.bfloat16" if op == "forward" else "torch.float32",
                                    use_fast_accum=False,
                                    calls=1,
                                )
                            )
    audit = dict(
        schema="gradpert-precision-preflight-1",
        status="passed",
        exact_full_state_restore=True,
        backend=dict(
            torch_version="2.13.0+cu130",
            cuda_version="13.0",
            device_name="synthetic SM120",
            compute_capability=[12, 0],
            matmul_allow_tf32=False,
            float32_matmul_precision="highest",
            scaled_mm_schema="aten::_scaled_mm",
            recipe="per-tensor current scaling, E4M3 forward / E5M2 gradient",
        ),
        head_linear_calls=[
            dict(phase=phase, path=path, calls=1)
            for phase in ("before_restore", "after_restore")
            for path in heads
        ],
        fp8_gemm_calls=gemms,
    )
    return dict(
        resume_checkpoint_sha256="a" * 64,
        precision_audit=copy.deepcopy(audit),
        rank_measurements=[
            dict(rank=i, physical_gpu=str(i), precision_audit=copy.deepcopy(audit)) for i in (0, 1)
        ],
    )


@pytest.mark.parametrize("name", ["PREC-SSL32", "PREC-PRED32", "PREC-FFN8"])
def test_exact_family_and_native_observation_contract(name):
    row = arm_row(name)
    assert runner.verify_manifest(ROOT, MANIFEST)["priority"] == list(runner.PRECISION_ARMS)
    assert runner.expected_epochs(row["plan"]) == 2
    runner.validate_precision_gate(row, receipt_for(row))


@pytest.mark.parametrize(
    "change",
    [
        "missing_rank",
        "duplicate_rank",
        "no_audit",
        "bare_pass",
        "no_restore",
        "missing_head",
        "extra_inactive_head",
        "wrong_gpu",
        "tf32",
        "primary_mismatch",
    ],
)
def test_head_gate_rejects_incomplete_or_misattributed_evidence(change):
    row = arm_row("PREC-SSL32")
    receipt = receipt_for(row)
    audit = receipt["rank_measurements"][1]["precision_audit"]
    if change == "missing_rank":
        receipt["rank_measurements"].pop()
    elif change == "duplicate_rank":
        receipt["rank_measurements"][1]["rank"] = 0
    elif change == "no_audit":
        receipt["rank_measurements"][1].pop("precision_audit")
    elif change == "bare_pass":
        receipt["rank_measurements"][1]["precision_audit"] = {"status": "passed"}
    elif change == "no_restore":
        audit["exact_full_state_restore"] = False
    elif change == "missing_head":
        audit["head_linear_calls"].pop()
    elif change == "extra_inactive_head":
        audit["head_linear_calls"].append(
            dict(phase="after_restore", path="student.ssl1_node.features.0", calls=1)
        )
    elif change == "wrong_gpu":
        receipt["rank_measurements"][1]["physical_gpu"] = "0"
    elif change == "tf32":
        audit["backend"]["matmul_allow_tf32"] = True
    else:
        receipt["precision_audit"]["exact_full_state_restore"] = False
    with pytest.raises(ValueError):
        runner.validate_precision_gate(row, receipt)


@pytest.mark.parametrize(
    "change",
    [
        "no_wgrad",
        "no_teacher",
        "no_after_restore",
        "fake_dtype",
        "fast_accum",
        "old_torch",
        "wrong_sm",
        "no_api",
    ],
)
def test_fp8_gate_rejects_simulation_and_incomplete_kernel_coverage(change):
    row = arm_row("PREC-FFN8")
    receipt = receipt_for(row)
    audit = receipt["rank_measurements"][1]["precision_audit"]
    if change in ("no_wgrad", "no_teacher", "no_after_restore"):
        audit["fp8_gemm_calls"] = [
            c
            for c in audit["fp8_gemm_calls"]
            if not (
                c["operation"] == "wgrad"
                if change == "no_wgrad"
                else c["path"].startswith("teacher.")
                if change == "no_teacher"
                else c["phase"] == "after_restore"
            )
        ]
    elif change == "fake_dtype":
        audit["fp8_gemm_calls"][0]["a"] = "torch.bfloat16"
    elif change == "fast_accum":
        audit["fp8_gemm_calls"][0]["use_fast_accum"] = True
    elif change == "old_torch":
        audit["backend"]["torch_version"] = "2.12.0"
    elif change == "wrong_sm":
        audit["backend"]["compute_capability"] = [9, 0]
    else:
        audit["backend"].pop("scaled_mm_schema")
    with pytest.raises(ValueError):
        runner.validate_precision_gate(row, receipt)


def test_two_epoch_rejects_altered_config_and_baseline():
    row = arm_row("PREC-PRED32")
    row["plan"]["resolved_config"]["training"]["max_epochs"]["value"] = 4
    with pytest.raises(ValueError):
        runner.validate_precision_plan(row["plan"])
    with pytest.raises(ValueError):
        runner.validate_precision_gate(dict(name="R3-RI"), {})


@pytest.mark.parametrize(
    "budget,counts", [([2, 150], [150, 300]), ([2, 149], [149, 298]), ([2, 150], [150, 299])]
)
def test_terminal_checks_successful_update_budget(tmp_path, monkeypatch, budget, counts):
    row = arm_row("PREC-PRED32")
    plan = row["plan"]
    plan["run_root"] = str(tmp_path)
    atomic_json(tmp_path / "fit/epoch_state.json", dict(budget=budget))
    atomic_json(
        tmp_path / "fit/history.json",
        [
            dict(optimizer_steps=n, validation=dict(validation_mode="disabled", performed=False))
            for n in counts
        ],
    )
    monkeypatch.setattr(
        runner,
        "validate_training_stage",
        lambda p: dict(epoch=2, last=dict(epoch=2), test_roles=["last"], best=None),
    )
    if budget == [2, 150] and counts == [150, 300]:
        runner.validate_training_contract(plan)
    else:
        with pytest.raises(ValueError, match="300 successful"):
            runner.validate_training_contract(plan)


@pytest.mark.parametrize("pollute", [None, "queue", "fit", "test"])
def test_precision_preflight_returns_before_fit_and_requires_explicit_promotion(
    tmp_path, monkeypatch, pollute
):
    directory = tmp_path
    rows = [arm_row(name) for name in runner.PRECISION_ARMS]
    for row in rows:
        row["plan"].update(
            run_root=str(tmp_path / f"run-{row['name']}"),
            test_root=str(tmp_path / f"test-{row['name']}"),
        )
        atomic_json(directory / f"{row['name']}.launch-plan.json", row["plan"])
    queue = dict(
        source=str(ROOT),
        output=str(directory),
        family=runner.PRECISION_SCHEMA,
        source_commit="a" * 40,
        preflight_only=True,
        manifest=str(MANIFEST),
        rows=rows,
        shared_gpu_policy={},
        eval_bindings=[
            dict(row_name=row["name"], gpu=gpu, output=str(directory / f"{row['name']}-g{gpu}"))
            for row in rows
            for gpu in ("0", "1")
        ],
    )
    atomic_json(directory / "queue.json", queue)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "verify", lambda q: None)
    monkeypatch.setattr(runner, "lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(runner, "snapshot", lambda *a: dict(processes=[]))
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    monkeypatch.setattr(runner, "validate_training_gate", lambda *a: {})
    monkeypatch.setattr(runner, "validate_eval_gate", lambda *a: {})
    monkeypatch.setattr(runner, "validate_common_gates", lambda *a: dict(frozen="common"))
    monkeypatch.setattr(runner, "probe", lambda *a, **k: ["synthetic-probe"])
    monkeypatch.setattr(runner, "eval_capacity_command", lambda *a: ["synthetic-eval"])
    calls = []

    def stage(q, row, phase, command, gpu, state, guard):
        calls.append(phase)
        if phase == "fit":
            atomic_json(Path(row["plan"]["run_root"]) / "TRAIN_COMPLETE.json", {"synthetic": True})
            return
        assert phase == "preflight" or phase.startswith("eval-capacity-g")
        path = (
            directory / f"{row['name']}-preflight/receipt.json"
            if phase == "preflight"
            else directory / f"{row['name']}-g{gpu}/receipt.json"
        )
        atomic_json(path, dict(synthetic=True))

    monkeypatch.setattr(runner, "run_stage", stage)
    if pollute:
        root = (
            directory
            if pollute == "queue"
            else Path(rows[0]["plan"]["run_root" if pollute == "fit" else "test_root"])
        )
        polluted = root / "nested" / "result.pkl"
        polluted.parent.mkdir(parents=True)
        polluted.write_bytes(b"must-preserve")
        with pytest.raises(ValueError, match="zero-PKL"):
            runner.execute(queue, directory, preflight_only=True)
        assert not (directory / "PREFLIGHT_COMPLETE.json").exists()
        assert polluted.read_bytes() == b"must-preserve"
        return
    runner.execute(queue, directory, preflight_only=True)
    assert calls.count("preflight") == 3 and len(calls) == 9
    state = read_json(directory / "state.json")
    assert state["status"] == "preflight_complete" and state["formal_fit_authorized"] is False
    with pytest.raises(FileNotFoundError):
        runner.validate_fit_promotion(queue, directory, state)
    ready = read_json(directory / "PREFLIGHT_COMPLETE.json")
    atomic_json(directory / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})
    assert runner.validate_fit_promotion(queue, directory, state) == ready
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: dict(last={"epoch": 2}))
    with pytest.raises(ValueError, match="fit-only-row"):
        runner.execute(queue, directory, resume_fit=True)
    runner.execute(queue, directory, resume_fit=True, fit_only_row="PREC-SSL32")
    assert calls.count("fit") == 1
    fitted = read_json(directory / "state.json")
    assert fitted["phase"] == "awaiting_main_precision_postfit"
    assert fitted["scientific_complete"] is False and fitted["completed"] == {}
    assert (directory / "FIT_COMPLETE.PREC-SSL32.json").is_file()
    assert not (directory / "RESEARCH_COMPLETE.json").exists()
    assert not (directory / "COMPLETE.json").exists()
    with pytest.raises(ValueError):
        runner.validate_fit_promotion(queue, directory, fitted, fit_only_row="PREC-SSL32")
    assert (
        runner.validate_fit_promotion(queue, directory, fitted, fit_only_row="PREC-PRED32") == ready
    )
    ready["queue_sha256"] = "b" * 64
    atomic_json(directory / "MAIN_ACCEPTANCE.json", {**ready, "approved": True})
    with pytest.raises(ValueError):
        runner.validate_fit_promotion(queue, directory, state)


@pytest.fixture
def precision_leaf_queue(tmp_path):
    """Real manifest/config verification and real shared-policy validator, without CUDA."""
    from types import SimpleNamespace

    from shared_gpu import TOTALVI_MEMORY90_AUTHORIZATION, TOTALVI_PROGRAM

    from gradpert.hashing import sha256_file

    policy = dict(
        schema="shared-gpu-admission-3",
        devices=[dict(gpu_index=i, gpu_uuid=f"GPU-{i}") for i in (0, 1)],
        foreign_uid=1003,
        command_basename="run_scbutterfly_archived_predictions.py",
        max_foreign_processes_per_gpu=1,
        max_foreign_memory_mib=1024,
        min_free_memory_mib=2048,
        max_own_memory_fraction=0.90,
        authorization_reference=TOTALVI_MEMORY90_AUTHORIZATION,
        additional_command_basenames=[TOTALVI_PROGRAM],
    )
    rows = [arm_row(name) for name in runner.PRECISION_ARMS]
    publication = tmp_path / "publication.json"
    atomic_json(publication, dict(synthetic=True))
    for row in rows:
        row["plan"].update(
            source_commit="a" * 40,
            config_sha256=sha256_file(Path(row["plan"]["config"])),
            data_root=str(tmp_path / "data"),
            publication=str(publication),
            publication_sha256=sha256_file(publication),
            run_root=str(tmp_path / f"run-{row['name']}"),
            test_root=str(tmp_path / f"test-{row['name']}"),
        )
    queue = dict(
        schema="unified-ddp-queue-1",
        family=runner.PRECISION_SCHEMA,
        mode=runner.PRECISION_SCHEMA,
        source=str(ROOT),
        source_commit="a" * 40,
        manifest=str(MANIFEST),
        manifest_sha256=sha256_file(MANIFEST),
        rows=rows,
        steps=10,
        shared_gpu_policy=policy,
        preflight_only=True,
        postfit_mode=runner.PRECISION_POSTFIT,
        precision_training_budget=copy.deepcopy(runner.PRECISION_BUDGET),
        output=str(tmp_path / "queue"),
        eval_bindings=[
            dict(
                row_name=row["name"],
                gpu=gpu,
                cell_batch=128,
                reference_batch=128,
                output=str(tmp_path / "queue" / f"{row['name']}-eval-b128-g{gpu}"),
            )
            for row in rows
            for gpu in ("0", "1")
        ],
    )
    args = SimpleNamespace(
        gpu="0",
        cell_batch=128,
        reference_batch=128,
        shared_queue=tmp_path / "queue" / "queue.json",
        data_root=tmp_path / "data",
        publication=publication,
        publication_sha256=sha256_file(publication),
    )
    return queue, args


@pytest.mark.parametrize("name", ["PREC-SSL32", "PREC-PRED32", "PREC-FFN8"])
@pytest.mark.parametrize("gpu", ["0", "1"])
def test_real_dual_eval_leaf_binding_for_each_precision_arm(precision_leaf_queue, name, gpu):
    from eval_batch_capacity import validate_eval_binding

    queue, args = precision_leaf_queue
    row = next(r for r in queue["rows"] if r["name"] == name)
    args.row_name, args.gpu = name, gpu
    args.config = Path(row["plan"]["config"])
    args.output = args.shared_queue.parent / f"{name}-eval-b128-g{gpu}"
    assert validate_eval_binding(queue, args, "a" * 40) is row


@pytest.mark.parametrize(
    "change",
    [
        "mode",
        "order",
        "missing_row",
        "steps",
        "eval_batch",
        "raw_config_hash",
        "resolved_config",
        "missing_resolved",
        "plan_source",
        "queue_source",
        "source_path",
        "missing_leaf",
        "manifest_hash",
        "baseline",
    ],
)
def test_real_eval_leaf_rejects_scope_or_identity_mutation(precision_leaf_queue, change):
    from eval_batch_capacity import validate_eval_binding

    queue, args = precision_leaf_queue
    args.row_name = "PREC-SSL32"
    args.config = Path(queue["rows"][0]["plan"]["config"])
    args.output = args.shared_queue.parent / "PREC-SSL32-eval-b128-g0"
    if change == "mode":
        queue["mode"] = "representation-round3-family-1"
    elif change == "order":
        queue["rows"].reverse()
    elif change == "missing_row":
        queue["rows"].pop()
    elif change == "steps":
        queue["steps"] = 1
    elif change == "eval_batch":
        args.cell_batch = 300
    elif change == "raw_config_hash":
        queue["rows"][0]["plan"]["config_sha256"] = "b" * 64
    elif change == "resolved_config":
        queue["rows"][0]["plan"]["resolved_config"]["training"]["max_epochs"]["value"] = 4
    elif change == "missing_resolved":
        queue["rows"][0]["plan"].pop("resolved_config")
    elif change == "plan_source":
        queue["rows"][0]["plan"]["source_commit"] = "b" * 40
    elif change == "queue_source":
        queue["source_commit"] = "b" * 40
    elif change == "source_path":
        queue["source"] = str(args.shared_queue.parent / "wrong-source")
    elif change == "missing_leaf":
        queue["eval_bindings"].pop()
    elif change == "manifest_hash":
        queue["manifest_sha256"] = "b" * 64
    else:
        queue["rows"][0]["name"] = "R3-RI"
    with pytest.raises((ValueError, OSError)):
        validate_eval_binding(queue, args, "a" * 40)


@pytest.mark.parametrize("root", ["queue", "fit", "test"])
def test_precision_zero_pkl_rejects_every_root_without_deleting(precision_leaf_queue, root):
    queue, _ = precision_leaf_queue
    base = (
        Path(queue["output"])
        if root == "queue"
        else Path(queue["rows"][0]["plan"]["run_root" if root == "fit" else "test_root"])
    )
    polluted = base / "nested" / "retained.pkl"
    polluted.parent.mkdir(parents=True)
    polluted.write_bytes(b"retained-test-evidence")
    with pytest.raises(ValueError, match="zero-PKL"):
        runner.validate_precision_zero_pkl(queue)
    with pytest.raises(ValueError, match="zero-PKL"):
        runner.precision_fit_evidence(queue, queue["rows"][0])
    assert polluted.read_bytes() == b"retained-test-evidence"


def test_retained_fitted_arm_pkl_cannot_promote_next_arm(precision_leaf_queue):
    queue, _ = precision_leaf_queue
    polluted = Path(queue["rows"][0]["plan"]["run_root"]) / "fit" / "adapter" / "result.pkl"
    polluted.parent.mkdir(parents=True)
    polluted.write_bytes(b"no-delete")
    state = dict(
        status="fit_complete",
        phase="awaiting_main_precision_postfit",
        precision_fits={"PREC-SSL32": {}},
    )
    with pytest.raises(ValueError, match="zero-PKL"):
        runner.validate_fit_promotion(
            queue, Path(queue["output"]), state, fit_only_row="PREC-PRED32"
        )
    assert polluted.is_file()
