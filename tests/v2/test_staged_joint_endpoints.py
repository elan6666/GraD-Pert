"""R9 endpoint roles, shared manifests and final-only postfit contracts."""

import copy
import json
import sys
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round9 as family  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as postfit  # noqa: E402
import run_staged_joint_medium as dual  # noqa: E402
from checkpoint_endpoint import select_checkpoint  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def endpoints(tmp_path):
    root = tmp_path / "run"
    config = family.configs(ROOT)[family.ARMS[0]]
    write(root / "resolved_config.json", config)
    write(root / "run_manifest.json", {"run_id": "R9"})
    entries = {}
    for role, epoch in (("pretrain", 1), ("last", 3)):
        p = root / "fit" / f"epoch-{epoch:04}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(role.encode())
        entries[role] = dict(file=p.name, epoch=epoch, sha256=sha256_file(p))
    write(root / "fit/epoch_state.json", {"last": entries["last"]})
    write(
        root / "fit/pretrain-checkpoint.json",
        dict(identity={"run_id": "R9"}, checkpoint=entries["pretrain"]),
    )
    return root, entries


def test_prefix_boundary_is_ssl_and_final_is_continuing_joint(endpoints):
    root, entries = endpoints
    for role, spec in family.ENDPOINTS.items():
        cp, endpoint = select_checkpoint(root, role)
        assert cp == entries[role]
        assert endpoint == {**spec, "full_training_epochs": 3}
    assert entries["pretrain"]["sha256"] != entries["last"]["sha256"]


@pytest.mark.parametrize("change", ["hash", "epoch", "identity", "not_staged"])
def test_prefix_endpoint_cannot_borrow_or_change_checkpoint(endpoints, change):
    root, _ = endpoints
    receipt = json.loads((root / "fit/pretrain-checkpoint.json").read_text())
    if change == "hash":
        receipt["checkpoint"]["sha256"] = "f" * 64
    elif change == "epoch":
        receipt["checkpoint"]["epoch"] = 2
    elif change == "identity":
        receipt["identity"] = {"run_id": "other"}
    else:
        config = json.loads((root / "resolved_config.json").read_text())
        config["model"]["parameters"]["training_mode"]["value"] = "joint"
        write(root / "resolved_config.json", config)
    write(root / "fit/pretrain-checkpoint.json", receipt)
    with pytest.raises(ValueError):
        select_checkpoint(root, "pretrain")


@pytest.fixture
def dual_case(tmp_path, endpoints, monkeypatch):
    run, _entries = endpoints
    output = tmp_path / "queue"
    queue = dict(family=family.SCHEMA, output=str(output), source_commit="a" * 40)
    write(output / "queue.json", queue)
    row = dict(name=family.ARMS[0], plan=dict(run_root=str(run)))
    root = output / f"{row['name']}-medium"
    manifest = root / "pretrain/manifest.json"
    write(manifest, {"schedule": ["sealed"]})
    proofs = {}
    for role in family.ENDPOINTS:
        proofs[role] = dict(receipt=role, sha256=role)
        write(
            root / role / "MEDIUM_COMPLETE.json",
            dict(
                checkpoint_endpoint=select_checkpoint(run, role)[1],
                manifest=str(manifest),
                manifest_sha256=sha256_file(manifest),
            ),
        )
    write(
        root / "MEDIUM_COMPLETE.json",
        dict(
            schema="staged-joint-dual-endpoint-medium-1",
            status="completed",
            updates=0,
            source_commit=queue["source_commit"],
            queue_sha256=sha256_file(output / "queue.json"),
            endpoints=proofs,
        ),
    )
    monkeypatch.setattr(
        medium,
        "validate_medium_complete",
        lambda q, r, *, root, checkpoint_role: proofs[checkpoint_role],
    )
    return queue, row, root


@pytest.mark.parametrize("change", [None, "missing", "different_manifest", "endpoint", "updates"])
def test_both_medium_endpoints_need_the_same_physical_manifest(dual_case, change):
    queue, row, root = dual_case
    if change in ("missing", "updates"):
        path = root / "MEDIUM_COMPLETE.json"
        value = json.loads(path.read_text())
        if change == "missing":
            value["endpoints"].pop("pretrain")
        else:
            value["updates"] = 1
        write(path, value)
    elif change in ("different_manifest", "endpoint"):
        path = root / "last/MEDIUM_COMPLETE.json"
        value = json.loads(path.read_text())
        if change == "endpoint":
            value["checkpoint_endpoint"]["epoch"] = 2
        else:
            alternate = root / "last/manifest.json"
            write(alternate, {"schedule": ["sealed"]})
            value.update(manifest=str(alternate), manifest_sha256=sha256_file(alternate))
        write(path, value)
    if change is None:
        assert dual.validate_complete(queue, row, root=root)["conditions"] == 1335
    else:
        with pytest.raises(ValueError):
            dual.validate_complete(queue, row, root=root)


def test_two_panel_commands_share_scope_and_require_teacher_signal(tmp_path):
    queue = dict(
        family=family.SCHEMA,
        source=str(ROOT),
        source_commit="a" * 40,
        output=str(tmp_path),
        medium_analysis={"parent_manifest": "sealed.json", "parent_manifest_sha256": "b" * 64},
    )
    write(tmp_path / "queue.json", queue)
    row = dict(name=family.ARMS[0], plan=dict(run_root="run", config="config", data_root="data"))
    commands = [
        postfit.panel_command(queue, row, checkpoint_role=role) for role in family.ENDPOINTS
    ]
    assert commands[0][commands[0].index("--checkpoint-role") + 1] == "pretrain"
    assert commands[1][commands[1].index("--checkpoint-role") + 1] == "last"
    for command in commands:
        assert "--teacher-signal-audit" in command
        assert command[command.index("--conditions") + 1] == "32"
        assert command[command.index("--controls") + 1] == "4"
        assert command[command.index("--parent-manifest-sha256") + 1] == "b" * 64
    normalized = copy.deepcopy(commands)
    for command in normalized:
        for flag in ("--output", "--checkpoint-role"):
            command[command.index(flag) + 1] = "same"
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize("change", [None, "prefix", "test", "population", "missing_group"])
def test_final_only_validation_keeps_all445_conditions_and18metrics(tmp_path, monkeypatch, change):
    import test_representation_postfit as old

    queue, row, root = old.validation_receipt.__wrapped__(tmp_path, monkeypatch)
    queue["family"] = family.SCHEMA
    plan_path, receipt_path = root / "plan.json", root / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["plan"]["checkpoint_epoch"] = 3
    groups = [
        receipt["result"]["metrics"],
        *(group["metrics"] for group in receipt["result"]["metric_gene_groups"].values()),
    ]
    for metrics in groups:
        for metric in metrics:
            metric["total_condition_count"] = 445
    if change == "prefix":
        receipt["plan"].update(checkpoint_role="pretrain", checkpoint_epoch=1)
    elif change == "test":
        receipt["plan"]["split"] = "test"
    elif change == "population":
        receipt["result"]["metrics"][0]["total_condition_count"] = 444
    elif change == "missing_group":
        receipt["result"]["metric_gene_groups"].pop("unseen_expression")
    write(plan_path, receipt["plan"])
    write(receipt_path, receipt)
    if change is None:
        assert postfit.validate_validation(queue, row)["split"] == "val"
    else:
        with pytest.raises(ValueError):
            postfit.validate_validation(queue, row)


def test_engineering5plus5_uses_actual_formal150_and300_lr_clocks():
    from capacity_probe import staged_joint_probe_lr

    from gradpert.config.step_schedule import EndpointLRWarmupCosine

    schedule = EndpointLRWarmupCosine(max_lr=0.001, min_lr=0.0002, warmup_fraction=0.16)
    # The original bug: five is a phase boundary for this short engineering check,
    # not a valid scientific LR horizon for this warmup fraction.
    with pytest.raises(ValueError):
        schedule.at_step(0, 5)
    rows = [
        staged_joint_probe_lr(schedule, step, formal_pretrain_steps=150, formal_total_steps=450)
        for step in range(10)
    ]
    assert [row["phase_local_step"] for row in rows] == list(range(5)) * 2
    assert [row["formal_clock_horizon"] for row in rows] == [150] * 5 + [300] * 5
    assert [row["formal_global_step"] for row in rows] == [0, 1, 2, 3, 4, 150, 151, 152, 153, 154]
    assert [row["learning_rate"] for row in rows[:5]] == [
        schedule.at_step(step, 150)["learning_rate"] for step in range(5)
    ]
    assert [row["learning_rate"] for row in rows[5:]] == [
        schedule.at_step(step, 300)["learning_rate"] for step in range(5)
    ]
    assert rows[0]["learning_rate"] == rows[5]["learning_rate"] == 0
    assert rows[4]["learning_rate"] > rows[9]["learning_rate"] > 0
    for step in (-1, 10):
        with pytest.raises(ValueError):
            staged_joint_probe_lr(schedule, step, formal_pretrain_steps=150, formal_total_steps=450)


@pytest.mark.parametrize(
    "field",
    [
        "stage_phase",
        "prediction_enabled",
        "teacher_updates_enabled",
        "decode_ssl_views",
        "pretrain_steps",
        "optimizer_phase",
        "optimizer_resets",
    ],
)
def test_restore_fingerprint_binds_real_phase_buffer_and_runtime_flags(field):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from capacity_probe import staged_restore_fingerprint, state_fingerprint
    from test_components import fixture

    from gradpert.training.v2.optimizer import V2Optimizer
    from gradpert.training.v2.staged import SSLThenJointObjective

    model, _batch = fixture()
    objective = SSLThenJointObjective(model, pretrain_steps=5, lambda1=1, lambda2=1)
    optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=optimizer,
        generator=np.random.default_rng(7),
        device=torch.device("cpu"),
    )
    # Historical PT1's exact full objective/optimizer/generator contract is retained.
    assert staged_restore_fingerprint(runtime, continuing_joint=False) == state_fingerprint(
        (objective.state_dict(), optimizer.state_dict(), runtime.generator.bit_generator.state)
    )
    before = staged_restore_fingerprint(runtime, continuing_joint=True)
    if field == "stage_phase":
        objective.stage_phase.fill_(1)
    elif field == "optimizer_phase":
        optimizer.training_stage = "joint"
    elif field == "optimizer_resets":
        optimizer.stage_reset_count = 1
    elif field == "pretrain_steps":
        objective.pretrain_steps = 6
    else:
        setattr(objective, field, not getattr(objective, field))
    assert staged_restore_fingerprint(runtime, continuing_joint=True) != before


@pytest.fixture(scope="module")
def native_teacher_signal():
    import test_teacher_signal_diagnostics as native
    from representation_panel import panel_schedule

    parent = {"analysis_scope": "medium_cap40_stratified", "schedule": []}
    for condition in range(32):
        rows = [
            dict(
                condition_id=f"p{condition}",
                control_row_id=f"c{control}",
                truth_row_id=f"t{condition}-{control}",
                context_id="T::b1",
                batch="b1",
                cell_line="Jurkat",
                cell_type="T",
            )
            for control in range(4)
        ]
        parent["schedule"].append(dict(condition_id=f"p{condition}", rows=rows, queries=[1, 3]))
    panel = panel_schedule(parent)
    objective = native.Objective()
    bank = native.module.TeacherSignalBank()
    for group in panel["schedule"]:
        with bank.capture(objective, group["rows"]):
            native.native(objective)
    return parent, panel, bank.result()


@pytest.mark.parametrize("role", ["pretrain", "last"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        "file",
        "output",
        "schema",
        "captures",
        "path",
        "scope",
        "pairs",
        "mode",
        "input",
        "unsealed_tamper",
    ],
)
def test_both_r9_panels_require_sealed_native_teacher_signal(
    tmp_path, monkeypatch, endpoints, native_teacher_signal, role, change
):
    run, checkpoints = endpoints
    parent, panel, signal = copy.deepcopy(native_teacher_signal)
    parent_path = tmp_path / "parent.json"
    write(parent_path, parent)
    queue = dict(
        family=family.SCHEMA,
        output=str(tmp_path),
        source_commit="a" * 40,
        medium_analysis=dict(
            parent_manifest=str(parent_path), parent_manifest_sha256=sha256_file(parent_path)
        ),
    )
    row = dict(
        name=family.ARMS[0], plan=dict(run_root=str(run), run_id="R9", config_sha256="b" * 64)
    )
    monkeypatch.setattr(postfit, "context", lambda *a: {"last": checkpoints["last"]})
    for endpoint in family.ENDPOINTS:
        root = tmp_path / f"{row['name']}-panel" / endpoint
        write(root / "panel.json", panel)
        write(root / "teacher_signal.json", signal)
        receipt = dict(
            status="completed",
            updates=0,
            method_sha=queue["source_commit"],
            run_id="R9",
            checkpoint_sha256=checkpoints[endpoint]["sha256"],
            checkpoint_endpoint=select_checkpoint(run, endpoint)[1],
            config_sha256="b" * 64,
            state_unchanged=True,
            state_before={"state": "same"},
            state_after={"state": "same"},
            parent_manifest=str(parent_path),
            parent_manifest_sha256=sha256_file(parent_path),
            panel_sha256=panel["panel_sha256"],
            panel_rows=128,
            diagnostic_mode="teacher_signal_audit",
            teacher_signal_input_mode="native_student_masked_teacher_clean",
            parity=dict(
                used_path_passed=True,
                tolerance=2e-6,
                cell_chunk_used=False,
                graph_max_abs=0,
                response_max_abs=0,
            ),
            outputs=postfit.output_seal(root),
        )
        if endpoint == role and change:
            if change in ("schema", "captures", "path", "scope", "pairs", "unsealed_tamper"):
                altered = copy.deepcopy(signal)
                if change in ("schema", "unsealed_tamper"):
                    altered["schema"] = "other"
                elif change == "captures":
                    altered["captures"].pop()
                elif change == "path":
                    altered["paths"].pop()
                elif change == "scope":
                    altered["paths"][0]["conditions"] = 31
                else:
                    altered["native_pairs"] = []
                write(root / "teacher_signal.json", altered)
                if change != "unsealed_tamper":
                    receipt["outputs"] = postfit.output_seal(root)
            elif change == "file":
                (root / "teacher_signal.json").unlink()
                receipt["outputs"].pop("teacher_signal.json")
            elif change == "output":
                receipt["outputs"].pop("teacher_signal.json")
            elif change == "mode":
                receipt["diagnostic_mode"] = "crossed_views"
            else:
                receipt["teacher_signal_input_mode"] = "clean_globals"
        write(root / "COMPLETE.json", receipt)
    if change is None:
        assert set(postfit.validate_panel(queue, row)["endpoints"]) == {"pretrain", "last"}
    else:
        with pytest.raises(ValueError):
            postfit.validate_panel(queue, row)
