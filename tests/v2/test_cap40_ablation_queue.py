"""Resource and stale-preflight boundaries for the approved cap40 queue."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
spec = importlib.util.spec_from_file_location(
    "cap40_queue", ROOT / "scripts/v2/run_cap40_ablations.py"
)
assert spec and spec.loader
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize("usage", ["0, 2\n1, 728\n", "0, 2\n", "0, 513\n1, 2\n", ""])
def test_waits_when_either_gpu_is_occupied_or_unknown(usage):
    assert not queue.idle_devices(usage)


def test_idle_boundary_and_extra_gpu_do_not_override_selected_devices():
    assert queue.idle_devices("0, 512\n1, 2\n2, 9999\n\n")


def test_capacity_failure_is_not_launch_permission(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"kind": "capacity_only", "status": "failed", "steps_completed": 127})
    )
    entry = {"receipt": str(receipt), "sha256": queue.sha256_file(receipt)}
    with pytest.raises(ValueError, match="passed"):
        queue.validate_preflight({}, entry)


def test_one_step_capacity_cannot_pass_as_sustained_capacity(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "capacity_only",
                "status": "passed",
                "steps_completed": 1,
                "resume_checkpoint_sha256": "x",
            }
        )
    )
    entry = {"receipt": str(receipt), "sha256": queue.sha256_file(receipt)}
    with pytest.raises(ValueError, match="update/checkpoint"):
        queue.validate_preflight({}, entry)


def test_incomplete_baseline_is_rejected_before_planning(tmp_path):
    (tmp_path / "fit").mkdir()
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"epoch": 5, "zero_pkl": True}))
    (tmp_path / "fit/epoch_state.json").write_text(json.dumps({"epoch": 5}))
    with pytest.raises(ValueError, match="prerequisite"):
        queue.prepare(Path("unused"), tmp_path)


def test_unapproved_config_hash_is_rejected_before_resolving(tmp_path, monkeypatch):
    (tmp_path / "fit").mkdir()
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"epoch": 6, "zero_pkl": True}))
    (tmp_path / "fit/epoch_state.json").write_text(json.dumps({"epoch": 6}))
    monkeypatch.setattr(queue, "sha256_file", lambda path: "changed")
    with pytest.raises(ValueError, match="configuration changed"):
        queue.prepare(Path("unused"), tmp_path)


def test_existing_run_never_enters_fresh_formal_launch(tmp_path):
    plan = {"run_root": str(tmp_path)}
    with pytest.raises(ValueError, match="saved launch plan"):
        queue.require_fresh_formal(plan)


def test_valid_committed_run_is_not_implicitly_resumed(tmp_path):
    plan = {"run_root": str(tmp_path)}
    (tmp_path / "launch.json").write_text(json.dumps(plan))
    (tmp_path / "fit").mkdir()
    (tmp_path / "fit/epoch_state.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot resume"):
        queue.require_fresh_formal(plan)


def imported_e1(tmp_path, monkeypatch):
    root, baseline = tmp_path / "e1", tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    manifest = {"data": {"selection": "frozen", "run_seed": 1}, "source": {"commit": "a" * 40}}
    for path in (root, baseline):
        (path / "run_manifest.json").write_text(json.dumps(manifest))
    (root / "COMPLETE.json").write_text(json.dumps({"epoch": 6}))
    config = ROOT / "configs/v2/cap40_ablations_jurkat/E1_no_mhc/gradpert_v2/nadig_jurkat.yaml"
    resolved = queue.load_experiment_config(config).model_dump(mode="json")
    (root / "resolved_config.json").write_text(json.dumps(resolved))
    rows = [
        {"role": role, "status": "complete", "config_sha256": queue.ROWS[0][1]}
        for role in ("best", "last")
    ]
    monkeypatch.setattr(queue, "collect_run", lambda path: rows)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *args, **kwargs: "same-src-tree\n")
    return root, baseline, rows


def test_completed_e1_is_reused_only_with_same_data_config_and_native_src(tmp_path, monkeypatch):
    root, baseline, rows = imported_e1(tmp_path, monkeypatch)
    accepted = queue.validate_completed_e1(root, baseline)
    assert accepted["collected_rows"] == rows
    assert accepted["complete_sha256"] == queue.sha256_file(root / "COMPLETE.json")
    assert accepted["native_src_tree"] == "same-src-tree"


@pytest.mark.parametrize("corruption", ["data", "config", "source", "failure", "partial"])
def test_completed_e1_import_fails_closed(tmp_path, monkeypatch, corruption):
    root, baseline, rows = imported_e1(tmp_path, monkeypatch)
    if corruption == "data":
        path = root / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["data"]["selection"] = "different"
        path.write_text(json.dumps(manifest))
    elif corruption == "config":
        (root / "resolved_config.json").write_text("{}")
    elif corruption == "source":
        trees = iter(["old-tree", "new-tree"])
        monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: next(trees))
    elif corruption == "failure":
        (root / "FAILURE.json").write_text("{}")
    else:
        rows[0]["status"] = "missing_test"
    with pytest.raises(ValueError):
        queue.validate_completed_e1(root, baseline)


def test_repaired_queue_plans_only_remaining_e2_e3(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    (baseline / "fit").mkdir(parents=True)
    (baseline / "COMPLETE.json").write_text(json.dumps({"epoch": 6, "zero_pkl": True}))
    (baseline / "fit/epoch_state.json").write_text(json.dumps({"epoch": 6}))
    for role in ("best", "last"):
        (baseline / "fit" / f"{role}-test.json").write_text(
            json.dumps({"identity": {"role": role}, "result": {"split": "test"}})
        )
    monkeypatch.setattr(queue, "validate_completed_e1", lambda *args: {"verified": True})
    configs = []

    def resolver(args):
        configs.append(args.config)
        return {"run_id": "new-distinct-" + args.config.parents[1].name}

    monkeypatch.setattr(queue, "resolve_plan", resolver)
    planned = queue.prepare(Path("runtime"), baseline, Path("completed-e1"))
    assert [row["name"] for row in planned["rows"]] == [
        "E2_prototypes16384",
        "E3_unit_distillation_no_spread_koleo",
    ]
    assert len(configs) == 2
    assert planned["completed_e1"] == {"verified": True}


def test_combined_twenty_plans_one_capacity_row(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    (baseline / "fit").mkdir(parents=True)
    (baseline / "COMPLETE.json").write_text(json.dumps({"epoch": 6, "zero_pkl": True}))
    (baseline / "fit/epoch_state.json").write_text(json.dumps({"epoch": 6}))
    for role in ("best", "last"):
        (baseline / "fit" / f"{role}-test.json").write_text(
            json.dumps({"identity": {"role": role}, "result": {"split": "test"}})
        )
    configs = []

    def resolver(args):
        configs.append(args.config)
        cfg = queue.load_experiment_config(args.config)
        assert cfg.continuation is None
        assert cfg.training.formal_run_policy == "v2_fixed_20"
        assert cfg.training.max_epochs.value == 20
        return {"run_id": "fresh-combined-twenty"}

    monkeypatch.setattr(queue, "resolve_plan", resolver)
    planned = queue.prepare(Path("runtime"), baseline, combined_twenty=True)
    assert planned["schema"] == "cap40-combined-twenty-queue-v1"
    assert len(configs) == len(planned["rows"]) == 1
    assert planned["rows"][0]["probe_kind"] == "preflight_only"
    assert planned["rows"][0]["name"] == "E23_prototypes16384_unit_distillation"
    combined = queue.load_experiment_config(configs[0]).model_dump(mode="json")
    parent = queue.load_experiment_config(
        ROOT
        / "configs/v2/cap40_ablations_jurkat"
        / "E3_unit_distillation_no_spread_koleo/gradpert_v2/nadig_jurkat.yaml"
    ).model_dump(mode="json")
    assert combined["model"]["parameters"]["prototypes"]["value"] == 16384
    combined["model"]["parameters"]["prototypes"]["value"] = 8192
    combined["training"]["formal_run_policy"] = parent["training"]["formal_run_policy"]
    combined["training"]["max_epochs"] = parent["training"]["max_epochs"]
    assert combined == parent


def test_combined_twenty_cannot_import_e1():
    with pytest.raises(ValueError, match="does not import"):
        queue.prepare(Path("runtime"), Path("baseline"), Path("e1"), combined_twenty=True)
