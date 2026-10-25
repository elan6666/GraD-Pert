"""A failed report must never cause a repeated R13 fit or overwrite its evidence."""

import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_relative_amplitude_analysis_repair as repair  # noqa: E402
from cap40_posthoc import read, write  # noqa: E402


def queue(tmp_path):
    return dict(
        family=repair.SCHEMA,
        source_commit=repair.TRAINING_SHA,
        source=str(tmp_path / "training-source"),
        output=str(tmp_path / "old-queue"),
        rows=[dict(name=name, plan=dict(run_root=str(tmp_path / name))) for name in repair.ARMS],
    )


def test_rejects_unrelated_source_config_or_training_diff(tmp_path, monkeypatch):
    q = queue(tmp_path)
    changes = dict(repair.ALLOWED_CHANGES)

    def git(source, *args):
        if args[0] == "status":
            return ""
        if args[0] == "rev-parse":
            return "a" * 40
        return "\n".join(f"{status}\t{path}" for path, status in changes.items())

    monkeypatch.setattr(repair, "git", git)
    assert repair.validate_repair_source(q, tmp_path)["commit"] == "a" * 40
    changes["src/gradpert/training/v2/objective.py"] = "M"
    with pytest.raises(ValueError, match="undeclared"):
        repair.validate_repair_source(q, tmp_path)
    changes.pop("src/gradpert/training/v2/objective.py")
    q["rows"].reverse()
    with pytest.raises(ValueError, match="original two-arm"):
        repair.validate_repair_source(q, tmp_path)


def test_original_failure_and_queued_nc_are_frozen(tmp_path, monkeypatch):
    q = queue(tmp_path)
    p = Path(q["output"]) / "queue.json"
    seals = {
        p: repair.QUEUE_SHA,
        p.parent / "FAILURE.json": repair.FAILURE_SHA,
        p.parent / "state.json": repair.FAILURE_SHA,
    }
    monkeypatch.setattr(repair, "sha", lambda path: seals[Path(path)])
    repair.validate_original(p, q)
    seals[p.parent / "FAILURE.json"] = "changed"
    with pytest.raises(ValueError, match="immutable"):
        repair.validate_original(p, q)
    seals[p.parent / "FAILURE.json"] = repair.FAILURE_SHA
    Path(q["rows"][1]["plan"]["run_root"]).mkdir()
    with pytest.raises(ValueError, match="unstarted"):
        repair.validate_original(p, q)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    import run_fgpo_medium as medium
    import run_group
    import run_representation_postfit as postfit

    import gradpert

    q = queue(tmp_path)
    p = Path(q["output"]) / "queue.json"
    p.parent.mkdir()
    write(p, q)
    src = Path(gradpert.__file__).resolve().parents[1]
    q["source"] = str(src.parent)
    write(p, q)
    expected = {"commit": "a" * 40, "dirty": False}
    identity = SimpleNamespace(commit="a" * 40, dirty=False, payload=lambda: expected)
    calls = []
    monkeypatch.setattr(repair, "validate_original", lambda *a: None)
    monkeypatch.setattr(repair, "validate_output", lambda q, out: Path(out))
    monkeypatch.setattr(repair, "validate_repair_source", lambda *a: {**expected, "diff": {}})
    monkeypatch.setattr(repair, "validate_repair_publication", lambda *a: identity)
    monkeypatch.setattr(repair, "preserved_files", lambda *a: {"frozen": "seal"})
    monkeypatch.setattr(repair, "validate_controller", lambda *a: {"admitted": True})
    monkeypatch.setattr(run_group, "lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(postfit, "context", lambda *a: {"last": {"sha256": repair.CHECKPOINT_SHA}})
    monkeypatch.setattr(postfit, "validate_validation", lambda *a: {"retained": True})
    monkeypatch.setattr(medium, "validate_medium_complete", lambda *a, **kw: {"medium": True})
    monkeypatch.setattr(postfit, "validate_panel", lambda *a, **kw: {"panel": True})

    def medium_execute(path, arm, **kw):
        calls.append(("medium", arm, kw))
        out = kw["repair_output"]
        out.mkdir()
        write(out / "MEDIUM_COMPLETE.json", {"report_controller_commit": "a" * 40})
        (out / "analysis").mkdir()
        write(out / "analysis/summary.json", {"analysis": expected})

    def panel_run(command, **kw):
        calls.append(("panel", command, kw))
        out = Path(command[-1])
        out.mkdir()
        write(out / "COMPLETE.json", {"analysis_source": expected})

    monkeypatch.setattr(medium, "execute", medium_execute)
    monkeypatch.setattr(repair, "repair_panel_command", lambda q, r, s, o: ["panel", str(o)])
    monkeypatch.setattr(repair.subprocess, "run", panel_run)
    return q, p, tmp_path / "repair", calls


def run_fixture(execution):
    _q, p, out, _calls = execution
    return repair.execute(
        p,
        out,
        analysis_sha="a" * 40,
        publication=p.parent / "publication.json",
        publication_sha256="pub",
    )


def test_reuses_fit_and_validation_and_writes_only_fresh_diagnostics(execution):
    q, p, _out, calls = execution
    before = p.read_bytes()
    completed = read(run_fixture(execution))
    assert [c[0] for c in calls] == ["medium", "panel"]
    assert calls[0][1] == "R13-SRAMP"
    assert calls[1][2]["env"]["PYTHONPATH"] == str(Path(q["source"]) / "src")
    assert completed["updates"] == 0
    assert not completed["retrained"] and not completed["validation_rerun"]
    assert not completed["scientific_complete"]
    assert completed["original_artifacts_unchanged"] and p.read_bytes() == before
    with pytest.raises(ValueError, match="fresh"):
        run_fixture(execution)


def test_model_or_original_artifact_drift_blocks_completion(execution, monkeypatch):
    _, _, out, _ = execution
    reads = iter([{"original": True}, {"original": False}])
    monkeypatch.setattr(repair, "preserved_files", lambda *a: next(reads))
    with pytest.raises(ValueError, match="provenance"):
        run_fixture(execution)
    assert (out / "FAILURE.json").is_file()
    assert not (out / "ANALYSIS_REPAIR_COMPLETE.json").exists()


def test_checkpoint_drift_stops_before_analysis(execution, monkeypatch):
    import run_representation_postfit as postfit

    _, _, out, calls = execution
    monkeypatch.setattr(postfit, "context", lambda *a: {"last": {"sha256": "changed"}})
    with pytest.raises(ValueError, match="checkpoint"):
        run_fixture(execution)
    assert not calls and not out.exists()


def test_publication_drift_during_analysis_blocks_completion(execution, monkeypatch):
    _, _, out, _ = execution
    calls = []

    def publication(*args):
        calls.append(True)
        value = {"commit": "a" * 40, "dirty": False, "receipt": len(calls)}
        return SimpleNamespace(commit="a" * 40, dirty=False, payload=lambda: value)

    monkeypatch.setattr(repair, "validate_repair_publication", publication)
    with pytest.raises(ValueError, match="provenance"):
        run_fixture(execution)
    assert len(calls) == 2
    assert (out / "FAILURE.json").exists()
    assert not (out / "ANALYSIS_REPAIR_COMPLETE.json").exists()


def test_queue_scope_not_mutated_by_source_validation(tmp_path, monkeypatch):
    q = queue(tmp_path)
    before = copy.deepcopy(q)
    monkeypatch.setattr(repair, "git", lambda *a: "dirty")
    with pytest.raises(ValueError, match="clean"):
        repair.validate_repair_source(q, tmp_path)
    assert q == before


@pytest.mark.parametrize("bad", [None, "tolerance", "state", "population", "endpoint"])
def test_native_panel_override_retains_population_state_and_parity(tmp_path, monkeypatch, bad):
    import run_readout_analysis_repair as base
    import run_representation_postfit as postfit
    from representation_panel import panel_schedule
    from test_representation_panel import parent

    q = queue(tmp_path)
    row = q["rows"][0]
    row["plan"].update(run_id="original-run", config_sha256="config-seal")
    frozen = parent()
    example = frozen["schedule"][0]
    frozen["schedule"] = []
    for index in range(32):
        group = copy.deepcopy(example)
        group["condition_id"] = f"p{index}"
        for control, record in enumerate(group["rows"]):
            record.update(condition_id=f"p{index}", truth_row_id=f"t{index}-{control}")
        frozen["schedule"].append(group)
    p = tmp_path / "parent.json"
    write(p, frozen)
    q["medium_analysis"] = dict(parent_manifest=str(p), parent_manifest_sha256=repair.sha(p))
    monkeypatch.setattr(base, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(postfit, "context", lambda *a: {"last": {"sha256": "checkpoint"}})
    panel = panel_schedule(frozen)
    root = tmp_path / "new-panel"
    write(root / "panel.json", panel)
    record = dict(
        status="completed",
        updates=0,
        method_sha=repair.TRAINING_SHA,
        run_id="original-run",
        checkpoint_sha256="checkpoint",
        config_sha256="config-seal",
        state_unchanged=True,
        state_before={"model": "same"},
        state_after={"model": "same"},
        parent_manifest=str(p),
        parent_manifest_sha256=repair.sha(p),
        panel_sha256=panel["panel_sha256"],
        panel_rows=128,
        checkpoint_endpoint=dict(
            role="last", epoch=2, training_stage="joint", full_training_epochs=2
        ),
        parity=dict(
            used_path_passed=True,
            tolerance=2e-6,
            cell_chunk_used=False,
            graph_max_abs=2e-6,
            response_max_abs=2e-6,
        ),
        outputs=postfit.output_seal(root),
    )
    if bad == "tolerance":
        record["parity"]["tolerance"] = 3e-6
    elif bad == "state":
        record["state_after"] = {"model": "changed"}
    elif bad == "population":
        record["panel_rows"] = 127
    elif bad == "endpoint":
        record["checkpoint_endpoint"]["training_stage"] = "frozen_readout"
    write(root / "COMPLETE.json", record)
    if bad:
        with pytest.raises(ValueError):
            postfit.validate_panel(q, row, root=root)
    else:
        assert postfit.validate_panel(q, row, root=root)["panel_sha256"] == panel["panel_sha256"]
    with pytest.raises(ValueError, match="scoped"):
        postfit.validate_panel(q, q["rows"][1], root=root)
