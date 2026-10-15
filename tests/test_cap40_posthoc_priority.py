"""Scheduling invariants, independent of native CUDA/model dependencies."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cap40_priority", Path(__file__).parents[1] / "scripts/v2/cap40_posthoc_priority.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def completed(tmp_path):
    folder = tmp_path / "DP"
    folder.mkdir()
    row = {"formal": str(folder), "arm": "DP", "checkpoint_sha256": "checkpoint"}
    q = {
        "manifest_sha256": "manifest",
        "analysis_commit": "frozen-worker",
        "expected_rows": 40,
        "expected_conditions": 2,
        "expected_deep_rows": 8,
    }
    summary = dict(
        arm="DP",
        rows=40,
        conditions=2,
        deep_rows=8,
        updates=0,
        analysis={"commit": "frozen-worker"},
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    m.write(folder / "summary.json", summary)
    receipt = dict(
        status="completed",
        scope="full_cap40_train",
        state_unchanged=True,
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
        outputs={"summary.json": {"sha256": m.sha(folder / "summary.json")}},
    )
    m.write(folder / "COMPLETE.json", receipt)
    return row, q


def test_adoption_requires_real_full_receipt_and_hashes(tmp_path):
    row, q = completed(tmp_path)
    assert m.validate_complete(row, q) == m.sha(Path(row["formal"]) / "COMPLETE.json")
    (Path(row["formal"]) / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        m.validate_complete(row, q)


@pytest.mark.parametrize(
    "field,value",
    [("scope", "gate_subset"), ("state_unchanged", False), ("checkpoint_sha256", "other")],
)
def test_gate_or_changed_checkpoint_cannot_be_adopted(tmp_path, field, value):
    row, q = completed(tmp_path)
    p = Path(row["formal"]) / "COMPLETE.json"
    rec = m.read(p)
    rec[field] = value
    m.write(p, rec)
    with pytest.raises(ValueError, match="identity"):
        m.validate_complete(row, q)


def test_pid_reuse_is_not_worker_completion(monkeypatch):
    expected = {"pid": 20, "start_ticks": "100", "command_sha256": "a"}
    monkeypatch.setattr(m, "proc", lambda pid: {**expected, "start_ticks": "101"})
    with pytest.raises(ValueError, match="identity changed"):
        m.same_process(expected)
    monkeypatch.setattr(m, "proc", lambda pid: None)
    assert not m.same_process(expected)


@pytest.mark.parametrize("failed", [False, True])
def test_priority_order_adoption_and_failure_no_retries(tmp_path, monkeypatch, failed):
    q = dict(
        adopted_worker={"pid": 42},
        retired_controller={"pid": 43},
        worker_source="/frozen",
        method_source="/method",
        data_root="/data",
        manifest="/manifest",
        manifest_sha256="manifest",
        resource_queue="/resource",
        original_analysis="/original",
        comparison=str(tmp_path / "comparison"),
        rows=[
            dict(
                arm=a,
                formal=str(tmp_path / a),
                report=str(tmp_path / (a + "-report")),
                run_root="/scientific/" + a,
                config="/config/" + a,
            )
            for a in ("DP", "FN", "FG", "FB", "PO")
        ],
    )
    path = tmp_path / "queue.json"
    m.write(path, q)
    launches, reports = [], []
    monkeypatch.setattr(m, "validate_plan", lambda q: None)
    monkeypatch.setattr(m, "validate_complete", lambda row, q: "validated")
    calls = iter([True, True, False])
    adoption = []

    def alive(expected):
        result = next(calls)
        adoption.append(result)
        return result

    monkeypatch.setattr(m, "same_process", alive)
    monkeypatch.setattr(m, "proc", lambda pid: None)
    monkeypatch.setattr(m.time, "sleep", lambda seconds: None)

    class Child:
        pid = 99

        def __init__(self):
            self.polls = iter([None, 3 if failed else 0])
            self.returncode = 3 if failed else 0

        def poll(self):
            return next(self.polls)

        def wait(self):
            return 0

    def spawn(cmd, **kwargs):
        assert adoption == [True, True, False]  # Never overlap independent GPU workers.
        launches.append((cmd[cmd.index("--arm") + 1], kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        assert kwargs["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
        assert kwargs["env"]["PYTHONPATH"] == "/method/src"
        return Child()

    def run(cmd, **kwargs):
        reports.append(Path(cmd[cmd.index("--output") + 1]).name)

    monkeypatch.setattr(m.subprocess, "Popen", spawn)
    monkeypatch.setattr(m.subprocess, "run", run)
    args = SimpleNamespace(queue=path, queue_sha256=m.sha(path))
    if failed:
        with pytest.raises(RuntimeError, match="PO worker"):
            m.execute(args)
        assert launches == [("PO", "1")]
        assert reports == ["DP-report"]
        assert m.read(tmp_path / "FAILURE.json")["status"] == "failed"
    else:
        m.execute(args)
        assert launches == [("PO", "1"), ("FN", "0"), ("FG", "0"), ("FB", "0")]
        assert reports == [
            "DP-report",
            "PO-report",
            "FN-report",
            "FG-report",
            "FB-report",
            "comparison",
        ]
        assert m.read(tmp_path / "state.json")["status"] == "completed"
    assert "DP" not in [arm for arm, gpu in launches]


def test_missing_adopted_worker_never_launches_replacement(tmp_path, monkeypatch):
    q = {"adopted_worker": {"pid": 42}}
    path = tmp_path / "queue.json"
    m.write(path, q)
    monkeypatch.setattr(m, "validate_plan", lambda q: None)
    monkeypatch.setattr(m, "same_process", lambda expected: False)
    with pytest.raises(ValueError, match="absent"):
        m.execute(SimpleNamespace(queue=path, queue_sha256=m.sha(path)))
    assert not (tmp_path / "formal-PO.log").exists()


def test_queue_hash_mismatch_prevents_any_process(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="queue hash"):
        m.execute(SimpleNamespace(queue=path, queue_sha256="bad"))
