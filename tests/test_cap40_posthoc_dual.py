"""Identity-scoped analysis peers preserve the original external resource gate."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
import cap40_posthoc_dual as dual
from cap40_analysis_admission import admit_analysis, dispatcher_owner
from shared_gpu import TOTALVI_FORMAL_AUTHORIZATION


def fixture():
    policy = dict(
        schema="shared-gpu-admission-3",
        devices=[dict(gpu_index=i, gpu_uuid=f"GPU-{i}") for i in (0, 1)],
        foreign_uid=1003,
        command_basename="run_scbutterfly_archived_predictions.py",
        additional_command_basenames=[
            "run_totalvi_source_validation_development.py",
            "run_totalvi_s2_formal.py",
        ],
        max_foreign_processes_per_gpu=1,
        max_foreign_memory_mib=1024,
        min_free_memory_mib=2048,
        max_own_memory_fraction=0.9,
        authorization_reference=TOTALVI_FORMAL_AUTHORIZATION,
    )
    identity = dict(pid=42, start_ticks="12", command_sha256="abc")
    info = dict(**identity, uid=1002, script_basename="cap40_posthoc.py")
    peer = dict(schema="cap40-exact-analysis-peer-1", arm="FN", gpu=0, uid=1002, process=identity)
    observed = dict(
        gpus={i: dict(uuid=f"GPU-{i}", total_mib=32000, free_mib=20000) for i in (0, 1)},
        processes=[dict(pid=42, identity=info, gpu_uuid="GPU-0", memory_mib=10000, owned=False)],
    )
    return policy, peer, observed, info


def test_only_exact_other_gpu_peer_is_recognized_without_mutating_snapshot():
    policy, peer, view, info = fixture()
    original = copy.deepcopy(view)
    assert admit_analysis(policy, view, 1, peer, lookup=lambda p: info, uid=1002)["processes"][0][
        "owned"
    ]
    assert view == original
    with pytest.raises(RuntimeError, match="foreign"):
        admit_analysis(policy, view, 1, uid=1002)


@pytest.mark.parametrize(
    "field,value", [("start_ticks", "13"), ("uid", 1003), ("script_basename", "other.py")]
)
def test_recycled_or_unrelated_process_is_not_a_peer(field, value):
    policy, peer, view, info = fixture()
    info = {**info, field: value}
    with pytest.raises(ValueError, match="identity changed"):
        admit_analysis(policy, view, 1, peer, lookup=lambda p: info, uid=1002)


@pytest.mark.parametrize("memory,free", [(30000, 20000), (10000, 1000)])
def test_peer_still_obeys_owned_memory_and_physical_headroom(memory, free):
    policy, peer, view, info = fixture()
    view["processes"][0]["memory_mib"] = memory
    view["gpus"][0]["free_mib"] = free
    with pytest.raises(RuntimeError, match=r"allowance|reserve"):
        admit_analysis(policy, view, 1, peer, lookup=lambda p: info, uid=1002)


def test_unknown_gpu_process_remains_rejected_even_with_valid_peer():
    policy, peer, view, info = fixture()
    view["processes"].append(
        dict(pid=99, identity={**info, "pid": 99}, gpu_uuid="GPU-1", memory_mib=800, owned=False)
    )
    with pytest.raises(RuntimeError, match="foreign"):
        admit_analysis(policy, view, 1, peer, lookup=lambda p: info, uid=1002)


def test_peer_cannot_share_same_gpu_or_change_physical_device():
    policy, peer, view, info = fixture()
    with pytest.raises(ValueError, match="other GPU"):
        admit_analysis(policy, view, 0, peer, lookup=lambda p: info, uid=1002)
    view["processes"][0]["gpu_uuid"] = "GPU-1"
    with pytest.raises(ValueError, match="telemetry"):
        admit_analysis(policy, view, 1, peer, lookup=lambda p: info, uid=1002)


def test_medium_adoption_needs_exact_counts_source_and_bytes(tmp_path):
    folder = tmp_path / "FN"
    folder.mkdir()
    row = dict(arm="FN", formal=str(folder), checkpoint_sha256="cp", analysis_commit="old")
    q = dict(manifest_sha256="manifest")
    s = dict(
        arm="FN",
        conditions=1335,
        rows=5317,
        deep_conditions=128,
        deep_rows=506,
        gradient_conditions=32,
        gradient_rows=125,
        updates=0,
        analysis={"commit": "old"},
        manifest_sha256="manifest",
        checkpoint_sha256="cp",
    )
    dual.write(folder / "summary.json", s)
    c = dict(
        status="completed",
        scope="medium_cap40_stratified",
        state_unchanged=True,
        manifest_sha256="manifest",
        checkpoint_sha256="cp",
        outputs={"summary.json": {"sha256": dual.sha(folder / "summary.json")}},
    )
    dual.write(folder / "COMPLETE.json", c)
    assert dual.complete(row, q)
    s["rows"] = 47836
    dual.write(folder / "summary.json", s)
    with pytest.raises(ValueError, match="hash"):
        dual.complete(row, q)
    c["outputs"]["summary.json"]["sha256"] = dual.sha(folder / "summary.json")
    dual.write(folder / "COMPLETE.json", c)
    with pytest.raises(ValueError, match="population"):
        dual.complete(row, q)


def test_adopted_slot_is_not_relaunched_and_only_other_device_is_available():
    assert dual.free_slots({"FN": {"gpu": 0}}) == [1]
    assert dual.free_slots({"FN": {"gpu": 0}, "FG": {"gpu": 1}}) == []
    with pytest.raises(ValueError, match="one analysis"):
        dual.free_slots({"FN": {"gpu": 0}, "FG": {"gpu": 0}})


def test_shared_dispatcher_owner_is_validated_on_every_snapshot():
    expected = dict(pid=12, start_ticks="45", command_sha256="hash")
    receipt = dict(schema="cap40-analysis-owner-1", uid=1002, process=expected)
    actual = dict(**expected, uid=1002, script_basename="cap40_posthoc_dual.py")
    assert dispatcher_owner(receipt, lookup=lambda _: actual, parent=12, uid=1002) == 12
    with pytest.raises(ValueError, match="parent identity"):
        dispatcher_owner(receipt, lookup=lambda _: None, parent=12, uid=1002)
    with pytest.raises(ValueError, match="parent identity"):
        dispatcher_owner(receipt, lookup=lambda _: actual, parent=13, uid=1002)
    with pytest.raises(ValueError, match="parent identity"):
        dispatcher_owner(
            receipt, lookup=lambda _: {**actual, "start_ticks": "46"}, parent=12, uid=1002
        )


def test_periodic_and_initial_checks_use_the_same_owner_path():
    # Regression: the first parallel attempt changed startup only. The next
    # per-condition native guard then rejected the other GPU analysis.
    text = (Path(__file__).parents[1] / "scripts/v2/cap40_posthoc.py").read_text()
    import ast

    run = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    loop = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.For) and ast.unparse(n.target) == "(number, g)"
    )
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "check_resources"
        for n in ast.walk(loop.body[0])
    )
    assert "snapshot(dispatcher_owner(owner)" in text


def test_two_new_slots_share_live_owner_and_finish_each_arm_once(tmp_path, monkeypatch):
    rows = [
        dict(
            arm=a,
            formal=str(tmp_path / "analysis" / a),
            report=str(tmp_path / "reports" / a),
            dual_gate="unused",
            run_root="/run/" + a,
            config="/config/" + a,
            checkpoint_sha256=a,
        )
        for a in ["DP", "FN", "FG", "FB", "PO"]
    ]
    q = dict(
        rows=rows,
        completed=["DP", "FN", "PO"],
        pending=["FG", "FB"],
        adopted={},
        worker_source="/worker",
        method_source="/method",
        data_root="/data",
        manifest="/manifest",
        manifest_sha256="manifest",
        resource_queue="/resource",
        original_analysis="/original",
        comparison=str(tmp_path / "comparison"),
    )
    queue = tmp_path / "queue.json"
    dual.write(queue, q)
    digest = dual.sha(queue)
    dual.write(tmp_path / "MAIN_ACCEPTANCE.json", dict(accepted=True, queue_sha256=digest))
    monkeypatch.setattr(dual, "validate", lambda _: None)
    monkeypatch.setattr(dual, "complete", lambda *_: "complete")
    monkeypatch.setattr(
        dual, "proc", lambda pid: dict(pid=pid, start_ticks="1", command_sha256="h")
    )
    launches = []

    def report_receipt(destination):
        a = Path(destination).name
        Path(destination).mkdir(parents=True, exist_ok=True)
        dual.write(
            Path(destination) / "receipt.json",
            dict(manifest_sha256="manifest", checkpoint_sha256=a, outputs={}),
        )

    for r in rows:
        if r["arm"] in q["completed"]:
            report_receipt(r["report"])

    class Child:
        returncode = 0

        def __init__(self, cmd, **kw):
            self.pid = 100 + len(launches)
            self.calls = 0
            assert "--owner-receipt" in cmd and "--peer-receipt" not in cmd
            owner = dual.read(cmd[cmd.index("--owner-receipt") + 1])
            assert owner["queue_sha256"] == digest
            launches.append((cmd[cmd.index("--arm") + 1], kw["env"]["CUDA_VISIBLE_DEVICES"]))
            Path(cmd[cmd.index("--output") + 1]).mkdir(parents=True)
            dual.write(Path(cmd[cmd.index("--output") + 1]) / "ADMISSION.json", {"passed": True})

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    def run(cmd, **kw):
        if "--input" in cmd:
            report_receipt(cmd[cmd.index("--output") + 1])

    monkeypatch.setattr(dual.subprocess, "Popen", Child)
    monkeypatch.setattr(dual.subprocess, "run", run)
    monkeypatch.setattr(dual.time, "sleep", lambda _: None)
    dual.execute(SimpleNamespace(queue=queue, queue_sha256=digest, gate_only=False))
    assert launches == [("FG", "0"), ("FB", "1")]
    state = dual.read(tmp_path / "state.json")
    assert state["status"] == "completed" and len(state["completed"]) == 5
