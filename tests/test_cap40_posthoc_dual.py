"""Identity-scoped analysis peers preserve the original external resource gate."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
import cap40_posthoc_dual as dual
from cap40_analysis_admission import admit_analysis
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
