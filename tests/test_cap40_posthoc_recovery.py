"""Recovery keeps exact stream sufficient statistics, identities and RNG bytes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc_recovery import load_recovery, save_recovery
from cap40_posthoc_stats import Distribution, Moments


def fixture():
    rng = np.random.default_rng(20)
    cells = rng.normal(size=(12, 8))
    logits = rng.normal(size=(12, 257))
    moments, dist = Moments(8), Distribution(257)
    moments.add(cells[:5])
    dist.add(logits[:5])
    return (
        cells,
        logits,
        {
            "next_index": 5,
            "features": {"student.main": moments},
            "heads": {"teacher.node": dist},
            "counts": {"c::p": 5},
            "rows": [{"truth_id": "c1", "control_id": "c0", "stat": 0.4}],
            "rng": rng.bytes(48),
        },
    )


def test_roundtrip_then_stream_continuation_is_bit_exact(tmp_path):
    cells, logits, state = fixture()
    state["rng"] = np.frombuffer(state["rng"], dtype=np.uint8).copy()
    binding = {"model": "sha", "manifest": "ids", "deterministic": True}
    path = save_recovery(tmp_path, binding=binding, state=state)
    restored, meta = load_recovery(path, binding=binding, schedule_size=12)
    assert meta["contains_model_or_cell_matrices"] is False
    assert restored["rows"] == state["rows"] and restored["counts"] == state["counts"]
    np.testing.assert_array_equal(restored["rng"], state["rng"])
    for value in (state, restored):
        value["features"]["student.main"].add(cells[5:])
        value["heads"]["teacher.node"].add(logits[5:])
    for key, names in (("features", ("mean", "m2")), ("heads", ("sum", "hits"))):
        a, b = next(iter(state[key].values())), next(iter(restored[key].values()))
        for name in names:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert a.result() == b.result()
    with pytest.raises(FileExistsError):
        save_recovery(tmp_path, binding=binding, state=state)


@pytest.mark.parametrize("change", ["binding", "bytes", "coverage", "path"])
def test_recovery_fails_closed_on_mutation_or_identity_change(tmp_path, change):
    _, _, state = fixture()
    state["rng"] = np.zeros(4, dtype=np.uint8)
    binding = {"source": "immutable", "queries": [1, 4, 7]}
    path = save_recovery(tmp_path, binding=binding, state=state)
    meta = json.loads(path.read_text())
    if change == "binding":
        binding = {**binding, "queries": [1, 4, 8]}
    elif change == "bytes":
        with Path(meta["file"]).open("ab") as f:
            f.write(b"tampered")
    elif change == "coverage":
        meta["next_index"] = 13
    elif change == "path":
        other = tmp_path / "other"
        other.mkdir()
        target = other / "copied.npz"
        target.write_bytes(Path(meta["file"]).read_bytes())
        meta["file"] = str(target)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="mismatch"):
        load_recovery(path, binding=binding, schedule_size=12)


def test_object_arrays_and_unknown_live_objects_cannot_be_serialized(tmp_path):
    for item in (np.array([object()]), object()):
        with pytest.raises((TypeError, ValueError)):
            save_recovery(tmp_path, binding={}, state={"next_index": 1, "unsafe": item})
