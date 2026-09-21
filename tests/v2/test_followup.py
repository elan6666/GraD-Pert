import runpy
from pathlib import Path

import pytest

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "configs/v2/capacity/m32/gradpert_v2/nadig_jurkat.yaml"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    return runpy.run_path(str(ROOT / "scripts/v2/prepare_followup.py"))


def test_later_group_without_selection_is_not_launchable(api, tmp_path):
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"group": "H2"})
    with pytest.raises(ValueError, match="require"):
        api["validate_dependency"](manifest, PARENT)
    atomic_json(manifest, {"group": "H1"})
    assert api["validate_dependency"](manifest, PARENT) is None


@pytest.mark.parametrize("tamper", [None, "selection", "manifest", "parent"])
def test_followup_materializes_selected_parent_and_rechecks_dependencies(
    api, monkeypatch, tmp_path, tamper
):
    upstream, selection = tmp_path / "upstream.json", tmp_path / "selection.json"
    atomic_json(upstream, {"group": "H1"})
    atomic_json(selection, {"winner": "synthetic; selection recomputation tested separately"})
    result = {
        "selection_sha256": sha256_file(selection),
        "manifest_sha256": sha256_file(upstream),
        "winner": {"config": str(PARENT), "sha256": sha256_file(PARENT)},
    }
    monkeypatch.setitem(api["prepare"].__globals__, "verify_selection", lambda *_: result)
    output = tmp_path / "H2"
    generated = api["prepare"](selection, upstream, PARENT, "H2", output)
    assert generated["parent_sha256"] == sha256_file(PARENT)
    assert len(generated["rows"]) == 2
    parent = PARENT
    if tamper == "selection":
        selection.write_text("changed")
    elif tamper == "manifest":
        upstream.write_text("changed")
    elif tamper == "parent":
        parent = tmp_path / "wrong-parent.yaml"
        parent.write_text("changed")
    if tamper:
        with pytest.raises(ValueError):
            api["validate_dependency"](output / "manifest.json", parent)
    else:
        assert api["validate_dependency"](output / "manifest.json", parent) == result
