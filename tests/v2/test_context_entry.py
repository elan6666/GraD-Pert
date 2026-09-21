import runpy
from pathlib import Path

import pytest

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file

API = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/v2/evaluate_context.py"))


@pytest.mark.parametrize("mutation", [None, "unfinished", "hash", "identity", "escape"])
def test_context_entry_requires_committed_checkpoint_identity(tmp_path, mutation):
    fit = tmp_path / "fit"
    fit.mkdir()
    checkpoint = fit / "epoch-0050.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    identity = {"source": "training"}
    selected = {"file": checkpoint.name, "sha256": sha256_file(checkpoint), "epoch": 50}
    journal = {"identity": identity, "epoch": 50, "budget": [50, 10], "last": selected}
    atomic_json(tmp_path / "run_manifest.json", identity)
    if mutation == "unfinished":
        journal["epoch"] = 49
    elif mutation == "hash":
        selected["sha256"] = "changed"
    elif mutation == "identity":
        journal["identity"] = {"source": "evaluation"}
    elif mutation == "escape":
        outside = tmp_path / "outside.pt"
        outside.write_bytes(checkpoint.read_bytes())
        selected["file"] = "../outside.pt"
    atomic_json(fit / "epoch_state.json", journal)
    if mutation:
        with pytest.raises(ValueError):
            API["selected_checkpoint"](tmp_path, "last")
    else:
        actual, role, path = API["selected_checkpoint"](tmp_path, "last")
        assert actual == identity and role == selected and path == checkpoint


@pytest.mark.parametrize("passed", [True, False])
def test_engineering_checkpoint_is_explicit_and_requires_terminal_evidence(tmp_path, passed):
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"synthetic engineering checkpoint")
    receipt = tmp_path / "receipt.json"
    atomic_json(
        receipt,
        {
            "status": "passed" if passed else "running",
            "kind": "integration_only",
            "source": {"dirty": False, "commit": "a", "published_commit": "a"},
            "resume_checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    if passed:
        training, selected, path = API["engineering_checkpoint"](tmp_path, sha256_file(receipt))
        assert training["kind"] == "integration_only"
        assert "epoch" not in selected
        assert path == checkpoint
    else:
        with pytest.raises(ValueError, match="passed probe"):
            API["engineering_checkpoint"](tmp_path, sha256_file(receipt))
