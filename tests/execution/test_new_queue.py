import json

import pytest

from scripts.server.run_new_ablation_queue import gpu_idle, step_valid
from scripts.server.run_r50_selection import sha


def test_step_gate_requires_exact_identity_and_capacity(tmp_path):
    identity = tmp_path / "source_identity.json"
    identity.write_text(json.dumps({"commit": "abc", "dirty": False}))
    receipt = tmp_path / "one_step_smoke.json"
    payload = dict(
        status="complete",
        config_sha256="cfg",
        source_identity_sha256=sha(identity),
        test_evaluations=0,
        steps=[
            dict(
                completed_steps=1,
                checkpoint_roundtrip_exact=True,
                resources={"cuda_acceptance": True},
            )
        ],
    )
    receipt.write_text(json.dumps(payload))
    assert step_valid(receipt, "cfg", "abc")
    assert not step_valid(receipt, "wrong", "abc")
    assert not step_valid(receipt, "cfg", "other")
    payload["steps"][0]["resources"]["cuda_acceptance"] = False
    receipt.write_text(json.dumps(payload))
    assert not step_valid(receipt, "cfg", "abc")


def test_occupied_gpu_is_never_used(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: "GPU-a, 123\n")
    with pytest.raises(RuntimeError, match="occupied"):
        gpu_idle("GPU-a")
    gpu_idle("GPU-b")
