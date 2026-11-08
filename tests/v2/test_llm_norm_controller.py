"""Actual controller preflight rejects changed/freshness-mismatched launch inputs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def controller_fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "llm_norm_controller", ROOT / "scripts/v2/run_llm_norm_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "F", tmp_path)
    monkeypatch.setattr(
        module,
        "inspect_source_identity",
        lambda *a, **k: SimpleNamespace(dirty=False, commit="commit"),
    )

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    runtime = write("RUNTIME.json", {})
    policy = write("POLICY.json", {"shared_gpu_policy": {"unchanged": True}})
    protocol = dict(
        source_commit="commit",
        publication_sha256="publication",
        runtime_sha256=sha256_file(runtime),
        policy_path=str(policy),
        policy_sha256=sha256_file(policy),
        order=["A_norm_mla", "B_norm_standard"],
        epochs_per_arm=2,
        updates_per_arm=300,
        plans={},
    )
    for arm, terminal in zip(protocol["order"], ["latent", "standard"], strict=True):
        config = write(arm + ".json", {})
        params = {
            k: {"value": v}
            for k, v in dict(
                normalization_layout="llm_pre",
                terminal_attention=terminal,
                attention="relay_full",
                graph_input_norm="historical",
                relay_validate_once=True,
                cache_kda_constants=True,
            ).items()
        }
        plan = dict(
            source_commit="commit",
            gpu="0",
            postfit_policy="deferred",
            cuda_memory_fraction=0.9,
            config=str(config),
            config_sha256=sha256_file(config),
            run_root=str(tmp_path / (arm + "-run")),
            resolved_config=dict(
                training=dict(max_epochs={"value": 2}), model=dict(parameters=params)
            ),
        )
        path = write(arm + "-plan.json", plan)
        protocol["plans"][arm] = dict(
            path=str(path), sha256=sha256_file(path), config_sha256=sha256_file(config)
        )
    write("PROTOCOL.json", protocol)
    return module, sha256_file(tmp_path / "PROTOCOL.json")


def test_controller_verifies_actual_order_and_fresh_plan_bindings(tmp_path, monkeypatch):
    m, sha = controller_fixture(tmp_path, monkeypatch)
    protocol, plans, policy = m.verify(sha)
    assert list(plans) == protocol["order"] == ["A_norm_mla", "B_norm_standard"]
    assert policy == {"unchanged": True}


@pytest.mark.parametrize(
    "change",
    ["protocol", "runtime", "policy", "config", "plan", "existing_run", "existing_integration"],
)
def test_controller_refuses_changed_or_reused_inputs(tmp_path, monkeypatch, change):
    m, sha = controller_fixture(tmp_path, monkeypatch)
    names = dict(
        protocol="PROTOCOL.json",
        runtime="RUNTIME.json",
        policy="POLICY.json",
        config="A_norm_mla.json",
        plan="A_norm_mla-plan.json",
    )
    if change in names:
        with (tmp_path / names[change]).open("a") as stream:
            stream.write(" ")
    elif change == "existing_run":
        (tmp_path / "A_norm_mla-run").mkdir()
    else:
        (tmp_path / "integration-A_norm_mla-a1").mkdir()
    with pytest.raises(AssertionError):
        m.verify(sha)
