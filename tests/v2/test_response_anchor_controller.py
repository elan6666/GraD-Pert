"""CPU-only synthetic controller metadata and owned-resource lifecycle contracts."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from scripts.v2 import run_response_anchor as controller  # noqa: E402
from scripts.v2.shared_gpu import validate_probe_binding  # noqa: E402


def calibration_fixture():
    root = Path(controller.__file__).resolve().parents[2]
    decision = json.loads(
        (root / "configs/v2/response_anchor_20261028/CALIBRATION_DECISION.json").read_text()
    )
    probe = root / "configs/v2/response_anchor_20261028/calibration/gradpert_v2/nadig_jurkat.yaml"
    fit = root / decision["fit_config"]
    raw = controller.yaml.safe_load(fit.read_text())["model"]["parameters"]
    calibration = dict(
        calibration_complete=True,
        state_restored=True,
        forward_precision="bfloat16_autocast",
        world_size=2,
        optimizer_updates=0,
        teacher_updates=0,
        center_updates=0,
        batches=[{} for _ in range(8)],
        source=dict(commit=decision["calibration_source"], dirty=False),
        config_sha256=controller.sha256_file(probe),
        resolved_config_sha256=controller.sha256_json(
            controller.load_experiment_config(probe).model_dump(mode="json")
        ),
        identity=dict(run_seed=1),
        bank_manifest_sha256=raw["response_anchor_sha256"]["value"],
        bank_arrays_sha256=raw["response_anchor_arrays_sha256"]["value"],
        selected_weight=raw["response_anchor_weight"]["value"],
    )
    return fit, calibration


def test_actual_calibration_validator_accepts_matching_synthetic_metadata():
    path, c = calibration_fixture()
    controller.validate_calibration(path, c)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source.commit", "wrong"),
        ("source.dirty", True),
        ("config_sha256", "wrong"),
        ("resolved_config_sha256", "wrong"),
        ("identity.run_seed", 2),
        ("forward_precision", "float32"),
        ("world_size", 1),
        ("optimizer_updates", 1),
        ("teacher_updates", 1),
        ("center_updates", 1),
        ("state_restored", False),
        ("calibration_complete", False),
        ("batches", [{}] * 7),
        ("bank_manifest_sha256", "wrong"),
        ("bank_arrays_sha256", "wrong"),
        ("selected_weight", 0.999),
    ],
)
def test_actual_calibration_validator_rejects_mismatch(key, value):
    path, c = calibration_fixture()
    c = copy.deepcopy(c)
    parts = key.split(".")
    if len(parts) == 2:
        c[parts[0]][parts[1]] = value
    else:
        c[key] = value
    with pytest.raises(ValueError):
        controller.validate_calibration(path, c)


@pytest.mark.parametrize("stage", ["preflight", "fit"])
def test_owned_cleanup_precedes_unlock_continues_errors_and_seals(tmp_path, monkeypatch, stage):
    root = tmp_path / "source"
    script = root / "scripts/v2/run_response_anchor.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(controller, "__file__", str(script))
    config = root / "config.yaml"
    config.write_text("synthetic")
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(dict(max_own_memory_fraction=0.85)))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(resource_policy_sha256="policyhash")))
    decision = root / "configs/v2/response_anchor_20261028/CALIBRATION_DECISION.json"
    decision.parent.mkdir(parents=True)
    decision.write_text(json.dumps(dict(calibration_complete_sha256="calhash")))
    spec = dict(
        schema="response-anchor-execution-v1",
        stage=stage,
        source_commit="source",
        gpu="0,1",
        data_root="/data/yilangliu/data",
        output=str(tmp_path / "output"),
    )
    # Test server-path safety separately from lifecycle: temporary synthetic sandbox only.
    output_path = Path(spec["output"])
    original_relative = Path.is_relative_to
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, other: (
            True if str(other) == "/data/yilangliu" else original_relative(self, other)
        ),
    )
    for key, path, digest in (
        ("config", config, controller.sha256_file(config)),
        ("publication", publication, "pubhash"),
        ("policy", policy, "policyhash"),
        ("calibration", calibration, "calhash"),
    ):
        spec[key] = dict(path=str(path), sha256=digest)
    plan_path = tmp_path / "plan.json"
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(dict(response_anchor_gate=dict(passed=True))))
    plan = dict(
        source_commit="source",
        config_sha256=spec["config"]["sha256"],
        postfit_policy="deferred",
        gpu="0,1",
        config=str(config),
        data_root=spec["data_root"],
        publication=str(publication),
        publication_sha256="pubhash",
        cuda_memory_fraction=0.85,
    )
    plan_path.write_text(json.dumps(plan))
    spec.update(
        plan=dict(path=str(plan_path), sha256="planhash"),
        preflight=dict(path=str(preflight), sha256="prehash"),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    events = []
    checked = []
    monkeypatch.setattr(
        controller, "checked", lambda seal: checked.append(seal["path"]) or Path(seal["path"])
    )
    monkeypatch.setattr(
        controller,
        "load_experiment_config",
        lambda path: SimpleNamespace(source_code=SimpleNamespace(repository="repo")),
    )
    monkeypatch.setattr(
        controller, "inspect_source_identity", lambda *a, **k: SimpleNamespace(commit="source")
    )
    monkeypatch.setattr(controller, "validate_calibration", lambda *a: None)
    monkeypatch.setattr(controller, "validate_preflight", lambda *a: None)

    @contextlib.contextmanager
    def lock(path):
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    monkeypatch.setattr(controller, "lock", lock)
    monkeypatch.setattr(controller, "snapshot", lambda pid: {})
    admissions = []

    def admit(*args):
        admissions.append(1)
        if len(admissions) > 1:
            raise RuntimeError("synthetic resource rejection")

    monkeypatch.setattr(controller, "admit", admit)
    monkeypatch.setattr(controller, "parent_map", lambda: {101: 1, 102: 101})
    monkeypatch.setattr(
        controller, "process_info", lambda pid: dict(pid=pid, start_ticks="x", command_sha256="y")
    )

    class Child:
        pid = 101
        stopped = False

        def poll(self):
            return 1 if self.stopped else None

        def terminate(self):
            events.append("child_terminate")
            self.stopped = True

        def wait(self, **kwargs):
            return 1

    child = Child()
    captured = []
    monkeypatch.setattr(
        controller.subprocess, "Popen", lambda command, **kwargs: captured.append(command) or child
    )

    def terminate(pid, identity):
        events.append(f"cleanup{pid}")
        if pid == 102:
            raise RuntimeError("synthetic recycled PID")

    monkeypatch.setattr(controller, "terminate_tree", terminate)
    atomic = controller.atomic_json

    def write(path, payload):
        if path.name == "FAILURE.json":
            events.append("failure")
        atomic(path, payload)

    monkeypatch.setattr(controller, "atomic_json", write)
    with pytest.raises(RuntimeError, match="resource rejection"):
        controller.execute(spec_path, "spechash")
    assert (
        events.index("cleanup102")
        < events.index("cleanup101")
        < events.index("failure")
        < events.index("unlock")
    )
    failure = json.loads((output_path / "FAILURE.json").read_text())
    assert failure["cleanup_errors"] == ["synthetic recycled PID"]
    assert failure["scientific_complete"] is False
    if stage == "fit":
        assert checked.count(str(plan_path)) >= 3 and checked.count(str(preflight)) >= 3
    else:
        command = captured[0]
        assert "--resource-owner-pid" in command and "--shared-queue" in command
        queue_path = Path(command[command.index("--shared-queue") + 1])
        queue = json.loads(queue_path.read_text())
        args = SimpleNamespace(
            config=config,
            data_root=Path(spec["data_root"]),
            publication=publication,
            publication_sha256="pubhash",
            gpu="0,1",
            steps=10,
            output=Path(command[command.index("--output") + 1]),
            shared_queue=queue_path,
        )
        validate_probe_binding(queue, args, "source", "preflight_only", 2)
        queue["rows"][0]["plan"]["config_sha256"] = "wrong"
        with pytest.raises(ValueError):
            validate_probe_binding(queue, args, "source", "preflight_only", 2)


@pytest.mark.parametrize("wrong", ["complete_sha", "policy_sha"])
def test_execute_rejects_frozen_calibration_or_policy_mismatch_before_launch(
    tmp_path, monkeypatch, wrong
):
    fit, calibration = calibration_fixture()
    root = Path(controller.__file__).resolve().parents[2]
    decision = json.loads(
        (root / "configs/v2/response_anchor_20261028/CALIBRATION_DECISION.json").read_text()
    )
    calibration["resource_policy_sha256"] = "acceptedpolicy"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    spec = dict(
        schema="response-anchor-execution-v1",
        stage="preflight",
        source_commit="current",
        gpu="0,1",
        config=dict(path=str(fit), sha256="config"),
        publication=dict(path=str(publication), sha256="pub"),
        calibration=dict(
            path=str(path),
            sha256="wrong" if wrong == "complete_sha" else decision["calibration_complete_sha256"],
        ),
        policy=dict(
            path=str(policy), sha256="wrong" if wrong == "policy_sha" else "acceptedpolicy"
        ),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(controller, "checked", lambda binding: Path(binding["path"]))
    monkeypatch.setattr(
        controller, "inspect_source_identity", lambda *a, **k: SimpleNamespace(commit="current")
    )

    def forbidden(*a, **k):
        raise AssertionError("must reject before GPU/process action")

    monkeypatch.setattr(controller.subprocess, "Popen", forbidden)
    monkeypatch.setattr(controller, "snapshot", forbidden)
    with pytest.raises(
        ValueError, match="calibration endpoint" if wrong == "complete_sha" else "policy differs"
    ):
        controller.execute(spec_path, "spec")
