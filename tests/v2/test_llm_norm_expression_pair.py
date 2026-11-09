"""CPU contract checks for the published two-arm expression evaluator."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "expression_pair", ROOT / "scripts/v2/run_llm_norm_expression_pair.py"
    )
    m = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(m)
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path)
    source = {"commit": "eval-source", "dirty": False}
    monkeypatch.setattr(
        m, "inspect_source_identity", lambda *a, **k: SimpleNamespace(payload=lambda: source)
    )

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    family = tmp_path / "family"
    root = family / "expression"
    pub = write(root / "PUBLICATION.json", {})
    runtime = write(
        root / "RUNTIME.json",
        {
            "publication_receipt": str(pub),
            "publication_sha256": sha256_file(pub),
            "data_root": str(tmp_path / "data"),
        },
    )
    policy = write(root / "POLICY.json", {"shared_gpu_policy": {"selected_gpu": 0}})
    spec = dict(
        order=list(m.ARMS),
        split="val",
        metric_scope="expression_only",
        no_DEG=True,
        no_test=True,
        no_training=True,
        publication=str(pub),
        publication_sha256=sha256_file(pub),
        runtime=str(runtime),
        runtime_sha256=sha256_file(runtime),
        source_commit="eval-source",
        policy_path=str(policy),
        policy_file_sha256=sha256_file(policy),
        plans={},
    )
    accept = dict(status="both_arms_main_independently_training_accepted", arms={})
    for arm in m.ARMS:
        train = family / "runs" / arm
        checkpoint = write(train / "fit/epoch-0002.pt", {"synthetic": arm})
        config = write(root / (arm + ".config"), {})
        identity = {"run_id": arm, "source": {"commit": "train-source"}}
        write(train / "run_manifest.json", identity)
        accept["arms"][arm] = dict(
            run_id=arm,
            run_root=str(train),
            source_commit="train-source",
            config_sha256=sha256_file(config),
            last_checkpoint={"file": checkpoint.name, "sha256": sha256_file(checkpoint)},
        )
        plan = dict(
            split="val",
            metric_scope="expression_only",
            checkpoint_epoch=2,
            checkpoint_role="last",
            gpu=["0"],
            cuda_memory_fraction=0.9,
            cpu_training_state=True,
            evaluation_source=source,
            runtime_sha256=sha256_file(runtime),
            data_root=str(tmp_path / "data"),
            training_identity=identity,
            training_run_root=str(train),
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            config=str(config),
            config_sha256=sha256_file(config),
            output_root=str(root / "results" / arm),
        )
        plan_path = write(root / (arm + ".plan.json"), plan)
        spec["plans"][arm] = dict(
            path=str(plan_path),
            sha256=sha256_file(plan_path),
            output=plan["output_root"],
            checkpoint_sha256=plan["checkpoint_sha256"],
        )
    receipt = write(family / "MAIN_TRAINING_ACCEPTANCE_A2.json", accept)
    spec["training_acceptance_sha256"] = sha256_file(receipt)
    return m, write(root / "SPEC.json", spec)


def test_accepts_exact_two_arm_scope(tmp_path, monkeypatch):
    m, spec = fixture(tmp_path, monkeypatch)
    sealed, policy = m.verify(spec, sha256_file(spec))
    assert sealed["order"] == list(m.ARMS) and policy == {"selected_gpu": 0}
    assert not (spec.parent / "OWNER.json").exists()


@pytest.mark.parametrize(
    "changed", ["policy", "checkpoint", "source", "test", "deg", "gpu", "role", "output"]
)
def test_rejects_wrong_scope_or_modified_evidence(tmp_path, monkeypatch, changed):
    m, path = fixture(tmp_path, monkeypatch)
    spec = json.loads(path.read_text())
    item = spec["plans"][m.ARMS[0]]
    plan_path = Path(item["path"])
    plan = json.loads(plan_path.read_text())
    if changed in ("policy", "checkpoint"):
        p = Path(spec["policy_path"] if changed == "policy" else plan["checkpoint"])
        p.write_text(p.read_text() + " ")
    elif changed == "source":
        spec["source_commit"] = "other"
    elif changed == "test":
        spec["split"] = "test"
    elif changed == "deg":
        spec["metric_scope"] = "all_and_deg"
    elif changed in ("gpu", "role"):
        plan.update({"gpu": ["1"]} if changed == "gpu" else {"checkpoint_role": "best"})
        plan_path.write_text(json.dumps(plan))
        item["sha256"] = sha256_file(plan_path)
    else:
        Path(item["output"]).mkdir(parents=True)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        m.verify(path, sha256_file(path))
    assert not (path.parent / "OWNER.json").exists()
