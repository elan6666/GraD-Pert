"""PT2 has a six-epoch scope and genuine phase-specific diagnostic checkpoints."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

import generate_pt2 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_pt2_medium as dual  # noqa: E402
from checkpoint_endpoint import select_checkpoint  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_completed_ablation_analysis import module as analysis  # noqa: E402
from test_completed_ablation_analysis import run as run  # noqa: E402
from test_completed_ablation_analysis import staged_run as staged_run  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


def test_pt2_self_contained_budget_and_native_entry():
    manifest = ROOT / "configs/v2/pt2_native_io_20261017/manifest.json"
    value = family.verify_manifest(ROOT, manifest)
    assert value["epochs"] == 6 and value["prediction_epochs"] == 4
    p = family.configuration(ROOT)
    options = {k: v["value"] for k, v in p["model"]["parameters"].items()}
    assert options["weighted_gram_backend"] == "triton_io"
    assert (options["ssl_pretrain_epochs"], options["finetune_learning_rate"]) == (2, 2e-4)
    assert list(value["analysis_endpoints"]) == ["after_ssl2", "after_finetune4"]
    plan = dict(
        repository_root=str(ROOT),
        config=str(ROOT / value["rows"][0]["config"]),
        resolved_config=load_experiment_config(ROOT / value["rows"][0]["config"]).model_dump(
            mode="json"
        ),
    )
    assert runner.expected_epochs(plan) == 6
    modified = copy.deepcopy(plan)
    modified["resolved_config"]["model"]["parameters"]["ssl_pretrain_epochs"]["value"] = 3
    with pytest.raises(ValueError, match="exact PT2"):
        runner.expected_epochs(modified)
    # The historical preparation catalog is not an executable queue manifest.
    with pytest.raises(ValueError, match="scope"):
        family.verify_manifest(ROOT, ROOT / "configs/v2/pt2_staged_2plus4_20261017/manifest.json")


def test_controller_can_prepare_only_the_exact_singleton(new_queue):
    old, oldout = new_queue
    q = runner.prepare(
        ROOT,
        ROOT / "configs/v2/pt2_native_io_20261017/manifest.json",
        Path(old["runtime"]),
        oldout.parent / "pt2-scope",
        copy.deepcopy(old["shared_gpu_policy"]),
    )
    runner.verify(q)
    assert q["family"] == family.SCHEMA and len(q["rows"]) == 1
    assert q["rows"][0]["plan"]["resolved_config"]["training"]["max_epochs"]["value"] == 6
    assert len(q["eval_bindings"]) == 2
    assert family.SCHEMA in runner.FGPO_FAMILIES and "PT2" in medium.AUTHORIZED_ARMS


def seal_pretrain(root):
    path = root / "fit/epoch-0002.pt"
    path.write_bytes(b"distinct SSL-only checkpoint")
    last = read_json(root / "COMPLETE.json")["last"]
    last["epoch"] = 4
    cp = dict(epoch=2, file=path.name, sha256=sha256_file(path))
    atomic_json(root / "fit/epoch_state.json", dict(last=last, pretrain=cp))
    return cp


def test_pretrain_analysis_is_read_only_and_never_borrows_final_test(staged_run, tmp_path):
    cp = seal_pretrain(staged_run)
    data = analysis.collect(
        staged_run, name="PT2", analysis_identity={"commit": "d" * 40}, checkpoint_role="pretrain"
    )
    assert data["checkpoint_sha256"] == cp["sha256"] and data["checkpoint_epoch"] == 2
    assert data["full_training_epochs"] == 4 and data["training_stage"] == "ssl_pretrain"
    assert [h["epoch"] for h in data["history"]] == [1, 2]
    assert [p["epoch"] for p in data["snapshots"]] == [0, 1, 2]
    assert not data["metrics"] and data["test_prediction_mse"] is None
    assert data["evaluation_source"] is None and not data["test_performed_for_endpoint"]
    output = tmp_path / "pretrain-report"
    analysis.export(data, output)
    text = (output / "report.md").read_text()
    assert "预测头尚未训练" in text and "不复用最终checkpoint" in text
    assert len(list(output.glob("*.png"))) == 4
    # The omitted later phase is still validated, not hidden by endpoint truncation.
    later = staged_run / "fit/loss-diagnostic-epoch-0004.json"
    d = read_json(later)
    d["process"]["input_sha256"] = "f" * 64
    atomic_json(later, d)
    history = read_json(staged_run / "fit/history.json")
    history[-1]["training_diagnostic"] = d
    atomic_json(staged_run / "fit/history.json", history)
    with pytest.raises(ValueError, match="within training stage"):
        analysis.collect(staged_run, name="PT2", analysis_identity={}, checkpoint_role="pretrain")


@pytest.mark.parametrize("change", ["hash", "epoch", "escape", "not_staged"])
def test_endpoint_cannot_select_a_different_or_unbound_checkpoint(staged_run, tmp_path, change):
    seal_pretrain(staged_run)
    select_checkpoint(staged_run, "pretrain")
    j = read_json(staged_run / "fit/epoch_state.json")
    if change == "hash":
        (staged_run / "fit/epoch-0002.pt").write_bytes(b"changed")
    elif change == "epoch":
        j["pretrain"]["epoch"] = 3
    elif change == "escape":
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"outside")
        j["pretrain"].update(file=str(outside), sha256=sha256_file(outside))
    else:
        c = read_json(staged_run / "resolved_config.json")
        c["model"]["parameters"]["training_mode"]["value"] = "joint"
        atomic_json(staged_run / "resolved_config.json", c)
    atomic_json(staged_run / "fit/epoch_state.json", j)
    with pytest.raises(ValueError):
        select_checkpoint(staged_run, "pretrain")


def test_legacy_config_does_not_acquire_six_epoch_dispatch():
    p = ROOT / "configs/v2/pt1_staged_20261017/PT1/gradpert_v2/nadig_jurkat.yaml"
    config = yaml.safe_load(p.read_text())
    assert config["training"]["max_epochs"]["value"] == 4
    assert (
        runner.expected_epochs(
            dict(resolved_config=load_experiment_config(p).model_dump(mode="json"))
        )
        == 4
    )


def test_pt2_completion_requires_both_endpoints_and_identical_physical_manifest(
    tmp_path, monkeypatch
):
    root = tmp_path / "PT2-medium"
    root.mkdir()
    queue = dict(family=family.SCHEMA, output=str(tmp_path), source_commit="c" * 40)
    atomic_json(tmp_path / "queue.json", queue)
    row = dict(name="PT2", plan=dict(run_root="unused"))
    shared = root / "after_ssl2/manifest/manifest.json"
    atomic_json(shared, dict(schedule=[]))
    endpoints = {}
    for i, (label, spec) in enumerate(family.ENDPOINTS.items()):
        endpoint = {**spec, "full_training_epochs": 6}
        cp = dict(sha256=str(i) * 64)
        child = root / label
        atomic_json(
            child / "MEDIUM_COMPLETE.json",
            dict(
                checkpoint_endpoint=endpoint,
                checkpoint_sha256=cp["sha256"],
                manifest=str(shared),
                manifest_sha256=sha256_file(shared),
            ),
        )
        endpoints[label] = dict(
            receipt=str(child / "MEDIUM_COMPLETE.json"),
            sha256=sha256_file(child / "MEDIUM_COMPLETE.json"),
            **medium.COUNTS,
        )
    atomic_json(
        root / "MEDIUM_COMPLETE.json",
        dict(
            schema="pt2-dual-endpoint-medium-complete-1",
            status="completed",
            source_commit=queue["source_commit"],
            queue_sha256=sha256_file(tmp_path / "queue.json"),
            updates=0,
            endpoints=endpoints,
            **medium.COUNTS,
        ),
    )

    def child_proof(q, r, *, root, checkpoint_role):
        assert checkpoint_role in ("pretrain", "last")
        return dict(
            receipt=str(root / "MEDIUM_COMPLETE.json"),
            sha256=sha256_file(root / "MEDIUM_COMPLETE.json"),
            **medium.COUNTS,
        )

    monkeypatch.setattr(medium, "validate_medium_complete", child_proof)
    monkeypatch.setattr(
        dual,
        "select_checkpoint",
        lambda root, role: (
            {},
            {
                **next(v for v in family.ENDPOINTS.values() if v["role"] == role),
                "full_training_epochs": 6,
            },
        ),
    )
    dual.validate_complete(queue, row, root=root)
    other = root / "after_finetune4/manifest.json"
    other.write_bytes(shared.read_bytes())
    child = root / "after_finetune4/MEDIUM_COMPLETE.json"
    v = read_json(child)
    v["manifest"] = str(other)
    atomic_json(child, v)
    complete = read_json(root / "MEDIUM_COMPLETE.json")
    complete["endpoints"]["after_finetune4"]["sha256"] = sha256_file(child)
    atomic_json(root / "MEDIUM_COMPLETE.json", complete)
    with pytest.raises(ValueError, match="identical physical"):
        dual.validate_complete(queue, row, root=root)
