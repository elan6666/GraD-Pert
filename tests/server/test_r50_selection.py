import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.server.run_r50_selection import command, phases, sha, validate


def test_build_campaign_never_includes_full_without_explicit_opt_in():
    assert phases(full_after_smoke=False) == (("smoke", 1),)
    assert phases(full_after_smoke=True) == (("smoke", 1), ("full", 50))


@pytest.mark.parametrize("full", [False, True])
def test_main_smoke_only_never_calls_full_or_test(tmp_path, monkeypatch, full):
    from types import SimpleNamespace

    from gradpert.execution import postfit
    from scripts.server import run_r50_selection as runner

    config = tmp_path / "configs/r50/g2_schedule/gradpert_b2/nadig_jurkat.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("synthetic fixture")
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    args = argparse.Namespace(
        source=tmp_path,
        root=tmp_path / "run",
        data_root=tmp_path / "data",
        row="g2_schedule",
        publication=publication,
        publication_sha=sha(publication),
        genept_receipt=publication,
        genept_sha=sha(publication),
        commit="a" * 40,
        config_sha=sha(config),
        memory_fraction=None,
        dry_run=False,
        full_after_smoke=full,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    # Synthetic filesystem only; real main still enforces the server root.
    monkeypatch.setattr(Path, "is_relative_to", lambda self, other: True)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **kwargs: args.commit if "rev-parse" in cmd else "",
    )
    observed = []

    def execute(cmd, **kwargs):
        observed.append(cmd[cmd.index("model") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner, "validate", lambda root, **kwargs: {"epochs": kwargs["epochs"]})
    monkeypatch.setattr(postfit, "evaluate_best_last", lambda **kwargs: observed.append("test"))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "cpu_array")
    runner.main()
    assert observed == (["smoke", "full", "test"] if full else ["smoke"])
    receipt = json.loads((args.root / "COMPLETE.json").read_text())
    assert receipt["full_after_smoke"] is full
    assert receipt["scientific_completion"] is False


def test_r50_command_carries_genept_and_source_receipts(tmp_path):
    args = argparse.Namespace(
        source=tmp_path,
        root=tmp_path / "run",
        data_root=tmp_path / "data",
        row="ref",
        publication=tmp_path / "publication.json",
        publication_sha="a" * 64,
        genept_receipt=tmp_path / "prior.json",
        genept_sha="b" * 64,
    )
    for phase in ("smoke", "full"):
        argv = command(args, phase)
        assert argv[argv.index("--genept-preflight-receipt") + 1] == str(args.genept_receipt)
        assert argv[argv.index("--genept-preflight-receipt-sha256") + 1] == "b" * 64
        assert argv[argv.index("--source-publication-remote-ref") + 1] == "refs/heads/main"
        assert argv[argv.index("--run-root") + 1] == str(args.root / phase)


def fixture(root: Path, epochs=50):
    small = root / "small_results"
    small.mkdir()
    cp = root / "checkpoints/best.pt"
    cp.parent.mkdir()
    cp.write_bytes(b"test-only checkpoint")
    files = {
        "run_manifest.json": dict(
            status="trained",
            test_evaluations=0,
            source_commit="a" * 40,
            source_dirty=False,
            formal_eligible=True,
            config_sha256="b" * 64,
            best_checkpoint_sha256=sha(cp),
        ),
        "selection_receipt.json": dict(
            status="complete",
            scientific_completion=False,
            epochs_completed=epochs,
            optimizer_steps=epochs,
            test_evaluations=0,
            test_deferred=True,
            control_manifest_scope="validation",
        ),
        "run_meta.json": dict(steps_per_epoch=1),
    }
    for name, payload in files.items():
        (small / name).write_text(json.dumps(payload))
    for epoch in range(epochs):
        (small / f"validation.epoch-{epoch:03d}.json").write_text(json.dumps(dict(epoch=epoch)))
    with (small / "train_steps.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "global_step"])
        w.writeheader()
        w.writerows(dict(epoch=e, global_step=e) for e in range(epochs))
    return small


def test_r50_terminal_budget_and_retention(tmp_path):
    fixture(tmp_path)
    assert validate(tmp_path, epochs=50, commit="a" * 40, config_sha="b" * 64)["epochs"] == 50
    with pytest.raises(ValueError):
        validate(tmp_path, epochs=1, commit="a" * 40, config_sha="b" * 64)


@pytest.mark.parametrize(
    "name", ["metrics_summary.json", "prediction_manifest.json", "evaluation_manifest.json"]
)
def test_r50_rejects_test_artifact(tmp_path, name):
    small = fixture(tmp_path)
    (small / name).write_text("{}")
    with pytest.raises(ValueError, match="test outputs"):
        validate(tmp_path, epochs=50, commit="a" * 40, config_sha="b" * 64)


def test_r50_rejects_hidden_pkl(tmp_path):
    fixture(tmp_path)
    (tmp_path / "unexpected.pkl").write_bytes(b"bad")
    with pytest.raises(ValueError, match="zero persistent PKL"):
        validate(tmp_path, epochs=50, commit="a" * 40, config_sha="b" * 64)


def test_r50_accepts_hash_bound_last_retention(tmp_path):
    small = fixture(tmp_path)
    last = tmp_path / "checkpoints/last.pt"
    last.write_bytes(b"final epoch checkpoint")
    (small / "checkpoint_retention.json").write_text(
        json.dumps(
            {
                "policy": "best_and_last_for_postfit_test",
                "last_checkpoint_sha256": sha(last),
            }
        )
    )
    validate(tmp_path, epochs=50, commit="a" * 40, config_sha="b" * 64)
    last.write_bytes(b"changed")
    with pytest.raises(ValueError, match="last checkpoint hash"):
        validate(tmp_path, epochs=50, commit="a" * 40, config_sha="b" * 64)
