"""The short entry preserves the historical default and fails closed."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.execution import train_entry as entry
from gradpert.hashing import sha256_file


@pytest.fixture
def setup(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(entry, "repository_root", lambda: root)
    monkeypatch.setattr(entry, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(
        "gradpert.execution.identity.inspect_source_identity",
        lambda *a, **kw: SimpleNamespace(commit="a" * 40),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "data_root": str(tmp_path),
                "runs_root": str(tmp_path / "runs"),
                "publication_receipt": str(receipt),
                "publication_sha256": sha256_file(receipt),
                "genept_receipt": str(receipt),
                "genept_sha256": sha256_file(receipt),
            }
        )
    )
    return argparse.Namespace(
        config=None, seed=None, gpu="0", data_root=None, runtime=runtime, dry_run=True
    )


def test_default_dry_run_no_artifacts(setup, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entry, "execute_plan", lambda p: pytest.fail("dry-run launched"))
    before = set(tmp_path.rglob("*"))
    assert entry.train_entry(setup) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"].endswith(entry.DEFAULT_CONFIG)
    assert plan["seed"] == 1
    assert plan["resolved_config"]["training"]["monitor"] == "val/txpert_macro_pearson_delta"
    assert set(tmp_path.rglob("*")) == before
    assert entry.resolve_plan(setup)["run_id"] != plan["run_id"]


@pytest.mark.parametrize("field,value", [("seed", 2), ("seed", -1), ("gpu", "0,1"), ("gpu", "-1")])
def test_invalid_overrides(setup, field, value):
    setattr(setup, field, value)
    with pytest.raises(ValueError):
        entry.resolve_plan(setup)


def test_bad_receipt_fails(setup):
    runtime = json.loads(setup.runtime.read_text())
    runtime["publication_sha256"] = "b" * 64
    setup.runtime.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="publication receipt hash"):
        entry.resolve_plan(setup)


def test_cli_parser_defaults():
    from gradpert.cli import _parser

    args = _parser().parse_args(["train", "--gpu", "1", "--dry-run"])
    assert args.config is None and args.seed is None and args.gpu == "1" and args.dry_run


def test_missing_prior_rejected(setup):
    runtime = json.loads(setup.runtime.read_text())
    del runtime["genept_receipt"]
    del runtime["genept_sha256"]
    setup.runtime.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="GenePT configuration requires"):
        entry.resolve_plan(setup)


def test_execute_uses_native_then_postfit(setup, monkeypatch):
    plan = entry.resolve_plan(setup)
    calls = []
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *a, **kw: "0, GPU-abcd\n")
    for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "GRADPERT_SPARSE_UNION_IMPL"):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setattr(
        "gradpert.evaluation.state.prepare_evaluation_state",
        lambda **kw: calls.append(("prepare", kw)),
    )
    monkeypatch.setattr(
        "gradpert.execution.native.run_native_experiment", lambda **kw: calls.append(("fit", kw))
    )
    monkeypatch.setattr(
        "gradpert.execution.postfit.evaluate_best_last", lambda **kw: calls.append(("test", kw))
    )
    entry.execute_plan(plan)
    assert [x[0] for x in calls] == ["prepare", "fit", "test"]
    assert calls[0][1]["validation_only"]
    assert calls[1][1]["report_all_validation_metrics"]
    assert calls[1][1]["formal"] and calls[1][1]["mode"] == "full"
    assert calls[2][1]["training_root"] == Path(plan["run_root"]) / "fit"
    assert (Path(plan["run_root"]) / "COMPLETE.json").is_file()
    with pytest.raises(FileExistsError):
        entry.execute_plan(plan)
