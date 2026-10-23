"""Singleton CR must retain native research gates and never refit old controls."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round4 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as research  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402
from test_fgpo_group import training_gate  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2/round4"
    family.generate(source, output)
    return source, output


def test_only_declared_loss_changes_no_repeat_baseline(family_root):
    source, output = family_root
    manifest = family.verify_manifest(source, output / "manifest.json")
    assert manifest["priority"] == manifest["preflight_priority"] == ["R4-CR"]
    assert manifest["repeat_baseline"] is False and manifest["epochs"] == 2
    assert manifest["initialization"].startswith("from_scratch")
    config = copy.deepcopy(family.configs(source)["R4-CR"])
    for key, value in family.CHANGES.items():
        assert config["model"]["parameters"].pop(key)["value"] == value
    assert config == first.configs(source)["RS-RI"]


@pytest.mark.parametrize("mutation", ["weight", "budget", "baseline", "evidence", "test"])
def test_rehashed_scientific_drift_rejected(family_root, mutation):
    source, output = family_root
    path = output / "manifest.json"
    manifest = read_json(path)
    if mutation in ("weight", "budget"):
        target = source / manifest["rows"][0]["config"]
        config = yaml.safe_load(target.read_text())
        if mutation == "weight":
            config["model"]["parameters"]["cr_weight"]["value"] = 0.001
        else:
            config["training"]["max_epochs"]["value"] = 4
        target.write_text(yaml.safe_dump(config))
        manifest["rows"][0]["sha256"] = sha256_file(target)
    elif mutation == "baseline":
        manifest["rows"].append({**manifest["rows"][0], "name": "R3-RI"})
    elif mutation == "evidence":
        manifest["coefficient_decision"]["complete_sha256"]["R3-RI"] = "a" * 64
    else:
        manifest["research_iteration"]["test_results_for_selection"] = True
    atomic_json(path, manifest)
    with pytest.raises(ValueError):
        family.verify_manifest(source, path)


def test_native_two_epoch_training_and_both_evaluation_bindings(
    family_root, new_queue, monkeypatch
):
    source, family_output = family_root
    old, output = new_queue
    runtime = Path(old["runtime"])
    atomic_json(runtime, {**read_json(runtime), "repository_root": str(source)})

    def resolve(args):
        plan = copy.deepcopy(old["rows"][0]["plan"])
        run_id = "fresh-r4-cr"
        plan.update(
            config=str(args.config),
            config_sha256=sha256_file(args.config),
            resolved_config=load_experiment_config(args.config).model_dump(mode="json"),
            runtime_sha256=sha256_file(runtime),
            repository_root=str(source),
            run_id=run_id,
            run_root=str(Path(plan["run_root"]).parent / run_id),
            test_root=str(Path(plan["test_root"]).parent / (run_id + "-test")),
        )
        return plan

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    monkeypatch.setattr(runner, "__file__", str(source / "scripts/v2/run_br400_group.py"))
    queue = runner.prepare(
        source,
        family_output / "manifest.json",
        runtime,
        output.parent / "r4",
        old["shared_gpu_policy"],
    )
    output = Path(queue["output"])
    assert len(queue["rows"]) == 1 and queue["steps"] == 10
    assert (
        list(medium.FAMILY_ARMS[family.SCHEMA])
        == list(research.FAMILY_ARMS[family.SCHEMA])
        == ["R4-CR"]
    )
    row = queue["rows"][0]
    plan = row["plan"]
    assert runner.expected_epochs(plan) == 2
    receipt = output / "R4-CR-preflight/receipt.json"
    atomic_json(receipt, training_gate(queue, row))
    runner.validate_training_gate(queue, row, runner.gate_entry(receipt))
    assert len(queue["eval_bindings"]) == 2
    for binding in queue["eval_bindings"]:
        args = SimpleNamespace(
            gpu=binding["gpu"],
            cell_batch=128,
            reference_batch=128,
            config=Path(plan["config"]),
            publication=Path(plan["publication"]),
            publication_sha256=plan["publication_sha256"],
            data_root=Path(plan["data_root"]),
            output=Path(binding["output"]),
            shared_queue=output / "queue.json",
            row_name="R4-CR",
        )
        assert validate_eval_binding(queue, args, queue["source_commit"]) == row
    changed = copy.deepcopy(queue)
    changed["rows"].append({**row, "name": "R3-RI"})
    with pytest.raises(ValueError):
        runner.verify(changed)
