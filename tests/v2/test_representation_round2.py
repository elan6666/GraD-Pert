"""Round2 isolates CLS-head input conditioning from backbone dispersion."""

import contextlib
import copy
import json
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
import generate_representation_round2 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as research  # noqa: E402
from eval_batch_capacity import validate_eval_binding  # noqa: E402
from test_br400_group import eval_receipt  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402
from test_fgpo_group import training_gate  # noqa: E402

MANIFEST = ROOT / "configs/v2/representation_round2_20261022/manifest.json"


def test_round2_exact_independent_axes_and_same_two_epoch_anchor():
    manifest = family.verify_manifest(ROOT, MANIFEST)
    values = family.configs(ROOT)
    assert values["R2-RI"] == first.configs(ROOT)["RS-RI"]
    assert manifest["epochs"] == 2 and manifest["global_batch"] == 320
    assert manifest["test_timing"] == "after_research_design_lock"
    assert manifest["koleo"]["context_restriction"] is False
    assert manifest["head_norm"]["other_heads"] == "unchanged; iBOT has no added input norm"
    for name, key in [("R2-HN", "ssl2_cls_input_layernorm"), ("R2-KL", "ssl2_koleo")]:
        changed = copy.deepcopy(values[name])
        before = values["R2-RI"]["model"]["parameters"].get(key)
        if before is None:
            changed["model"]["parameters"].pop(key)
        else:
            changed["model"]["parameters"][key] = before
        assert changed == values["R2-RI"]
    assert values["R2-KL"]["model"]["parameters"]["koleo_exclude_same_condition"]["value"]


@pytest.mark.parametrize("mutation", ["mix_axes", "budget", "test", "extra_arm", "coefficient"])
def test_seal_rejects_scope_or_config_changes_even_after_rehash(tmp_path, mutation):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2/round2"
    data = family.generate(source, output)
    if mutation == "test":
        data["research_iteration"]["test_results_for_selection"] = True
    elif mutation == "extra_arm":
        data["rows"].append(data["rows"][0])
    else:
        row = data["rows"][1]
        path = source / row["config"]
        cfg = yaml.safe_load(path.read_text())
        if mutation == "budget":
            cfg["training"]["max_epochs"]["value"] = 4
        else:
            cfg["model"]["parameters"]["ssl2_koleo"]["value"] = 0.1 if mutation == "mix_axes" else 1
        path.write_text(yaml.safe_dump(cfg))
        row["sha256"] = sha256_file(path)
    (output / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        family.verify_manifest(source, output / "manifest.json")


@pytest.fixture
def round2_queue(new_queue):
    old, output = new_queue
    q = runner.prepare(
        ROOT, MANIFEST, Path(old["runtime"]), output.parent / "round2", old["shared_gpu_policy"]
    )
    return q, Path(q["output"])


def test_round2_native_training_and_eval_bindings(round2_queue):
    q, output = round2_queue
    assert list(research.FAMILY_ARMS[q["family"]]) == list(family.ARMS)
    assert list(medium.FAMILY_ARMS[q["family"]]) == list(family.ARMS)
    for row in q["rows"]:
        plan = row["plan"]
        assert runner.expected_epochs(plan) == 2
        assert load_experiment_config(plan["config"]).training.max_epochs.value == 2
        atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(q, row))
        runner.validate_training_gate(
            q, row, runner.gate_entry(output / f"{row['name']}-preflight/receipt.json")
        )
        for binding in [b for b in q["eval_bindings"] if b["row_name"] == row["name"]]:
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
                row_name=row["name"],
            )
            assert validate_eval_binding(q, args, q["source_commit"]) == row


@pytest.mark.parametrize("fail", [None, "postfit-validation", "research-analysis"])
def test_round2_controller_stops_before_test_and_before_next_round(round2_queue, monkeypatch, fail):
    q, output = round2_queue
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", runner.ALLOCATOR)
    monkeypatch.setattr(runner, "snapshot", lambda pid: dict(processes=[], gpus={}))
    monkeypatch.setattr(runner, "admit", lambda *a: None)
    locks, calls = set(), []

    @contextlib.contextmanager
    def lock(path):
        assert path.name not in locks
        locks.add(path.name)
        try:
            yield
        finally:
            locks.remove(path.name)

    def stage(queue, row, name, command, gpu, state, guard):
        calls.append((row["name"], name))
        assert name not in ("last", "best-last")
        if row["name"] == "R2-RI" and name == fail:
            raise RuntimeError("injected round2 failure")
        if name == "preflight":
            atomic_json(output / f"{row['name']}-preflight/receipt.json", training_gate(q, row))
        elif name.startswith("eval-capacity"):
            binding = next(
                b for b in q["eval_bindings"] if b["row_name"] == row["name"] and b["gpu"] == gpu
            )
            receipt = eval_receipt(q, row, binding)
            receipt["cell_batch"] = 128
            atomic_json(Path(binding["output"]) / "receipt.json", receipt)

    def accept(_):
        atomic_json(
            output / "MAIN_ACCEPTANCE.json",
            {**read_json(output / "PREFLIGHT_COMPLETE.json"), "approved": True},
        )

    monkeypatch.setattr(runner, "lock", lock)
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner.time, "sleep", accept)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: {})
    monkeypatch.setattr(runner, "validate_complete", lambda p: pytest.fail("test finalizer used"))
    monkeypatch.setattr(research, "validate_research_complete", lambda *a: {"research": True})
    if fail:
        with pytest.raises(RuntimeError, match="injected round2"):
            runner.execute(q, output)
        assert ("R2-HN", "fit") not in calls
    else:
        runner.execute(q, output)
        assert calls[9:] == [
            (a, s) for a in family.ARMS for s in ("fit", "postfit-validation", "research-analysis")
        ]
        result = read_json(output / "RESEARCH_COMPLETE.json")
        assert result["scientific_complete"] is False
        assert set(result["research_completed"]) == set(family.ARMS)
    assert not (output / "COMPLETE.json").exists() and not locks


def test_koleo_partial_probe_report_exports_na_and_explicit_scope(tmp_path):
    import csv

    root, run = tmp_path / "R2-KL-medium", tmp_path / "run"
    row = dict(name="R2-KL", plan=dict(run_root=str(run), config_sha256="c" * 64))
    q = dict(output=str(tmp_path), source_commit="a" * 40)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p1",
                joint=2.0,
                loss_terms=dict(prediction=1.0, ssl2_dino=1.0),
                diagnostic_objective_scope=dict(
                    unavailable_terms=dict(ssl2_koleo="requires_complete_global_population"),
                    joint_scope="prediction_and_ssl_excluding_unavailable_koleo",
                ),
            )
        ],
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(run / "fit/history.json", [dict(epoch=1), dict(epoch=2)])
    atomic_json(
        research.validation_path(q, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    research.render_medium_report(q, row, root)
    rows = list(csv.DictReader((root / "report/frozen_losses.csv").open()))
    assert rows[0]["ssl2_koleo"] == ""
    assert rows[0]["unavailable_terms.ssl2_koleo"] == "requires_complete_global_population"
    assert rows[0]["joint_scope"] == "prediction_and_ssl_excluding_unavailable_koleo"
