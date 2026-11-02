"""Synthetic immutable last300 terminal/axis fixtures; no H5AD or CUDA access."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.state import EvaluationStateLayout
from gradpert.execution import dataset_transfer_validation as adapter
from gradpert.execution import v2_checkpoint_eval as native
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.expression_policy import expression_policy

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr("gradpert.execution.train_entry.repository_root", lambda: ROOT)
    monkeypatch.setattr(
        native,
        "inspect_source_identity",
        lambda *a, **kw: SimpleNamespace(
            payload=lambda: dict(repository_root=str(ROOT), commit="b" * 40, dirty=False)
        ),
    )

    def create(steps=263):
        folder = tmp_path / str(steps)
        folder.mkdir()
        config = load_experiment_config(
            ROOT
            / "configs/v2/dataset_transfer_300_20261102/gradpert_v2/replogle_k562_essential.yaml"
        ).model_dump(mode="json")
        epochs = (300 + steps - 1) // steps
        config["training"]["max_epochs"]["value"] = epochs
        cfg = folder / "gradpert_v2/replogle_k562_essential.yaml"
        atomic_json(cfg, config)
        parsed = load_experiment_config(cfg)
        run = folder / "run"
        (run / "fit").mkdir(parents=True)
        data_root = folder / "data"
        layout = DatasetLayout(data_root, parsed.dataset_id, parsed.data.protocol_id)
        layout.manifests.mkdir(parents=True)
        layout.canonical.mkdir(parents=True)
        genes = tuple(f"g{i}" for i in range(8))
        (layout.canonical / "expression_gene_ids.txt").write_text("\n".join(genes) + "\n")
        atomic_json(
            layout.manifests / "split.json",
            dict(
                dataset_id=parsed.dataset_id,
                protocol_id=parsed.data.protocol_id,
                val_conditions=["g1+ctrl", "g2+ctrl"],
                test_conditions=["g7+ctrl"],
                control_condition_id="ctrl",
            ),
        )
        atomic_json(layout.manifests / "evaluation_controls.val.json", {"frozen": "controls"})
        ref = EvaluationStateLayout(
            data_root,
            parsed.dataset_id,
            parsed.data.protocol_id,
            validation_only=True,
            evaluation_protocol="v2",
        ).manifest
        atomic_json(ref, {"frozen": "train-only reference"})
        _, policy = expression_policy(genes, ("g7+ctrl",), "ctrl", enabled=True)
        original_source = folder / "immutable-training-source"
        original_source.mkdir()
        identity = dict(
            source=dict(
                commit="a" * 40,
                published_commit="a" * 40,
                dirty=False,
                repository_root=str(original_source),
            ),
            config_sha256=sha256_file(cfg),
            resolved_config_sha256=sha256_json(config),
            run_id="synthetic300",
            training_budget=dict(
                unit="optimizer_update",
                count=300,
                steps_per_data_epoch=steps,
                partial_final_epoch=300 % steps != 0,
            ),
            data=dict(
                run_seed=1,
                training_expression_policy=policy,
                effective_training_expression_ids_sha256=policy[
                    "allowed_expression_gene_ids_sha256"
                ],
            ),
        )
        plan = dict(
            config=str(cfg),
            config_sha256=sha256_file(cfg),
            resolved_config=config,
            source_commit="a" * 40,
            run_id="synthetic300",
            run_root=str(run),
            data_root=str(data_root),
            seed=1,
            gpu="0,1",
            postfit_policy="deferred",
        )
        atomic_json(run / "launch.json", plan)
        atomic_json(run / "resolved_config.json", config)
        atomic_json(run / "run_manifest.json", identity)
        checkpoint = run / "fit/final.pt"
        checkpoint.write_bytes(b"synthetic state identity; not model weights")
        role = dict(file="final.pt", sha256=sha256_file(checkpoint), epoch=epochs)
        history = [
            dict(
                epoch=i + 1,
                optimizer_steps=min((i + 1) * steps, 300),
                updates_this_data_epoch=min(steps, 300 - i * steps),
                data_epoch_complete=min(steps, 300 - i * steps) == steps,
                data_epoch_fraction=min(steps, 300 - i * steps) / steps,
                seen_expression_gene_indices=[0, 1, 2],
            )
            for i in range(epochs)
        ]
        journal = dict(
            identity=identity,
            budget=[epochs, steps],
            epoch=epochs,
            last=role,
            best=role,
            contract=dict(optimizer_update_budget=300),
        )
        atomic_json(run / "fit/history.json", history)
        atomic_json(run / "fit/epoch_state.json", journal)
        atomic_json(run / "TRAIN_COMPLETE.json", training_stage_evidence(run, identity, epochs))
        original = folder / "PLAN.json"
        atomic_json(original, plan)
        pub = folder / "PUBLICATION.json"
        atomic_json(pub, {"source": "synthetic"})
        runtime = folder / "RUNTIME.json"
        atomic_json(
            runtime,
            dict(
                publication_receipt=str(pub),
                publication_sha256=sha256_file(pub),
                data_root=str(data_root),
            ),
        )
        compat = folder / "COMPATIBILITY.json"
        atomic_json(
            compat,
            dict(
                approved=True,
                model_checkpoint_compatible=True,
                training_commit="a" * 40,
                evaluation_commit="b" * 40,
                checks=["synthetic immutable compatibility receipt"],
            ),
        )
        args = dict(
            training_plan=original,
            training_plan_sha256=sha256_file(original),
            runtime=runtime,
            output=folder / "eval",
            gpu="0,1",
            compatibility=adapter.binding(compat),
        )
        return args, plan, identity, history, genes

    return create


@pytest.mark.parametrize("steps", [263, 393, 150])
def test_partial_and_complete_data_epochs_last300(terminal, steps):
    args, _training, identity, history, genes = terminal(steps)
    plan = adapter.build_validation_plan(**args)
    assert plan["checkpoint_role"] == "last" and plan["split"] == "val"
    assert plan["training_identity"] == identity
    assert adapter.validate_hook(plan)
    frozen = plan[adapter.HOOK]["frozen_axis"]
    groups, exposure = adapter.validation_groups(
        plan,
        history=history,
        genes=genes,
        allowed=tuple(range(7)),
        test_conditions=("g7+ctrl",),
        control_condition="ctrl",
        condition_order=("g1+ctrl", "g2+ctrl"),
        reference_hash=frozen["reference_manifest"]["sha256"],
        controls_hash=frozen["val_controls"]["sha256"],
    )
    assert groups == {"allowed_expression": tuple(range(7)), "heldout_expression": (7,)}
    assert not exposure["actual_seen_equals_allowed"]
    assert exposure["allowed_but_not_observed"]["gene_count"] == 4
    assert "g7+ctrl" not in frozen["conditions"]


@pytest.mark.parametrize(
    "target", ["history", "checkpoint", "config", "dataset", "count", "fraction"]
)
def test_changed_or_incomplete_training_rejected(terminal, target):
    args, training, _, _, _ = terminal()
    run = Path(training["run_root"])
    if target == "checkpoint":
        (run / "fit/final.pt").write_bytes(b"changed")
    elif target == "config":
        Path(training["config"]).write_text("{}")
    elif target in ("history", "fraction"):
        h = read_json(run / "fit/history.json")
        h[-1]["optimizer_steps" if target == "history" else "data_epoch_fraction"] = 299
        atomic_json(run / "fit/history.json", h)
    elif target == "dataset":
        changed = copy.deepcopy(training)
        changed["resolved_config"]["dataset_id"] = "nadig_jurkat"
        atomic_json(args["training_plan"], changed)
    else:
        j = read_json(run / "fit/epoch_state.json")
        j["contract"]["optimizer_update_budget"] = 299
        atomic_json(run / "fit/epoch_state.json", j)
    with pytest.raises((ValueError, KeyError)):
        adapter.build_validation_plan(**args)


def test_wrong_policy_and_missing_compatibility_fail_closed(terminal):
    args, *_ = terminal()
    args["compatibility"] = None
    with pytest.raises(ValueError, match="compatibility"):
        adapter.build_validation_plan(**args)
    assert adapter.validate_hook({}) is False
    for value in (None, {}, dict(schema="unknown")):
        with pytest.raises(ValueError, match="unknown"):
            adapter.validate_hook({adapter.HOOK: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("checkpoint_role", "best"),
        ("diagnostic_condition_count", 1),
        ("evaluation_precision", "common_ri_native"),
    ],
)
def test_hook_cannot_override_other_protocols(terminal, key, value):
    args, *_ = terminal()
    plan = adapter.build_validation_plan(**args)
    plan[key] = value
    with pytest.raises(ValueError):
        adapter.validate_hook(plan)


def test_allowed_axis_or_reference_tamper_rejected(terminal):
    args, *_ = terminal()
    plan = adapter.build_validation_plan(**args)
    plan[adapter.HOOK]["frozen_axis"]["groups"]["heldout_expression"] = [6, 7]
    with pytest.raises(ValueError):
        adapter.validate_hook(plan)


def test_native_unautocast_context_does_not_touch_default():
    import torch

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with adapter.validation_context({}, "cpu"):
            assert torch.is_autocast_enabled("cpu")
        with adapter.validation_context({adapter.HOOK: {}}, "cpu"):
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")


def test_complete_merge_and_missing_duplicate_conditions():
    from test_eval_shards import shard

    from gradpert.training.v2.evaluation import merge_evaluation_shards

    shards = [shard("A", 0.2), shard("B", 0.4)]
    for s in shards:
        s["split"] = "val"
        g = s["metric_gene_groups"]
        g["allowed_expression"] = g.pop("seen_expression")
        g["heldout_expression"] = g.pop("unseen_expression")
        g["heldout_expression"]["gene_count"] = 1
        for row in s["conditions"]:
            rg = row["metric_gene_groups"]
            rg["allowed_expression"] = rg.pop("seen_expression")
            rg.pop("unseen_expression")
            rg["heldout_expression"] = copy.deepcopy(row["metrics"])
    plan = {adapter.HOOK: dict(frozen_axis=dict(conditions=["A", "B"]))}
    merged = merge_evaluation_shards(shards, ordered_conditions=("A", "B"))
    # Native empty heldout group legitimately has no metric rows, handled separately below.
    for group in merged["metric_gene_groups"].values():
        group["metrics"] = copy.deepcopy(merged["metrics"])
    adapter.validate_merged(plan, merged, ("A", "B"))
    for incomplete in ([shards[0]], [shards[0], shards[0]]):
        with pytest.raises(ValueError):
            merge_evaluation_shards(incomplete, ordered_conditions=("A", "B"))
    with pytest.raises(ValueError):
        adapter.validate_merged(plan, merged, ("A",))


def test_missing_reference_is_not_prepared_implicitly(terminal):
    args, training, *_ = terminal()
    config = load_experiment_config(training["config"])
    reference = EvaluationStateLayout(
        Path(training["data_root"]),
        config.dataset_id,
        config.data.protocol_id,
        validation_only=True,
        evaluation_protocol="v2",
    ).manifest
    reference.unlink()
    with pytest.raises(FileNotFoundError):
        adapter.build_validation_plan(**args)
    assert not reference.exists()


def test_runtime_reference_and_out_of_allowed_actual_exposure_rejected(terminal):
    args, training, identity, history, genes = terminal()
    history[-1]["seen_expression_gene_indices"] = [0, 1, 7]
    root = Path(training["run_root"])
    atomic_json(root / "fit/history.json", history)
    atomic_json(root / "TRAIN_COMPLETE.json", training_stage_evidence(root, identity, 2))
    plan = adapter.build_validation_plan(**args)
    frozen = plan[adapter.HOOK]["frozen_axis"]
    kwargs = dict(
        history=history,
        genes=genes,
        allowed=tuple(range(7)),
        test_conditions=("g7+ctrl",),
        control_condition="ctrl",
        condition_order=("g1+ctrl", "g2+ctrl"),
        reference_hash=frozen["reference_manifest"]["sha256"],
        controls_hash=frozen["val_controls"]["sha256"],
    )
    with pytest.raises(ValueError, match="exposure"):
        adapter.validation_groups(plan, **kwargs)
    kwargs["reference_hash"] = "changed"
    with pytest.raises(ValueError, match="reference"):
        adapter.validation_groups(plan, **kwargs)


def test_eval_source_identity_tamper_is_not_a_compatibility_bypass(terminal):
    args, *_ = terminal()
    plan = adapter.build_validation_plan(**args)
    plan["evaluation_source"]["commit"] = "a" * 40
    with pytest.raises(ValueError, match="publication/source"):
        adapter.validate_hook(plan)


def test_independent_output_cannot_mutate_original_run_or_data(terminal):
    args, training, *_ = terminal()
    for protected in (training["run_root"], training["data_root"]):
        changed = {**args, "output": Path(protected) / "new-eval"}
        with pytest.raises(ValueError, match="overlaps"):
            adapter.build_validation_plan(**changed)


def test_cross_version_output_must_not_enter_original_training_checkout(terminal):
    args, training, identity, *_ = terminal()
    source = Path(identity["source"]["repository_root"])
    with pytest.raises(ValueError, match="protected"):
        adapter.build_validation_plan(**{**args, "output": source / "evaluation"})
    assert not (source / "evaluation").exists()
    del identity["source"]["repository_root"]
    with pytest.raises(ValueError, match="repository_root"):
        adapter.protected_roots(training, identity)


@pytest.mark.parametrize(
    "location",
    [
        "train_source",
        "train_run",
        "data",
        "eval_source",
        "eval_output",
        "inside_eval",
        "eval_ancestor",
        "outside_server",
    ],
)
def test_cli_plan_output_cannot_mutate_protected_or_future_output(terminal, monkeypatch, location):
    import sys

    args, training, identity, *_ = terminal()
    plan = adapter.build_validation_plan(**args)
    output = Path(plan["output_root"])
    paths = dict(
        train_source=Path(identity["source"]["repository_root"]) / "plan.json",
        train_run=Path(training["run_root"]) / "plan.json",
        data=Path(training["data_root"]) / "plan.json",
        eval_source=ROOT / "prohibited-plan.json",
        eval_output=output,
        inside_eval=output / "plan.json",
        eval_ancestor=output.parent,
        outside_server=Path("/outside-authorized-server/plan.json"),
    )
    monkeypatch.setattr(adapter, "build_validation_plan", lambda **kw: plan)
    monkeypatch.setattr(adapter, "atomic_json", lambda *a: pytest.fail("unsafe plan output write"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation",
            "--training-plan",
            str(args["training_plan"]),
            "--training-plan-sha256",
            args["training_plan_sha256"],
            "--runtime",
            str(args["runtime"]),
            "--output",
            str(output),
            "--gpu",
            "0,1",
            "--plan-output",
            str(paths[location]),
        ],
    )
    with pytest.raises(ValueError):
        adapter.main()
    assert not output.exists()


def test_cli_safe_sibling_plan_output_leaves_evaluation_root_absent(terminal, monkeypatch):
    import sys

    args, *_ = terminal()
    plan = adapter.build_validation_plan(**args)
    destination = Path(plan["output_root"]).with_name("last300-val.plan.json")
    monkeypatch.setattr(adapter, "build_validation_plan", lambda **kw: plan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation",
            "--training-plan",
            str(args["training_plan"]),
            "--training-plan-sha256",
            args["training_plan_sha256"],
            "--runtime",
            str(args["runtime"]),
            "--output",
            plan["output_root"],
            "--gpu",
            "0,1",
            "--plan-output",
            str(destination),
        ],
    )
    adapter.main()
    assert read_json(destination) == plan
    assert not Path(plan["output_root"]).exists()
