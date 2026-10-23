"""Strict AU panel identity binding and temporary observer assembly options."""

import copy
import dataclasses
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from cap40_posthoc import sha  # noqa: E402
from generate_representation_round3 import SCHEMA  # noqa: E402
from representation_panel import assemble_control_only, panel_schedule  # noqa: E402
from run_representation_panel import panel_parent_binding  # noqa: E402


def case(tmp_path, arm):
    config = ROOT / f"configs/v2/representation_round3_20261023/{arm}/gradpert_v2/nadig_jurkat.yaml"
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    parent = dict(
        analysis_scope="medium_cap40_stratified",
        data=dict(canonical_sha256="a" * 64, loss_protocol={}),
        schedule=[],
    )
    for index, condition in enumerate(("p", "q")):
        rows = [
            dict(
                condition_id=condition,
                control_row_id=f"c{i}",
                truth_row_id=f"t{index}{i}",
                cell_line="j",
                cell_type="j",
                context_id="x",
                batch="b",
            )
            for i in range(2)
        ]
        parent["schedule"].append(
            dict(
                index=index,
                condition_id=condition,
                rows=rows,
                deep_rows=[],
                gradient_rows=[],
                queries=[1, 2],
                view_seed=3,
            )
        )
    identity = dict(run_id="run", source=dict(commit="c" * 40), data=copy.deepcopy(parent["data"]))
    identity["data"]["loss_protocol"]["alignment_uniformity"] = dict(
        alignment_weight=options.au_alignment_weight,
        uniformity_weight=options.au_uniformity_weight,
        temperature=2.0,
        warmup_completed_optimizer_updates=30,
        representation="student_response_cls_after_final_rmsnorm_before_projector",
        population_scope="complete_effective_batch_all_microbatches_and_ranks",
        deduplication="control_row_id_and_condition_id_before_rank_partition",
        alignment="same_control_and_condition_two_global_views_both_student_gradients",
        uniformity_pairs="unordered_same_cell_type_batch_context_different_condition",
        reduction="log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
        partial_diagnostics="unavailable_requires_complete_global_population",
    )
    args = SimpleNamespace(
        config=config,
        run_root=tmp_path / "run",
        method_sha="c" * 40,
        parent_manifest=tmp_path / "original.json",
        parent_manifest_sha256="b" * 64,
    )
    args.parent_manifest.write_text("original parent stays untouched")
    queue = dict(
        family=SCHEMA,
        source_commit=args.method_sha,
        medium_analysis=dict(
            parent_manifest=str(args.parent_manifest),
            parent_manifest_sha256=args.parent_manifest_sha256,
        ),
        rows=[
            dict(
                name=arm,
                plan=dict(
                    run_root=str(args.run_root),
                    run_id="run",
                    config=str(config),
                    config_sha256=sha(config),
                ),
            )
        ],
    )
    return parent, identity, queue, args, options


@pytest.mark.parametrize("arm", ["R3-AUB", "R3-AUS"])
def test_au_panel_binding_preserves_parent_and_exact_panel(tmp_path, arm):
    parent, identity, queue, args, _ = case(tmp_path, arm)
    before = copy.deepcopy((parent, identity, queue))
    panel = panel_schedule(parent, conditions=2, controls=2)
    binding = panel_parent_binding(parent, identity, queue, args)
    assert (parent, identity, queue) == before
    assert panel_schedule(parent, conditions=2, controls=2) == panel
    assert binding["original_parent_manifest"] == str(args.parent_manifest)
    assert binding["original_parent_sha256"] == args.parent_manifest_sha256
    assert binding["sampling_changed"] is False
    assert args.parent_manifest.read_text() == "original parent stays untouched"


@pytest.mark.parametrize(
    "drift",
    [
        "data",
        "coefficient",
        "run",
        "root",
        "config",
        "hash",
        "arm",
        "family",
        "parent",
        "duplicate",
        "source",
    ],
)
def test_au_panel_binding_rejects_identity_drift(tmp_path, drift):
    parent, identity, queue, args, _ = case(tmp_path, "R3-AUB")
    plan = queue["rows"][0]["plan"]
    if drift == "data":
        identity["data"]["canonical_sha256"] = "wrong"
    elif drift == "coefficient":
        identity["data"]["loss_protocol"]["alignment_uniformity"]["alignment_weight"] = 9
    elif drift == "run":
        plan["run_id"] = "other"
    elif drift == "root":
        plan["run_root"] += "other"
    elif drift == "config":
        plan["config"] += "other"
    elif drift == "hash":
        plan["config_sha256"] = "wrong"
    elif drift == "arm":
        queue["rows"][0]["name"] = "R3-AUS"
    elif drift == "family":
        queue["family"] = "other"
    elif drift == "parent":
        queue["medium_analysis"]["parent_manifest_sha256"] = "wrong"
    elif drift == "duplicate":
        queue["rows"].append(copy.deepcopy(queue["rows"][0]))
    else:
        queue["source_commit"] = "wrong"
    with pytest.raises(ValueError):
        panel_parent_binding(parent, identity, queue, args)


def test_non_au_still_requires_exact_data(tmp_path):
    parent, identity, queue, args, _ = case(tmp_path, "R3-AUB")
    queue["rows"][0]["name"] = "R3-RI"
    with pytest.raises(ValueError):
        panel_parent_binding(parent, identity, queue, args)
    identity["data"] = copy.deepcopy(parent["data"])
    assert panel_parent_binding(parent, identity, queue, args) is None


@pytest.mark.parametrize("arm", ["R3-AUB", "R3-AUS", "historical"])
@pytest.mark.parametrize("deep", [False, True])
def test_control_assembly_only_changes_transient_options(tmp_path, monkeypatch, arm, deep):
    if arm == "historical":

        @dataclasses.dataclass(frozen=True)
        class HistoricalOptions:
            lambda1: float = 0.5
            lambda2: float = 0.5

        options = HistoricalOptions()
    else:
        *_, options = case(tmp_path, arm)
    before = copy.deepcopy(options)
    data = SimpleNamespace(
        _row_index={"c": 0},
        anchors_by_condition={"p": (1,)},
        _read_expression_indices=lambda indices: np.zeros((len(indices), 2), dtype=np.float32),
    )
    runtime = SimpleNamespace(options=options, data=data, index=object())
    views = ModuleType("gradpert.training.v2.views")
    views.assemble_batch = lambda raw, index, options, rng, **kwargs: (raw, options)
    monkeypatch.setitem(sys.modules, views.__name__, views)
    raw, actual = assemble_control_only(
        runtime,
        [dict(control_row_id="c", truth_row_id="unread", condition_id="p")],
        [1],
        3,
        deep=deep,
    )
    changes = dict(lambda1=float(deep), lambda2=float(deep))
    if not deep and arm != "historical":
        changes.update(au_alignment_weight=0.0, au_uniformity_weight=0.0)
    assert actual == dataclasses.replace(options, **changes)
    assert runtime.options is options and options == before
    assert raw.target_expression.count_nonzero() == 0


def test_panel_binding_rejects_unknown_native_config_coefficients(tmp_path):
    import yaml

    parent, identity, queue, args, _ = case(tmp_path, "R3-AUB")
    raw = yaml.safe_load(args.config.read_text())
    raw["model"]["parameters"]["au_alignment_weight"]["value"] = 0.2
    args.config = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    args.config.parent.mkdir()
    args.config.write_text(yaml.safe_dump(raw))
    queue["rows"][0]["plan"].update(config=str(args.config), config_sha256=sha(args.config))
    with pytest.raises(ValueError, match="frozen arm"):
        panel_parent_binding(parent, identity, queue, args)
