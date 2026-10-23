"""R4/R5/R6 observers preserve exact metadata and the original frozen population."""

import copy
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gradpert.data._io import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round4 as round4  # noqa: E402
import generate_representation_round5 as round5  # noqa: E402
import generate_representation_round6 as round6  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as research  # noqa: E402
from representation_panel import panel_schedule  # noqa: E402
from run_representation_panel import panel_parent_binding  # noqa: E402


@pytest.fixture(params=[round4, round5, round6], ids=["R4-CR", "R5-PROJ", "R6-EMA99"])
def family(request):
    return request.param


@pytest.fixture
def cr_case(tmp_path, family):
    arm = family.ARMS[0]
    config = tmp_path / family.FAMILY / arm / "gradpert_v2/nadig_jurkat.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(family.configs(ROOT)[arm]))
    representation = (
        "student_ssl2_cls_features_before_l2_and_prototypes"
        if family in (round5, round6)
        else "student_response_cls_after_final_rmsnorm_before_projector"
    )
    schedule = []
    for index, condition in enumerate(("p", "q")):
        rows = [
            dict(
                control_row_id=f"c{control}",
                truth_row_id=f"t{index}-{control}",
                condition_id=condition,
                cell_line="Jurkat",
                cell_type="jurkat",
                context_id="jurkat::1",
                batch="1",
            )
            for control in range(4)
        ]
        schedule.append(
            dict(
                index=index,
                condition_id=condition,
                rows=rows,
                deep_rows=copy.deepcopy(rows),
                gradient_rows=copy.deepcopy(rows[:1]),
                queries=[1, 3],
                view_seed=3,
            )
        )
    parent = dict(
        analysis_scope="medium_cap40_stratified",
        data=dict(canonical_sha256="a" * 64, loss_protocol=dict(reduction="row_mean")),
        schedule=schedule,
    )
    identity = dict(
        run_id=arm + "-run", source=dict(commit="c" * 40), data=copy.deepcopy(parent["data"])
    )
    identity["data"]["loss_protocol"].update(
        alignment_uniformity=dict(
            alignment_weight=0.05,
            uniformity_weight=0.0,
            temperature=2.0,
            warmup_completed_optimizer_updates=30,
            representation=representation,
            population_scope="complete_effective_batch_all_microbatches_and_ranks",
            deduplication="control_row_id_and_condition_id_before_rank_partition",
            alignment="same_control_and_condition_two_global_views_both_student_gradients",
            uniformity_pairs="unordered_same_cell_type_batch_context_different_condition",
            reduction="log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
            partial_diagnostics="unavailable_requires_complete_global_population",
        ),
        coding_rate=dict(
            weight=0.00004,
            epsilon_squared=0.05,
            warmup_completed_optimizer_updates=30,
            representation=representation,
            population_scope="complete_effective_batch_all_microbatches_and_ranks_all_contexts",
            deduplication="control_row_id_and_condition_id_before_rank_partition",
            formula="mean_two_globals_negative_half_logdet_I_plus_d_over_N_epsilon_squared_UtU",
            normalization="separate_global_unit_rows_uncentered_fp64_cholesky",
            partial_diagnostics="unavailable_requires_complete_global_population",
        ),
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
        family=family.SCHEMA,
        output=str(tmp_path),
        source=str(ROOT),
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
                    run_id=identity["run_id"],
                    config=str(config),
                    config_sha256=medium.sha(config),
                    data_root=str(tmp_path / "data"),
                ),
            )
        ],
    )
    kwargs = dict(
        arm=arm,
        config=config,
        parent_path=args.parent_manifest,
        parent_sha256=args.parent_manifest_sha256,
    )
    return parent, identity, queue, args, kwargs


def test_cr_binding_preserves_original_parent_and_panel_binding(cr_case):
    parent, identity, queue, args, kwargs = cr_case
    before = copy.deepcopy((parent, identity, queue))
    original_panel = panel_schedule(parent, conditions=2, controls=4)
    bound = medium.cr_parent_binding(parent, identity, **kwargs)
    binding = bound["training_protocol_binding"]
    assert bound["data"] == identity["data"]
    assert bound["schedule"] == parent["schedule"]
    assert panel_schedule(bound, conditions=2, controls=4) == original_panel
    assert binding["changed_fields"] == [
        "data.loss_protocol.alignment_uniformity",
        "data.loss_protocol.coding_rate",
    ]
    assert binding["sampling_changed"] is False
    assert binding["diagnostic_population_sha256"] == medium.population_identity(parent["schedule"])
    assert panel_parent_binding(parent, identity, queue, args) == binding
    assert (parent, identity, queue) == before
    assert args.parent_manifest.read_text() == "original parent stays untouched"
    bound["schedule"][0]["queries"].append(9)
    assert parent == before[0]
    with pytest.raises(ValueError, match="original parent"):
        medium.cr_parent_binding(bound, identity, **kwargs)
    with pytest.raises(ValueError, match="scoped"):
        medium.cr_parent_binding(parent, identity, **dict(kwargs, arm="R3-AUB"))


@pytest.mark.parametrize(
    "path,value",
    [
        (("canonical_sha256",), "wrong"),
        (("loss_protocol", "reduction"), "other"),
        (("loss_protocol", "extra"), True),
        (("loss_protocol", "coding_rate", "weight"), 0.00008),
        (("loss_protocol", "coding_rate", "epsilon_squared"), 0.1),
        (("loss_protocol", "coding_rate", "warmup_completed_optimizer_updates"), 31),
        (("loss_protocol", "coding_rate", "extra"), True),
        (("loss_protocol", "coding_rate", "population_scope"), "per_context"),
        (("loss_protocol", "coding_rate", "normalization"), "centered"),
        (("loss_protocol", "coding_rate", "partial_diagnostics"), "computed"),
        (("loss_protocol", "coding_rate", "representation"), "wrong_space"),
        (("loss_protocol", "alignment_uniformity", "representation"), "wrong_space"),
        (("loss_protocol", "alignment_uniformity", "uniformity_weight"), 0.005),
        (("loss_protocol", "alignment_uniformity", "alignment_weight"), 0.1),
    ],
)
def test_cr_binding_rejects_every_undeclared_data_change(cr_case, path, value):
    parent, identity, queue, args, kwargs = cr_case
    target = identity["data"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="differs beyond"):
        medium.cr_parent_binding(parent, identity, **kwargs)
    with pytest.raises(ValueError):
        panel_parent_binding(parent, identity, queue, args)


@pytest.mark.parametrize("protocol", ["alignment_uniformity", "coding_rate"])
def test_cr_binding_requires_complete_exact_protocol(cr_case, protocol):
    parent, identity, _, _, kwargs = cr_case
    identity["data"]["loss_protocol"][protocol].pop("deduplication")
    with pytest.raises(ValueError, match="differs beyond"):
        medium.cr_parent_binding(parent, identity, **kwargs)


@pytest.mark.parametrize(
    "drift", ["family", "run", "root", "config", "hash", "parent", "source", "duplicate"]
)
def test_cr_panel_rejects_queue_endpoint_drift(cr_case, drift):
    parent, identity, queue, args, _ = cr_case
    plan = queue["rows"][0]["plan"]
    if drift == "family":
        queue["family"] = medium.RS3_SCHEMA
    elif drift in ("run", "root", "config", "hash"):
        field = dict(run="run_id", root="run_root", config="config", hash="config_sha256")[drift]
        plan[field] += "wrong"
    elif drift == "parent":
        queue["medium_analysis"]["parent_manifest_sha256"] = "wrong"
    elif drift == "source":
        queue["source_commit"] = "wrong"
    else:
        queue["rows"].append(copy.deepcopy(queue["rows"][0]))
    with pytest.raises(ValueError):
        panel_parent_binding(parent, identity, queue, args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("au_alignment_weight", 0.1),
        ("au_uniformity_weight", 0.005),
        ("au_temperature", 3.0),
        ("au_warmup_steps", 31),
        ("cr_weight", 0.00008),
        ("cr_warmup_steps", 31),
        ("cr_epsilon_squared", 0.1),
        ("population_aux_space", "opposite_space"),
    ],
)
def test_cr_binding_rejects_unapproved_config(cr_case, tmp_path, field, value):
    parent, identity, _, _, kwargs = cr_case
    raw = yaml.safe_load(kwargs["config"].read_text())
    if value == "opposite_space":
        value = (
            "response_cls" if kwargs["arm"] in ("R5-PROJ", "R6-EMA99") else "ssl2_cls_bottleneck"
        )
    raw["model"]["parameters"].setdefault(
        field, dict(source="project_preregistered", reference="test:unapproved-space")
    )["value"] = value
    target = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    target.parent.mkdir()
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"frozen arm|representation space|epsilon_squared"):
        medium.cr_parent_binding(parent, identity, **dict(kwargs, config=target))


@pytest.mark.parametrize("population_drift", [False, True])
def test_cr_medium_execute_binds_before_native_seal(cr_case, monkeypatch, population_drift):
    parent, identity, queue, args, _ = cr_case
    row = queue["rows"][0]
    atomic_json(args.run_root / "run_manifest.json", identity)
    queue_path = Path(queue["output"]) / "queue.json"
    atomic_json(queue_path, queue)
    before = copy.deepcopy(parent)
    monkeypatch.setattr(medium, "SERVER_ROOT", args.run_root.parent)
    monkeypatch.setattr(runner, "verify", lambda q: None)
    monkeypatch.setattr(runner, "validate_training_contract", lambda p: None)
    monkeypatch.setattr(medium, "validate_analysis_spec", lambda spec: parent)
    monkeypatch.setattr(medium, "select_checkpoint", lambda *a: (dict(sha256="d" * 64), {}))

    def run(command, **kwargs):
        assert "seal-medium" in command and "--reuse-medium-schedule" in command
        assert command[command.index("--method-sha") + 1] == args.method_sha
        assert kwargs["env"]["PYTHONPATH"] == str(ROOT / "src")
        bound_path = Path(command[command.index("--parent-manifest") + 1])
        assert medium.sha(bound_path) == command[command.index("--parent-manifest-sha256") + 1]
        bound = read_json(bound_path)
        assert bound["data"] == identity["data"] and bound["schedule"] == before["schedule"]
        if not population_drift:
            raise RuntimeError("verified CR native seal dispatch")
        bound["schedule"][0]["rows"][0]["truth_row_id"] = "wrong"
        bound.update(
            method_sha=args.method_sha, view_recipe_config_sha256=row["plan"]["config_sha256"]
        )
        atomic_json(Path(command[command.index("--output") + 1]) / "manifest.json", bound)

    monkeypatch.setattr(medium.subprocess, "run", run)
    error, message = (
        (ValueError, "resealing changed the common diagnostic population")
        if population_drift
        else (RuntimeError, "verified CR native seal dispatch")
    )
    with pytest.raises(error, match=message):
        medium.execute(queue_path, row["name"])
    assert parent == before
    root = Path(queue["output"]) / (row["name"] + "-medium")
    assert read_json(root / "FAILURE.json")["status"] == "failed"


def test_cr_report_keeps_missing_population_terms_na(tmp_path, family):
    arm = family.ARMS[0]
    root, run = tmp_path / "medium", tmp_path / "run"
    row = dict(name=arm, plan=dict(run_root=str(run), config_sha256="c" * 64))
    queue = dict(output=str(tmp_path), source_commit="a" * 40, family=family.SCHEMA)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    unavailable = {
        key: "requires_complete_global_population" for key in ("au_alignment", "cr_negative")
    }
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p",
                joint=2,
                loss_terms=dict(prediction=1, ssl2_dino=1),
                diagnostic_objective_scope=dict(
                    unavailable_terms=unavailable,
                    joint_scope="prediction_and_ssl_excluding_unavailable_representation_terms",
                ),
            )
        ],
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(run / "fit/history.json", [dict(epoch=1), dict(epoch=2)])
    atomic_json(
        research.validation_path(queue, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    research.render_medium_report(queue, row, root)
    loss = next(csv.DictReader((root / "report/frozen_losses.csv").open()))
    for key, reason in unavailable.items():
        assert loss[key] == "" and loss["unavailable_terms." + key] == reason
    assert loss["joint"] == "2"
    report = (root / "report/REPORT_CN.md").read_text()
    assert ".00004" in report
    if family is round6:
        assert "恒定.99" in report and "A/CR仍为NA" in report
    elif family is round5:
        assert "256维瓶颈" in report and "A/CR均为NA" in report
    else:
        assert "CR权重=.00004" in report and "A与CR均记NA" in report


def test_r6_parent_rejects_nonconstant_ema(cr_case, tmp_path):
    parent, identity, _, _, kwargs = cr_case
    if kwargs["arm"] != "R6-EMA99":
        return
    raw = yaml.safe_load(kwargs["config"].read_text())
    raw["model"]["parameters"]["teacher_end"]["value"] = 1.0
    target = tmp_path / "changed/gradpert_v2/nadig_jurkat.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"constant \.99"):
        medium.cr_parent_binding(parent, identity, **dict(kwargs, config=target))
