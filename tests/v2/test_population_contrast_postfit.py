"""R18 full sealed postfit dispatch chain with explicit synthetic CUDA boundary."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from test_population_head_postfit import completed_parent, head_orchestration_fixture
from test_population_postfit import m
from test_population_postfit import recipe as recipe
from test_response_anchor_r15_postfit import binding, write

from gradpert.config import load_experiment_config


def r18_fixture(tmp_path, recipe, parent_spec, parent_binding):
    from test_population_head_postfit import r17_fixture

    spec = r17_fixture(tmp_path, recipe, parent_spec, parent_binding)
    ep = spec["endpoints"][0]
    ep["name"] = "R18-CPD"
    path = Path(ep["config"]["path"])
    value = yaml.safe_load(path.read_text())
    value["model"]["parameters"]["response_anchor_loss"]["value"] = "population_prediction_contrast"
    value["model"]["parameters"]["response_anchor_prediction_head_grad"]["value"] = True
    path.write_text(yaml.safe_dump(value))
    ep["config"] = binding(path)
    config = load_experiment_config(path)
    ep["run_manifest"] = write(
        Path(ep["run_manifest"]["path"]),
        dict(
            source=dict(commit=ep["source"]["commit"], dirty=False),
            config_sha256=m.sha(path),
            resolved_config_sha256=m.digest(config.model_dump(mode="json")),
            data=dict(response_anchor=m.population_descriptor(m.population_options(config))),
        ),
    )
    spec["schema"] = m.R18_SCHEMA
    return spec


def test_actual_r18_cli_dispatch_merge_compare_and_validate(tmp_path, monkeypatch, recipe):
    import test_population_head_postfit as shared

    parent = completed_parent(tmp_path / "parent", monkeypatch, recipe)
    original = shared.r17_fixture

    def candidate(*args):
        # Keep r18_fixture's historical donor construction while overriding dispatcher input.
        with monkeypatch.context() as local:
            local.setattr(shared, "r17_fixture", original)
            return r18_fixture(*args)

    monkeypatch.setattr(shared, "r17_fixture", candidate)
    spec, seal, trace, invoke = head_orchestration_fixture(
        tmp_path / "r18", monkeypatch, recipe, *parent
    )
    assert [x for x in trace if x[0] == "worker"] == [
        ("worker", "R18-CPD", 0),
        ("worker", "R18-CPD", 1),
    ]
    assert (
        json.loads((Path(spec["output"]) / "COMPLETE.json").read_text())["scientific_complete"]
        is False
    )
    invoke(["validate", "--spec", seal["path"], "--spec-sha256", seal["sha256"]])
    m.validate_r18_result(spec, spec_file=seal, server=False)
    # Changing only the schema cannot relabel signed contrast as historical absolute GPM.
    bad = copy.deepcopy(spec)
    bad["schema"] = m.R17_SCHEMA
    bad["endpoints"][0]["name"] = "R17-GPM-HG0"
    with pytest.raises(ValueError):
        m.validate_spec(bad, server=False)


def test_full_r18_medium_spec_preserves_frozen_population_and_strict_identity(
    tmp_path, monkeypatch, recipe
):
    from test_response_anchor_medium import _panel

    import scripts.v2.response_anchor_medium as medium

    parent = completed_parent(tmp_path / "parent", monkeypatch, recipe)
    post = r18_fixture(tmp_path / "candidate", recipe, *parent)
    ep = post["endpoints"][0]
    run = tmp_path / "native-run"
    run.mkdir()
    identity = json.loads(Path(ep["run_manifest"]["path"]).read_text())
    identity["run_id"] = "synthetic-r18"
    config = load_experiment_config(ep["config"]["path"])
    resolved = write(run / "resolved_config.json", config.model_dump(mode="json"))
    manifest = write(run / "run_manifest.json", identity)
    journal = write(run / "fit/epoch_state.json", {})
    panel, _ = _panel()
    conditions = [g["condition_id"] for g in panel["schedule"]] + [f"extra{i}" for i in range(1303)]
    schedule = []
    for i, condition in enumerate(conditions):
        rows = [
            dict(truth_row_id=f"t{i}-{j}", control_row_id=f"c{i}-{j}")
            for j in range(3 if 128 <= i < 151 else 4)
        ]
        schedule.append(
            dict(
                index=i,
                condition_id=condition,
                rows=rows,
                deep_rows=rows[: 3 if i < 6 else 4] if i < 128 else [],
                gradient_rows=rows[: 3 if i < 3 else 4] if i < 32 else [],
                queries=[0, 2, 4],
                view_seed=1,
            )
        )
    parent_data = copy.deepcopy(identity["data"])
    del parent_data["response_anchor"]
    frozen = dict(
        analysis_scope="medium_cap40_stratified",
        schedule=schedule,
        data=parent_data,
        **medium.COUNTS,
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    def source(original):
        return dict(
            root=original["root"],
            commit=original["commit"],
            publication=original["publication"]["path"],
            publication_sha256=original["publication"]["sha256"],
        )

    spec = dict(
        schema=medium.R18_SCHEMA,
        name="R18-CPD",
        analysis_source=source(post["analysis_source"]),
        native_source=source(ep["source"]),
        run_root=str(run),
        run_manifest=manifest,
        resolved_config=resolved,
        config=ep["config"],
        journal=journal,
        checkpoint=dict(role="last", epoch=2, sha256=ep["checkpoint"]["sha256"]),
        data_root=post["data_root"],
        medium=dict(
            parent=write(inputs / "parent.json", frozen),
            performance=write(inputs / "performance.json", {}),
            population_sha256=medium.population(schedule),
            identity_changes=medium.identity_diff(parent_data, identity["data"]),
        ),
        panel=dict(
            population=write(inputs / "panel.json", panel),
            matched_views=write(inputs / "views.json", {}),
        ),
        allowed_axis=dict(count=4775, sha256="a" * 64),
        resource_policy=write(inputs / "policy.json", {}),
        resource_owner=write(inputs / "owner.json", {}),
        output=str(tmp_path / "new-medium"),
        gpu="0",
    )
    bound = medium.validate_spec(spec, server=False)
    assert (
        bound["schedule"] == schedule
        and bound["training_protocol_binding"]["sampling_changed"] is False
    )
    medium.validate_r18_medium_identity(spec, server=False)
    for mutation in ("schema", "name", "allowed", "checkpoint", "condition", "population"):
        bad = copy.deepcopy(spec)
        if mutation == "schema":
            bad["schema"] = medium.R16_SCHEMA
        elif mutation == "name":
            bad["name"] = "R17-GPM-HG0"
        elif mutation == "allowed":
            bad["allowed_axis"]["count"] = 5000
        elif mutation == "checkpoint":
            bad["checkpoint"]["epoch"] = 1
        elif mutation == "condition":
            bad["medium"]["identity_changes"] = []
        else:
            bad["medium"]["population_sha256"] = "f" * 64
        with pytest.raises(ValueError):
            medium.validate_spec(bad, server=False)
    with pytest.raises(ValueError):
        m.validate_r16_identity(
            dict(source=ep["source"], config=ep["config"], run_manifest=ep["run_manifest"]),
            server=False,
        )
