"""Frozen precision diagnostic populations and arithmetic labels; no native materialization."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
from test_precision_postfit import CHANGES, ROOT, bound, m, precision_fixture, write
from test_response_anchor_medium import _panel, _runtime

from gradpert.config import load_experiment_config

SCRIPT = ROOT / "scripts/v2/response_anchor_medium.py"
loader = importlib.util.spec_from_file_location("precision_medium_test", SCRIPT)
medium = importlib.util.module_from_spec(loader)
loader.loader.exec_module(medium)


def medium_fixture(tmp_path, arm="PREC-SSL32"):
    val, queue = precision_fixture(tmp_path, arm)
    endpoint = val["endpoints"][0]
    panel, _ = _panel()
    schedule = []
    for i in range(1335):
        name = panel["schedule"][i]["condition_id"] if i < 32 else f"p{i}"
        count = 4 if i < 1312 else 3
        rows = [dict(truth_row_id=f"t{i}_{j}", control_row_id=f"c{i}_{j}") for j in range(count)]
        schedule.append(
            dict(
                index=i,
                condition_id=name,
                rows=rows,
                deep_rows=rows[: 4 if i < 122 else 3] if i < 128 else [],
                gradient_rows=rows[: 4 if i < 29 else 3] if i < 32 else [],
                queries=[0, 2, 4],
                view_seed=i,
            )
        )
    parent = dict(
        analysis_scope="medium_cap40_stratified",
        schedule=schedule,
        data=dict(run_seed=1),
        **medium.COUNTS,
    )
    parent_file = write(tmp_path / "population/parent.json", parent)
    panel_file = write(tmp_path / "population/panel.json", panel)
    matched_file = write(tmp_path / "population/matched.json", [])
    performance = write(tmp_path / "population/performance.json", {})

    def source_binding(source):
        return dict(
            root=source["root"],
            commit=source["commit"],
            publication=source["publication"]["path"],
            publication_sha256=source["publication"]["sha256"],
        )

    policy = dict.fromkeys(
        ("ffn_compute_precision", "ssl_head_precision", "prediction_precision"), "autocast"
    )
    policy.update(CHANGES[arm])
    spec = dict(
        schema=medium.PRECISION_SCHEMA,
        name=arm,
        analysis_source=source_binding(val["analysis_source"]),
        native_source=source_binding(endpoint["source"]),
        run_root=endpoint["run_root"],
        run_manifest=endpoint["run_manifest"],
        resolved_config=bound(Path(endpoint["run_root"]) / "resolved_config.json"),
        config=endpoint["config"],
        journal=endpoint["journal"],
        checkpoint=dict(role="last", epoch=2, sha256=endpoint["checkpoint"]["sha256"]),
        data_root=val["data_root"],
        medium=dict(
            parent=parent_file,
            performance=performance,
            population_sha256=medium.population(schedule),
            identity_changes=[],
        ),
        panel=dict(population=panel_file, matched_views=matched_file),
        allowed_axis=dict(count=4775, sha256=medium.digest(list(range(4775)))),
        resource_policy=val["shared_policy"],
        resource_owner=write(tmp_path / "owner.json", {}),
        output=str(tmp_path / "medium-output"),
        gpu="0",
        precision=val["precision"],
        diagnostic_precision=dict(
            endpoint="as_trained_medium_bf16_context",
            outer_autocast_dtype="bfloat16",
            student=policy,
            teacher=policy,
        ),
    )
    # Original RI SPEC is metadata only here, not executable native evidence.
    reference = copy.deepcopy(spec)
    for field in ("precision", "diagnostic_precision"):
        del reference[field]
    reference.update(schema=medium.SCHEMA, name="R3-RI", output=str(tmp_path / "retained-medium"))
    old_source, old_run = tmp_path / "ri-source", tmp_path / "ri-run"
    baseline = old_source / "configs/gradpert_v2/nadig_jurkat.yaml"
    baseline.parent.mkdir(parents=True)
    from generate_precision_family import BASELINE

    baseline.write_bytes((ROOT / BASELINE).read_bytes())
    resolved = load_experiment_config(baseline).model_dump(mode="json")
    reference.update(
        native_source=dict(
            root=str(old_source),
            commit="f0cb20ebb2efab749970f2bf5d205ed2f358592c",
            publication=str(old_source / "publication.json"),
            publication_sha256="a" * 64,
        ),
        run_root=str(old_run),
        config=bound(baseline),
        run_manifest=write(
            old_run / "run_manifest.json",
            dict(
                run_id="retained-ri-synthetic",
                source=dict(commit="f0cb20ebb2efab749970f2bf5d205ed2f358592c", dirty=False),
                data=dict(run_seed=1),
                config_sha256=m.sha(baseline),
                resolved_config_sha256=m.digest(resolved),
            ),
        ),
        resolved_config=write(old_run / "resolved_config.json", resolved),
        journal=write(old_run / "fit/epoch_state.json", {}),
        checkpoint=dict(
            role="last",
            epoch=2,
            sha256="96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4",
        ),
    )
    spec["population_reference"] = write(tmp_path / "retained-RI-SPEC.json", reference)
    return spec, queue


@pytest.mark.parametrize("arm", m.PRECISION_ARMS)
def test_three_precision_profiles_keep_exact_parent_and_no_anchor(tmp_path, arm):
    spec, _ = medium_fixture(tmp_path, arm)
    bound_parent = medium.validate_spec(spec, server=False)
    assert bound_parent["training_protocol_binding"]["changed_fields"] == []
    assert not bound_parent["training_protocol_binding"]["sampling_changed"]
    assert spec["diagnostic_precision"]["outer_autocast_dtype"] == "bfloat16"
    metadata = medium.precision_metadata(spec)
    medium.validate_precision_metadata(spec, metadata)
    metadata["diagnostic_precision"]["student"]["ssl_head_precision"] = "wrong"
    assert spec["diagnostic_precision"]["student"]["ssl_head_precision"] != "wrong"
    with pytest.raises(ValueError, match="arithmetic"):
        medium.validate_precision_metadata(spec, metadata)


@pytest.mark.parametrize(
    "case",
    [
        "arm",
        "schema",
        "precision",
        "teacher",
        "role",
        "cp",
        "axis",
        "rows",
        "panel",
        "parent",
        "matched",
        "protocol",
        "resource",
        "reference",
    ],
)
def test_precision_medium_rejects_resampling_relabeling_and_own_policy(tmp_path, case):
    spec, _ = medium_fixture(tmp_path)
    if case == "arm":
        spec["name"] = "R3-RI"
    elif case == "schema":
        spec["schema"] = medium.R18_SCHEMA
    elif case == "precision":
        spec["diagnostic_precision"]["outer_autocast_dtype"] = "float32"
    elif case == "teacher":
        spec["diagnostic_precision"]["teacher"]["ssl_head_precision"] = "autocast"
    elif case == "role":
        spec["checkpoint"]["role"] = "best"
    elif case == "cp":
        spec["checkpoint"]["sha256"] = "b" * 64
    elif case == "axis":
        spec["allowed_axis"]["sha256"] = "b" * 64
    elif case in ("rows", "parent"):
        parent = medium.read(spec["medium"]["parent"]["path"])
        if case == "rows":
            parent["schedule"][0]["rows"].pop()
        else:
            parent["schedule"][0]["view_seed"] += 1
            spec["medium"]["population_sha256"] = medium.population(parent["schedule"])
        spec["medium"]["parent"] = write(
            Path(spec["medium"]["parent"]["path"]).with_name("changed.json"), parent
        )
    elif case == "panel":
        panel = medium.read(spec["panel"]["population"]["path"])
        panel["schedule"][0]["rows"].reverse()
        panel["panel_sha256"] = medium.digest(panel["schedule"])
        spec["panel"]["population"] = write(
            Path(spec["panel"]["population"]["path"]).with_name("changed.json"), panel
        )
    elif case == "matched":
        spec["panel"]["matched_views"] = write(
            tmp_path / "changed-matched.json", [{"changed": True}]
        )
    elif case == "protocol":
        spec["medium"]["identity_changes"] = [{"path": "/loss_protocol/changed"}]
    elif case == "resource":
        spec["resource_policy"] = write(
            tmp_path / "changed-policy.json", {"max_own_memory_fraction": 1.0}
        )
    elif case == "reference":
        old = medium.read(spec["population_reference"]["path"])
        old["name"] = "R14-FRGA"
        spec["population_reference"] = write(tmp_path / "wrong-reference.json", old)
    with pytest.raises((ValueError, KeyError)):
        medium.validate_spec(spec, server=False)


@pytest.mark.parametrize("location", ["output", "panel", "other_test"])
def test_precision_medium_rejects_pkl_anywhere_without_deleting(tmp_path, location):
    spec, queue = medium_fixture(tmp_path)
    root = Path(spec["output"]) / "panel" if location == "panel" else Path(spec["output"])
    if location == "other_test":
        root = Path(queue["rows"][1]["plan"]["test_root"])
    polluted = root / "nested/untouched.pkl"
    polluted.parent.mkdir(parents=True)
    polluted.write_bytes(b"keep")
    with pytest.raises(ValueError, match="zero-PKL"):
        medium.validate_spec(spec, server=False)
    assert polluted.read_bytes() == b"keep"


def test_precision_uses_existing_allowed_first_guard_without_broadening_reads():
    runtime, records, values = _runtime()
    with medium.allowed_train_reads(runtime, records, controls_only=True, receipt={}):
        exposed = runtime.data._read_expression_indices([4, 5])
        assert exposed[:, [0, 2, 4]].tolist() == values[[4, 5]][:, [0, 2, 4]].tolist()
        assert exposed[:, [1, 3]].tolist() == [[0.0, 0.0], [0.0, 0.0]]
        with pytest.raises(ValueError, match="whitelist"):
            runtime.data._read_expression_indices([0])
    assert medium.precision_metadata(dict(schema=medium.SCHEMA)) == {}


def protected_runtime_fixture():
    from contextlib import contextmanager
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np
    import torch
    from test_components import fixture

    from gradpert.modeling.v2 import GraDPertV2
    from gradpert.training.v2.objective import JointObjective

    model, _ = fixture()
    objective = JointObjective(
        GraDPertV2(
            model.graph.embedding.weight.detach(),
            replace(model.options, ssl_head_precision="float32"),
        )
    ).eval()
    objective.register_buffer("private_clock", torch.zeros(()))
    runtime, records, _ = _runtime()
    runtime.objective = objective
    runtime.optimizer = SimpleNamespace(state_dict=lambda: {})
    runtime.generator = np.random.default_rng(1)
    runtime.device = torch.device("cpu")

    @contextmanager
    def prepare(**kwargs):
        yield runtime

    config = SimpleNamespace(model=SimpleNamespace(parameters={}))
    cap = SimpleNamespace(runtime_args=lambda _: (torch, config, {}, prepare, None))
    return runtime, records, cap


@pytest.mark.parametrize("worker", ["medium", "panel"])
@pytest.mark.parametrize(
    "mutation", ["temperature", "private_buffer", "completion_temperature", "completion_buffer"]
)
def test_precision_adapter_rejects_direct_native_and_completion_mutations(worker, mutation):
    import capacity_probe

    runtime, records, cap = protected_runtime_fixture()
    original_snapshot, original_args = capacity_probe.validation_state, cap.runtime_args
    legacy_before = original_snapshot(runtime)
    from types import SimpleNamespace

    panel = (
        SimpleNamespace(
            runtime_args=cap.runtime_args, panel_parent_binding=object(), panel_schedule=object()
        )
        if worker == "panel"
        else None
    )
    selected = panel if panel is not None else cap
    with (
        pytest.raises(ValueError, match="mutated protected objective"),
        medium.diagnostic_adapter(
            cap,
            panel,
            records,
            controls_only=worker == "panel",
            read_receipts=[],
            protect_objective=True,
        ),
    ):
        # The unchanged native workers import this local function at call entry.
        from capacity_probe import validation_state

        with selected.runtime_args(None)[3](purpose="evaluation"):
            validation_state(runtime)  # Native post-checkpoint baseline.
            if mutation.startswith("completion"):
                validation_state(runtime)  # Native completed snapshot, before context exit.
            if "temperature" in mutation:
                runtime.objective._temperature_updates += 1
            else:
                runtime.objective.private_clock.add_(1)
            if not mutation.startswith("completion"):
                validation_state(runtime)
    assert legacy_before == original_snapshot(runtime)  # Legacy snapshot missed this change.
    assert capacity_probe.validation_state is original_snapshot
    assert cap.runtime_args is original_args
    assert not runtime.data._adata.calls  # Protection itself never reads expression.


def test_precision_runtime_proof_uses_post_restore_baseline_and_keeps_autocast_rng_and_scope():
    import capacity_probe
    import torch

    runtime, records, cap = protected_runtime_fixture()
    other, _, _ = protected_runtime_fixture()
    original_snapshot, original_args = capacity_probe.validation_state, cap.runtime_args
    rng = torch.get_rng_state()
    reads = []
    with medium.diagnostic_adapter(
        cap, None, records, controls_only=False, read_receipts=reads, protect_objective=True
    ):
        from capacity_probe import validation_state

        with cap.runtime_args(None)[3](purpose="evaluation"):
            # Simulate strict restore before the native baseline; never compare against
            # fresh initialization as though checkpoint loading were a forbidden update.
            runtime.objective._temperature_updates = 17
            assert "objective_buffers" not in validation_state(other)
            before = validation_state(runtime)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert torch.is_autocast_enabled("cpu")
                assert before == validation_state(runtime)
                assert torch.is_autocast_enabled("cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert (
        capacity_probe.validation_state is original_snapshot and cap.runtime_args is original_args
    )
    proof = reads[0]["state_protection"]
    assert proof["native_snapshot_calls"] == 2 and proof["state_before"] == proof["state_after"]
    native = dict(state_before=before, state_after=before)
    medium.validate_precision_protection(
        dict(reads=reads, state_protection=medium.PRECISION_STATE_PROTECTION),
        expected_reads=1,
        native=native,
    )
    assert runtime.objective._temperature_updates == 17


def test_legacy_adapter_does_not_patch_snapshot_or_add_precision_receipt_fields():
    import capacity_probe

    runtime, records, cap = protected_runtime_fixture()
    original = capacity_probe.validation_state
    reads = []
    with (
        medium.diagnostic_adapter(cap, None, records, controls_only=False, read_receipts=reads),
        cap.runtime_args(None)[3](purpose="evaluation"),
    ):
        assert capacity_probe.validation_state is original
        runtime.objective._temperature_updates += 1
    assert "state_protection" not in reads[0]
    assert capacity_probe.validation_state is original


def test_precision_adapter_requires_native_snapshots_and_restores_after_missing_proof():
    import capacity_probe

    _, records, cap = protected_runtime_fixture()
    original = capacity_probe.validation_state
    with (
        pytest.raises(ValueError, match="requires native before/after"),
        medium.diagnostic_adapter(
            cap, None, records, controls_only=False, read_receipts=[], protect_objective=True
        ),
        cap.runtime_args(None)[3](purpose="evaluation"),
    ):
        pass
    assert capacity_probe.validation_state is original


@pytest.mark.parametrize(
    "mutation", ["legacy", "missing", "calls", "changed", "empty", "native", "bad_hash"]
)
def test_precision_phase_and_native_completion_require_expanded_matching_protection(mutation):
    import copy

    state = dict(
        student="a" * 64,
        teacher="b" * 64,
        objective_state="c" * 64,
        objective_buffers="d" * 64,
        objective_scalar_clocks="e" * 64,
    )
    proof = dict(
        policy=medium.PRECISION_STATE_PROTECTION,
        native_snapshot_calls=2,
        state_before=state,
        state_after=copy.deepcopy(state),
        state_unchanged=True,
    )
    receipt = dict(
        state_protection=medium.PRECISION_STATE_PROTECTION, reads=[dict(state_protection=proof)]
    )
    native = dict(state_before=copy.deepcopy(state), state_after=copy.deepcopy(state))
    if mutation == "legacy":
        del proof["state_before"]["objective_buffers"]
        del proof["state_after"]["objective_buffers"]
    elif mutation == "missing":
        del receipt["reads"][0]["state_protection"]
    elif mutation == "calls":
        proof["native_snapshot_calls"] = 1
    elif mutation == "changed":
        proof["state_after"]["objective_scalar_clocks"] = "f" * 64
    elif mutation == "empty":
        proof["state_before"] = proof["state_after"] = {}
    elif mutation == "native":
        native["state_after"]["objective_buffers"] = "f" * 64
    else:
        proof["state_before"]["objective_state"] = proof["state_after"]["objective_state"] = (
            "invalid"
        )
    with pytest.raises(ValueError):
        medium.validate_precision_protection(receipt, expected_reads=1, native=native)
