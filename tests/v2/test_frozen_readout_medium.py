"""Actual frozen medium/panel assembly retains the parent diagnostic recipes."""

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_relay_method import small_index

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.training.v2.views import assemble_batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from cap40_posthoc import (  # noqa: E402
    Collector,
    assemble,
    frozen_readout_scope,
    make_raw,
    recipe,
)
from diagnostic_view_options import diagnostic_view_options  # noqa: E402
from representation_panel import assemble_control_only  # noqa: E402


def runtime(*, independent=True):
    path = (
        ROOT
        / "configs/v2/representation_round11_20261025"
        / "R11-READOUT/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(load_experiment_config(path).model.parameters)
    options = replace(
        options,
        query_count=6,
        independent_view_rng=independent,
        mask_probability=1.0,
        graph_mask_ratio=0.5,
        graph_edge_dropout=0.2,
    )
    values = np.arange(8 * 16, dtype=np.float32).reshape(8, 16) / 128
    ids = [f"t{i}" for i in range(4)] + [f"c{i}" for i in range(4)]
    reads = []

    def read(indices):
        reads.extend(indices)
        return values[indices].copy()

    data = SimpleNamespace(
        _row_index={name: i for i, name in enumerate(ids)},
        _read_expression_indices=read,
        anchors_by_condition={"g0": (0,), "g1": (1,)},
        manifest=SimpleNamespace(n_expression_genes=16),
    )
    records = [
        dict(
            condition_id=f"g{i % 2}",
            truth_row_id=f"t{i}",
            control_row_id=f"c{i}",
            context_id="T::b1",
            cell_type="T",
            cell_line="Jurkat",
            batch="b1",
        )
        for i in range(4)
    ]
    return SimpleNamespace(options=options, data=data, index=small_index(16)), records, reads


def test_actual_medium_frozen_deep_assembly_keeps_guard():
    rt, records, reads = runtime()
    original = copy.deepcopy(rt.options)
    with pytest.raises(ValueError, match="MSE-only"):
        replace(rt.options, lambda1=1.0, lambda2=1.0)
    batch = assemble(rt, records, [0, 2, 4, 6, 8, 10], 101, deep=True, dummy=True)
    assert len(batch.graph_views) == 2 + rt.options.ssl1_local_views
    assert len(batch.cell_views) == 2 + rt.options.ssl2_local_views
    assert rt.options == original and reads == []
    assert rt.options.training_mode == "frozen_readout"
    assert rt.options.lambda1 == rt.options.lambda2 == 0.0


def parent_options(options):
    path = (
        ROOT
        / "configs/v2/representation_round9_20261024"
        / "R9-SSL1J2/gradpert_v2/nadig_jurkat.yaml"
    )
    _, parent = V2Options.parse_parameters(load_experiment_config(path).model.parameters)
    return replace(
        parent,
        query_count=options.query_count,
        independent_view_rng=options.independent_view_rng,
        mask_probability=options.mask_probability,
        graph_mask_ratio=options.graph_mask_ratio,
        graph_edge_dropout=options.graph_edge_dropout,
        # Historical medium activates both view families regardless of stage.
        lambda1=1.0,
        lambda2=1.0,
    )


@pytest.mark.parametrize("seed", [0, 101, 20261021])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("deep", [False, True])
def test_actual_medium_and_panel_match_parent_native_recipe(seed, independent, deep):
    rt, records, reads = runtime(independent=independent)
    original = copy.deepcopy(rt.options)
    queries = [0, 2, 4, 6, 8, 10]
    raw = make_raw(rt, records)
    reads.clear()
    expected_options = replace(
        parent_options(rt.options),
        lambda1=float(deep),
        lambda2=float(deep),
        **(
            {"au_alignment_weight": 0.0, "au_uniformity_weight": 0.0, "cr_weight": 0.0}
            if not deep
            else {}
        ),
    )
    expected = assemble_batch(
        raw,
        small_index(16),
        expected_options,
        np.random.default_rng(seed),
        allowed_expression_ids=np.array(queries),
    )
    actual = assemble(rt, records, queries, seed, deep=deep)
    assert reads == [4, 5, 6, 7, 0, 1, 2, 3]
    reads.clear()
    panel = assemble_control_only(rt, records, queries, seed, deep=deep)
    assert reads == [4, 5, 6, 7]  # No perturbed expression is read by the panel.
    assert recipe(actual) == recipe(panel) == recipe(expected)
    torch.testing.assert_close(actual.control, expected.control, rtol=0, atol=0)
    torch.testing.assert_close(actual.truth, expected.truth, rtol=0, atol=0)
    torch.testing.assert_close(panel.control, expected.control, rtol=0, atol=0)
    assert not panel.truth.any()
    assert rt.options == original
    assert bool(actual.graph_views) == bool(actual.cell_views) == deep
    if deep:
        assert len(actual.graph_views) == 2 + original.ssl1_local_views
        assert len(actual.cell_views) == 2 + original.ssl2_local_views
        assert any(v.mask.any() for v in actual.cell_views[:2])


@pytest.mark.parametrize("deep", [False, True])
def test_sampling_clone_is_valid_but_cannot_relabel_the_frozen_training_contract(deep):
    rt, _, _ = runtime()
    sample = diagnostic_view_options(rt.options, deep=deep)
    assert isinstance(sample, V2Options) and sample is not rt.options
    assert replace(sample) == sample  # Run its native constructor/guards again.
    assert rt.options.training_mode == "frozen_readout"
    assert rt.options.readout_parent_epoch == 3 and rt.options.readout_parent_run_id
    if deep:
        assert sample.training_mode == "joint" and sample.readout_parent_epoch == 0
        assert not sample.readout_parent_run_id
        with pytest.raises(ValueError, match="MSE-only"):
            replace(sample, training_mode="frozen_readout")
    else:
        assert sample == rt.options
    with pytest.raises(ValueError, match="MSE-only"):
        replace(rt.options, lambda1=1.0, lambda2=1.0)


@pytest.mark.parametrize("deep", [False, True])
def test_historical_joint_sampling_options_are_exactly_unchanged(deep):
    rt, _, _ = runtime()
    parent = parent_options(rt.options)
    # Cover the prior shallow AU/CR correction and its active deep path too.
    for options in (
        parent,
        replace(parent, au_alignment_weight=0.3, au_uniformity_weight=0.2, cr_weight=0.1),
    ):
        changes = dict(lambda1=float(deep), lambda2=float(deep))
        if not deep:
            changes.update(au_alignment_weight=0.0, au_uniformity_weight=0.0, cr_weight=0.0)
        assert diagnostic_view_options(options, deep=deep) == replace(options, **changes)
    assert frozen_readout_scope(SimpleNamespace(training_stage="joint")) == {}


def test_real_frozen_forward_observes_student_views_without_teacher_or_ssl_updates():
    from test_frozen_readout import fixture
    from test_staged_joint_training import assert_nested_equal

    from gradpert.training.v2.diagnostic_objective import diagnostic_objective
    from gradpert.training.v2.readout import frozen_digest, frozen_state_proof

    torch.set_num_threads(1)
    rt, records, _ = runtime()
    obj, _, optimizer = fixture()
    obj.eval()
    batch = assemble(rt, records, [0, 2, 4, 6, 8, 10], 101, deep=True)
    state, opt_state = copy.deepcopy(obj.state_dict()), copy.deepcopy(optimizer.state_dict())
    modes, rng = [m.training for m in obj.modules()], torch.get_rng_state().clone()
    teacher_calls = []
    hooks = [
        module.register_forward_hook(lambda *args: teacher_calls.append(True))
        for module in obj.teacher.modules()
    ]
    collector = Collector(batch, records, 101, stats_backend="fast")
    collector.install(obj)
    try:
        with torch.no_grad(), torch.random.fork_rng():
            loss, terms, _ = diagnostic_objective(obj, batch)
            result = collector.result()
    finally:
        collector.close()
        for hook in hooks:
            hook.remove()
    assert torch.isfinite(loss)
    assert set(terms) == {"prediction", "prediction_mse", "control_copy_mse"}
    assert teacher_calls == [] and result["heads"] == {} and result["pairs"] == []
    assert len(collector.response_cache) == 1 + len(batch.cell_views)
    assert len(collector.condition_cache) == len(batch.graph_views)
    assert {role for role, _ in collector.response_cache} == {"student"}
    assert {role for role, _ in collector.condition_cache} == {"student"}
    assert obj.lambda1 == obj.lambda2 == 0 and not obj.teacher_updates_enabled
    assert obj.pending == {} and all(p.grad is None for p in obj.parameters())
    assert_nested_equal(obj.state_dict(), state)
    assert_nested_equal(optimizer.state_dict(), opt_state)
    assert [m.training for m in obj.modules()] == modes and torch.equal(torch.get_rng_state(), rng)
    assert frozen_digest(obj) == obj.parent_frozen_sha256
    assert frozen_state_proof(obj, optimizer)["frozen_equal"]
    scope = frozen_readout_scope(obj)
    assert scope["diagnostic_objective_scope"]["joint_scope"] == "frozen_readout_prediction_only"
    assert "not_forwarded:inactive_ssl" in scope["teacher_reference_status"]
