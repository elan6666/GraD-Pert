"""R16 independent profile and headless evaluation contracts, synthetic CPU only."""

import builtins
import copy
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_response_anchor_parity import fixture, pilot
from test_response_anchor_postfit import m, spec_fixture
from test_response_anchor_r15_postfit import binding, retained_fixture, write
from torch import nn

import gradpert
from gradpert.config import load_experiment_config
from gradpert.training.v2 import population_anchor, response_anchor_runtime
from scripts.v2.generate_response_anchor import configuration


@pytest.fixture
def recipe(tmp_path):
    root = Path(gradpert.__file__).resolve().parents[2]
    value = configuration(
        root,
        bank_path="/data/yilangliu/bank/server-only/manifest.json",
        manifest_sha256="a" * 64,
        arrays_sha256="b" * 64,
        weight=0.03,
        loss_mode="population_prediction",
        population_path="/data/yilangliu/pop/server-only/manifest.json",
        population_sha256="c" * 64,
        population_arrays_sha256="d" * 64,
    )
    path = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(value))
    return path, load_experiment_config(path)


def r16_fixture(tmp_path, recipe):
    old, refs = retained_fixture(tmp_path)
    spec = spec_fixture(tmp_path / "new")
    endpoint = spec["endpoints"][1]
    endpoint["name"] = "R16-GPM"
    endpoint["source"]["commit"] = "1" * 40
    path = Path(endpoint["source"]["root"]) / "configs/R16-GPM/gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir(parents=True)
    path.write_bytes(recipe[0].read_bytes())
    endpoint["config"] = binding(path)
    config = load_experiment_config(path)
    endpoint["run_manifest"] = write(
        Path(endpoint["run_manifest"]["path"]),
        dict(
            source=dict(commit="1" * 40, dirty=False),
            config_sha256=m.sha(path),
            resolved_config_sha256=m.digest(config.model_dump(mode="json")),
            data=dict(response_anchor=m.population_descriptor(m.population_options(config))),
        ),
    )
    spec.update(
        schema=m.R16_SCHEMA, endpoints=[endpoint], retained_references=refs, observer_parity=None
    )
    spec["validation"] = copy.deepcopy(old["validation"])
    return spec


def test_r16_profile_binds_only_metadata_without_auxiliary_reads(tmp_path, recipe, monkeypatch):
    spec = r16_fixture(tmp_path, recipe)

    def reject(*args, **kwargs):
        raise AssertionError("TRAIN numerical loader must not run")

    monkeypatch.setattr(population_anchor, "load_population_targets", reject)
    monkeypatch.setattr(response_anchor_runtime, "load_response_bank", reject)
    m.validate_spec(spec, server=False)
    assert spec["endpoints"][0]["name"] == "R16-GPM"
    with pytest.raises(ValueError, match="first-condition"):
        m.validate_r16_observer_parity(spec, server=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss", "cosine_direction"),
        ("head", "Linear256_256_default_random_bias_true_private_rng"),
        ("teacher_readout", "unused_ema_mirror_no_forward_or_target_values"),
        ("population_manifest_sha256", "f" * 64),
        ("population_arrays_sha256", "f" * 64),
        ("formula", "square-before-cell-mean"),
        ("weight", 1.0),
    ],
)
def test_r16_rejects_resealed_wrong_descriptor(tmp_path, recipe, field, value):
    spec = r16_fixture(tmp_path, recipe)
    ep = spec["endpoints"][0]
    identity = json.loads(Path(ep["run_manifest"]["path"]).read_text())
    identity["data"]["response_anchor"][field] = value
    ep["run_manifest"] = write(Path(ep["run_manifest"]["path"]), identity)
    with pytest.raises(ValueError, match="descriptor"):
        m.validate_spec(spec, server=False)


@pytest.mark.parametrize("where", ["student", "teacher"])
def test_unused_auxiliary_head_is_rejected(recipe, where):
    args, _, _ = fixture()
    runtime = args[1]
    getattr(runtime.objective, where).response_anchor_readout = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="auxiliary Student or Teacher head"):  # noqa: SIM117
        with m.population_no_training_reads(recipe[1], runtime):
            pass


def test_auxiliary_controller_is_rejected_before_evaluation(recipe):
    args, _, _ = fixture()
    runtime = args[1]
    runtime.objective.response_anchor_backward = lambda *a: None
    with pytest.raises(ValueError, match="TRAIN auxiliary controller"):  # noqa: SIM117
        with m.population_no_training_reads(recipe[1], runtime):
            pass


@pytest.mark.parametrize(
    "entry", ["bank", "population", "builtin", "io", "os", "np", "path", "callback"]
)
def test_auxiliary_loaders_and_reads_fail_before_io_and_restore(recipe, entry, tmp_path):
    args, _, _ = fixture()
    runtime = args[1]
    original = (
        builtins.open,
        io.open,
        os.open,
        np.load,
        population_anchor.load_population_targets,
        response_anchor_runtime.load_response_bank,
    )
    bank = "/data/yilangliu/bank/server-only/anchor_bank.npz"
    population = "/data/yilangliu/pop/server-only/population_targets.npz"
    with m.population_no_training_reads(recipe[1], runtime) as counts:
        assert not hasattr(runtime.objective.student, "response_anchor_readout")
        with pytest.raises(ValueError, match="forbidden evaluation path"):
            if entry == "bank":
                response_anchor_runtime.load_response_bank("bad", "bad", None)
            elif entry == "population":
                population_anchor.load_population_targets("bad", "bad", "bad", None)
            elif entry == "builtin":
                builtins.open(bank, "rb")  # noqa: SIM115
            elif entry == "io":
                io.open(population, "rb")  # noqa: UP020, SIM115
            elif entry == "os":
                os.open(bank, os.O_RDONLY)
            elif entry == "np":
                np.load(population)
            elif entry == "path":
                Path(population).read_bytes()
            else:
                runtime.objective.response_anchor_backward()
        safe = tmp_path / "safe.txt"
        safe.write_text("safe")
        assert safe.read_text() == "safe"
        assert sum(counts.values()) == 1
    assert original == (
        builtins.open,
        io.open,
        os.open,
        np.load,
        population_anchor.load_population_targets,
        response_anchor_runtime.load_response_bank,
    )
    assert not hasattr(runtime.objective, "response_anchor_backward")


def test_headless_parity_uses_native_prediction_and_preserves_private_state(recipe):
    args, calls, _ = fixture(width=5000, controls=300, query=1000, batch=300)
    runtime = args[1]
    original = pilot.postfit
    pilot.postfit = m
    try:
        parity, _result, _arrays = pilot.paired_evaluate(*args, population_config=recipe[1])
    finally:
        pilot.postfit = original
    assert len(calls) == 2
    assert parity["max_abs"] == 0
    assert parity["state_before"] == parity["state_after"]
    assert all(
        set(row)
        == {
            "teacher_forward",
            "optimizer_step",
            "statistics_commit",
            "backward",
            "autograd_grad",
            *m.POPULATION_COUNTERS,
        }
        for row in parity["forbidden_calls"]
    )
    assert all(not any(row.values()) for row in parity["forbidden_calls"])
    assert not hasattr(runtime.objective.student, "response_anchor_readout")
    assert not hasattr(runtime.objective.teacher, "response_anchor_readout")


@pytest.mark.parametrize("mutation", ["temperature", "private_buffer", "teacher", "rng"])
def test_headless_parity_rejects_full_protected_state_mutations(recipe, mutation):
    from scripts.v2.capacity_probe import state_fingerprint, validation_state

    args, _, _ = fixture()
    args = list(args)
    native, runtime = args[:2]
    args[6] = lambda: m.population_protected_state(runtime, validation_state, state_fingerprint)
    original_evaluate = native.evaluate

    def evaluate(*a, **kw):
        result = original_evaluate(*a, **kw)
        if mutation == "temperature":
            runtime.objective._temperature_updates += 1
        elif mutation == "private_buffer":
            runtime.objective.private_clock.add_(1)
        elif mutation == "teacher":
            with torch.no_grad():
                runtime.objective.teacher.weight.add_(1)
        else:
            torch.rand(1)
        return result

    native.evaluate = evaluate
    old = pilot.postfit
    pilot.postfit = m
    try:
        with pytest.raises(ValueError, match="protected state or RNG"):
            pilot.paired_evaluate(*args, population_config=recipe[1])
    finally:
        pilot.postfit = old


def test_r16_medium_identity_and_na_reuse_without_numerical_target_reads(tmp_path, recipe):
    import importlib.util

    path = Path(m.__file__).with_name("response_anchor_medium.py")
    spec = importlib.util.spec_from_file_location("r16_medium_test", path)
    medium = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(medium)
    post = r16_fixture(tmp_path, recipe)
    e = post["endpoints"][0]
    medium.validate_r16_medium_identity(
        dict(native_source=e["source"], config=e["config"], run_manifest=e["run_manifest"]),
        server=False,
    )
    scope = [
        {
            "diagnostic_objective_scope": {
                "unavailable_terms": {"response_anchor": "requires_complete_train_condition_bank"}
            },
            "loss_terms": {},
        }
    ]
    medium.validate_scopes(scope, anchor=True)
    scope[0]["loss_terms"]["response_anchor"] = 0.0
    with pytest.raises(ValueError, match="unavailable"):
        medium.validate_scopes(scope, anchor=True)


def test_r16_medium_guard_rejects_loader_before_constructor_and_restores(recipe):
    import importlib.util
    from contextlib import contextmanager
    from types import SimpleNamespace

    path = Path(m.__file__).with_name("response_anchor_medium.py")
    spec = importlib.util.spec_from_file_location("r16_medium_guard_test", path)
    medium = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(medium)
    calls = []

    @contextmanager
    def prepare(*a, **kw):
        calls.append(kw["purpose"])
        population_anchor.load_population_targets("forbidden", "bad", "bad", None)
        yield

    def runtime_args(args):
        return torch, recipe[1], {}, prepare, Path("/native")

    cap = SimpleNamespace(runtime_args=runtime_args)
    original_loader = population_anchor.load_population_targets
    with (
        medium.diagnostic_adapter(cap, None, [], controls_only=True, read_receipts=[]),
        pytest.raises(ValueError, match="population_loader"),
        cap.runtime_args(None)[3](purpose="evaluation"),
    ):
        raise AssertionError("unreachable")
    assert calls == ["evaluation"]
    assert cap.runtime_args is runtime_args
    assert population_anchor.load_population_targets is original_loader
