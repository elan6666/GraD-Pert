"""Explicit R17 analysis profile, historical closures and real native CPU routing."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from test_population_postfit import m, r16_fixture
from test_population_postfit import recipe as recipe
from test_population_postfit_chain import population_parity_fixture
from test_response_anchor_postfit import synthetic_arrays
from test_response_anchor_r15_postfit import binding, real_retained_fixture, write

from gradpert.config import load_experiment_config


def completed_parent(tmp_path, monkeypatch, recipe):
    import test_population_postfit_chain as old_chain

    original = old_chain.r16_fixture

    def exact_source(*args):
        spec = original(*args)
        ep = spec["endpoints"][0]
        ep["source"]["commit"] = "3cd32cf1e327bae35815666ff0a1552725e06c4b"
        identity = json.loads(Path(ep["run_manifest"]["path"]).read_text())
        identity["source"]["commit"] = ep["source"]["commit"]
        ep["run_manifest"] = write(Path(ep["run_manifest"]["path"]), identity)
        return spec

    with monkeypatch.context() as local:
        local.setattr(old_chain, "r16_fixture", exact_source)
        spec, seal, _, invoke = old_chain.population_orchestration_fixture(tmp_path, local, recipe)
        invoke(["validate", "--spec", seal["path"], "--spec-sha256", seal["sha256"]])
    return spec, seal


def r17_fixture(tmp_path, recipe, parent_spec, parent_binding):
    spec = r16_fixture(tmp_path, recipe)
    ep = spec["endpoints"][0]
    ep["name"] = "R17-GPM-HG0"
    ep["source"]["commit"] = "d" * 40
    config_path = Path(ep["config"]["path"])
    config = yaml.safe_load(config_path.read_text())
    config["model"]["parameters"]["response_anchor_prediction_head_grad"] = dict(
        value=False, source="project_preregistered", reference="synthetic"
    )
    config_path.write_text(yaml.safe_dump(config))
    ep["config"] = binding(config_path)
    config = load_experiment_config(config_path)
    ep["run_manifest"] = write(
        Path(ep["run_manifest"]["path"]),
        dict(
            source=dict(commit=ep["source"]["commit"], dirty=False),
            config_sha256=m.sha(config_path),
            resolved_config_sha256=m.digest(config.model_dump(mode="json")),
            data=dict(response_anchor=m.population_descriptor(m.population_options(config))),
        ),
    )
    spec["schema"] = m.R17_SCHEMA
    spec["retained_references"] = copy.deepcopy(parent_spec["retained_references"])
    spec["retained_references"]["R16-GPM"] = dict(
        postfit_spec=parent_binding,
        endpoint_name="R16-GPM",
        complete=binding(Path(parent_spec["output"]) / "R16-GPM/COMPLETE.json"),
    )
    return spec


def head_parity_fixture(tmp_path, base):
    spec, receipt = population_parity_fixture(tmp_path, base)
    receipt["endpoint"] = "R17-GPM-HG0"
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    return spec, receipt


def head_orchestration_fixture(
    tmp_path, monkeypatch, recipe, parent_spec, parent_binding, *, fail_statistics=False
):
    """Actual CLI/dispatch/merge/compare/validate; GPU and costly stats are explicit boundaries."""
    import shutil
    import sys
    from types import SimpleNamespace

    import gradpert

    root = Path(gradpert.__file__).resolve().parents[2]
    spec = r17_fixture(tmp_path / "candidate", recipe, parent_spec, parent_binding)
    for name in ("R3-RI", "R14-FRGA"):
        _, b, _ = real_retained_fixture(tmp_path / name, name)
        spec["retained_references"][name] = b
    analysis = Path(spec["analysis_source"]["root"])
    script = analysis / "scripts/v2/response_anchor_postfit.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(m.__file__, script)
    helper = analysis / "src/gradpert/evaluation/response_geometry.py"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(root / "src/gradpert/evaluation/response_geometry.py", helper)
    spec, _ = head_parity_fixture(tmp_path / "pilot", spec)
    input_spec = write(tmp_path / "INPUT_SPEC.json", spec)
    _, _, donor = real_retained_fixture(tmp_path / "donor")
    monkeypatch.setattr(m, "__file__", str(script))
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path.parent.resolve())
    original_retained = m.load_retained_reference
    monkeypatch.setattr(
        m, "load_retained_reference", lambda b, n, v, **kw: original_retained(b, n, v, server=False)
    )
    original_validate = m.validate_spec
    monkeypatch.setattr(
        m, "validate_spec", lambda value, **kw: original_validate(value, server=False)
    )
    monkeypatch.setattr(m, "preimport_source", lambda source: None)
    monkeypatch.setattr(m, "_source", lambda source: dict(commit=source["commit"], dirty=False))
    monkeypatch.setattr(m, "make_dispatcher_owner", lambda *a, **kw: dict(synthetic_owner=True))
    trace = []
    original_geometry = m.geometry_module

    def geometry(path):
        real = original_geometry(path)  # Real private file-load/dataclass path is executed.

        def statistics(a, b, **kwargs):
            if fail_statistics:
                raise ValueError("synthetic statistics failure")
            assert kwargs == spec["bootstrap"]
            small = synthetic_arrays()
            result = real.compare_endpoints(small, small, draws=10, seed=kwargs["seed"])
            result.update(draws=kwargs["draws"], condition_N=445)
            for key in ("ownmask_Pearson_delta", "ownmask_MSE_delta", "RSA_delta"):
                row = result[key]
                row["valid_draws"] = kwargs["draws"] if row["CI95"] is not None else 0
                row["invalid_draws"] = kwargs["draws"] - row["valid_draws"]
            return result

        return SimpleNamespace(
            compare_endpoints=statistics,
            summarize_endpoint=lambda **kw: dict(synthetic_statistics_boundary=True),
        )

    monkeypatch.setattr(m, "geometry_module", geometry)

    def worker(value, name, index, *, spec_file, resource_owner):
        trace.append(("worker", name, index))
        assert name == "R17-GPM-HG0"
        e = value["endpoints"][0]
        target = Path(value["output"]) / name / f"worker-{index}"
        target.mkdir()
        source = donor / f"worker-{index}"
        for file in ("PRIVATE_ENDPOINT.npz", "PRIVATE_NATIVE_RESULT.json", "SUMMARY.json"):
            shutil.copyfile(source / file, target / file)
        r = json.loads((source / "COMPLETE.json").read_text())
        r.update(
            endpoint=name,
            source=dict(commit=e["source"]["commit"], dirty=False),
            analysis_source=dict(commit=value["analysis_source"]["commit"], dirty=False),
            postfit_spec_sha256=m.digest(value),
            endpoint_spec_sha256=m.digest(e),
            checkpoint_sha256=e["checkpoint"]["sha256"],
            resource_owner=resource_owner,
            arrays=binding(target / "PRIVATE_ENDPOINT.npz"),
            native_result=binding(target / "PRIVATE_NATIVE_RESULT.json"),
            summary_sha256=m.sha(target / "SUMMARY.json"),
        )
        r["forbidden_calls"].update(dict.fromkeys(m.POPULATION_COUNTERS, 0))
        inputs = [
            spec_file,
            value["shared_policy"],
            value["analysis_source"]["publication"],
            e["source"]["publication"],
            resource_owner,
            *[e[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
        ]
        r["input_files_sha256"] = {str(Path(v["path"]).resolve()): v["sha256"] for v in inputs}
        write(target / "COMPLETE.json", r)

    monkeypatch.setattr(m, "run_endpoint", worker)

    def invoke(argv):
        old = sys.argv
        sys.argv = [str(script), *argv]
        try:
            m.main()
        finally:
            sys.argv = old

    class Child:
        def __init__(self, argv, **kw):
            assert argv[1] == str(script) and argv[2] == "worker"
            assert (
                kw["env"]["PYTHONPATH"]
                == str(Path(spec["endpoints"][0]["source"]["root"]) / "src")
                + ":"
                + spec["endpoints"][0]["source"]["root"]
            )
            assert kw["env"]["CUDA_VISIBLE_DEVICES"] in ("0", "1")
            trace.append(("spawn", tuple(argv[2:])))
            kw["stdout"].write("synthetic CUDA boundary\n")
            kw["stdout"].flush()
            invoke(argv[2:])
            self.pid = 100 + len(trace)

        def wait(self):
            return 0

        def poll(self):
            return 0

    def aggregate(argv, **kw):
        assert argv[2] == "compare" and kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
        trace.append(("compare",))
        invoke(argv[2:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "Popen", Child)
    monkeypatch.setattr(m.subprocess, "run", aggregate)
    invoke(["run", "--spec", input_spec["path"], "--spec-sha256", input_spec["sha256"]])
    return spec, input_spec, trace, invoke


@pytest.fixture
def parent(tmp_path, monkeypatch, recipe):
    return completed_parent(tmp_path / "parent", monkeypatch, recipe)


def test_explicit_profile_and_three_retained_references(tmp_path, recipe, parent, monkeypatch):
    spec = r17_fixture(tmp_path / "new", recipe, *parent)
    m.validate_spec(spec, server=False)
    assert [name for name, _ in m.population_profile(spec)[3]] == ["R3-RI", "R14-FRGA", "R16-GPM"]
    identity = json.loads(Path(spec["endpoints"][0]["run_manifest"]["path"]).read_text())
    assert identity["data"]["response_anchor"]["prediction_head_grad"] is False
    assert (
        identity["data"]["response_anchor"]["gradient_routing"]
        == "gpm_head_parameters_detached_preserve_input_jacobian"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "flag_true",
        "r16_schema",
        "missing_parent",
        "changed_parent_role",
        "changed_validation",
        "parent_receipt_extra",
    ],
)
def test_profile_mismatch_and_retained_protocol_fail_closed(tmp_path, recipe, parent, mutation):
    spec = r17_fixture(tmp_path / "new", recipe, *parent)
    if mutation == "flag_true":
        ep = spec["endpoints"][0]
        value = yaml.safe_load(Path(ep["config"]["path"]).read_text())
        value["model"]["parameters"]["response_anchor_prediction_head_grad"]["value"] = True
        Path(ep["config"]["path"]).write_text(yaml.safe_dump(value))
        ep["config"] = binding(Path(ep["config"]["path"]))
        config = load_experiment_config(ep["config"]["path"])
        identity = json.loads(Path(ep["run_manifest"]["path"]).read_text())
        identity.update(
            config_sha256=ep["config"]["sha256"],
            resolved_config_sha256=m.digest(config.model_dump(mode="json")),
        )
        identity["data"]["response_anchor"] = m.population_descriptor(m.population_options(config))
        ep["run_manifest"] = write(Path(ep["run_manifest"]["path"]), identity)
    elif mutation == "r16_schema":
        spec["schema"] = m.R16_SCHEMA
        spec["endpoints"][0]["name"] = "R16-GPM"
        spec["retained_references"].pop("R16-GPM")
    elif mutation == "missing_parent":
        spec["retained_references"].pop("R16-GPM")
    elif mutation == "changed_parent_role":
        spec["retained_references"]["R16-GPM"]["endpoint_name"] = "R17-GPM-HG0"
    elif mutation == "changed_validation":
        spec["validation"]["conditions"][0], spec["validation"]["conditions"][1] = (
            spec["validation"]["conditions"][1],
            spec["validation"]["conditions"][0],
        )
    else:
        path = Path(parent[0]["output"]) / "unexpected.txt"
        path.write_text("extra")
    with pytest.raises(ValueError):
        m.validate_spec(spec, server=False)


@pytest.mark.parametrize(
    "mutation", ["r16_endpoint", "temperature", "missing_counter", "bad_source"]
)
def test_r17_observer_parity_rejects_resealed_wrong_proof(tmp_path, recipe, parent, mutation):
    spec, receipt = head_parity_fixture(
        tmp_path / "pilot", r17_fixture(tmp_path / "new", recipe, *parent)
    )
    m.validate_r17_observer_parity(spec, server=False)
    if mutation == "r16_endpoint":
        receipt["endpoint"] = "R16-GPM"
    elif mutation == "temperature":
        receipt["parity"]["state_after"]["protected"]["objective_scalar_clocks"] = "0" * 64
    elif mutation == "missing_counter":
        receipt["parity"]["forbidden_calls"][0].pop("training_file_read")
    else:
        receipt["source"]["commit"] = "0" * 40
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    with pytest.raises(ValueError):
        m.validate_r17_observer_parity(spec, server=False)
    with pytest.raises(ValueError, match="relabel"):
        m.validate_r16_observer_parity(spec, server=False)


def test_actual_three_reference_cli_result_chain(tmp_path, monkeypatch, recipe, parent, capsys):
    spec, seal, trace, invoke = head_orchestration_fixture(
        tmp_path / "candidate", monkeypatch, recipe, *parent
    )
    invoke(["validate", "--spec", seal["path"], "--spec-sha256", seal["sha256"]])
    # Discard parent validation output and read the last actual CLI response.
    assert (
        json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"]
        == "validated_completed_result"
    )
    result = json.loads((Path(spec["output"]) / "COMPARISON.json").read_text())
    assert result["candidate"] == "R17-GPM-HG0"
    assert set(result["comparisons"]) == {"primary_vs_RI", "secondary_vs_R14", "parent_vs_R16"}
    assert [x for x in trace if x[0] == "worker"] == [
        ("worker", "R17-GPM-HG0", 0),
        ("worker", "R17-GPM-HG0", 1),
    ]
    with pytest.raises(ValueError, match="relabel"):
        m.validate_r16_result(spec, spec_file=seal, server=False)


@pytest.mark.parametrize("mutation", ["extra", "missing", "parent_changed"])
def test_actual_result_output_and_parent_closures_fail(
    tmp_path, monkeypatch, recipe, parent, mutation
):
    spec, seal, _, _ = head_orchestration_fixture(
        tmp_path / "candidate", monkeypatch, recipe, *parent
    )
    root = Path(spec["output"])
    if mutation == "extra":
        (root / "unexpected.txt").write_text("extra")
    elif mutation == "missing":
        (root / "COMPARISON.json").unlink()
    else:
        (Path(parent[0]["output"]) / "COMPARISON.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        m.validate_r17_result(spec, spec_file=seal, server=False)


def test_medium_uses_same_r17_identity_guard_without_relabel(tmp_path, recipe, parent):
    module_spec = importlib.util.spec_from_file_location(
        "r17_medium", Path(m.__file__).with_name("response_anchor_medium.py")
    )
    medium = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(medium)
    post = r17_fixture(tmp_path / "new", recipe, *parent)
    ep = post["endpoints"][0]
    spec = dict(
        schema=medium.R17_SCHEMA,
        native_source=ep["source"],
        config=ep["config"],
        run_manifest=ep["run_manifest"],
    )
    medium.validate_r17_medium_identity(spec, server=False)
    with pytest.raises(ValueError):
        medium.validate_r16_medium_identity(spec, server=False)
    spec["schema"] = medium.R16_SCHEMA
    with pytest.raises(ValueError):
        medium.validate_population_medium_identity(spec, server=False)


@pytest.mark.parametrize("entry", ["population_loader", "training_file_read"])
def test_r17_rejects_additional_train_loader_or_read(tmp_path, recipe, parent, entry):
    from gradpert.training.v2 import population_anchor

    spec = r17_fixture(tmp_path / "new", recipe, *parent)
    config = load_experiment_config(spec["endpoints"][0]["config"]["path"])
    with m.population_no_training_reads(config) as counts:
        with pytest.raises(ValueError, match=entry):
            if entry == "population_loader":
                population_anchor.load_population_targets("forbidden", "bad", "bad", None)
            else:
                Path(config.model.parameters["response_anchor_population_path"].value).read_bytes()
        assert counts[entry] == 1


def test_existing_r16_small_terminal_metadata_keeps_original_schema_and_roles():
    root = Path(
        "/Users/elan/code/grad-pert/experiment-analysis/representation-adaptive-20261028/r16-postfit-20261030/val-report-platform-candidate/inputs"
    )
    if not root.is_dir():
        pytest.skip("permanent local small-result archive is not shipped with source")
    complete = json.loads((root / "R16_NATIVE_VAL_COMPLETE.json").read_text())
    comparison = json.loads((root / "R16_COMPARISON.json").read_text())
    name, slug, head_grad, refs = m.population_profile({"schema": m.R16_SCHEMA})
    assert head_grad is True
    assert complete["schema"] == f"response-anchor-{slug}-val-postfit-complete-1"
    assert set(complete["endpoint_receipts"]) == {name}
    assert comparison["candidate"] == name
    assert comparison["schema"] == f"response-anchor-{slug}-paired-comparison-1"
    assert set(comparison["comparisons"]) == {label for _, label in refs}
    assert m.sha(root / "R16_COMPARISON.json") == complete["comparison_sha256"]
