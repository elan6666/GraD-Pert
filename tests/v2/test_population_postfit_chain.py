"""Actual R16 CPU orchestration and strict parity receipt closure tests."""

import copy
import json
from pathlib import Path

import pytest
from test_population_postfit import m, r16_fixture
from test_population_postfit import recipe as recipe
from test_response_anchor_postfit import synthetic_arrays
from test_response_anchor_r15_postfit import binding, real_retained_fixture, write


def population_parity_fixture(tmp_path, base_spec):
    from test_response_anchor_parity import fixture, pilot

    from gradpert.config import load_experiment_config
    from scripts.v2.capacity_probe import state_fingerprint, validation_state

    spec = copy.deepcopy(base_spec)
    cell_batch = int(
        load_experiment_config(
            spec["endpoints"][0]["config"]["path"]
        ).training.eval_batch_size.value
    )
    args, _, _ = fixture(controls=300, width=5000, query=1000, batch=cell_batch)
    args = list(args)
    runtime = args[1]

    def state():
        return dict(
            protected=pilot.protected_state(
                runtime, validation_state=validation_state, fingerprint=state_fingerprint
            ),
            cuda_rng=state_fingerprint([]),
            canonical_stat=[100, 200],
        )

    args[6] = state
    proof, _, _ = pilot.paired_evaluate(
        *args, population_config=load_experiment_config(spec["endpoints"][0]["config"]["path"])
    )
    pilot_spec = write(tmp_path / "pilot-spec.json", spec)
    parity_root = tmp_path / "parity"
    arrays = write(parity_root / "PRIVATE_ENDPOINT.npz", dict(synthetic=True))
    native = write(parity_root / "PRIVATE_NATIVE_RESULT.json", dict(synthetic=True))
    e = spec["endpoints"][0]
    inputs = [
        pilot_spec,
        spec["shared_policy"],
        spec["analysis_source"]["publication"],
        e["source"]["publication"],
        *[e[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
    ]
    receipt = dict(
        status="completed",
        endpoint="R16-GPM",
        spec=pilot_spec,
        spec_digest=m.digest(spec),
        endpoint_digest=m.digest(e),
        source=dict(commit=e["source"]["commit"]),
        analysis_source=dict(commit=spec["analysis_source"]["commit"]),
        checkpoint=e["checkpoint"],
        condition=spec["validation"]["conditions"][0],
        full_validation_conditions=445,
        executed_conditions=1,
        controls=300,
        expression_genes=5000,
        query_count=1000,
        epoch=2,
        role="last",
        split="val",
        updates=0,
        teacher_updates=0,
        center_updates=0,
        zero_pkl=True,
        parity=proof,
        input_files_sha256={str(Path(v["path"]).resolve()): v["sha256"] for v in inputs},
        arrays=arrays,
        native_result_sha256=native["sha256"],
    )
    spec["observer_parity"] = write(parity_root / "COMPLETE.json", receipt)
    return spec, receipt


def population_orchestration_fixture(tmp_path, monkeypatch, recipe, *, fail_statistics=False):
    """Actual CLI/dispatch/merge/compare/validate; GPU and costly stats are explicit boundaries."""
    import shutil
    import sys
    from types import SimpleNamespace

    import gradpert

    root = Path(gradpert.__file__).resolve().parents[2]
    spec = r16_fixture(tmp_path / "candidate", recipe)
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
    spec, _ = population_parity_fixture(tmp_path / "pilot", spec)
    input_spec = write(tmp_path / "INPUT_SPEC.json", spec)
    _, _, donor = real_retained_fixture(tmp_path / "donor")
    monkeypatch.setattr(m, "__file__", str(script))
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path.resolve())
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
        assert name == "R16-GPM"
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


def test_r16_actual_cli_dispatch_merge_compare_validate(tmp_path, monkeypatch, recipe, capsys):
    spec, input_spec, trace, invoke = population_orchestration_fixture(
        tmp_path, monkeypatch, recipe
    )
    assert [r for r in trace if r[0] == "worker"] == [
        ("worker", "R16-GPM", 0),
        ("worker", "R16-GPM", 1),
    ]
    invoke(["validate", "--spec", input_spec["path"], "--spec-sha256", input_spec["sha256"]])
    assert json.loads(capsys.readouterr().out.strip())["status"] == "validated_completed_result"
    complete = json.loads((Path(spec["output"]) / "COMPLETE.json").read_text())
    assert complete["scientific_complete"] is False
    assert complete["postfit_complete"] is False
    assert set(complete["endpoint_receipts"]) == {"R16-GPM"}


@pytest.mark.parametrize("change", ["counter", "state", "pilot", "endpoint"])
def test_r16_resealed_parity_fails(tmp_path, recipe, change):
    spec, receipt = population_parity_fixture(
        tmp_path / "pilot", r16_fixture(tmp_path / "base", recipe)
    )
    m.validate_r16_observer_parity(spec, server=False)
    if change == "counter":
        receipt["parity"]["forbidden_calls"][0].pop("population_loader")
    elif change == "state":
        receipt["parity"]["state_after"]["protected"]["objective_buffers"] = "0" * 64
    elif change == "pilot":
        receipt["spec_digest"] = "0" * 64
    else:
        receipt["endpoint"] = "R15-DIR"
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    with pytest.raises(ValueError):
        m.validate_r16_observer_parity(spec, server=False)
