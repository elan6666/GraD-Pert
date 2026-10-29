"""Staged synthetic CPU tests only; no server values, checkpoint model or GPU."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_response_anchor_postfit import m, spec_fixture, synthetic_arrays


def binding(path):
    return dict(path=str(path), sha256=m.sha(path))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return binding(path)


def retained_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    old = spec_fixture(tmp_path / "old")
    old["endpoints"][0]["source"]["commit"] = "f0cb20ebb2efab749970f2bf5d205ed2f358592c"
    old["endpoints"][1]["source"]["commit"] = "dc5233ef1a243b2144107e5441a5a8db4f800fc6"
    old_binding = write(tmp_path / "old/SPEC.json", old)
    reference = {}
    for name, endpoint_name in [("R3-RI", "R3-RI"), ("R14-FRGA", "FixedResponseGroupAnchor")]:
        reference[name] = dict(
            postfit_spec=old_binding,
            endpoint_name=endpoint_name,
            complete=write(Path(old["output"]) / endpoint_name / "COMPLETE.json", {}),
        )
    return old, reference


def r15_fixture(tmp_path):
    old, references = retained_fixture(tmp_path)
    new = spec_fixture(tmp_path / "new")
    endpoint = new["endpoints"][1]
    endpoint["name"] = "R15-DIR"
    endpoint["source"]["commit"] = m.R15_SOURCE
    import gradpert

    native_root = Path(gradpert.__file__).resolve().parents[2]
    config = (
        native_root
        / "configs/v2/response_anchor_direction_20261029/R15-DIR/gradpert_v2/nadig_jurkat.yaml"
    )
    target = Path(endpoint["source"]["root"]) / "configs/R15-DIR/gradpert_v2/nadig_jurkat.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(config.read_bytes())
    endpoint["config"] = binding(target)
    endpoint["run_manifest"] = write(
        Path(endpoint["run_manifest"]["path"]),
        dict(
            source=dict(commit=m.R15_SOURCE, dirty=False),
            config_sha256=m.R15_CONFIG_SHA256,
            data=dict(response_anchor=m.r15_descriptor()),
        ),
    )
    new.update(
        schema=m.R15_SCHEMA,
        endpoints=[endpoint],
        retained_references=references,
        observer_parity=None,
    )
    new["validation"] = copy.deepcopy(old["validation"])
    return new


def test_explicit_new_profile_validates_metadata_only_and_does_not_invent_cp(tmp_path):
    spec = r15_fixture(tmp_path)
    m.validate_spec(spec, server=False)
    with pytest.raises(ValueError, match="off/on parity"):
        m.validate_observer_parity(spec, server=False)
    spec["schema"] = m.SCHEMA
    with pytest.raises(ValueError):
        m.validate_spec(spec, server=False)


@pytest.mark.parametrize(
    "change", ["mode", "epsilon", "formula", "weight", "source", "config", "missing_field"]
)
def test_r15_descriptor_rejects_resealed_wrong_mechanism(tmp_path, change):
    spec = r15_fixture(tmp_path)
    endpoint = spec["endpoints"][0]
    path = Path(endpoint["run_manifest"]["path"])
    identity = json.loads(path.read_text())
    descriptor = identity["data"]["response_anchor"]
    if change == "mode":
        descriptor["loss"] = "mse"
    if change == "epsilon":
        descriptor["direction_epsilon"] = 1e-5
    if change == "formula":
        descriptor["direction_formula"] = "mean(normalize(cell))"
    if change == "weight":
        descriptor["weight"] = 0.03148168297045377
    if change == "source":
        identity["source"]["commit"] = "d" * 40
    if change == "config":
        identity["config_sha256"] = "0" * 64
    if change == "missing_field":
        del descriptor["loss"]
    endpoint["run_manifest"] = write(path, identity)
    with pytest.raises(ValueError, match="R15 requires"):
        m.validate_spec(spec, server=False)


def full_native(ids, validation):
    from gradpert.training.v2.evaluation import merge_evaluation_shards

    keys = ("txpert_macro_pearson_delta_all", "trishift_pearson_delta_all", "systema_pearson_all")
    results = [dict(metric_id=k, value=0.2, reason=None, gene_count=4775) for k in keys]
    rows = [
        dict(
            condition_id=p,
            control_row_ids_sha256="controls-" + p,
            truth_row_ids_sha256="truth-" + p,
            loss=1.0,
            metrics=dict(results=results),
            metric_gene_groups={"seen_expression": dict(results=results)},
        )
        for p in ids
    ]
    value = dict(
        split="val",
        selection_gene_ids=[],
        control_manifest_sha256=validation["control_manifest_sha256"],
        reference_sha256=validation["reference_sha256"],
        query_recipe="synthetic native boundary",
        conditions=rows,
        metric_gene_groups={"seen_expression": dict(gene_count=4775, gene_ids_sha256="a" * 64)},
    )
    return merge_evaluation_shards([value], ordered_conditions=tuple(ids))


def real_retained_fixture(tmp_path, name="R3-RI"):
    old, references = retained_fixture(tmp_path)
    b = references[name]
    endpoint = old["endpoints"][0 if name == "R3-RI" else 1]
    root = Path(b["complete"]["path"]).parent
    owner = write(root / "DISPATCH_OWNER.json", dict(synthetic_owner=True))
    ids = old["validation"]["conditions"]
    arrays = dict(
        condition_ids=np.array(ids),
        allowed_gene_indices=np.arange(4775),
        own_target_indices=np.zeros((445, 1), np.int64),
        prediction_delta=np.zeros((445, 4775), np.float32),
        truth_delta=np.zeros((445, 4775), np.float32),
        cls_means=np.zeros((445, 256), np.float32),
    )
    native = full_native(ids, old["validation"])

    def save_arrays(path, value):
        with path.open("wb") as f:
            np.savez_compressed(f, **value)
        return binding(path)

    receipts = []
    for i in (0, 1):
        worker = root / f"worker-{i}"
        worker.mkdir()
        shard = {k: v if k == "allowed_gene_indices" else v[i::2] for k, v in arrays.items()}
        a = save_arrays(worker / "PRIVATE_ENDPOINT.npz", shard)
        n = write(
            worker / "PRIVATE_NATIVE_RESULT.json",
            dict(native, conditions=native["conditions"][i::2]),
        )
        summary = write(worker / "SUMMARY.json", {})
        inputs = [
            b["postfit_spec"],
            old["shared_policy"],
            old["analysis_source"]["publication"],
            endpoint["source"]["publication"],
            owner,
            *[endpoint[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
        ]
        receipt = dict(
            status="completed",
            endpoint=endpoint["name"],
            worker_index=i,
            postfit_spec_sha256=m.digest(old),
            endpoint_spec_sha256=m.digest(endpoint),
            validation_spec_sha256=m.digest(old["validation"]),
            conditions=ids[i::2],
            source=dict(commit=endpoint["source"]["commit"], dirty=False),
            analysis_source=dict(commit=old["analysis_source"]["commit"], dirty=False),
            checkpoint_sha256=endpoint["checkpoint"]["sha256"],
            control_manifest_sha256=old["validation"]["control_manifest_sha256"],
            reference_sha256=old["validation"]["reference_sha256"],
            epoch=2,
            role="last",
            split="val",
            state_before=dict(student="a" * 64, teacher="b" * 64),
            state_unchanged=True,
            cuda_rng_unchanged=True,
            zero_pkl=True,
            updates=0,
            teacher_updates=0,
            center_updates=0,
            forbidden_calls={
                k: 0
                for k in (
                    "teacher_forward",
                    "optimizer_step",
                    "statistics_commit",
                    "backward",
                    "autograd_grad",
                )
            },
            resource_checks=dict(conditions_checked=len(ids[i::2]), calls=len(ids[i::2]) + 2),
            resource_owner=owner,
            arrays=a,
            native_result=n,
            summary_sha256=summary["sha256"],
            input_files_sha256={str(Path(v["path"]).resolve()): v["sha256"] for v in inputs},
        )
        receipts.append(write(worker / "COMPLETE.json", receipt)["sha256"])
    a = save_arrays(root / "PRIVATE_ENDPOINT.npz", arrays)
    n = write(root / "PRIVATE_NATIVE_RESULT.json", native)
    s = write(root / "SUMMARY.json", {})
    complete = dict(
        status="completed",
        endpoint=endpoint["name"],
        arrays=a,
        native_result_sha256=n["sha256"],
        summary_sha256=s["sha256"],
        worker_receipts_sha256=receipts,
        validation_spec_sha256=m.digest(old["validation"]),
        conditions=445,
        updates=0,
        zero_pkl=True,
        scientific_complete=False,
    )
    b["complete"] = write(root / "COMPLETE.json", complete)
    return old, b, root


def test_real_retained_loader_is_read_only_and_checks_worker_array_closure(tmp_path):
    old, b, root = real_retained_fixture(tmp_path)
    before = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    arrays, native = m.load_retained_reference(b, "R3-RI", old["validation"], server=False)
    assert arrays["truth_delta"].shape == (445, 4775) and len(native["conditions"]) == 445
    assert before == {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "tamper", "missing_required_input", "resealed_axis"]
)
def test_retained_closure_failures_even_with_resealed_receipt(tmp_path, mutation):
    old, b, root = real_retained_fixture(tmp_path)
    if mutation == "extra":
        (root / "worker-0/unknown.json").write_text("{}")
    elif mutation == "missing":
        (root / "worker-0/SUMMARY.json").unlink()
    elif mutation == "tamper":
        (root / "SUMMARY.json").write_text("tampered")
    else:
        path = root / "worker-0/COMPLETE.json"
        r = json.loads(path.read_text())
        if mutation == "missing_required_input":
            r["input_files_sha256"] = {}
        else:
            a = m.load_arrays(r["arrays"], server=False)
            a["truth_delta"][0, 0] = 1
            with Path(r["arrays"]["path"]).open("wb") as f:
                np.savez_compressed(f, **a)
            r["arrays"] = binding(Path(r["arrays"]["path"]))
        write(path, r)
        cp = root / "COMPLETE.json"
        c = json.loads(cp.read_text())
        c["worker_receipts_sha256"][0] = m.sha(path)
        b["complete"] = write(cp, c)
    with pytest.raises(ValueError):
        m.load_retained_reference(b, "R3-RI", old["validation"], server=False)


def native_result(arrays):
    return dict(
        prediction_loss=1,
        conditions=[
            dict(condition_id=p, control_row_ids_sha256="c" + p, truth_row_ids_sha256="t" + p)
            for p in arrays["condition_ids"]
        ],
        metric_gene_groups=dict(
            seen_expression=dict(
                metrics=[
                    dict(metric_id=k, macro_mean=0.2, finite_condition_count=4)
                    for k in (
                        "txpert_macro_pearson_delta_all",
                        "trishift_pearson_delta_all",
                        "systema_pearson_all",
                    )
                ]
            )
        ),
    )


def test_actual_paired_geometry_comparison_and_mismatched_rows_rejected():
    from gradpert.evaluation import response_geometry as geometry

    a = synthetic_arrays()
    n = native_result(a)
    result = m.paired_comparison(a, a, n, n, geometry, dict(draws=10, seed=20261028))
    assert result["ownmask_MSE_delta"]["candidate_minus_reference"] == 0
    b = copy.deepcopy(n)
    b["conditions"][0]["truth_row_ids_sha256"] = "changed"
    with pytest.raises(ValueError, match="ordered control/truth"):
        m.paired_comparison(a, a, n, b, geometry, dict(draws=10))
    wrong = copy.deepcopy(a)
    wrong["truth_delta"][0, 0] += 1
    with pytest.raises(ValueError, match="truth or axes"):
        m.paired_comparison(a, wrong, n, n, geometry, dict(draws=10))


def test_medium_new_profile_uses_same_exact_descriptor_guard(tmp_path):
    path = Path(m.__file__).with_name("response_anchor_medium.py")
    spec = importlib.util.spec_from_file_location("r15_medium_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    post = r15_fixture(tmp_path)
    e = post["endpoints"][0]
    medium = dict(native_source=e["source"], config=e["config"], run_manifest=e["run_manifest"])
    module.validate_r15_medium_identity(medium, server=False)
    identity = json.loads(Path(e["run_manifest"]["path"]).read_text())
    identity["data"]["response_anchor"]["weight"] = 0.1
    medium["run_manifest"] = write(Path(e["run_manifest"]["path"]), identity)
    with pytest.raises(ValueError):
        module.validate_r15_medium_identity(medium, server=False)


def parity_fixture(tmp_path, base_spec=None):
    from test_response_anchor_parity import fixture, pilot

    from gradpert.config import load_experiment_config
    from scripts.v2.capacity_probe import state_fingerprint, validation_state

    spec = r15_fixture(tmp_path) if base_spec is None else copy.deepcopy(base_spec)
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
    proof, _, _ = pilot.paired_evaluate(*args)
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
        endpoint="R15-DIR",
        spec=pilot_spec,
        spec_digest=m.digest(spec),
        endpoint_digest=m.digest(e),
        source=dict(commit=m.R15_SOURCE),
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


def test_r15_parity_receipt_binds_same_pilot_except_appended_proof(tmp_path):
    spec, _ = parity_fixture(tmp_path)
    m.validate_spec(spec, server=False)
    m.validate_observer_parity(spec, server=False)


@pytest.mark.parametrize(
    "mutation", ["endpoint", "state", "nonzero", "missing_counter", "different_pilot"]
)
def test_r15_parity_resealed_bad_proof_is_rejected(tmp_path, mutation):
    spec, receipt = parity_fixture(tmp_path)
    if mutation == "endpoint":
        receipt["endpoint"] = "FixedResponseGroupAnchor"
    elif mutation == "state":
        receipt["parity"]["state_after"] = {"state": "changed"}
    elif mutation == "nonzero":
        receipt["parity"]["max_abs"] = 1e-8
    elif mutation == "missing_counter":
        receipt["parity"]["forbidden_calls"] = []
    else:
        spec["validation"]["conditions"][0] = "changed"
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    with pytest.raises(ValueError, match="observer parity"):
        m.validate_observer_parity(spec, server=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_inputs",
        "missing_input",
        "wrong_input",
        "empty_counters",
        "missing_counter_key",
        "no_states",
        "empty_states",
        "bad_state_hash",
    ],
)
def test_actual_parity_proof_rejects_vacuous_or_incomplete_resealed_evidence(tmp_path, mutation):
    spec, receipt = parity_fixture(tmp_path)
    if mutation == "empty_inputs":
        receipt["input_files_sha256"] = {}
    elif mutation == "missing_input":
        del receipt["input_files_sha256"][
            str(Path(spec["endpoints"][0]["checkpoint"]["path"]).resolve())
        ]
    elif mutation == "wrong_input":
        receipt["input_files_sha256"][str(Path(spec["shared_policy"]["path"]).resolve())] = "0" * 64
    elif mutation == "empty_counters":
        receipt["parity"]["forbidden_calls"] = [{}, {}]
    elif mutation == "missing_counter_key":
        del receipt["parity"]["forbidden_calls"][0]["autograd_grad"]
    elif mutation == "no_states":
        receipt["parity"].pop("state_before")
        receipt["parity"].pop("state_after")
    elif mutation == "empty_states":
        receipt["parity"].update(state_before={}, state_after={})
    else:
        receipt["parity"]["state_before"]["protected"]["objective_state"] = ""
        receipt["parity"]["state_after"] = copy.deepcopy(receipt["parity"]["state_before"])
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    with pytest.raises(ValueError):
        m.validate_observer_parity(spec, server=False)


def orchestration_fixture(tmp_path, monkeypatch, *, fail_statistics=False):
    """Actual CLI/dispatch/merge/compare/validate; GPU and costly stats are explicit boundaries."""
    import shutil
    import sys
    from types import SimpleNamespace

    import gradpert

    root = Path(gradpert.__file__).resolve().parents[2]
    spec = r15_fixture(tmp_path / "candidate")
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
    spec, _ = parity_fixture(tmp_path / "pilot", spec)
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
        assert name == "R15-DIR"
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
        r["forbidden_calls"]["anchor_forward"] = 0
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


def test_actual_new_profile_cli_dispatch_worker_merge_compare_validate_chain(
    tmp_path, monkeypatch, capsys
):
    spec, input_spec, trace, invoke = orchestration_fixture(tmp_path, monkeypatch)
    assert [v for v in trace if v[0] == "worker"] == [
        ("worker", "R15-DIR", 0),
        ("worker", "R15-DIR", 1),
    ]
    assert len([v for v in trace if v[0] == "spawn"]) == 2 and ("compare",) in trace
    root = Path(spec["output"])
    before = {str(p): m.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        m,
        "merge_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("validate must not merge")),
    )
    invoke(["validate", "--spec", input_spec["path"], "--spec-sha256", input_spec["sha256"]])
    assert "validated_completed_result" in capsys.readouterr().out
    assert before == {str(p): m.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    c = json.loads((root / "COMPLETE.json").read_text())
    assert len(c["outputs"]) == 17 and c["input_spec"] == input_spec


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "extra_resealed",
        "missing",
        "tamper",
        "missing_complete_field",
        "comparison_roles_resealed",
        "candidate_counter_resealed",
        "retained_tamper",
        "parity_tamper",
        "pkl",
    ],
)
def test_new_profile_validate_rejects_output_or_external_provenance_failures(
    tmp_path, monkeypatch, mutation
):
    spec, input_spec, _, _ = orchestration_fixture(tmp_path, monkeypatch)
    root = Path(spec["output"])
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if mutation.startswith("extra"):
        write(root / "unlisted.json", {})
        if mutation == "extra_resealed":
            complete["outputs"]["unlisted.json"] = m.sha(root / "unlisted.json")
    elif mutation == "missing":
        (root / "R15-DIR-worker-0.log").unlink()
    elif mutation == "tamper":
        (root / "R15-DIR-worker-0.log").write_text("changed")
    elif mutation == "missing_complete_field":
        del complete["comparison_sha256"]
    elif mutation == "comparison_roles_resealed":
        p = root / "COMPARISON.json"
        v = json.loads(p.read_text())
        v["candidate"] = "FixedResponseGroupAnchor"
        write(p, v)
        complete["comparison_sha256"] = m.sha(p)
        complete["outputs"]["COMPARISON.json"] = m.sha(p)
    elif mutation == "candidate_counter_resealed":
        p = root / "R15-DIR/worker-0/COMPLETE.json"
        v = json.loads(p.read_text())
        v["forbidden_calls"] = {}
        write(p, v)
        a = root / "R15-DIR/COMPLETE.json"
        v = json.loads(a.read_text())
        v["worker_receipts_sha256"][0] = m.sha(p)
        write(a, v)
        complete["endpoint_receipts"]["R15-DIR"] = m.sha(a)
        complete["outputs"]["R15-DIR/COMPLETE.json"] = m.sha(a)
        complete["outputs"]["R15-DIR/worker-0/COMPLETE.json"] = m.sha(p)
    elif mutation == "retained_tamper":
        Path(spec["retained_references"]["R3-RI"]["complete"]["path"]).write_text("{}")
    elif mutation == "parity_tamper":
        Path(spec["observer_parity"]["path"]).write_text("{}")
    else:
        (root / "hidden.pkl").write_bytes(b"synthetic")
    write(complete_path, complete)
    monkeypatch.setattr(
        m,
        "merge_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("validate must not merge")),
    )
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        m.validate_r15_result(spec, spec_file=input_spec, server=False)


@pytest.mark.parametrize("missing_helper", [False, True])
def test_actual_worker_preloads_private_geometry_before_any_cuda_boundary(
    tmp_path, monkeypatch, missing_helper
):
    import shutil

    import torch

    import gradpert

    native_root = Path(gradpert.__file__).resolve().parents[2]
    spec = r15_fixture(tmp_path / "candidate")
    e = spec["endpoints"][0]
    e["source"]["root"] = str(native_root)
    e["config"] = binding(
        native_root
        / "configs/v2/response_anchor_direction_20261029/R15-DIR/gradpert_v2/nadig_jurkat.yaml"
    )
    analysis = Path(spec["analysis_source"]["root"])
    script = analysis / "scripts/v2/response_anchor_postfit.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(m.__file__, script)
    helper = analysis / "src/gradpert/evaluation/response_geometry.py"
    helper.parent.mkdir(parents=True)
    if not missing_helper:
        shutil.copyfile(native_root / "src/gradpert/evaluation/response_geometry.py", helper)
    spec, _ = parity_fixture(tmp_path / "pilot", spec)
    seal = write(tmp_path / "SPEC.json", spec)
    original = m.validate_spec
    monkeypatch.setattr(m, "validate_spec", lambda s, **kw: original(s, server=False))
    monkeypatch.setattr(m, "SERVER_ROOT", Path("/"))
    monkeypatch.setattr(m, "__file__", str(script))
    trace = []
    monkeypatch.setattr(m, "preimport_source", lambda binding: trace.append("source_guard"))
    real_geometry = m.geometry_module

    def load(root):
        trace.append("geometry_preload")
        return real_geometry(root)

    monkeypatch.setattr(m, "geometry_module", load)

    def cuda_boundary():
        trace.append("cuda_boundary")
        raise RuntimeError("synthetic CUDA boundary reached")

    monkeypatch.setattr(torch.cuda, "is_available", cuda_boundary)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = Path(gradpert.__file__).resolve()
    expected = (ValueError, FileNotFoundError) if missing_helper else RuntimeError
    with pytest.raises(expected):
        m.run_endpoint(
            spec,
            "R15-DIR",
            0,
            spec_file=seal,
            resource_owner=dict(path="not_reached", sha256="0" * 64),
        )
    assert trace[:3] == ["source_guard", "source_guard", "geometry_preload"]
    assert ("cuda_boundary" in trace) is (not missing_helper)
    assert Path(gradpert.__file__).resolve() == before


def test_new_dispatch_failure_preserves_partial_evidence_without_complete(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="synthetic statistics failure"):
        orchestration_fixture(tmp_path, monkeypatch, fail_statistics=True)
    root = tmp_path / "candidate/new/output"
    assert (root / "FAILURE.json").is_file() and not (root / "COMPLETE.json").exists()
    assert (root / "R15-DIR/COMPLETE.json").is_file()  # Computed candidate summaries survive.
    receipt = json.loads((root / "FAILURE.json").read_text())
    assert receipt["updates"] == 0 and receipt["scientific_complete"] is False


def test_legacy_validate_cli_remains_metadata_only_without_result_directory(
    tmp_path, monkeypatch, capsys
):
    import sys

    spec = spec_fixture(tmp_path)
    seal = write(tmp_path / "INPUT.json", spec)
    original = m.validate_spec
    monkeypatch.setattr(m, "validate_spec", lambda value, **kw: original(value, server=False))
    monkeypatch.setattr(m, "SERVER_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        sys,
        "argv",
        [m.__file__, "validate", "--spec", seal["path"], "--spec-sha256", seal["sha256"]],
    )
    m.main()
    assert "validated_metadata_only" in capsys.readouterr().out
    assert not Path(spec["output"]).exists()
