"""Precision native pilot contracts on synthetic CPU inputs; CUDA acceptance is separate."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_precision_postfit import precision_fixture, write
from test_response_anchor_parity import fixture, pilot
from torch import nn

from gradpert.config import load_experiment_config
from gradpert.data import DatasetLayout
from gradpert.evaluation.state import EvaluationStateLayout
from scripts.v2.capacity_probe import state_fingerprint, validation_state

m = pilot.postfit


def paired_fixture(*, failure=None, controls=5, width=7, query=3, batch=2):
    args, calls, _admission = fixture(controls=controls, width=width, query=query, batch=batch)
    args = list(args)
    native, runtime = args[:2]
    original = runtime.objective.student
    original.options = SimpleNamespace(
        ffn_compute_precision="autocast",
        ssl_head_precision="float32",
        prediction_precision="autocast",
    )
    student = copy.deepcopy(original)
    student.options.ssl_head_precision = "autocast"
    student.register_buffer("private_clock", torch.tensor(0), persistent=False)
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        evaluation_precision=m.PRECISION_POLICY,
        endpoints=[dict(name="PREC-SSL32")],
    )
    receipt = dict(
        evaluation_precision=m.PRECISION_POLICY,
        training=dict(
            ffn_compute_precision="autocast",
            ssl_head_precision="float32",
            prediction_precision="autocast",
        ),
        evaluation=dict(
            ffn_compute_precision="autocast",
            ssl_head_precision="autocast",
            prediction_precision="autocast",
        ),
        autocast_enabled=False,
        autocast_dtype=None,
        parameter_dtype="float32",
        backend=dict(
            cuda_matmul_allow_tf32=False, float32_matmul_precision="highest", cudnn_allow_tf32=False
        ),
    )
    args[5]["device"] = torch.device("cpu")

    def state():
        return dict(
            protected=pilot.protected_state(
                runtime, validation_state=validation_state, fingerprint=state_fingerprint
            ),
            evaluation_student=m.precision_student_state(student, state_fingerprint),
            cuda_rng=state_fingerprint([]),
            canonical_stat=[100, 200],
        )

    args[6] = state
    evaluate = native.evaluate

    def checked(model, *positional, **kwargs):
        assert model is student and not torch.is_autocast_enabled("cpu")
        result = evaluate(model, *positional, **kwargs)
        if failure == "student":
            with torch.no_grad():
                student.weight.add_(1)
        elif failure == "buffer":
            student.private_clock.add_(1)
        elif failure == "clock":
            runtime.objective._temperature_updates += 1
        elif failure == "rng":
            torch.rand(1)
        elif failure == "original_forward":
            original.response(torch.ones(1))
        elif failure == "anchor":
            student.response_anchor_readout(torch.ones(1))
        elif failure == "coverage":
            kwargs["on_condition_complete"](1, 1, "p0")
        return result

    native.evaluate = checked
    if failure == "anchor":
        student.response_anchor_readout = nn.Identity()
    return args, dict(precision_spec=spec, student=student, precision_receipt=receipt), calls


def test_both_native_passes_and_observer_share_separate_common_student_and_disabled_autocast():
    args, kwargs, calls = paired_fixture()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        proof, _, _ = pilot.paired_evaluate(*args, **kwargs)
        assert torch.is_autocast_enabled("cpu")
    assert len(calls) == 2 and proof["state_unchanged"]
    assert all(row["training_student_forward"] == 0 for row in proof["forbidden_calls"])


@pytest.mark.parametrize(
    "failure", ["student", "buffer", "clock", "rng", "original_forward", "anchor", "coverage"]
)
def test_mutations_forbidden_forward_and_incomplete_coverage_fail_restore(failure):
    args, kwargs, _ = paired_fixture(failure=failure)
    native, runtime = args[:2]
    methods = native.predict_controls, native.predict_query_set
    with pytest.raises(ValueError):
        pilot.paired_evaluate(*args, **kwargs)
    assert methods == (native.predict_controls, native.predict_query_set)
    assert not kwargs["student"].response._forward_hooks
    assert not runtime.objective.student.response._forward_pre_hooks
    assert not runtime.objective.teacher.response._forward_pre_hooks


@pytest.mark.parametrize("case", ["policy", "model", "alias", "weights"])
def test_wrong_policy_or_model_fails_before_forward(case):
    args, kwargs, calls = paired_fixture()
    if case == "policy":
        kwargs["precision_spec"]["evaluation_precision"] = "native"
    elif case == "model":
        kwargs["student"] = args[1].objective.student
    elif case == "alias":
        kwargs["student"].weight = args[1].objective.student.weight
    else:
        with torch.no_grad():
            kwargs["student"].weight.add_(1)
    with pytest.raises(ValueError):
        pilot.paired_evaluate(*args, **kwargs)
    assert not calls


def sealed_fixture(tmp_path):
    spec, queue = precision_fixture(tmp_path)
    endpoint = spec["endpoints"][0]
    config = load_experiment_config(endpoint["config"]["path"])
    batch = int(config.training.eval_batch_size.value)
    args, kw, _ = paired_fixture(controls=300, width=5000, query=1000, batch=batch)
    proof, _, _ = pilot.paired_evaluate(*args, **kw)
    spec["observer_parity"] = None
    sb = write(tmp_path / "pilot-spec.json", spec)
    root = tmp_path / "pilot"
    arrays = write(root / "PRIVATE_ENDPOINT.npz", {"synthetic": True})
    native = write(root / "PRIVATE_NATIVE_RESULT.json", {"synthetic": True})
    bindings = [
        sb,
        spec["shared_policy"],
        spec["analysis_source"]["publication"],
        endpoint["source"]["publication"],
        *[endpoint[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
        *m.precision_inputs(spec, endpoint, server=False),
    ]
    layout = DatasetLayout(Path(spec["data_root"]), config.dataset_id, config.data.protocol_id)
    reference = EvaluationStateLayout(
        Path(spec["data_root"]), config.dataset_id, config.data.protocol_id, True, "v2"
    )
    bindings.extend(
        write(layout.manifests / n, {})
        for n in ("canonical.json", "split.json", "evaluation_controls.val.json")
    )
    bindings.extend([write(reference.manifest, {}), write(reference.arrays, {})])

    def source(value):
        return dict(
            commit=value["commit"],
            repository_root=value["root"],
            dirty=False,
            publication_receipt_sha256=value["publication"]["sha256"],
        )

    receipt = dict(
        schema=m.PRECISION_PARITY_SCHEMA,
        status="completed",
        endpoint=endpoint["name"],
        source=source(endpoint["source"]),
        analysis_source=source(spec["analysis_source"]),
        spec=sb,
        spec_digest=m.digest(spec),
        endpoint_digest=m.digest(endpoint),
        condition=spec["validation"]["conditions"][0],
        full_validation_conditions=445,
        executed_conditions=1,
        controls=300,
        expression_genes=5000,
        query_count=1000,
        checkpoint=endpoint["checkpoint"],
        epoch=2,
        role="last",
        split="val",
        parity=proof,
        arrays=arrays,
        native_result_sha256=native["sha256"],
        input_files_sha256={str(Path(b["path"]).resolve()): b["sha256"] for b in bindings},
        updates=0,
        teacher_updates=0,
        center_updates=0,
        zero_pkl=True,
        postfit_complete=False,
        scientific_complete=False,
        precision_policy=kw["precision_receipt"],
    )
    spec["observer_parity"] = write(root / "COMPLETE.json", receipt)
    return spec, receipt, queue


def test_metadata_preparation_accepts_unpiloted_full445_but_dispatch_and_worker_refuse(
    tmp_path, monkeypatch
):
    spec, _ = precision_fixture(tmp_path)
    m.validate_spec(spec, server=False)
    binding = write(tmp_path / "spec.json", spec)
    checked = m.checked
    monkeypatch.setattr(m, "checked", lambda b, **kw: checked(b, server=False))
    validate = m.validate_spec
    monkeypatch.setattr(m, "validate_spec", lambda s, **kw: validate(s, server=False))
    with pytest.raises(ValueError, match="requires native"):
        m.dispatch(Path(binding["path"]), binding["sha256"])
    with pytest.raises(ValueError, match="requires native"):
        m.run_endpoint(spec, "PREC-SSL32", 0, spec_file=binding, resource_owner={})


def test_sealed_proof_binds_original_spec_without_circular_hash_and_no_baseline_forward(tmp_path):
    spec, receipt, _ = sealed_fixture(tmp_path)
    m.validate_spec(spec, server=False)
    assert m.validate_precision_parity(spec, server=False) == receipt
    original = json.loads(Path(receipt["spec"]["path"]).read_text())
    assert original["observer_parity"] is None and len(original["validation"]["conditions"]) == 445
    assert len(original["endpoints"]) == 1


@pytest.mark.parametrize(
    "case",
    [
        "source",
        "spec",
        "input",
        "input_missing",
        "coverage",
        "state",
        "policy",
        "counter",
        "forbiddenanchor",
        "extraendpoint",
        "pkl_other_arm",
        "pkl_pilot",
        "failed",
    ],
)
def test_resealed_bad_precision_parity_proofs_fail(tmp_path, case):
    spec, receipt, queue = sealed_fixture(tmp_path)
    if case == "source":
        receipt["source"]["commit"] = "f" * 40
    elif case == "spec":
        spec["medium"] = {"changed": True}
    elif case == "input":
        Path(spec["precision"]["controller_state"]["path"]).write_text("{}")
    elif case == "input_missing":
        del receipt["input_files_sha256"][spec["precision"]["family_manifest"]["path"]]
    elif case == "coverage":
        receipt["parity"]["observer_calls"]["query_blocks"] = 4
    elif case == "state":
        receipt["parity"]["state_after"]["evaluation_student"]["buffers"] = "f" * 64
    elif case == "policy":
        receipt["precision_policy"]["evaluation_precision"] = "native"
    elif case == "counter":
        receipt["parity"]["forbidden_calls"] = [{}, {}]
    elif case == "forbiddenanchor":
        receipt["parity"]["forbidden_calls"][0]["anchor_forward"] = 1
    elif case == "extraendpoint":
        spec["endpoints"].append(copy.deepcopy(spec["endpoints"][0]))
    elif case == "pkl_other_arm":
        write(Path(queue["rows"][1]["plan"]["test_root"]) / "hidden.pkl", {})
    elif case == "pkl_pilot":
        write(Path(spec["observer_parity"]["path"]).parent / "hidden.pkl", {})
    elif case == "failed":
        write(Path(spec["observer_parity"]["path"]).parent / "FAILURE.json", {})
    spec["observer_parity"] = write(Path(spec["observer_parity"]["path"]), receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.validate_precision_parity(spec, server=False)


@pytest.mark.parametrize("mutation", ["student", "buffer", "clock", "mode", "policy"])
def test_full_precision_worker_state_helper_rejects_independent_student_mutations(mutation):
    args, kwargs, _ = paired_fixture()
    runtime, student, spec = args[1], kwargs["student"], kwargs["precision_spec"]
    before = m.evaluation_protected_state(
        spec, runtime, validation_state, state_fingerprint, student=student
    )
    if mutation == "student":
        with torch.no_grad():
            student.weight.add_(1)
    elif mutation == "buffer":
        student.private_clock.add_(1)
    elif mutation == "clock":
        student.private_counter = 1
    elif mutation == "mode":
        student.response.train()
    else:
        student.options.ssl_head_precision = "float32"
    with pytest.raises(ValueError, match="mutated protected state"):
        m.verify_evaluation_state(
            spec, runtime, before, validation_state, state_fingerprint, student=student
        )


@pytest.mark.parametrize("forward", ["original", "anchor"])
def test_full_precision_worker_forbids_original_student_and_anchor_forward(forward):
    args, kwargs, _ = paired_fixture()
    runtime, student = args[1], kwargs["student"]
    student.response_anchor_readout = nn.Identity()
    with m.forbid_precision_training_student(runtime, student=student) as counts:
        with pytest.raises(ValueError, match="forbidden postfit path"):
            (
                runtime.objective.student.response
                if forward == "original"
                else student.response_anchor_readout
            )(torch.ones(1))
        assert (
            counts["training_student_forward" if forward == "original" else "anchor_forward"] == 1
        )
    assert not runtime.objective.student.response._forward_pre_hooks
    assert not student.response_anchor_readout._forward_pre_hooks


@pytest.mark.parametrize("case", ["missing_state", "missing_counter", "forward_nonzero"])
def test_full_worker_readback_rejects_missing_or_nonzero_protection(case):
    receipt = dict(
        state_before={
            "evaluation_student_" + k: "a" * 64
            for k in ("state", "buffers", "modes", "scalar_clocks")
        },
        forbidden_calls=dict.fromkeys(
            (
                "teacher_forward",
                "optimizer_step",
                "statistics_commit",
                "backward",
                "autograd_grad",
                "anchor_forward",
                "training_student_forward",
            ),
            0,
        ),
    )
    m.validate_precision_worker_protection(receipt)
    if case == "missing_state":
        del receipt["state_before"]["evaluation_student_buffers"]
    elif case == "missing_counter":
        del receipt["forbidden_calls"]["training_student_forward"]
    else:
        receipt["forbidden_calls"]["anchor_forward"] = 1
    with pytest.raises(ValueError):
        m.validate_precision_worker_protection(receipt)


def native_precision_models():
    from dataclasses import replace

    from test_components import fixture as native_fixture

    from gradpert.modeling.v2 import GraDPertV2
    from gradpert.training.v2.objective import JointObjective

    original, _ = native_fixture()
    model = GraDPertV2(
        original.graph.embedding.weight.detach(),
        replace(original.options, ssl_head_precision="float32"),
    )
    objective = JointObjective(model).eval()
    spec = dict(
        schema=m.PRECISION_SCHEMA,
        evaluation_precision=m.PRECISION_POLICY,
        endpoints=[dict(name="PREC-SSL32")],
    )
    return objective, spec


def test_native_cpu_lazy_sinkhorn_reproduced_then_primed_without_forward_tensor_or_rng_change():
    from gradpert.modeling.v2.operators import ManifoldResidual
    from gradpert.modeling.v2.precision_evaluation import evaluation_student

    objective, spec = native_precision_models()
    unprimed = evaluation_student(objective.student, m.PRECISION_POLICY).eval()
    layer = next(
        layer
        for layer in unprimed.modules()
        if type(layer) is ManifoldResidual and layer.streams > 1
    )
    x = torch.zeros(2, layer.streams, layer.norm.normalized_shape[0])
    before = m.precision_student_state(unprimed, state_fingerprint)
    with torch.inference_mode():
        layer.maps(x)
    after = m.precision_student_state(unprimed, state_fingerprint)
    assert m.changed_state_paths(before, after) == ["scalar_clocks"]
    assert layer.resolved_sinkhorn_backend == "native"

    objective_before = state_fingerprint(objective.state_dict())
    rng = torch.get_rng_state().clone()
    student, _ = m.evaluation_model(spec, objective, torch.device("cpu"))
    assert state_fingerprint(objective.state_dict()) == objective_before
    assert torch.equal(rng, torch.get_rng_state())
    before = m.precision_student_state(student, state_fingerprint)
    forward_hooks = [
        module.register_forward_pre_hook(
            lambda *args: pytest.fail("priming must not forward any module")
        )
        for module in student.modules()
    ]
    try:
        m.prime_precision_sinkhorn_diagnostics(student, torch.device("cpu"))
    finally:
        for hook in forward_hooks:
            hook.remove()
    assert before == m.precision_student_state(student, state_fingerprint)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.inference_mode():
        for layer in student.modules():
            if type(layer) is ManifoldResidual and layer.streams > 1:
                assert layer.resolved_sinkhorn_backend == "native"
                layer.maps(torch.zeros(2, layer.streams, layer.norm.normalized_shape[0]))
    assert before == m.precision_student_state(student, state_fingerprint)
    assert torch.equal(rng, torch.get_rng_state())
    assert state_fingerprint(objective.state_dict()) == objective_before


@pytest.mark.parametrize(
    "field,value",
    [
        ("sinkhorn_backend", "auto"),
        ("resolved_sinkhorn_backend", "unused"),
        ("fused_sinkhorn_diagnostic", True),
        ("sinkhorn_save_no_grad_diagnostic", True),
    ],
)
def test_primed_native_sinkhorn_requested_and_resolved_diagnostics_remain_guarded(field, value):
    from gradpert.modeling.v2.operators import ManifoldResidual

    objective, spec = native_precision_models()
    student, _ = m.evaluation_model(spec, objective, torch.device("cpu"))
    runtime = SimpleNamespace(objective=objective)

    def native_state(runtime):
        return {"original": state_fingerprint(runtime.objective.state_dict())}

    before = m.evaluation_protected_state(
        spec, runtime, native_state, state_fingerprint, student=student
    )
    layer = next(
        layer
        for layer in student.modules()
        if type(layer) is ManifoldResidual and layer.streams > 1
    )
    setattr(layer, field, value)
    with pytest.raises(ValueError, match="evaluation_student_scalar_clocks"):
        m.verify_evaluation_state(
            spec, runtime, before, native_state, state_fingerprint, student=student
        )


def test_state_mutation_error_names_only_changed_fingerprint_paths():
    args, kwargs, _ = paired_fixture(failure="buffer")
    with pytest.raises(ValueError, match=r"evaluation_student\.buffers") as error:
        pilot.paired_evaluate(*args, **kwargs)
    assert "tensor(" not in str(error.value)
    assert "tensor:" not in str(error.value)
