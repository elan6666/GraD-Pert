"""Bounded observer-off/on parity on the first frozen validation condition.

Run each endpoint separately, with its original native source and checkpoint.
The full445 postfit SPEC is unchanged. No full-val, training, or test path runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

import numpy as np


def postfit_module():
    path = Path(__file__).with_name("response_anchor_postfit.py")
    spec = importlib.util.spec_from_file_location("sealed_postfit_parity_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


postfit = postfit_module()


def expected_anchor_enabled(endpoint_name):
    if endpoint_name in ("R3-RI", *postfit.PRECISION_ARMS):
        return False
    if endpoint_name in (
        "FixedResponseGroupAnchor",
        "R15-DIR",
        "R16-GPM",
        "R17-GPM-HG0",
        "R18-CPD",
    ):
        return True
    raise ValueError("unknown native parity endpoint")


def first_condition(validation, ordered):
    if tuple(ordered) != tuple(validation["conditions"]) or len(ordered) != 445:
        raise ValueError("full frozen445 condition order differs")
    return tuple(ordered[:1])


def array_sha(array):
    array = np.ascontiguousarray(array)
    header = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":"))
    return hashlib.sha256(header.encode() + b"\0" + array.tobytes()).hexdigest()


def protected_state(runtime, *, validation_state, fingerprint):
    """Extend historical snapshots without modifying either native source.

    state_dict covers every objective parameter and persistent buffer, while
    named_buffers also covers nonpersistent clocks. Root scalar attributes bind
    unregistered clocks such as _temperature_updates and other private counters.
    """
    objective = runtime.objective
    scalars = {
        name: value
        for name, value in vars(objective).items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
    return dict(
        native=validation_state(runtime),
        objective_state=fingerprint(objective.state_dict()),
        objective_buffers=fingerprint(dict(objective.named_buffers())),
        objective_scalar_clocks=fingerprint(scalars),
    )


@contextmanager
def forbid_anchor_forward(runtime, student=None):
    """The optional Student anchor readout is outside this inference recipe."""
    counts = dict(anchor_forward=0)
    models = [runtime.objective.student]
    if student is not None and student is not runtime.objective.student:
        models.append(student)
    readouts = [getattr(model, "response_anchor_readout", None) for model in models]

    def forbidden(*args, **kwargs):
        counts["anchor_forward"] += 1
        raise ValueError("forbidden postfit path: anchor_forward")

    handles = [
        readout.register_forward_pre_hook(forbidden) for readout in readouts if readout is not None
    ]
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_prediction(native, *, controls, width):
    """Pass through the original object; retain a private RAM copy for parity."""
    original = native.predict_controls
    predictions = []

    @wraps(original)
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (controls, width)
            or not np.isfinite(value).all()
            or predictions
        ):
            raise ValueError("pilot requires exactly one finite full prediction")
        predictions.append(value.copy())
        return value

    native.predict_controls = capture
    try:
        yield predictions
    finally:
        native.predict_controls = original


def paired_evaluate(
    native,
    runtime,
    data,
    reference,
    collector,
    kwargs,
    state,
    admit,
    *,
    population_config=None,
    precision_spec=None,
    student=None,
    precision_receipt=None,
):
    """Delegate both passes to native.evaluate; compare its complete result."""
    if precision_spec is not None:
        postfit.assert_precision_evaluation_student(
            precision_spec, runtime.objective, student, precision_receipt
        )
    elif student is not None:
        raise ValueError("independent Student requires a precision SPEC")
    student = runtime.objective.student if student is None else student
    selected = tuple(collector.conditions)
    if len(selected) != 1 or tuple(kwargs["condition_ids"]) != selected:
        raise ValueError("pilot must evaluate exactly the first fixed condition")
    initial = state()
    if precision_spec is not None:
        from scripts.v2.capacity_probe import state_fingerprint

        scalar_diagnostics_before = postfit.precision_scalar_diagnostics(student, state_fingerprint)
    original_methods = (native.predict_controls, native.predict_query_set)
    original_hooks = dict(student.response._forward_hooks)
    predictions, results, counters, callback_counts = [], [], [], []
    for observed in (False, True):
        callbacks = []

        def condition_complete(completed, total, condition, callbacks=callbacks):
            if (completed, total, condition) != (1, 1, selected[0]) or callbacks:
                raise ValueError("pilot native condition callback differs")
            admit()
            callbacks.append(condition)

        admit()
        with (
            postfit.population_no_training_reads(population_config, runtime)
            if population_config is not None
            else nullcontext({}) as population_counts,
            postfit.forbid_updates(runtime) as forbidden,
            forbid_anchor_forward(runtime, student) as anchor_counts,
            postfit.forbid_precision_training_student(runtime)
            if precision_spec is not None
            else nullcontext({}) as precision_counts,
            postfit.precision_context(precision_spec or {}, kwargs["device"].type)
            if precision_spec is not None
            else nullcontext(),
            capture_prediction(native, controls=collector.controls, width=collector.width) as raw,
        ):
            arguments = dict(kwargs, on_condition_complete=condition_complete)
            if observed:
                with collector.observe(native, student, data):
                    result = native.evaluate(student, runtime.index, data, reference, **arguments)
            else:
                result = native.evaluate(student, runtime.index, data, reference, **arguments)
        if len(raw) != 1 or len(callbacks) != 1:
            raise ValueError("pilot native forward/callback coverage differs")
        after = state()
        if after != initial:
            paths = postfit.changed_state_paths(initial, after)
            if precision_spec is not None:
                paths.extend(
                    postfit.changed_state_paths(
                        scalar_diagnostics_before,
                        postfit.precision_scalar_diagnostics(student, state_fingerprint),
                        "evaluation_student.modules",
                    )
                )
            raise ValueError("pilot mutated protected state or RNG: " + ", ".join(paths))
        if (native.predict_controls, native.predict_query_set) != original_methods or dict(
            student.response._forward_hooks
        ) != original_hooks:
            raise ValueError("pilot failed to restore native hooks/methods")
        predictions.append(raw[0])
        results.append(result)
        counters.append(
            dict(
                forbidden,
                **(population_counts if population_config is not None else anchor_counts),
                **precision_counts,
            )
        )
        callback_counts.append(len(callbacks))
    a, b = predictions
    if a.dtype != b.dtype or not np.array_equal(a, b):
        raise ValueError("observer changed native predictions; exact parity required")
    if postfit.digest(results[0]) != postfit.digest(results[1]):
        raise ValueError("observer changed native metrics or ordered row identities")
    postfit.validate_native_result(results[1], collector)
    expected = dict(
        predict_controls=1,
        query_blocks=math.ceil(collector.width / collector.query_count),
        response_forward=math.ceil(collector.width / collector.query_count)
        * math.ceil(collector.controls / kwargs["cell_batch"]),
        truth_reads=1,
        control_reads=1,
    )
    if collector.calls != expected or any(any(c.values()) for c in counters):
        raise ValueError("observer forward coverage or zero-update contract differs")
    admit()
    return (
        dict(
            passed=True,
            comparison="exact_array_and_native_result_identity",
            atol=0,
            rtol=0,
            max_abs=0.0,
            prediction_shape=list(a.shape),
            prediction_sha256=[array_sha(v) for v in predictions],
            native_result_sha256=[postfit.digest(v) for v in results],
            observer_calls=collector.calls,
            forbidden_calls=counters,
            callback_counts=callback_counts,
            state_before=initial,
            state_after=state(),
            state_unchanged=True,
        ),
        results[1],
        collector.arrays(),
    )


def output_path(spec, endpoint, output):
    root = postfit.server_path(output, exists=False)
    protected = [
        spec["data_root"],
        spec["analysis_source"]["root"],
        spec["output"],
        *[e[k] for e in spec["endpoints"] for k in ("run_root",)],
        *[e["source"]["root"] for e in spec["endpoints"]],
    ]
    if spec.get("schema") == postfit.PRECISION_SCHEMA:
        queue = json.loads(postfit.checked(spec["precision"]["queue"]).read_text())
        protected.extend(
            [
                queue["output"],
                *[r["plan"][k] for r in queue["rows"] for k in ("run_root", "test_root")],
            ]
        )
        retained = spec["retained_references"]["R3-RI"]
        old = json.loads(postfit.checked(retained["postfit_spec"]).read_text())
        protected.append(old["output"])
    if any(
        root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root)
        for p in protected
    ):
        raise ValueError("pilot output overlaps protected input or full-val output")
    if endpoint not in [e["name"] for e in spec["endpoints"]] or root.exists():
        raise ValueError("fresh endpoint pilot output required")
    return root


def execute(spec, endpoint_name, *, spec_file, output):
    postfit.validate_spec(spec)
    precision = spec["schema"] == postfit.PRECISION_SCHEMA
    if precision and spec.get("observer_parity") is not None:
        raise ValueError("precision pilot requires an unpiloted immutable SPEC")
    if endpoint_name not in [e["name"] for e in spec["endpoints"]]:
        raise ValueError("pilot endpoint differs from sealed SPEC")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("native CUDA allocator contract missing")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1"):
        raise ValueError("one isolated physical GPU required")
    endpoint = next(e for e in spec["endpoints"] if e["name"] == endpoint_name)
    root = output_path(spec, endpoint_name, output)
    analysis_root = Path(spec["analysis_source"]["root"])
    if Path(__file__).resolve() != analysis_root / "scripts/v2/response_anchor_parity.py":
        raise ValueError("pilot must belong to the sealed analysis source")
    postfit.preimport_source(endpoint["source"])
    postfit.preimport_source(spec["analysis_source"])
    if spec["schema"] in (postfit.R15_SCHEMA, *postfit.POPULATION_SCHEMAS):
        postfit.geometry_module(Path(spec["analysis_source"]["root"]))
    native_root = Path(endpoint["source"]["root"])
    sys.path[:0] = [str(native_root / "src"), str(native_root)]
    import torch

    import gradpert

    if not Path(gradpert.__file__).resolve().is_relative_to(native_root / "src"):
        raise ValueError("mixed native source imports")
    from gradpert.config import load_experiment_config
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.evaluation.state import EvaluationStateLayout, load_evaluation_state
    from gradpert.execution.identity import inspect_environment
    from gradpert.training.v2 import evaluation as native
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.exposure import checkpoint_expression_groups
    from gradpert.training.v2.runtime import prepare_runtime
    from scripts.v2.capacity_probe import state_fingerprint, validation_state
    from scripts.v2.run_group import lock
    from scripts.v2.shared_gpu import admit, process_info, snapshot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("pilot needs exactly one visible native CUDA device")
    sources = [postfit._source(e) for e in (endpoint["source"], spec["analysis_source"])]
    config = load_experiment_config(postfit.checked(endpoint["config"]))
    identity = postfit._bind_checkpoint(endpoint, config)
    bindings = [
        spec_file,
        spec["shared_policy"],
        endpoint["run_manifest"],
        endpoint["journal"],
        endpoint["config"],
        endpoint["checkpoint"],
        endpoint["source"]["publication"],
        spec["analysis_source"]["publication"],
    ]
    if precision:
        bindings.extend(postfit.precision_inputs(spec, endpoint))
    files = {str(postfit.checked(b)): b["sha256"] for b in bindings}
    policy_doc = json.loads(postfit.checked(spec["shared_policy"]).read_text())
    policy = policy_doc.get("shared_gpu_policy", policy_doc)
    owner = process_info(os.getpid())
    if owner is None or owner["uid"] != os.getuid():
        raise ValueError("pilot live process identity unavailable")
    snapshots = []

    def admission():
        if process_info(os.getpid()) != owner:
            raise ValueError("pilot live process identity changed")
        observed = snapshot(os.getpid())
        admit(policy, observed)
        snapshots.append(postfit.digest(observed))

    root.mkdir(parents=True, exist_ok=False)
    try:
        with lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock"):
            admission()
            device = torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], device)
            torch.set_num_threads(1)
            common = dict(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=Path(spec["data_root"]),
            )
            with (
                postfit.population_no_training_reads(config)
                if spec["schema"] in postfit.POPULATION_SCHEMAS
                else nullcontext(),
                prepare_runtime(
                    config,
                    data_root=common["data_root"],
                    run_seed=int(identity["data"]["run_seed"]),
                    device=torch.device("cpu"),
                    purpose="evaluation",
                ) as runtime,
            ):
                if spec["schema"] in postfit.POPULATION_SCHEMAS:
                    postfit.assert_population_headless(runtime)
                if (
                    bool(getattr(runtime.options, "response_anchor_weight", 0))
                    != expected_anchor_enabled(endpoint_name)
                    or runtime.identity != identity["data"]
                ):
                    raise ValueError("native evaluation identity/anchor differs")
                progress = load_evaluation_checkpoint(
                    postfit.checked(endpoint["checkpoint"]),
                    runtime.objective,
                    training_identity=identity,
                    checkpoint_sha256=endpoint["checkpoint"]["sha256"],
                )
                runtime.objective.eval()
                student, precision_receipt = postfit.evaluation_model(
                    spec, runtime.objective, device
                )
                if runtime.objective.pending:
                    raise ValueError("checkpoint has pending Teacher statistics")
                reference = load_evaluation_state(
                    **common, validation_only=True, evaluation_protocol="v2"
                )
                layout = EvaluationStateLayout(
                    common["data_root"], common["dataset_id"], common["protocol_id"], True, "v2"
                )
                with CanonicalEvaluationData(**common, split_name="val") as data:
                    v = spec["validation"]
                    metadata = [
                        data.layout.manifests / "canonical.json",
                        data.layout.manifests / "split.json",
                        data.layout.manifests / "evaluation_controls.val.json",
                        layout.manifest,
                        layout.arrays,
                    ]
                    files.update({str(postfit.server_path(p)): postfit.sha(p) for p in metadata})
                    ordered = tuple(d.condition_id for d in data.control_manifest.draws)
                    allowed = np.asarray(v["allowed_gene_indices"], dtype=np.int64)
                    if (
                        ordered != tuple(v["conditions"])
                        or data.control_manifest_file_sha256 != v["control_manifest_sha256"]
                        or reference.manifest_file_sha256 != v["reference_sha256"]
                        or postfit.digest(list(data.expression_gene_ids))
                        != v["expression_gene_ids_sha256"]
                        or not np.array_equal(allowed, runtime.allowed_expression_ids)
                        or postfit.digest([data.expression_gene_ids[i] for i in allowed])
                        != v["allowed_gene_ids_sha256"]
                        or runtime.options.eval_query_count != v["query_count"]
                    ):
                        raise ValueError("frozen validation population/axis/recipe differs")
                    selected = first_condition(v, ordered)
                    gene_index = {g: i for i, g in enumerate(runtime.index.gene_ids)}
                    targets = {
                        p: tuple(
                            gene_index[g]
                            for g in p.split("+")
                            if g != data.split.control_condition_id
                        )
                        for p in selected
                    }
                    controls = {
                        d.condition_id: tuple(d.ordered_row_ids)
                        for d in data.control_manifest.draws
                        if d.condition_id in selected
                    }
                    groups, exposure = checkpoint_expression_groups(
                        progress["history"],
                        checkpoint_epoch=2,
                        expression_gene_ids=data.expression_gene_ids,
                        allowed_gene_indices=tuple(int(i) for i in allowed),
                    )
                    if tuple(groups["seen_expression"]) != tuple(allowed):
                        raise ValueError("checkpoint seen-expression axis differs")
                    collector = postfit.NativeValCollector(selected, targets, controls, allowed)
                    adata = data.layout.canonical_adata
                    original_stat = (adata.stat().st_size, adata.stat().st_mtime_ns)

                    def state():
                        return dict(
                            protected=protected_state(
                                runtime,
                                validation_state=validation_state,
                                fingerprint=state_fingerprint,
                            ),
                            **(
                                {
                                    "evaluation_student": postfit.precision_student_state(
                                        student, state_fingerprint
                                    )
                                }
                                if precision
                                else {}
                            ),
                            cuda_rng=state_fingerprint(torch.cuda.get_rng_state_all()),
                            canonical_stat=[adata.stat().st_size, adata.stat().st_mtime_ns],
                        )

                    parity, result, arrays = paired_evaluate(
                        native,
                        runtime,
                        data,
                        reference,
                        collector,
                        dict(
                            expected_split="val",
                            device=device,
                            cell_batch=int(config.training.eval_batch_size.value),
                            query_count=runtime.options.eval_query_count,
                            metric_gene_groups=groups,
                            condition_ids=selected,
                        ),
                        state,
                        admission,
                        precision_spec=spec if precision else None,
                        student=student if precision else None,
                        precision_receipt=precision_receipt.get("precision_policy"),
                        population_config=config
                        if spec["schema"] in postfit.POPULATION_SCHEMAS
                        else None,
                    )
                    if arrays["cls_means"].shape != (1, 256) or original_stat != (
                        adata.stat().st_size,
                        adata.stat().st_mtime_ns,
                    ):
                        raise ValueError("native CLS width or canonical data stat differs")
                    array_binding = postfit.save_arrays(root / "PRIVATE_ENDPOINT.npz", arrays)
                    result["expression_exposure"] = exposure
                    postfit.write_new(root / "PRIVATE_NATIVE_RESULT.json", result)
            admission()
            if any(postfit.sha(p) != h for p, h in files.items()) or sources != [
                postfit._source(e) for e in (endpoint["source"], spec["analysis_source"])
            ]:
                raise ValueError("pilot immutable input/source closure failed")
            if precision:
                queue = postfit.validate_precision_identity(endpoint, spec["precision"])
                postfit.precision_zero_pkl(queue, root, spec["output"])
            if list(root.rglob("*.pkl")):
                raise ValueError("pilot requires zero PKL")
            receipt = dict(
                schema=postfit.PRECISION_PARITY_SCHEMA
                if precision
                else "response-anchor-first-condition-parity-1",
                **precision_receipt,
                status="completed",
                endpoint=endpoint_name,
                source=sources[0],
                analysis_source=sources[1],
                spec=spec_file,
                spec_digest=postfit.digest(spec),
                endpoint_digest=postfit.digest(endpoint),
                condition=selected[0],
                full_validation_conditions=445,
                executed_conditions=1,
                controls=300,
                expression_genes=5000,
                query_count=1000,
                checkpoint=endpoint["checkpoint"],
                epoch=2,
                role="last",
                split="val",
                parity=parity,
                arrays=array_binding,
                native_result_sha256=postfit.sha(root / "PRIVATE_NATIVE_RESULT.json"),
                input_files_sha256=files,
                process=owner,
                admission_snapshot_sha256=snapshots,
                updates=0,
                teacher_updates=0,
                center_updates=0,
                zero_pkl=True,
                postfit_complete=False,
                scientific_complete=False,
                environment=inspect_environment(native_root, device_name="cuda:0").payload(),
            )
            postfit.write_new(root / "COMPLETE.json", receipt)
            return receipt
    except BaseException as error:
        postfit.write_new(
            root / "FAILURE.json",
            dict(status="failed", error_type=type(error).__name__, scientific_complete=False),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument(
        "--endpoint",
        choices=(
            "R3-RI",
            "FixedResponseGroupAnchor",
            "R15-DIR",
            "R16-GPM",
            "R17-GPM-HG0",
            "R18-CPD",
            *postfit.PRECISION_ARMS,
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec_file = dict(path=str(args.spec.resolve()), sha256=args.spec_sha256)
    spec = json.loads(postfit.checked(spec_file).read_text())
    execute(spec, args.endpoint, spec_file=spec_file, output=args.output)


if __name__ == "__main__":
    main()
