"""Frozen train-input panel audit; no objective, backward, or training updates.

Each endpoint runs in a fresh process importing its original training checkout.
Only aggregate feature statistics and small input/provenance receipts are saved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "postfit-spectral-audit-input-1"
NAMES = ("RI", "R12BAL", "R13SRAMP")
REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(binding):
    path = Path(binding["path"]).resolve(strict=True)
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise ValueError(f"immutable input hash differs: {path}")
    return path


def validate_spec(spec, *, server=True):
    if (
        set(spec)
        != {
            "schema",
            "analysis_source",
            "parent",
            "panel",
            "resource_queue",
            "data_root",
            "output",
            "endpoints",
        }
        or spec["schema"] != SCHEMA
    ):
        raise ValueError("audit input schema differs")
    endpoints = spec["endpoints"]
    if not isinstance(endpoints, list) or [e.get("name") for e in endpoints] != list(NAMES):
        raise ValueError("exact ordered RI/R12BAL/R13SRAMP endpoints required")
    paths = [Path(spec[k]).resolve() for k in ("data_root", "output")]
    for key in ("parent", "panel", "resource_queue"):
        paths.append(checked(spec[key]))
    for source in [spec["analysis_source"], *(e["source"] for e in endpoints)]:
        if set(source) != {"root", "commit", "publication", "publication_sha256"}:
            raise ValueError("source binding schema differs")
        if len(source["commit"]) != 40 or any(
            c not in "0123456789abcdef" for c in source["commit"]
        ):
            raise ValueError("full source commit required")
        paths.extend(
            (
                Path(source["root"]).resolve(),
                checked(
                    {
                        "path": source["publication"],
                        "sha256": source["publication_sha256"],
                    }
                ),
            )
        )
    protected = [Path(spec["data_root"]).resolve(), Path(spec["analysis_source"]["root"]).resolve()]
    for endpoint in endpoints:
        if set(endpoint) != {
            "name",
            "source",
            "run_root",
            "run_manifest_sha256",
            "config",
            "config_sha256",
            "checkpoint_sha256",
        }:
            raise ValueError("endpoint binding schema differs")
        root = Path(endpoint["run_root"]).resolve()
        source = Path(endpoint["source"]["root"]).resolve()
        config = checked({"path": endpoint["config"], "sha256": endpoint["config_sha256"]})
        if not config.is_relative_to(source / "configs"):
            raise ValueError("original training configuration must remain in its source")
        checked({"path": root / "run_manifest.json", "sha256": endpoint["run_manifest_sha256"]})
        protected.extend((root, source))
        paths.extend((root, source, config))
    output = Path(spec["output"]).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("audit output overlaps source, data, or retained training run")
    if server and any(not p.is_relative_to("/data/yilangliu") for p in paths):
        raise ValueError("scientific audit is server-only")


def verify_source(binding):
    from gradpert.execution.identity import inspect_source_identity

    identity = inspect_source_identity(
        binding["root"],
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=binding["publication"],
        expected_publication_receipt_sha256=binding["publication_sha256"],
    ).payload()
    if identity["commit"] != binding["commit"] or identity["dirty"]:
        raise ValueError("source differs from the sealed published commit")
    return identity


def physical_data(data, name):
    """Only the explicitly registered R12/R13 spectral metadata may differ."""
    result = copy.deepcopy(data)
    if not isinstance(result, dict) or not isinstance(result.get("loss_protocol"), dict):
        raise ValueError("original data/loss protocol identity required")
    protocol = result["loss_protocol"]
    if name == "RI":
        if "spectral_pair" in protocol:
            raise ValueError("original RI must not contain spectral-pair metadata")
    elif name in NAMES[1:]:
        if not isinstance(protocol.pop("spectral_pair", None), dict):
            raise ValueError("R12/R13 must retain their sealed spectral-pair metadata")
    else:
        raise ValueError("unknown postfit endpoint")
    return result


def validate_parity(parity):
    if set(parity) != {"student", "teacher"}:
        raise ValueError("both native role parity receipts required")
    for row in parity.values():
        if row.get("used_path_passed") is not True or row.get("cell_chunk_used") is not False:
            raise ValueError("unverified cache or rebatching execution path")
        for key in ("graph_max_abs", "response_max_abs"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 2e-6
            ):
                raise ValueError("native cached graph/response parity failed")


def spectral_statistics(values):
    """FP64 population covariance; no jitter, scale floor, or invented rank."""
    import torch

    x = values.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("finite nonempty row-by-feature matrix required")
    n, d = x.shape
    mean = x.mean(0)
    centered = x - mean
    total = x.square().sum(1).mean().item()
    trace = centered.square().sum(1).mean().item()
    mean_energy = mean.square().sum().item()
    ceiling = min(d, n - 1)
    rank = None
    if ceiling:
        if trace == 0:
            rank = 0.0
        else:
            # SVD avoids a covariance eigensolver's negative roundoff eigenvalues.
            eigenvalues = torch.linalg.svdvals(centered).square() / n
            probabilities = eigenvalues / eigenvalues.sum()
            positive = probabilities[probabilities > 0]
            rank = (-torch.sum(positive * positive.log())).exp().item()
    return dict(
        n=n,
        dimension=d,
        rank_ceiling=ceiling,
        trace=trace,
        total_energy=total,
        mean_energy=mean_energy,
        q=trace / total if total > 0 else None,
        effective_rank=rank,
        rho=rank / ceiling if ceiling and rank is not None else None,
        covariance_denominator="N",
        spectral_method="FP64 centered singular values; no jitter",
    )


def normalization_statistics(values, gamma):
    import torch

    x = values.detach().to(device="cpu", dtype=torch.float64)
    g = gamma.detach().to(device="cpu", dtype=torch.float64)
    if g.shape != (x.shape[-1],) or not torch.isfinite(g).all():
        raise ValueError("finite matching final RMSNorm gamma required")
    raw = spectral_statistics(x)
    zero_gamma = int((g == 0).sum())
    row_rms = x.square().mean(1).sqrt()
    zero_rows = int((row_rms == 0).sum())
    return dict(
        raw=raw,
        affine_free=(None if zero_gamma else spectral_statistics(x / g)),
        affine_free_status="NA_zero_gamma"
        if zero_gamma
        else "observed_same_features_divided_by_gamma",
        fixed_row_rms=(None if zero_rows else spectral_statistics(x / row_rms[:, None])),
        fixed_row_rms_status="NA_zero_row_energy"
        if zero_rows
        else "observed_same_features_unit_row_RMS",
        zero_gamma_features=zero_gamma,
        zero_energy_rows=zero_rows,
        gamma=dict(
            min=g.min().item(),
            median=g.median().item(),
            max=g.max().item(),
            norm=g.norm().item(),
            mean_abs=g.abs().mean().item(),
        ),
        raw_activations_saved=False,
    )


@contextmanager
def frozen_observation(runtime):
    """Restore RNG and modes; detect rather than conceal parameter/buffer changes."""
    import numpy as np
    import torch
    from capacity_probe import validation_state

    objective = runtime.objective
    before = validation_state(runtime)
    modes = [(m, m.training) for m in objective.modules()]
    rng = (
        random.getstate(),
        np.random.get_state(),
        copy.deepcopy(runtime.generator.bit_generator.state),
    )
    proof = {"before": before}
    try:
        with torch.random.fork_rng(devices=[0] if runtime.device.type == "cuda" else []):
            objective.eval()
            with torch.no_grad():
                yield proof
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        runtime.generator.bit_generator.state = rng[2]
        for module, training in modes:
            module.training = training
        proof["after"] = validation_state(runtime)
        proof["unchanged"] = proof["before"] == proof["after"]
        if not proof["unchanged"]:
            raise ValueError(
                "frozen observation changed parameters, buffers, gradients, optimizer, or state"
            )


def observed_cls(model, forward):
    """Check the returned CLS against the existing final RMSNorm hook, then close."""
    import torch

    captured = []

    def capture(_module, _args, output):
        captured.append(output[:, -1].detach())

    handle = model.response.norm.register_forward_hook(capture)
    try:
        result = forward()["response_cls"]
        if len(captured) != 1 or not torch.equal(captured[0], result):
            raise ValueError("ResponseCLS no longer equals the observed final RMSNorm CLS")
        if result.requires_grad or not torch.isfinite(result).all():
            raise ValueError("audit response must be finite and without autograd")
        return result.detach().to(device="cpu", dtype=torch.float64)
    finally:
        handle.remove()


def role_runtime(runtime, role):
    obj = runtime.objective
    proxy = SimpleNamespace(
        student=getattr(obj, role),
        _graph=obj._graph,
        _batch_graph=obj._batch_graph,
        _select_graph=obj._select_graph,
        response_metadata=obj.response_metadata,
    )
    return SimpleNamespace(objective=proxy, index=runtime.index, device=runtime.device)


def collect_group(runtime, batch):
    """Mirror native cell-loss encoder inputs without invoking its loss/objective."""
    obj = runtime.objective
    result = {}
    views = batch.cell_views if obj.weights[1][0] else batch.cell_views[:2]
    if not obj.lambda2 or len(views) < 2:
        raise ValueError("the sealed endpoints require native SSL2 Global views")
    for role in ("student", "teacher"):
        model = getattr(obj, role)
        prefix = (
            model.graph.relay_prefix(batch.graph.context)
            if model.options.control_conditioned_graph and batch.graph.context is not None
            else None
        )
        graph, condition = obj._batch_graph(model, batch, prefix=prefix)
        for i, view in enumerate(views if role == "student" else views[:2]):
            p = view.positions
            sg, sc = (
                obj._batch_graph(model, batch, view.mask if role == "student" else None, p, prefix)
                if model.options.control_conditioned_graph
                else (graph, condition)
            )
            args = (obj._select_graph(sg, batch.query_positions[p]), batch.control[:, p], sc)
            if role == "student":
                args += (view.mask,)
            name = f"{role}.global{i}" if i < 2 else f"{role}.local{i - 2}"
            result[name] = observed_cls(
                model,
                partial(
                    model.encode_response, *args, **obj.response_metadata(batch, p), decode=False
                ),
            )
    return result


def load_panel(spec):
    from representation_panel import panel_schedule

    panel = read(checked(spec["panel"]))
    expected = panel_schedule(
        read(checked(spec["parent"])), seed=panel["seed"], conditions=32, controls=4
    )
    if panel != expected or panel["conditions"] != 32 or panel["rows"] != 128:
        raise ValueError("panel differs from the frozen native 32-condition x 4-control schedule")
    return panel


def worker(spec, endpoint):
    import torch
    from cap40_posthoc import main_response, runtime_args, static_graph, verify_cache
    from checkpoint_endpoint import select_checkpoint
    from representation_panel import MatchedCellViews, assemble_control_only, validate_panel_runtime
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import _move_batch_tree

    source = verify_source(endpoint["source"])
    analysis = verify_source(spec["analysis_source"])
    _, config, identity, prepare, method_root = runtime_args(
        SimpleNamespace(
            method_sha=endpoint["source"]["commit"],
            config=Path(endpoint["config"]),
            run_root=Path(endpoint["run_root"]),
        )
    )
    if method_root != Path(endpoint["source"]["root"]).resolve() or identity["source"] != source:
        raise ValueError("native training imports or original run source identity differs")
    cp, checkpoint_endpoint = select_checkpoint(Path(endpoint["run_root"]), "last")
    checkpoint = Path(endpoint["run_root"]) / "fit" / cp["file"]
    if cp["sha256"] != endpoint["checkpoint_sha256"]:
        raise ValueError("retained last checkpoint differs from the sealed audit endpoint")
    panel = load_panel(spec)
    policy = read(checked(spec["resource_queue"]))["shared_gpu_policy"]
    owner = read(Path(spec["output"]) / "OWNER.json")
    observed = snapshot(owner["pid"])
    admit(policy, observed)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.current_stream(0) != torch.cuda.default_stream(0):
        raise ValueError("audit requires native default CUDA stream0")
    output = Path(spec["output"]) / endpoint["name"]
    output.mkdir()
    write_new(
        output / "ADMISSION.json", dict(policy=policy, observed=observed, gpu=0, stream="default")
    )
    banks, recipes, parities = {}, [], {}
    with prepare(
        config,
        data_root=Path(spec["data_root"]),
        run_seed=1,
        device=torch.device("cuda:0"),
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != identity["data"] or physical_data(
            runtime.identity, endpoint["name"]
        ) != physical_data(read(checked(spec["parent"]))["data"], "RI"):
            raise ValueError("runtime differs from the original training data identity")
        validate_panel_runtime(panel, runtime)
        load_evaluation_checkpoint(
            checkpoint,
            runtime.objective,
            training_identity=identity,
            checkpoint_sha256=cp["sha256"],
        )
        if (
            getattr(runtime.objective, "training_stage", "joint")
            != checkpoint_endpoint["training_stage"]
        ):
            raise ValueError("checkpoint training stage differs")
        gamma = {
            r: getattr(runtime.objective, r).response.norm.weight.detach().cpu().double().clone()
            for r in ("student", "teacher")
        }
        matched = MatchedCellViews()
        with frozen_observation(runtime) as state, torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(panel["seed"])
            caches = {r: static_graph(role_runtime(runtime, r)) for r in ("student", "teacher")}
            for number, group in enumerate(panel["schedule"]):
                admit(policy, snapshot(owner["pid"]))
                shallow = _move_batch_tree(
                    assemble_control_only(
                        runtime, group["rows"], group["queries"], group["view_seed"], deep=False
                    ),
                    runtime.device,
                )
                if shallow.graph.ids[shallow.query_positions].tolist() != group["queries"]:
                    raise ValueError("native query order differs")
                features = {}
                for role in ("student", "teacher"):
                    proxy = role_runtime(runtime, role)
                    if number == 0:
                        parities[role] = verify_cache(proxy, shallow, caches[role])
                    features[f"{role}.main"] = observed_cls(
                        getattr(runtime.objective, role),
                        partial(main_response, proxy, shallow, caches[role]),
                    )
                deep = _move_batch_tree(
                    assemble_control_only(
                        runtime, group["rows"], group["queries"], group["view_seed"], deep=True
                    ),
                    runtime.device,
                )
                if deep.graph.ids[deep.query_positions].tolist() != group["queries"]:
                    raise ValueError("deep native query order differs")
                deep, recipe = matched.bind(deep, group["rows"])
                recipes.append(
                    dict(
                        group_index=number,
                        condition_id=group["condition_id"],
                        rows=group["rows"],
                        **recipe,
                    )
                )
                features.update(collect_group(runtime, deep))
                for key, value in features.items():
                    if value.shape[0] != len(group["rows"]):
                        raise ValueError("native response row order/count differs")
                    banks.setdefault(key, []).append(value)
        statistics = {
            key: normalization_statistics(torch.cat(values), gamma[key.split(".")[0]])
            for key, values in banks.items()
        }
    if any(row["raw"]["n"] != 128 for row in statistics.values()):
        raise ValueError("audit statistics require the complete128-row population")
    validate_parity(parities)
    validate_spec(spec)
    if (
        sha(checkpoint) != endpoint["checkpoint_sha256"]
        or verify_source(endpoint["source"]) != source
    ):
        raise ValueError("training inputs changed during audit")
    if verify_source(spec["analysis_source"]) != analysis:
        raise ValueError("analysis source changed during audit")
    result = dict(
        schema="postfit-spectral-audit-result-1",
        endpoint=endpoint,
        source=source,
        analysis_source=analysis,
        training_identity=identity,
        checkpoint=cp,
        checkpoint_endpoint=checkpoint_endpoint,
        panel_sha256=spec["panel"]["sha256"],
        panel_schedule_sha256=panel["panel_sha256"],
        query_sha256=panel["query_sha256"],
        matched_views_sha256=digest(recipes),
        matched_views=recipes,
        parity=parities,
        state=state,
        updates=0,
        backward_calls=0,
        objective_calls=0,
        truth_expression_reads=0,
        raw_activations_saved=False,
        statistics=statistics,
        teacher_main="additional frozen diagnostic forward; not a native training main path",
        teacher_locals=None,
        teacher_locals_status="NA_not_native_teacher_views",
        population="128 crossed control-condition interventions; not medium506; ceiling=min(D,127)",
    )
    write_new(output / "COMPLETE.json", result)


def combine(results):
    if [r["endpoint"]["name"] for r in results] != list(NAMES):
        raise ValueError("complete ordered endpoint bank required")
    baseline = results[0]
    for result in results:
        if any(
            result[k] != baseline[k]
            for k in (
                "panel_sha256",
                "panel_schedule_sha256",
                "query_sha256",
                "matched_views_sha256",
            )
        ):
            raise ValueError("actual crossed population or matched views differ across endpoints")
        if (
            result["updates"] != 0
            or result["backward_calls"] != 0
            or result["objective_calls"] != 0
            or not result["state"]["unchanged"]
            or result["state"]["before"] != result["state"]["after"]
        ):
            raise ValueError("endpoint lacks exact zero-update/state parity proof")
        validate_parity(result["parity"])
        if physical_data(
            result["training_identity"]["data"], result["endpoint"]["name"]
        ) != physical_data(baseline["training_identity"]["data"], "RI"):
            raise ValueError("original endpoint data identities differ")
    return dict(
        schema="postfit-spectral-audit-complete-1",
        updates=0,
        endpoints=list(NAMES),
        panel_sha256=baseline["panel_sha256"],
        matched_views_sha256=baseline["matched_views_sha256"],
        interpretation=(
            "frozen diagnostic supplement; no recalibration, training, or scientific acceptance"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--endpoint", choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sha(args.spec) != args.spec_sha256:
        raise ValueError("input specification hash differs")
    spec = read(args.spec)
    validate_spec(spec)
    if Path(spec["analysis_source"]["root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("executed analysis script is outside the sealed analysis source")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0")
    ):
        raise ValueError("native allocator/determinism/GPU0 environment contract differs")
    from run_group import lock
    from shared_gpu import descendant, parent_map, process_info

    output = Path(spec["output"])
    if args.endpoint:
        if sha(output / "INPUT.json") != args.spec_sha256:
            raise ValueError("worker input differs from the coordinator's sealed input")
        owner = read(output / "OWNER.json")
        if process_info(owner["pid"]) != owner or not descendant(
            os.getpid(), owner["pid"], parent_map()
        ):
            raise ValueError("worker lacks the live owning coordinator and GPU lease")
        endpoint = next(e for e in spec["endpoints"] if e["name"] == args.endpoint)
        worker(spec, endpoint)
        return
    verify_source(spec["analysis_source"])
    with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):
        output.mkdir(parents=True, exist_ok=False)
        (output / "INPUT.json").write_bytes(args.spec.read_bytes())
        owner = process_info(os.getpid())
        if owner is None:
            raise ValueError("live coordinator process identity missing")
        write_new(output / "OWNER.json", owner)
        try:
            for endpoint in spec["endpoints"]:
                env = dict(os.environ, PYTHONPATH=str(Path(endpoint["source"]["root"]) / "src"))
                with (output / f"{endpoint['name']}.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--spec",
                            str(args.spec.resolve()),
                            "--spec-sha256",
                            args.spec_sha256,
                            "--endpoint",
                            endpoint["name"],
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            result = combine([read(output / name / "COMPLETE.json") for name in NAMES])
            result["result_sha256"] = {name: sha(output / name / "COMPLETE.json") for name in NAMES}
            result["spec_sha256"] = args.spec_sha256
            write_new(output / "COMPLETE.json", result)
        except BaseException as error:
            write_new(
                output / "FAILURE.json",
                dict(type=type(error).__name__, error=str(error), updates=0),
            )
            raise


if __name__ == "__main__":
    main()
