"""Sealed, val-only native postfit collection; no training or test execution.

Endpoint workers import their own original native source. The observer returns
all native outputs unchanged and collects ResponseCLS from the existing forward.
Scientific matrices stay in server-private NPZ; only small summaries are public.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

SERVER_ROOT = Path("/data/yilangliu")
SCHEMA = "response-anchor-postfit-input-1"
R15_SCHEMA = "response-anchor-postfit-r15-input-1"
R15_SOURCE = "aa65a42554d2f971adf0419bc1421e43c5a335d6"
R15_CONFIG_SHA256 = "4f50d4848d9f4ac2f19f81b5b44ab33d5371e1c327b2911023ba64f61c9519e7"
R15_WEIGHT = 0.04871788142525519
# Frozen Jurkat graph IDs include non-expression targets after the 5,000-gene prefix.
FROZEN_GRAPH_GENES = 6506
ARRAY_FIELDS = {
    "condition_ids",
    "allowed_gene_indices",
    "own_target_indices",
    "prediction_delta",
    "truth_delta",
    "cls_means",
}
REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if path.name.startswith("PRIVATE_"):
        path.chmod(0o600)


def server_path(path: str | Path, *, exists: bool = True) -> Path:
    p = Path(path)
    result = p.resolve(strict=exists)
    if not result.is_relative_to(SERVER_ROOT) or (exists and result != p.absolute()):
        raise ValueError("scientific input/output must be a canonical server path")
    return result


def checked(binding: dict, *, server: bool = True) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError("exact file path/hash binding required")
    path = server_path(binding["path"]) if server else Path(binding["path"]).resolve(strict=True)
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise ValueError("sealed file checksum mismatch")
    return path


def validate_spec(spec: dict, *, server: bool = True) -> None:
    required = {
        "schema",
        "analysis_source",
        "data_root",
        "shared_policy",
        "output",
        "validation",
        "endpoints",
        "bootstrap",
        "medium",
    }
    r15 = spec.get("schema") == R15_SCHEMA
    if r15:
        required |= {"retained_references", "observer_parity"}
    if set(spec) != required or spec["schema"] not in (SCHEMA, R15_SCHEMA):
        raise ValueError("postfit schema differs")
    v = spec["validation"]
    if set(v) != {
        "conditions",
        "controls_per_condition",
        "expression_genes",
        "query_count",
        "allowed_gene_indices",
        "allowed_gene_ids_sha256",
        "expression_gene_ids_sha256",
        "control_manifest_sha256",
        "reference_sha256",
    }:
        raise ValueError("complete frozen validation contract required")
    conditions, allowed = v["conditions"], np.asarray(v["allowed_gene_indices"])
    if (
        len(conditions) != 445
        or len(set(conditions)) != 445
        or any(not isinstance(p, str) or not p for p in conditions)
        or v["controls_per_condition"] != 300
        or v["expression_genes"] != 5000
        or v["query_count"] != 1000
        or allowed.dtype.kind not in "iu"
        or allowed.shape != (4775,)
        or (allowed < 0).any()
        or (allowed >= 5000).any()
        or (np.diff(allowed) <= 0).any()
    ):
        raise ValueError("full val445/300control/5000output/4775allowed contract differs")
    if spec["bootstrap"] != {"draws": 10000, "seed": 20261028}:
        raise ValueError("paired condition bootstrap is frozen")
    endpoints = spec["endpoints"]
    names = ["R15-DIR"] if r15 else ["R3-RI", "FixedResponseGroupAnchor"]
    if [e["name"] for e in endpoints] != names:
        raise ValueError("exact historical reference and one candidate required")
    protected = [Path(spec["data_root"]).resolve()]
    for e in endpoints:
        if set(e) != {
            "name",
            "source",
            "run_root",
            "run_manifest",
            "journal",
            "config",
            "checkpoint",
        }:
            raise ValueError("endpoint binding differs")
        for key in ("run_manifest", "journal", "config", "checkpoint"):
            checked(e[key], server=server)
        root = Path(e["run_root"]).resolve(strict=True)
        if Path(e["run_manifest"]["path"]).resolve() != root / "run_manifest.json":
            raise ValueError("run manifest escaped endpoint")
        if Path(e["journal"]["path"]).resolve() != root / "fit/epoch_state.json":
            raise ValueError("journal escaped endpoint")
        if not Path(e["checkpoint"]["path"]).resolve().is_relative_to(root / "fit"):
            raise ValueError("checkpoint escaped endpoint fit root")
        if (
            not Path(e["config"]["path"])
            .resolve()
            .is_relative_to(Path(e["source"]["root"]).resolve() / "configs")
        ):
            raise ValueError("endpoint config must belong to its frozen native source")
        protected.extend([root, Path(e["source"]["root"]).resolve()])
        _check_source_shape(e["source"], server=server)
        if server:
            server_path(root)
    if server and not r15:
        reference = endpoints[0]
        if (
            reference["source"]["commit"] != "f0cb20ebb2efab749970f2bf5d205ed2f358592c"
            or reference["checkpoint"]["sha256"]
            != "96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4"
            or reference["config"]["sha256"]
            != "65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7"
        ):
            raise ValueError("historical R3-RI reference differs from frozen design")
    _check_source_shape(spec["analysis_source"], server=server)
    checked(spec["shared_policy"], server=server)
    protected.append(Path(spec["analysis_source"]["root"]).resolve())
    output = Path(spec["output"]).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("postfit output overlaps protected source/data/run")
    if server:
        server_path(spec["data_root"])
        server_path(output, exists=False)
    # Metadata-only future medium binding; endpoint CLI never executes it.
    if not isinstance(spec["medium"], dict):
        raise ValueError("explicit medium binding/status required")
    if r15:
        validate_r15_payload(spec, server=server)


def _check_source_shape(source: dict, *, server: bool) -> None:
    if set(source) != {"root", "commit", "publication"}:
        raise ValueError("source root/full commit/publication binding required")
    if (
        not isinstance(source["commit"], str)
        or len(source["commit"]) != 40
        or any(c not in "0123456789abcdef" for c in source["commit"])
    ):
        raise ValueError("full source commit required")
    checked(source["publication"], server=server)
    if server:
        server_path(source["root"])


class NativeValCollector:
    """Bounded observer, exact control/chunk/block identities, no extra forward."""

    def __init__(
        self,
        conditions,
        targets_by_condition,
        controls_by_condition,
        allowed,
        *,
        controls=300,
        expression_genes=5000,
        query_count=1000,
    ):
        self.conditions = tuple(conditions)
        self.targets = {p: tuple(targets_by_condition[p]) for p in self.conditions}
        self.control_ids = {p: tuple(controls_by_condition[p]) for p in self.conditions}
        self.target_lookup = {target: p for p, target in self.targets.items()}
        self.allowed = np.asarray(allowed, dtype=np.int64)
        if (
            not self.conditions
            or len(set(self.conditions)) != len(self.conditions)
            or len(self.target_lookup) != len(self.conditions)
            or any(len(ids) != controls for ids in self.control_ids.values())
            or self.allowed.ndim != 1
            or not len(self.allowed)
            or (self.allowed < 0).any()
            or (self.allowed >= expression_genes).any()
            or (np.diff(self.allowed) <= 0).any()
        ):
            raise ValueError("collector axis/identity contract invalid")
        self.controls, self.width, self.query_count = controls, expression_genes, query_count
        self.rows = {}
        self.active = None
        self.block = None
        self.last_control_ids = None
        self.calls = {
            "predict_controls": 0,
            "query_blocks": 0,
            "response_forward": 0,
            "truth_reads": 0,
            "control_reads": 0,
        }

    def begin(self, controls, targets):
        targets = tuple(int(t) for t in targets)
        if targets not in self.target_lookup:
            raise ValueError("prediction condition outside frozen validation shard")
        p = self.target_lookup[targets]
        if p in self.rows or (self.active is not None and "truth" not in self.active):
            raise ValueError("duplicate condition or preceding truth not consumed")
        if self.last_control_ids != self.control_ids[p]:
            raise ValueError("ordered control IDs differ from frozen draw")
        x = np.asarray(controls)
        if x.shape != (self.controls, self.width) or not np.isfinite(x).all():
            raise ValueError("native validation control matrix differs")
        self.active = dict(
            condition=p,
            control=x.mean(0, dtype=np.float64),
            cls_sum=None,
            coverage=np.zeros(self.width, dtype=np.int64),
            next_query=0,
            control_hash=digest(list(self.last_control_ids)),
        )
        self.calls["predict_controls"] += 1

    def begin_block(self, queries):
        q = np.asarray(queries, dtype=np.int64)
        if self.active is None or self.block is not None:
            raise ValueError("query block outside active native prediction")
        start = self.active["next_query"]
        expected = np.arange(start, min(start + self.query_count, self.width))
        if not np.array_equal(q, expected):
            raise ValueError("native ordered-disjoint query recipe changed")
        self.block = dict(queries=q, rows=0)
        self.calls["query_blocks"] += 1

    def response_hook(self, _module, _args, output):
        if self.block is None or self.active is None:
            raise ValueError("Response forward outside native query block")
        # RelayResponseEncoder returns post-final-RMSNorm [B,Q+CLS,D].
        if not hasattr(output, "detach") or output.ndim != 3:
            raise ValueError("native Response output is not a token tensor")
        n, tokens = output.shape[:2]
        if tokens != len(self.block["queries"]) + 1 or n < 1:
            raise ValueError("native Response query/control shape differs")
        end = self.block["rows"] + n
        if end > self.controls:
            raise ValueError("Response chunks exceed frozen control count")
        z = output[:, -1].detach().double().cpu().numpy()
        if not np.isfinite(z).all():
            raise ValueError("nonfinite prehead ResponseCLS")
        if self.active["cls_sum"] is None:
            self.active["cls_sum"] = np.zeros(z.shape[1], dtype=np.float64)
        if self.active["cls_sum"].shape != (z.shape[1],):
            raise ValueError("ResponseCLS width changed")
        self.active["cls_sum"] += len(self.block["queries"]) * z.sum(0, dtype=np.float64)
        self.block["rows"] = end
        self.calls["response_forward"] += 1
        # None means the forward hook leaves native output untouched.
        return None

    def end_block(self):
        if self.block is None or self.block["rows"] != self.controls:
            raise ValueError("native query block omitted/repeated control rows")
        self.active["coverage"][self.block["queries"]] += self.block["rows"]
        self.active["next_query"] += len(self.block["queries"])
        self.block = None

    def end_prediction(self, prediction):
        x = np.asarray(prediction)
        if (
            self.active is None
            or self.block is not None
            or x.shape != (self.controls, self.width)
            or not np.isfinite(x).all()
            or not np.all(self.active["coverage"] == self.controls)
            or self.active["next_query"] != self.width
        ):
            raise ValueError("incomplete native prediction coverage")
        self.active["prediction"] = x.mean(0, dtype=np.float64)
        self.active["cls"] = self.active["cls_sum"] / (self.controls * self.width)

    def consume_truth(self, p, truth):
        x = np.asarray(truth.expression)
        if (
            self.active is None
            or self.active["condition"] != p
            or "prediction" not in self.active
            or p in self.rows
            or x.ndim != 2
            or not len(x)
            or x.shape[1] != self.width
            or not np.isfinite(x).all()
        ):
            raise ValueError("truth outside completed native condition")
        y = x.mean(0, dtype=np.float64)
        self.active["truth"] = y
        self.rows[p] = dict(
            prediction_delta=(self.active["prediction"] - self.active["control"])[self.allowed],
            truth_delta=(y - self.active["control"])[self.allowed],
            cls_means=self.active["cls"],
            control_sha256=self.active["control_hash"],
            truth_sha256=digest(list(truth.ordered_row_ids)),
        )
        self.calls["truth_reads"] += 1

    def arrays(self):
        if set(self.rows) != set(self.conditions) or self.block is not None:
            raise ValueError("incomplete frozen condition coverage")
        k = max(map(len, self.targets.values()))
        own = np.full((len(self.conditions), k), -1, dtype=np.int64)
        for i, p in enumerate(self.conditions):
            own[i, : len(self.targets[p])] = self.targets[p]
        return dict(
            condition_ids=np.asarray(self.conditions, dtype=str),
            allowed_gene_indices=self.allowed,
            own_target_indices=own,
            **{
                name: np.stack([self.rows[p][name] for p in self.conditions])
                for name in ("prediction_delta", "truth_delta", "cls_means")
            },
        )

    @contextmanager
    def observe(self, native, model, data):
        """Restore every method/global/hook, including all exception paths."""
        original_controls, original_query = native.predict_controls, native.predict_query_set
        originals = {
            name: (name in vars(data), vars(data).get(name), getattr(data, name))
            for name in ("load_control_rows", "load_truth_rows")
        }

        @wraps(original_controls)
        def predict(*args, **kwargs):
            bound = inspect.signature(original_controls).bind(*args, **kwargs)
            self.begin(bound.arguments["controls"], bound.arguments["targets"])
            result = original_controls(*args, **kwargs)
            self.end_prediction(result)
            return result

        @wraps(original_query)
        def query(*args, **kwargs):
            bound = inspect.signature(original_query).bind(*args, **kwargs)
            self.begin_block(bound.arguments["queries"])
            result = original_query(*args, **kwargs)
            self.end_block()
            return result

        def controls(ids):
            result = originals["load_control_rows"][2](ids)
            if tuple(result.ordered_row_ids) != tuple(ids):
                raise ValueError("native loader changed control order")
            self.last_control_ids = tuple(ids)
            self.calls["control_reads"] += 1
            return result

        def truth(p):
            result = originals["load_truth_rows"][2](p)
            self.consume_truth(p, result)
            return result

        handle = model.response.register_forward_hook(self.response_hook)
        native.predict_controls, native.predict_query_set = predict, query
        data.load_control_rows, data.load_truth_rows = controls, truth
        try:
            yield self
        finally:
            handle.remove()
            native.predict_controls, native.predict_query_set = original_controls, original_query
            for name, (owned, value, _) in originals.items():
                if owned:
                    setattr(data, name, value)
                else:
                    delattr(data, name)


def validate_native_result(result, collector):
    expected_metrics = {
        "txpert_macro_pearson_delta_all",
        "txpert_macro_pearson_delta_deg",
        "trishift_pearson_delta_all",
        "trishift_pearson_delta_deg",
        "systema_pearson_all",
        "systema_pearson_deg",
    }
    if (
        result["split"] != "val"
        or len(result["conditions"]) != len(collector.conditions)
        or set(result["metric_gene_groups"]) != {"seen_expression", "unseen_expression"}
    ):
        raise ValueError("native result lacks complete val18 population")
    groups = [result["metrics"]] + [v["metrics"] for v in result["metric_gene_groups"].values()]
    for values in groups:
        if len(values) != 6 or {v["metric_id"] for v in values} != expected_metrics:
            raise ValueError("native metric18 identities differ")
        for metric in values:
            if metric["total_condition_count"] != len(collector.conditions) or not 0 <= metric[
                "finite_condition_count"
            ] <= len(collector.conditions):
                raise ValueError("native finite counts differ")
    if not np.isfinite(result["prediction_loss"]):
        raise ValueError("nonfinite native validation loss")
    for row, p in zip(result["conditions"], collector.conditions, strict=True):
        if (
            row["condition_id"] != p
            or row["control_row_ids_sha256"] != collector.rows[p]["control_sha256"]
            or row["truth_row_ids_sha256"] != collector.rows[p]["truth_sha256"]
        ):
            raise ValueError("native/observer ordered row identity differs")


def native_point_gates(reference, candidate):
    metrics = (
        "txpert_macro_pearson_delta_all",
        "trishift_pearson_delta_all",
        "systema_pearson_all",
    )
    rows = {}
    for name in metrics:
        values = []
        for result in (reference, candidate):
            items = result["metric_gene_groups"]["seen_expression"]["metrics"]
            values.append(next(v for v in items if v["metric_id"] == name))
        r, c = values
        finite = (
            r["finite_condition_count"] == c["finite_condition_count"] == 445
            and r["macro_mean"] is not None
            and c["macro_mean"] is not None
            and np.isfinite([r["macro_mean"], c["macro_mean"]]).all()
        )
        delta = c["macro_mean"] - r["macro_mean"] if finite else None
        rows[name] = dict(candidate_minus_reference=delta, passed=bool(finite and delta >= 0))
    loss_delta = candidate["prediction_loss"] - reference["prediction_loss"]
    loss_passed = bool(np.isfinite(loss_delta) and loss_delta <= 0)
    return dict(
        seen_expression_all=rows,
        native_prediction_loss_delta=loss_delta,
        native_loss_passed=loss_passed,
        passed=loss_passed and all(row["passed"] for row in rows.values()),
    )


@contextmanager
def forbid_updates(runtime, *, forbid_anchor=False):
    """Count and forbid accidental Teacher/optimizer/commit/backward entry."""
    import torch

    counts = dict(
        teacher_forward=0, optimizer_step=0, statistics_commit=0, backward=0, autograd_grad=0
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            counts[name] += 1
            raise ValueError("forbidden postfit path: " + name)

        return fail

    originals = []

    def patch(owner, name, value):
        owned = name in vars(owner)
        originals.append((owner, name, owned, vars(owner).get(name), getattr(owner, name)))
        setattr(owner, name, value)

    readout = None
    if forbid_anchor:
        readout = getattr(runtime.objective.student, "response_anchor_readout", None)
        if readout is None:
            raise ValueError("R15 inference requires its retained anchor head")
    handles = [
        module.register_forward_pre_hook(forbidden("teacher_forward"))
        for module in runtime.objective.teacher.modules()
    ]
    if forbid_anchor:
        counts["anchor_forward"] = 0
        handles.append(readout.register_forward_pre_hook(forbidden("anchor_forward")))
    patch(runtime.optimizer, "step", forbidden("optimizer_step"))
    patch(runtime.objective, "commit_statistics", forbidden("statistics_commit"))
    patch(torch.Tensor, "backward", forbidden("backward"))
    patch(torch.autograd, "grad", forbidden("autograd_grad"))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, owned, own_value, _ in reversed(originals):
            if owned:
                setattr(owner, name, own_value)
            else:
                delattr(owner, name)


def geometry_module(analysis_root: Path):
    path = analysis_root / "src/gradpert/evaluation/response_geometry.py"
    spec = importlib.util.spec_from_file_location("_sealed_response_geometry", path)
    if spec is None or spec.loader is None:
        raise ValueError("published analysis statistics module missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_arrays(path: Path, arrays: dict, *, server: bool = True):
    if server:
        server_path(path, exists=False)
    if set(arrays) != ARRAY_FIELDS or path.exists():
        raise ValueError("fresh exact-six-field scientific NPZ required")
    # A file handle avoids NumPy silently appending a different filename.
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    path.chmod(0o600)
    return {"path": str(path), "sha256": sha(path)}


def load_arrays(binding: dict, *, server: bool = True):
    path = checked(binding, server=server)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_FIELDS or len(archive.files) != len(ARRAY_FIELDS):
            raise ValueError("scientific NPZ schema differs")
        return {name: archive[name] for name in ARRAY_FIELDS}


def preimport_source(binding):
    """Check clean commit before importing any endpoint model/CUDA module."""
    checked(binding["publication"])
    root = server_path(binding["root"])
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if commit != binding["commit"] or status.strip():
        raise ValueError("native/analysis source is dirty or differs before import")


def _source(binding):
    from gradpert.execution.identity import inspect_source_identity

    publication = checked(binding["publication"])
    method = json.loads(publication.read_text()).get("verification_method")
    if method not in {"git_ls_remote_live", "github_rest_live"}:
        raise ValueError("unsupported sealed source publication verification method")
    result = inspect_source_identity(
        binding["root"],
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=binding["publication"]["path"],
        expected_publication_receipt_sha256=binding["publication"]["sha256"],
        verification_method=method,
    ).payload()
    if result["commit"] != binding["commit"] or result["dirty"]:
        raise ValueError("native/analysis source differs from clean publication")
    return result


def _bind_checkpoint(endpoint, config):
    identity = json.loads(checked(endpoint["run_manifest"]).read_text())
    journal = json.loads(checked(endpoint["journal"]).read_text())
    last = journal["last"]
    cp = checked(endpoint["checkpoint"])
    if (
        journal["identity"] != identity
        or last["epoch"] != 2
        or last["sha256"] != endpoint["checkpoint"]["sha256"]
        or cp != Path(endpoint["run_root"]).resolve() / "fit" / last["file"]
        or identity["source"]["commit"] != endpoint["source"]["commit"]
    ):
        raise ValueError("exact original epoch2 last/source required")
    if digest(config.model_dump(mode="json")) != identity["resolved_config_sha256"]:
        raise ValueError("native config differs from training checkpoint identity")
    return identity


def _command_arguments(command: bytes):
    if not isinstance(command, bytes) or not command or command[-1:] != b"\0":
        raise ValueError("live raw NUL command bytes required")
    return [part.decode("utf-8") for part in command.split(b"\0") if part]


def _flag_value(arguments, flag):
    values = []
    for i, arg in enumerate(arguments):
        if arg == flag:
            if i + 1 >= len(arguments):
                raise ValueError("dispatcher command flag lacks a value")
            values.append(arguments[i + 1])
        elif arg.startswith(flag + "="):
            values.append(arg[len(flag) + 1 :])
    if len(values) != 1:
        raise ValueError("dispatcher command flag missing or repeated")
    return values[0]


def _raw_command(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes()


def make_dispatcher_owner(
    spec, endpoint, spec_file, *, lookup, pid=None, uid=None, command_reader=_raw_command
):
    """Seal our exact live dispatcher, never a foreign-job whitelist entry."""
    pid = os.getpid() if pid is None else pid
    uid = os.getuid() if uid is None else uid
    actual = lookup(pid)
    raw = command_reader(pid)
    arguments = _command_arguments(raw)
    scripts = [(i, arg) for i, arg in enumerate(arguments) if arg.endswith(".py")]
    script = str(
        Path(spec["analysis_source"]["root"]).resolve() / "scripts/v2/response_anchor_postfit.py"
    )
    if (
        len(scripts) != 1
        or Path(scripts[0][1]).resolve() != Path(script)
        or scripts[0][0] + 1 >= len(arguments)
        or arguments[scripts[0][0] + 1] != "run"
        or actual is None
        or actual.get("pid") != pid
        or actual.get("uid") != uid
        or actual.get("script_basename") != "response_anchor_postfit.py"
        or actual.get("command_sha256") != hashlib.sha256(raw).hexdigest()
        or not isinstance(actual.get("start_ticks"), str)
        or not actual["start_ticks"].isdigit()
    ):
        raise ValueError("owner is not our live sealed postfit dispatcher")
    spec_argument = _flag_value(arguments, "--spec")
    if (
        Path(spec_argument).resolve() != Path(spec_file["path"]).resolve()
        or _flag_value(arguments, "--spec-sha256") != spec_file["sha256"]
    ):
        raise ValueError("dispatcher command does not bind this POSTFIT_SPEC")
    command_contract = dict(
        script_path=script,
        script_argument=scripts[0][1],
        spec_path=str(Path(spec_file["path"]).resolve()),
        spec_argument=spec_argument,
        spec_file_sha256=spec_file["sha256"],
    )
    return dict(
        schema="response-anchor-postfit-dispatch-owner-1",
        uid=uid,
        process={
            key: actual[key]
            for key in ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
        },
        command_contract=command_contract,
        spec_file=dict(path=command_contract["spec_path"], sha256=spec_file["sha256"]),
        spec_sha256=digest(spec),
        endpoint=endpoint["name"],
        endpoint_spec_sha256=digest(endpoint),
        output_root=str(Path(spec["output"]).resolve()),
        analysis_source_commit=spec["analysis_source"]["commit"],
    )


def verified_dispatcher_owner(
    receipt,
    spec,
    endpoint,
    spec_file,
    *,
    lookup,
    parents,
    descendant,
    worker_pid=None,
    uid=None,
    command_reader=_raw_command,
):
    """Reject PID reuse, exec, nonancestor and another-run receipts before snapshot."""
    worker_pid = os.getpid() if worker_pid is None else worker_pid
    uid = os.getuid() if uid is None else uid
    expected_keys = {
        "schema",
        "uid",
        "process",
        "command_contract",
        "spec_file",
        "spec_sha256",
        "endpoint",
        "endpoint_spec_sha256",
        "output_root",
        "analysis_source_commit",
    }
    expected = receipt.get("process", {})
    pid = expected.get("pid")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != "response-anchor-postfit-dispatch-owner-1"
        or type(pid) is not int
        or pid <= 0
        or pid == worker_pid
        or receipt["uid"] != uid
        or expected.get("uid") != uid
        or receipt["spec_sha256"] != digest(spec)
        or receipt["endpoint"] != endpoint["name"]
        or receipt["endpoint_spec_sha256"] != digest(endpoint)
        or receipt["output_root"] != str(Path(spec["output"]).resolve())
        or receipt["analysis_source_commit"] != spec["analysis_source"]["commit"]
        or receipt["spec_file"]
        != dict(path=str(Path(spec_file["path"]).resolve()), sha256=spec_file["sha256"])
        or not descendant(worker_pid, pid, parents)
    ):
        raise ValueError("dispatcher owner is not a same-run ancestor")
    actual = lookup(pid)
    if (
        actual is None
        or any(
            actual.get(k) != expected.get(k)
            for k in ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
        )
        or actual.get("script_basename") != "response_anchor_postfit.py"
    ):
        raise ValueError("dispatcher PID/start ticks/raw command identity changed")
    raw = command_reader(pid)
    if hashlib.sha256(raw).hexdigest() != expected["command_sha256"]:
        raise ValueError("dispatcher raw NUL command changed")
    arguments = _command_arguments(raw)
    contract = receipt["command_contract"]
    script = str(
        Path(spec["analysis_source"]["root"]).resolve() / "scripts/v2/response_anchor_postfit.py"
    )
    positions = [i for i, arg in enumerate(arguments) if arg == contract["script_argument"]]
    if (
        contract["script_path"] != script
        or contract["spec_path"] != receipt["spec_file"]["path"]
        or contract["spec_file_sha256"] != spec_file["sha256"]
        or len(positions) != 1
        or positions[0] + 1 >= len(arguments)
        or arguments[positions[0] + 1] != "run"
        or _flag_value(arguments, "--spec") != contract["spec_argument"]
        or _flag_value(arguments, "--spec-sha256") != spec_file["sha256"]
    ):
        raise ValueError("live dispatcher command differs from same-run contract")
    return pid


class DispatcherAdmission:
    """Fresh native ownership/admission check at entry and each native condition callback."""

    def __init__(
        self,
        receipt,
        spec,
        endpoint,
        spec_file,
        policy,
        *,
        lookup,
        parent_lookup,
        descendant,
        snapshot,
        admit,
        worker_pid=None,
        uid=None,
        worker_index=None,
        command_reader=_raw_command,
    ):
        self.worker_index = worker_index
        self.receipt, self.spec, self.endpoint, self.spec_file = receipt, spec, endpoint, spec_file
        self.policy, self.lookup, self.parent_lookup = policy, lookup, parent_lookup
        self.descendant, self.snapshot, self.admit = descendant, snapshot, admit
        self.worker_pid, self.uid, self.command_reader = worker_pid, uid, command_reader
        self.calls, self.conditions_checked = 0, 0
        self.last_snapshot_sha256 = None

    def check(self):
        owner = verified_dispatcher_owner(
            self.receipt,
            self.spec,
            self.endpoint,
            self.spec_file,
            lookup=self.lookup,
            parents=self.parent_lookup(),
            descendant=self.descendant,
            worker_pid=self.worker_pid,
            uid=self.uid,
            command_reader=self.command_reader,
        )
        observed = self.snapshot(owner)
        # Original admit sees all dispatcher descendants as owned. No telemetry
        # edits, peer exceptions, foreign whitelist changes or increased budgets.
        self.admit(self.policy, observed)
        self.calls += 1
        self.last_snapshot_sha256 = digest(observed)
        return observed

    def on_condition_complete(self, done, total, condition):
        if self.worker_index not in (0, 1):
            raise ValueError("native admission callback requires worker index0/1")
        expected = self.spec["validation"]["conditions"][self.worker_index :: 2]
        if (
            done != self.conditions_checked + 1
            or total != len(expected)
            or not 1 <= done <= len(expected)
            or condition != expected[done - 1]
        ):
            raise ValueError("native admission callback condition order differs")
        self.check()
        self.conditions_checked = done


def run_endpoint(spec, name, worker_index, *, spec_file, resource_owner):
    """Formal server-only worker. Synthetic tests never call this entry point."""
    validate_spec(spec)
    if spec["schema"] == R15_SCHEMA:
        validate_observer_parity(spec)
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("native CUDA allocator contract missing")
    endpoints = {e["name"]: e for e in spec["endpoints"]}
    endpoint = endpoints[name]
    policy_document = json.loads(checked(spec["shared_policy"]).read_text())
    policy = policy_document.get("shared_gpu_policy", policy_document)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1") or worker_index not in (0, 1) or gpu != str(worker_index):
        raise ValueError("one isolated physical GPU and worker index0/1 required")
    analysis_script = (
        Path(spec["analysis_source"]["root"]) / "scripts/v2/response_anchor_postfit.py"
    )
    if Path(__file__).resolve() != analysis_script.resolve():
        raise ValueError("worker is not the sealed analysis script")
    native_root = Path(endpoint["source"]["root"]).resolve()
    preimport_source(endpoint["source"])
    preimport_source(spec["analysis_source"])
    if spec["schema"] == R15_SCHEMA:
        geometry_module(Path(spec["analysis_source"]["root"]))
    # Child is a fresh interpreter; pin native imports before importing torch/model code.
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
    from scripts.v2.shared_gpu import admit, descendant, parent_map, process_info, snapshot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("formal postfit needs one visible CUDA device per worker")
    native_source, analysis_source = _source(endpoint["source"]), _source(spec["analysis_source"])
    config = load_experiment_config(checked(endpoint["config"]))
    identity = _bind_checkpoint(endpoint, config)
    owner_path = checked(resource_owner)
    expected_owner_path = Path(spec["output"]).resolve() / name / "DISPATCH_OWNER.json"
    if owner_path != expected_owner_path:
        raise ValueError("resource owner receipt belongs to another endpoint")
    owner_receipt = json.loads(owner_path.read_text())
    admission_guard = DispatcherAdmission(
        owner_receipt,
        spec,
        endpoint,
        spec_file,
        policy,
        lookup=process_info,
        parent_lookup=parent_map,
        descendant=descendant,
        snapshot=snapshot,
        admit=admit,
        worker_index=worker_index,
    )
    file_bindings = [
        spec["shared_policy"],
        resource_owner,
        spec_file,
        endpoint["run_manifest"],
        endpoint["journal"],
        endpoint["config"],
        endpoint["checkpoint"],
        endpoint["source"]["publication"],
        spec["analysis_source"]["publication"],
    ]
    before_files = {str(checked(b)): b["sha256"] for b in file_bindings}
    root = server_path(Path(spec["output"]) / name / f"worker-{worker_index}", exists=False)
    if root.exists():
        raise ValueError("fresh immutable postfit worker output required")
    root.mkdir(parents=True)
    device = torch.device("cuda:0")
    try:
        with lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock"):
            admission = admission_guard.check()
            torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], device)
            torch.set_num_threads(1)
            common = dict(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=Path(spec["data_root"]),
            )
            with prepare_runtime(
                config,
                data_root=common["data_root"],
                run_seed=int(identity["data"]["run_seed"]),
                device=torch.device("cpu"),
                purpose="evaluation",
            ) as runtime:
                enabled = bool(getattr(runtime.options, "response_anchor_weight", 0))
                if enabled != (name in ("FixedResponseGroupAnchor", "R15-DIR")):
                    raise ValueError("candidate/reference anchor mechanism differs")
                if runtime.identity != identity["data"]:
                    raise ValueError("native evaluation identity differs from training data")
                if runtime.options.eval_query_count != spec["validation"]["query_count"]:
                    raise ValueError("native query count changed")
                progress = load_evaluation_checkpoint(
                    checked(endpoint["checkpoint"]),
                    runtime.objective,
                    training_identity=identity,
                    checkpoint_sha256=endpoint["checkpoint"]["sha256"],
                )
                runtime.objective.student.to(device)
                runtime.objective.eval()
                if runtime.objective.pending:
                    raise ValueError("evaluation checkpoint has pending Teacher statistics")
                state_layout = EvaluationStateLayout(
                    common["data_root"], common["dataset_id"], common["protocol_id"], True, "v2"
                )
                reference = load_evaluation_state(
                    **common, validation_only=True, evaluation_protocol="v2"
                )
                with CanonicalEvaluationData(**common, split_name="val") as data:
                    v = spec["validation"]
                    metadata_paths = [
                        data.layout.manifests / "canonical.json",
                        data.layout.manifests / "split.json",
                        data.layout.manifests / "evaluation_controls.val.json",
                        state_layout.manifest,
                        state_layout.arrays,
                    ]
                    before_files.update(
                        {str(server_path(path)): sha(path) for path in metadata_paths}
                    )
                    ordered = tuple(draw.condition_id for draw in data.control_manifest.draws)
                    if (
                        ordered != tuple(v["conditions"])
                        or data.control_manifest_file_sha256 != v["control_manifest_sha256"]
                        or reference.manifest_file_sha256 != v["reference_sha256"]
                        or digest(list(data.expression_gene_ids)) != v["expression_gene_ids_sha256"]
                    ):
                        raise ValueError("val population/reference/axis differs")
                    allowed = np.asarray(v["allowed_gene_indices"], dtype=np.int64)
                    if (
                        not np.array_equal(allowed, runtime.allowed_expression_ids)
                        or digest([data.expression_gene_ids[i] for i in allowed])
                        != v["allowed_gene_ids_sha256"]
                    ):
                        raise ValueError("training allowed gene axis differs")
                    selected = ordered[worker_index::2]
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
                        raise ValueError("native seen-expression axis is not the frozen4775")
                    data_binding = dict(
                        canonical_adata_declared_sha256=data.manifest.canonical_adata_sha256,
                        observation_order_sha256=data.manifest.observation_order_sha256,
                        split_content_sha256=data.split.split_content_sha256,
                    )
                    collector = NativeValCollector(selected, targets, controls, allowed)
                    before = validation_state(runtime)
                    cuda_before = state_fingerprint(torch.cuda.get_rng_state_all())
                    adata_path = data.layout.canonical_adata
                    data_stat = (adata_path.stat().st_size, adata_path.stat().st_mtime_ns)
                    with (
                        forbid_updates(
                            runtime, forbid_anchor=spec["schema"] == R15_SCHEMA
                        ) as forbidden_counts,
                        collector.observe(native, runtime.objective.student, data),
                    ):
                        result = native.evaluate(
                            runtime.objective.student,
                            runtime.index,
                            data,
                            reference,
                            expected_split="val",
                            device=device,
                            cell_batch=int(config.training.eval_batch_size.value),
                            query_count=runtime.options.eval_query_count,
                            metric_gene_groups=groups,
                            condition_ids=selected,
                            on_condition_complete=admission_guard.on_condition_complete,
                        )
                    if (
                        validation_state(runtime) != before
                        or state_fingerprint(torch.cuda.get_rng_state_all()) != cuda_before
                        or data_stat != (adata_path.stat().st_size, adata_path.stat().st_mtime_ns)
                    ):
                        raise ValueError("evaluation mutated protected state/RNG/data file")
                    if admission_guard.conditions_checked != len(selected):
                        raise ValueError("native admission callbacks did not cover all conditions")
                    arrays = collector.arrays()
                    if arrays["cls_means"].shape != (len(selected), 256):
                        raise ValueError("formal ResponseCLS must have frozen width256")
                    validate_native_result(result, collector)
                    arrays_binding = save_arrays(root / "PRIVATE_ENDPOINT.npz", arrays)
                    result["expression_exposure"] = exposure
                    write_new(root / "PRIVATE_NATIVE_RESULT.json", result)
                    geo = geometry_module(Path(spec["analysis_source"]["root"]))
                    summary = geo.summarize_endpoint(**arrays)
                    # Row IDs remain in server-private native results, never the public summary.
                    write_new(root / "SUMMARY.json", summary)
                    rows_identity = digest(
                        [
                            dict(
                                condition=p,
                                control=collector.rows[p]["control_sha256"],
                                truth=collector.rows[p]["truth_sha256"],
                            )
                            for p in selected
                        ]
                    )
            admission_guard.check()
            if (
                any(sha(p) != h for p, h in before_files.items())
                or _source(endpoint["source"]) != native_source
                or _source(spec["analysis_source"]) != analysis_source
            ):
                raise ValueError("postfit immutable source/input closure failed")
            if list(root.rglob("*.pkl")):
                raise ValueError("postfit requires zero PKL")
            complete = dict(
                schema="response-anchor-postfit-endpoint-worker-1",
                status="completed",
                endpoint=name,
                worker_index=worker_index,
                source=native_source,
                analysis_source=analysis_source,
                postfit_spec_sha256=digest(spec),
                endpoint_spec_sha256=digest(endpoint),
                validation_spec_sha256=digest(v),
                checkpoint_sha256=endpoint["checkpoint"]["sha256"],
                epoch=2,
                role="last",
                split="val",
                conditions=list(selected),
                arrays=arrays_binding,
                summary_sha256=sha(root / "SUMMARY.json"),
                calls=collector.calls,
                forbidden_calls=forbidden_counts,
                data_binding=data_binding,
                canonical_adata_stat_before=list(data_stat),
                native_result={
                    "path": str(root / "PRIVATE_NATIVE_RESULT.json"),
                    "sha256": sha(root / "PRIVATE_NATIVE_RESULT.json"),
                },
                row_identity_sha256=rows_identity,
                control_manifest_sha256=v["control_manifest_sha256"],
                reference_sha256=v["reference_sha256"],
                input_files_sha256=before_files,
                state_before=before,
                state_unchanged=True,
                cuda_rng_unchanged=True,
                updates=0,
                teacher_updates=0,
                center_updates=0,
                zero_pkl=True,
                resource_admission=admission,
                resource_owner=resource_owner,
                dispatcher_process=owner_receipt["process"],
                resource_checks=dict(
                    calls=admission_guard.calls,
                    conditions_checked=admission_guard.conditions_checked,
                    last_snapshot_sha256=admission_guard.last_snapshot_sha256,
                ),
                environment=inspect_environment(native_root, device_name="cuda:0").payload(),
                scientific_complete=False,
            )
            write_new(root / "COMPLETE.json", complete)
            return complete
    except BaseException as error:
        write_new(
            root / "FAILURE.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                scientific_complete=False,
                updates=0,
            ),
        )
        raise


def merge_endpoint(spec, name):
    """CPU aggregate merge; scientific matrices never leave the server."""
    from gradpert.training.v2.evaluation import merge_evaluation_shards

    root = server_path(Path(spec["output"]) / name)
    receipts = [json.loads((root / f"worker-{i}/COMPLETE.json").read_text()) for i in range(2)]
    endpoint = next(e for e in spec["endpoints"] if e["name"] == name)
    if receipts[0]["resource_owner"] != receipts[1]["resource_owner"]:
        raise ValueError("endpoint workers did not share a sealed dispatcher owner")
    native = []
    by_condition = {}
    first = None
    for i, r in enumerate(receipts):
        if (
            r["status"] != "completed"
            or r["endpoint"] != name
            or r["worker_index"] != i
            or r["postfit_spec_sha256"] != digest(spec)
            or r["endpoint_spec_sha256"] != digest(endpoint)
            or r["validation_spec_sha256"] != digest(spec["validation"])
            or r["conditions"] != spec["validation"]["conditions"][i::2]
            or not r["state_unchanged"]
            or not r["cuda_rng_unchanged"]
            or not r["zero_pkl"]
            or any(r[k] != 0 for k in ("updates", "teacher_updates", "center_updates"))
            or (root / f"worker-{i}/FAILURE.json").exists()
        ):
            raise ValueError("endpoint worker acceptance differs")
        if (
            r["source"]["commit"] != endpoint["source"]["commit"]
            or r["analysis_source"]["commit"] != spec["analysis_source"]["commit"]
            or r["summary_sha256"] != sha(root / f"worker-{i}/SUMMARY.json")
            or r["resource_checks"]["conditions_checked"] != len(r["conditions"])
            or r["resource_checks"]["calls"] != len(r["conditions"]) + 2
            or any(r["forbidden_calls"].values())
            or any(sha(path) != h for path, h in r["input_files_sha256"].items())
        ):
            raise ValueError("worker source/input/forbidden-call closure changed")
        a = load_arrays(r["arrays"])
        if a["condition_ids"].tolist() != r["conditions"]:
            raise ValueError("array condition order differs from receipt")
        if first is None:
            first = a
        elif not np.array_equal(a["allowed_gene_indices"], first["allowed_gene_indices"]):
            raise ValueError("worker allowed axes differ")
        for row, p in enumerate(a["condition_ids"].tolist()):
            if p in by_condition:
                raise ValueError("condition duplicated across shards")
            by_condition[p] = {
                k: a[k][row] for k in ARRAY_FIELDS - {"condition_ids", "allowed_gene_indices"}
            }
        native.append(json.loads(checked(r["native_result"]).read_text()))
    ordered = spec["validation"]["conditions"]
    if set(by_condition) != set(ordered):
        raise ValueError("not full val445")
    arrays = dict(
        condition_ids=np.asarray(ordered, dtype=str),
        allowed_gene_indices=first["allowed_gene_indices"],
        **{
            k: np.stack([by_condition[p][k] for p in ordered])
            for k in ARRAY_FIELDS - {"condition_ids", "allowed_gene_indices"}
        },
    )
    geo = geometry_module(Path(spec["analysis_source"]["root"]))
    native_merged = merge_evaluation_shards(native, ordered_conditions=tuple(ordered))
    if len(native_merged["conditions"]) != 445:
        raise ValueError("native val coverage differs")
    binding = save_arrays(root / "PRIVATE_ENDPOINT.npz", arrays)
    write_new(root / "PRIVATE_NATIVE_RESULT.json", native_merged)
    write_new(
        root / "SUMMARY.json",
        dict(
            native={k: v for k, v in native_merged.items() if k != "conditions"},
            geometry=geo.summarize_endpoint(**arrays),
            scientific_complete=False,
        ),
    )
    write_new(
        root / "COMPLETE.json",
        dict(
            status="completed",
            endpoint=name,
            arrays=binding,
            native_result_sha256=sha(root / "PRIVATE_NATIVE_RESULT.json"),
            summary_sha256=sha(root / "SUMMARY.json"),
            worker_receipts_sha256=[sha(root / f"worker-{i}/COMPLETE.json") for i in range(2)],
            validation_spec_sha256=digest(spec["validation"]),
            conditions=445,
            updates=0,
            zero_pkl=not list(root.rglob("*.pkl")),
            scientific_complete=False,
        ),
    )
    return arrays, native_merged


def compare(spec, *, spec_file=None):
    validate_spec(spec)
    _source(spec["analysis_source"])
    root = server_path(spec["output"])
    if (root / "FAILURE.json").exists():
        raise ValueError("failed postfit output cannot be promoted")
    geo = geometry_module(Path(spec["analysis_source"]["root"]))
    if spec["schema"] == R15_SCHEMA:
        return compare_r15(spec, geo, spec_file=spec_file)
    ref, refnative = merge_endpoint(spec, "R3-RI")
    candidate, candidate_native = merge_endpoint(spec, "FixedResponseGroupAnchor")
    # Exact per-condition control/truth hashes, not merely equal macro counts/seeds.
    for a, b in zip(refnative["conditions"], candidate_native["conditions"], strict=True):
        for key in ("condition_id", "control_row_ids_sha256", "truth_row_ids_sha256"):
            if a[key] != b[key]:
                raise ValueError("endpoint frozen row identities differ")
    if not np.array_equal(ref["truth_delta"], candidate["truth_delta"]) or not np.array_equal(
        ref["own_target_indices"], candidate["own_target_indices"]
    ):
        raise ValueError("endpoint raw truth/target axes differ")
    result = geo.compare_endpoints(ref, candidate, **spec["bootstrap"])
    result["native_point_gates"] = native_point_gates(refnative, candidate_native)

    def positive_ci(name):
        ci = result[name]["CI95"]
        return ci is not None and ci[0] > 0

    result["single_candidate_supported"] = bool(
        all(result["full_axis_finite"].values())
        and result["candidate_new_Pearson_NA"] == 0
        and positive_ci("ownmask_Pearson_delta")
        and positive_ci("RSA_delta")
        and result["ownmask_MSE_delta"]["candidate_minus_reference"] is not None
        and result["ownmask_MSE_delta"]["candidate_minus_reference"] <= 0
        and result["native_point_gates"]["passed"]
    )
    result["scientific_complete"] = False
    if list(root.rglob("*.pkl")):
        raise ValueError("postfit requires zero PKL across full output")
    write_new(root / "COMPARISON.json", result)
    write_new(
        root / "COMPLETE.json",
        dict(
            schema="response-anchor-val-postfit-complete-1",
            stage="validation_and_response_geometry",
            postfit_complete=False,
            status="completed",
            spec_sha256=digest(spec),
            comparison_sha256=sha(root / "COMPARISON.json"),
            endpoint_receipts={
                n: sha(root / n / "COMPLETE.json") for n in ("R3-RI", "FixedResponseGroupAnchor")
            },
            medium_status="not_executed_by_val_worker",
            test_status="deferred",
            updates=0,
            scientific_complete=False,
            zero_pkl=not list(root.rglob("*.pkl")),
        ),
    )
    return result


def dispatch(spec_path: Path, expected_sha: str):
    spec_path = checked(dict(path=str(spec_path), sha256=expected_sha))
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    if spec["schema"] == R15_SCHEMA:
        validate_observer_parity(spec)
    root = server_path(spec["output"], exists=False)
    analysis = Path(spec["analysis_source"]["root"])
    script = analysis / "scripts/v2/response_anchor_postfit.py"
    if Path(__file__).resolve() != script.resolve():
        raise ValueError("dispatcher is not sealed analysis script")
    preimport_source(spec["analysis_source"])
    sys.path.insert(0, str(analysis.resolve()))
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "SPEC.json", spec)
    try:
        from scripts.v2.shared_gpu import process_info

        for endpoint in spec["endpoints"]:
            owner = make_dispatcher_owner(
                spec, endpoint, dict(path=str(spec_path), sha256=expected_sha), lookup=process_info
            )
            owner_path = root / endpoint["name"] / "DISPATCH_OWNER.json"
            write_new(owner_path, owner)
            owner_sha256 = sha(owner_path)
            processes = []
            try:
                for i, gpu in enumerate(("0", "1")):
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu,
                        PYTORCH_ALLOC_CONF="expandable_segments:True",
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    )
                    env["PYTHONPATH"] = os.pathsep.join(
                        [str(Path(endpoint["source"]["root"]) / "src"), endpoint["source"]["root"]]
                    )
                    with (root / f"{endpoint['name']}-worker-{i}.log").open("x") as log:
                        processes.append(
                            subprocess.Popen(
                                [
                                    sys.executable,
                                    str(script),
                                    "worker",
                                    "--spec",
                                    str(spec_path),
                                    "--spec-sha256",
                                    expected_sha,
                                    "--endpoint",
                                    endpoint["name"],
                                    "--worker-index",
                                    str(i),
                                    "--resource-owner",
                                    str(owner_path),
                                    "--resource-owner-sha256",
                                    owner_sha256,
                                ],
                                cwd=endpoint["source"]["root"],
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        )
                if any(p.wait() != 0 for p in processes):
                    raise RuntimeError("native endpoint worker failed")
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                        p.wait()
        # Aggregation is launched in the analysis source, not an original model process.
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(analysis / "src"), str(analysis)]),
            CUDA_VISIBLE_DEVICES="",
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "compare",
                "--spec",
                str(spec_path),
                "--spec-sha256",
                expected_sha,
            ],
            cwd=analysis,
            env=env,
            check=True,
        )
    except BaseException as error:
        write_new(
            root / "FAILURE.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                scientific_complete=False,
                updates=0,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "worker", "compare", "validate"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--spec-sha256", required=True)
    p.add_argument("--endpoint", choices=("R3-RI", "FixedResponseGroupAnchor", "R15-DIR"))
    p.add_argument("--worker-index", type=int, choices=(0, 1))
    p.add_argument("--resource-owner", type=Path)
    p.add_argument("--resource-owner-sha256")
    args = p.parse_args()
    spec = json.loads(checked(dict(path=str(args.spec), sha256=args.spec_sha256)).read_text())
    validate_spec(spec)
    if args.mode == "run":
        dispatch(args.spec, args.spec_sha256)
    elif args.mode == "worker":
        if (
            args.endpoint is None
            or args.worker_index is None
            or args.resource_owner is None
            or args.resource_owner_sha256 is None
        ):
            p.error("worker endpoint/index and sealed dispatcher owner required")
        run_endpoint(
            spec,
            args.endpoint,
            args.worker_index,
            spec_file=dict(path=str(args.spec.resolve()), sha256=args.spec_sha256),
            resource_owner=dict(path=str(args.resource_owner), sha256=args.resource_owner_sha256),
        )
    elif args.mode == "compare":
        compare(spec, spec_file=dict(path=str(args.spec.resolve()), sha256=args.spec_sha256))
    elif spec["schema"] == R15_SCHEMA:
        print(
            json.dumps(
                validate_r15_result(
                    spec, spec_file=dict(path=str(args.spec.resolve()), sha256=args.spec_sha256)
                )
            )
        )
    else:
        print(
            json.dumps(
                dict(
                    status="validated_metadata_only",
                    spec_sha256=digest(spec),
                    scientific_complete=False,
                )
            )
        )


def r15_descriptor():
    return dict(
        manifest_sha256="86901e2829d7964f41d39bc958d81f1b150584810861d98fa6fb1b4c5eaf012a",
        arrays_sha256="28bc66a2b02f5435ee7fdec46be808461e1a33cd13fd9be35dd3d6a7bd21ff6d",
        weight=R15_WEIGHT,
        seed=20261028,
        head="Linear256_256_default_random_bias_true_private_rng",
        teacher_readout="unused_ema_mirror_no_forward_or_target_values",
        schedule="hash_permutation_step_i_mod_T_rank_floor_i_div_T_mod_world",
        loss="cosine_direction",
        direction_epsilon=1e-6,
        direction_formula="1-dot(normalize_fp32(Linear(mean_group_cls)),normalize_fp32_fixed_target)",
    )


def validate_r15_identity(endpoint, *, server=True):
    identity = json.loads(checked(endpoint["run_manifest"], server=server).read_text())
    if (
        endpoint["source"]["commit"] != R15_SOURCE
        or identity["source"]["commit"] != R15_SOURCE
        or identity["source"].get("dirty") is not False
        or endpoint["config"]["sha256"] != R15_CONFIG_SHA256
        or identity.get("config_sha256") != R15_CONFIG_SHA256
        or identity["data"].get("response_anchor") != r15_descriptor()
    ):
        raise ValueError("R15 requires its exact final training source/config/direction descriptor")
    checked(endpoint["config"], server=server)


def validate_r15_payload(spec, *, server=True):
    validate_r15_identity(spec["endpoints"][0], server=server)
    if set(spec["retained_references"]) != {"R3-RI", "R14-FRGA"}:
        raise ValueError("R15 requires explicitly retained RI primary and R14 secondary")
    for name, binding in spec["retained_references"].items():
        old, _ = retained_metadata(binding, name, spec["validation"], server=server)
        output, retained = Path(spec["output"]).resolve(), Path(old["output"]).resolve()
        if output.is_relative_to(retained) or retained.is_relative_to(output):
            raise ValueError("new R15 output overlaps retained historical output")
    if spec["observer_parity"] is not None:
        checked(spec["observer_parity"], server=server)


def retained_metadata(binding, name, validation, *, server=True):
    if set(binding) != {"postfit_spec", "endpoint_name", "complete"}:
        raise ValueError("exact retained spec/endpoint/COMPLETE binding required")
    expected = {"R3-RI": "R3-RI", "R14-FRGA": "FixedResponseGroupAnchor"}
    if name not in expected or binding["endpoint_name"] != expected[name]:
        raise ValueError("retained historical endpoint role changed")
    old = json.loads(checked(binding["postfit_spec"], server=server).read_text())
    if old.get("schema") != SCHEMA:
        raise ValueError("retained reference must use its original v1 spec")
    validate_spec(old, server=server)
    if old["validation"] != validation:
        raise ValueError("retained validation/control/query/allowed axes differ")
    endpoint = next(e for e in old["endpoints"] if e["name"] == expected[name])
    native = {
        "R3-RI": "f0cb20ebb2efab749970f2bf5d205ed2f358592c",
        "R14-FRGA": "dc5233ef1a243b2144107e5441a5a8db4f800fc6",
    }
    if endpoint["source"]["commit"] != native[name]:
        raise ValueError("retained original native source changed")
    complete = checked(binding["complete"], server=server)
    if complete.resolve() != (Path(old["output"]) / expected[name] / "COMPLETE.json").resolve():
        raise ValueError("retained COMPLETE escaped its original endpoint")
    return old, endpoint


def load_retained_reference(binding, name, validation, *, server=True):
    """Read and verify original aggregate and both worker closures, without writes."""
    old, endpoint = retained_metadata(binding, name, validation, server=server)
    return load_completed_endpoint(
        old, endpoint, binding["complete"], binding["postfit_spec"], server=server
    )


def load_completed_endpoint(old, endpoint, complete_binding, spec_binding, *, server=True):
    """Independently read an already completed endpoint; never merge or rewrite."""
    validation = old["validation"]
    root = Path(complete_binding["path"]).resolve().parent
    top = root.parent
    if (top / "FAILURE.json").exists() or list(top.rglob("*.pkl")):
        raise ValueError("failed or nonzero-PKL retained output")
    complete = json.loads(checked(complete_binding, server=server).read_text())
    if set(complete) != {
        "status",
        "endpoint",
        "arrays",
        "native_result_sha256",
        "summary_sha256",
        "worker_receipts_sha256",
        "validation_spec_sha256",
        "conditions",
        "updates",
        "zero_pkl",
        "scientific_complete",
    } or (
        complete["status"] != "completed"
        or complete["endpoint"] != endpoint["name"]
        or complete["conditions"] != 445
        or complete["updates"] != 0
        or complete["zero_pkl"] is not True
        or complete["scientific_complete"] is not False
        or complete["validation_spec_sha256"] != digest(validation)
        or len(complete["worker_receipts_sha256"]) != 2
    ):
        raise ValueError("retained aggregate receipt differs")
    expected = {
        "COMPLETE.json",
        "PRIVATE_ENDPOINT.npz",
        "PRIVATE_NATIVE_RESULT.json",
        "SUMMARY.json",
        "DISPATCH_OWNER.json",
    }
    expected |= {
        f"worker-{i}/{n}"
        for i in (0, 1)
        for n in (
            "COMPLETE.json",
            "PRIVATE_ENDPOINT.npz",
            "PRIVATE_NATIVE_RESULT.json",
            "SUMMARY.json",
        )
    }
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("retained recursive closure has missing or extra files")
    for file, digest_value in [
        ("PRIVATE_NATIVE_RESULT.json", complete["native_result_sha256"]),
        ("SUMMARY.json", complete["summary_sha256"]),
    ]:
        checked(dict(path=str(root / file), sha256=digest_value), server=server)
    if Path(complete["arrays"]["path"]).resolve() != root / "PRIVATE_ENDPOINT.npz":
        raise ValueError("retained aggregate array escaped endpoint")
    arrays = load_arrays(complete["arrays"], server=server)
    ordered = validation["conditions"]
    if (
        arrays["prediction_delta"].shape != (445, 4775)
        or arrays["truth_delta"].shape != (445, 4775)
        or arrays["cls_means"].shape != (445, 256)
        or arrays["own_target_indices"].shape != (445, 1)
    ):
        raise ValueError("retained aggregate array shapes differ from frozen Jurkat axes")
    if (
        arrays["own_target_indices"].dtype.kind not in "iu"
        or (arrays["own_target_indices"] < -1).any()
        or (arrays["own_target_indices"] >= FROZEN_GRAPH_GENES).any()
        or any(
            not np.isfinite(arrays[k]).all()
            for k in ("prediction_delta", "truth_delta", "cls_means")
        )
    ):
        raise ValueError("completed endpoint contains nonfinite arrays or invalid own-target IDs")
    if arrays["condition_ids"].tolist() != ordered or not np.array_equal(
        arrays["allowed_gene_indices"], validation["allowed_gene_indices"]
    ):
        raise ValueError("retained aggregate condition/allowed axes changed")
    native = json.loads((root / "PRIVATE_NATIVE_RESULT.json").read_text())
    if [r["condition_id"] for r in native["conditions"]] != ordered:
        raise ValueError("retained native condition axis differs")
    owners = []
    for i in (0, 1):
        worker = root / f"worker-{i}"
        path = checked(
            dict(path=str(worker / "COMPLETE.json"), sha256=complete["worker_receipts_sha256"][i]),
            server=server,
        )
        receipt = json.loads(path.read_text())
        if (
            receipt["status"] != "completed"
            or receipt["endpoint"] != endpoint["name"]
            or receipt["worker_index"] != i
            or receipt["postfit_spec_sha256"] != digest(old)
            or receipt["endpoint_spec_sha256"] != digest(endpoint)
            or receipt["validation_spec_sha256"] != digest(validation)
            or receipt["conditions"] != ordered[i::2]
            or receipt["source"]["commit"] != endpoint["source"]["commit"]
            or receipt["analysis_source"]["commit"] != old["analysis_source"]["commit"]
            or receipt["checkpoint_sha256"] != endpoint["checkpoint"]["sha256"]
            or receipt["control_manifest_sha256"] != validation["control_manifest_sha256"]
            or receipt["reference_sha256"] != validation["reference_sha256"]
            or (receipt["epoch"], receipt["role"], receipt["split"]) != (2, "last", "val")
            or not receipt["state_unchanged"]
            or not receipt["cuda_rng_unchanged"]
            or not receipt["zero_pkl"]
            or any(receipt[k] != 0 for k in ("updates", "teacher_updates", "center_updates"))
            or any(receipt["forbidden_calls"].values())
            or receipt["resource_checks"]["conditions_checked"] != len(ordered[i::2])
            or receipt["resource_checks"]["calls"] != len(ordered[i::2]) + 2
        ):
            raise ValueError("retained worker source/state/CP/spec/coverage differs")
        counter_keys = {
            "teacher_forward",
            "optimizer_step",
            "statistics_commit",
            "backward",
            "autograd_grad",
        }
        if old["schema"] == R15_SCHEMA:
            counter_keys.add("anchor_forward")
        if (
            set(receipt["forbidden_calls"]) != counter_keys
            or any(type(v) is not int or v != 0 for v in receipt["forbidden_calls"].values())
            or receipt["source"].get("dirty") is not False
            or receipt["analysis_source"].get("dirty") is not False
            or not isinstance(receipt.get("state_before"), dict)
            or not receipt["state_before"]
            or not all(valid_sha256(v) for v in receipt["state_before"].values())
        ):
            raise ValueError("completed worker missing clean-source/state/counter proof")
        if Path(receipt["resource_owner"]["path"]).resolve() != root / "DISPATCH_OWNER.json":
            raise ValueError("retained dispatcher owner escaped endpoint")
        checked(receipt["resource_owner"], server=server)
        owners.append(receipt["resource_owner"])
        for filename, declared in (
            ("PRIVATE_ENDPOINT.npz", receipt["arrays"]),
            ("PRIVATE_NATIVE_RESULT.json", receipt["native_result"]),
        ):
            if Path(declared["path"]).resolve() != worker / filename:
                raise ValueError("retained worker output escaped its closure")
            checked(declared, server=server)
        checked(
            dict(path=str(worker / "SUMMARY.json"), sha256=receipt["summary_sha256"]), server=server
        )
        required_bindings = [
            spec_binding,
            old["shared_policy"],
            old["analysis_source"]["publication"],
            endpoint["source"]["publication"],
            receipt["resource_owner"],
            *[endpoint[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
        ]
        if any(
            receipt["input_files_sha256"].get(str(Path(b["path"]).resolve())) != b["sha256"]
            for b in required_bindings
        ):
            raise ValueError("retained worker omitted required immutable input bindings")
        for file, value in receipt["input_files_sha256"].items():
            checked(dict(path=file, sha256=value), server=server)
        shard = load_arrays(receipt["arrays"], server=server)
        if not np.array_equal(shard["allowed_gene_indices"], arrays["allowed_gene_indices"]):
            raise ValueError("retained worker allowed axis differs")
        for field in ARRAY_FIELDS - {"allowed_gene_indices"}:
            if not np.array_equal(shard[field], arrays[field][i::2]):
                raise ValueError("retained worker and aggregate arrays differ")
        shard_native = json.loads((worker / "PRIVATE_NATIVE_RESULT.json").read_text())
        if shard_native["conditions"] != native["conditions"][i::2]:
            raise ValueError("retained worker and aggregate native conditions differ")
    if owners[0] != owners[1]:
        raise ValueError("retained workers used different dispatcher owners")
    return arrays, native


def paired_comparison(
    reference, candidate, reference_native, candidate_native, geometry, bootstrap
):
    """Same primitive for RI-primary and R14-secondary; no baseline model forward."""
    assert_paired_identity(reference, candidate, reference_native, candidate_native)
    result = geometry.compare_endpoints(reference, candidate, **bootstrap)
    result["native_point_gates"] = native_point_gates(reference_native, candidate_native)
    result["scientific_complete"] = False
    return result


def assert_paired_identity(reference, candidate, reference_native, candidate_native):
    for field in ("condition_ids", "allowed_gene_indices", "own_target_indices", "truth_delta"):
        if not np.array_equal(reference[field], candidate[field]):
            raise ValueError("paired reference/candidate truth or axes differ")
    a, b = reference_native["conditions"], candidate_native["conditions"]
    if len(a) != len(b):
        raise ValueError("paired native condition counts differ")
    for before, after in zip(a, b, strict=True):
        if any(
            before[k] != after[k]
            for k in ("condition_id", "control_row_ids_sha256", "truth_row_ids_sha256")
        ):
            raise ValueError("paired native ordered control/truth identities differ")


def validate_observer_parity(spec, *, server=True):
    if spec["schema"] != R15_SCHEMA:
        return
    if spec["observer_parity"] is None:
        raise ValueError("R15 execution requires its native first-condition off/on parity receipt")
    receipt = json.loads(checked(spec["observer_parity"], server=server).read_text())
    pilot = json.loads(checked(receipt["spec"], server=server).read_text())
    expected_pilot = dict(spec, observer_parity=None)
    parity = receipt.get("parity", {})
    endpoint = spec["endpoints"][0]
    if (
        pilot != expected_pilot
        or receipt["spec_digest"] != digest(pilot)
        or receipt["status"] != "completed"
        or receipt["endpoint"] != "R15-DIR"
        or receipt["endpoint_digest"] != digest(endpoint)
        or receipt["source"]["commit"] != R15_SOURCE
        or receipt["analysis_source"]["commit"] != spec["analysis_source"]["commit"]
        or receipt["checkpoint"] != endpoint["checkpoint"]
        or receipt["condition"] != spec["validation"]["conditions"][0]
        or (
            receipt["full_validation_conditions"],
            receipt["executed_conditions"],
            receipt["controls"],
            receipt["expression_genes"],
            receipt["query_count"],
        )
        != (445, 1, 300, 5000, 1000)
        or (receipt["epoch"], receipt["role"], receipt["split"]) != (2, "last", "val")
        or any(receipt[k] != 0 for k in ("updates", "teacher_updates", "center_updates"))
        or receipt.get("zero_pkl") is not True
        or parity.get("passed") is not True
        or parity.get("atol") != 0
        or parity.get("rtol") != 0
        or parity.get("max_abs") != 0
        or parity.get("state_unchanged") is not True
        or parity.get("state_before") != parity.get("state_after")
        or parity.get("callback_counts") != [1, 1]
        or parity.get("prediction_shape") != [300, 5000]
        or len(parity.get("forbidden_calls", [])) != 2
        or len(parity.get("prediction_sha256", [])) != 2
        or len(set(parity.get("prediction_sha256", []))) != 1
        or len(parity.get("native_result_sha256", [])) != 2
        or len(set(parity.get("native_result_sha256", []))) != 1
        or any(any(row.values()) for row in parity["forbidden_calls"])
    ):
        raise ValueError("R15 observer parity source/CP/coverage/state proof differs")
    required_bindings = [
        receipt["spec"],
        spec["shared_policy"],
        spec["analysis_source"]["publication"],
        endpoint["source"]["publication"],
        *[endpoint[k] for k in ("run_manifest", "journal", "config", "checkpoint")],
    ]
    inputs = receipt.get("input_files_sha256")
    if not isinstance(inputs, dict) or any(
        inputs.get(str(Path(b["path"]).resolve())) != b["sha256"] for b in required_bindings
    ):
        raise ValueError("R15 observer parity omitted required immutable input bindings")
    counters = {
        "teacher_forward",
        "optimizer_step",
        "statistics_commit",
        "backward",
        "autograd_grad",
        "anchor_forward",
    }
    if any(
        set(row) != counters or any(type(v) is not int or v != 0 for v in row.values())
        for row in parity["forbidden_calls"]
    ):
        raise ValueError("R15 observer parity counter proof missing or nonzero")
    if not valid_parity_state(parity.get("state_before")) or not valid_parity_state(
        parity.get("state_after")
    ):
        raise ValueError("R15 observer parity state proof missing or malformed")
    if any(
        not valid_sha256(v)
        for key in ("prediction_sha256", "native_result_sha256")
        for v in parity[key]
    ):
        raise ValueError("R15 observer parity prediction/result fingerprints invalid")
    from gradpert.config import load_experiment_config

    cell_batch = int(
        load_experiment_config(
            checked(endpoint["config"], server=server)
        ).training.eval_batch_size.value
    )
    expected_calls = dict(
        predict_controls=1,
        query_blocks=5,
        response_forward=5 * math.ceil(300 / cell_batch),
        truth_reads=1,
        control_reads=1,
    )
    if parity.get("observer_calls") != expected_calls:
        raise ValueError("R15 observer parity actual forward/read coverage differs")
    for file, value in receipt["input_files_sha256"].items():
        checked(dict(path=file, sha256=value), server=server)
    checked(receipt["arrays"], server=server)
    checked(
        dict(
            path=str(Path(spec["observer_parity"]["path"]).parent / "PRIVATE_NATIVE_RESULT.json"),
            sha256=receipt["native_result_sha256"],
        ),
        server=server,
    )


def valid_sha256(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def valid_parity_state(value):
    native_keys = {
        "student",
        "teacher",
        "centers",
        "optimizer",
        "gradients",
        "numpy_rng",
        "python_rng",
        "torch_rng",
        "pending_statistics",
        "auxiliary_statistics",
        "module_modes",
    }
    if not isinstance(value, dict) or set(value) != {"protected", "cuda_rng", "canonical_stat"}:
        return False
    protected = value["protected"]
    if not isinstance(protected, dict) or set(protected) != {
        "native",
        "objective_state",
        "objective_buffers",
        "objective_scalar_clocks",
    }:
        return False
    native = protected["native"]
    stat = value["canonical_stat"]
    return (
        isinstance(native, dict)
        and set(native) == native_keys
        and all(valid_sha256(v) for v in native.values())
        and all(valid_sha256(protected[k]) for k in protected if k != "native")
        and valid_sha256(value["cuda_rng"])
        and isinstance(stat, list)
        and len(stat) == 2
        and all(type(v) is int and v > 0 for v in stat)
    )


def compare_r15(spec, geometry, *, spec_file):
    validate_observer_parity(spec)
    if json.loads(checked(spec_file).read_text()) != spec:
        raise ValueError("R15 input SPEC binding differs")
    root = server_path(spec["output"])
    candidate, native = merge_endpoint(spec, "R15-DIR")
    comparisons = {}
    for name, label in (("R3-RI", "primary_vs_RI"), ("R14-FRGA", "secondary_vs_R14")):
        reference, original = load_retained_reference(
            spec["retained_references"][name], name, spec["validation"]
        )
        comparisons[label] = paired_comparison(
            reference, candidate, original, native, geometry, spec["bootstrap"]
        )
    result = dict(
        schema="response-anchor-r15-paired-comparison-1",
        candidate="R15-DIR",
        comparisons=comparisons,
        scientific_complete=False,
    )
    write_new(root / "COMPARISON.json", result)
    if list(root.rglob("*.pkl")):
        raise ValueError("R15 postfit requires whole-output zero PKL")
    write_new(
        root / "COMPLETE.json",
        dict(
            schema="response-anchor-r15-val-postfit-complete-1",
            status="completed",
            spec_sha256=digest(spec),
            input_spec=spec_file,
            comparison_sha256=sha(root / "COMPARISON.json"),
            endpoint_receipts={"R15-DIR": sha(root / "R15-DIR/COMPLETE.json")},
            retained_references=spec["retained_references"],
            observer_parity=spec["observer_parity"],
            outputs={
                str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
            updates=0,
            zero_pkl=True,
            scientific_complete=False,
            postfit_complete=False,
            medium_status="not_executed_by_val_worker",
            test_status="deferred",
        ),
    )
    return result


def validate_comparison_contract(result, spec, reference_native, candidate_native):
    expected = {
        "schema",
        "draws",
        "seed",
        "condition_N",
        "resampling_unit",
        "ownmask_Pearson_delta",
        "ownmask_MSE_delta",
        "RSA_delta",
        "full_axis_finite",
        "candidate_new_Pearson_NA",
        "conditioned_on_fixed_truth_control_bank",
        "scientific_complete",
        "native_point_gates",
    }
    if (
        set(result) != expected
        or result["schema"] != "paired-condition-response-geometry-v1"
        or result["draws"] != spec["bootstrap"]["draws"]
        or result["seed"] != spec["bootstrap"]["seed"]
        or result["condition_N"] != 445
        or result["resampling_unit"]
        != "original condition; pair multiplicity ci*cj; same-original-ID pairs excluded"
        or result["conditioned_on_fixed_truth_control_bank"] is not True
        or result["scientific_complete"] is not False
        or set(result["full_axis_finite"]) != {"Pearson", "MSE", "RSA"}
        or any(type(v) is not bool for v in result["full_axis_finite"].values())
        or type(result["candidate_new_Pearson_NA"]) is not int
        or not 0 <= result["candidate_new_Pearson_NA"] <= 445
        or result["native_point_gates"] != native_point_gates(reference_native, candidate_native)
    ):
        raise ValueError("R15 paired comparison protocol or native point gates differ")
    for key in ("ownmask_Pearson_delta", "ownmask_MSE_delta", "RSA_delta"):
        row = result[key]
        if set(row) != {"candidate_minus_reference", "CI95", "valid_draws", "invalid_draws"}:
            raise ValueError("R15 comparison missing interval fields")
        a, b, point, ci = (
            row["valid_draws"],
            row["invalid_draws"],
            row["candidate_minus_reference"],
            row["CI95"],
        )
        if (
            type(a) is not int
            or type(b) is not int
            or min(a, b) < 0
            or a + b != result["draws"]
            or (
                point is not None
                and (
                    isinstance(point, bool)
                    or not isinstance(point, (int, float))
                    or not math.isfinite(point)
                )
            )
            or (a == 0 and ci is not None)
            or (
                a > 0
                and (
                    not isinstance(ci, list)
                    or len(ci) != 2
                    or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in ci
                    )
                    or ci[0] > ci[1]
                )
            )
        ):
            raise ValueError("R15 comparison invalid finite-count/CI proof")


def validate_r15_result(spec, *, spec_file, server=True):
    """Verify terminal receipts and exact closure only; never merge, write or fit."""
    validate_spec(spec, server=server)
    if server:
        _source(spec["analysis_source"])
    if spec["schema"] != R15_SCHEMA:
        raise ValueError("R15 result validator cannot relabel an old profile")
    if json.loads(checked(spec_file, server=server).read_text()) != spec:
        raise ValueError("R15 terminal input SPEC differs")
    validate_observer_parity(spec, server=server)
    root = server_path(spec["output"]) if server else Path(spec["output"]).resolve()
    if (root / "FAILURE.json").exists() or list(root.rglob("*.pkl")):
        raise ValueError("failed or nonzero-PKL R15 result cannot validate")
    complete = json.loads((root / "COMPLETE.json").read_text())
    required = {
        "schema",
        "status",
        "spec_sha256",
        "input_spec",
        "comparison_sha256",
        "endpoint_receipts",
        "retained_references",
        "observer_parity",
        "outputs",
        "updates",
        "zero_pkl",
        "scientific_complete",
        "postfit_complete",
        "medium_status",
        "test_status",
    }
    if (
        set(complete) != required
        or complete["schema"] != "response-anchor-r15-val-postfit-complete-1"
        or complete["status"] != "completed"
        or complete["spec_sha256"] != digest(spec)
        or complete["input_spec"] != spec_file
        or complete["retained_references"] != spec["retained_references"]
        or complete["observer_parity"] != spec["observer_parity"]
        or complete["updates"] != 0
        or complete["zero_pkl"] is not True
        or complete["scientific_complete"] is not False
        or complete["postfit_complete"] is not False
        or complete["medium_status"] != "not_executed_by_val_worker"
        or complete["test_status"] != "deferred"
        or set(complete["endpoint_receipts"]) != {"R15-DIR"}
    ):
        raise ValueError("R15 terminal COMPLETE protocol differs")
    if json.loads((root / "SPEC.json").read_text()) != spec:
        raise ValueError("R15 output SPEC snapshot differs")
    endpoint = spec["endpoints"][0]
    candidate_root = root / "R15-DIR"
    cp = dict(
        path=str(candidate_root / "COMPLETE.json"), sha256=complete["endpoint_receipts"]["R15-DIR"]
    )
    candidate, native = load_completed_endpoint(spec, endpoint, cp, spec_file, server=server)
    expected = {"SPEC.json", "COMPARISON.json", "R15-DIR-worker-0.log", "R15-DIR-worker-1.log"}
    expected |= {
        "R15-DIR/" + str(p.relative_to(candidate_root))
        for p in candidate_root.rglob("*")
        if p.is_file()
    }
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "COMPLETE.json"
    }
    if (
        actual != expected
        or set(complete["outputs"]) != actual
        or any(p.is_symlink() for p in root.rglob("*"))
    ):
        raise ValueError("R15 recursive closure missing or extra output")
    for relative, value in complete["outputs"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not valid_sha256(value):
            raise ValueError("R15 output escaped closure or has invalid hash")
        checked(dict(path=str(path), sha256=value), server=server)
    comparison_path = checked(
        dict(path=str(root / "COMPARISON.json"), sha256=complete["comparison_sha256"]),
        server=server,
    )
    comparison = json.loads(comparison_path.read_text())
    if (
        set(comparison) != {"schema", "candidate", "comparisons", "scientific_complete"}
        or comparison["schema"] != "response-anchor-r15-paired-comparison-1"
        or comparison["candidate"] != "R15-DIR"
        or comparison["scientific_complete"] is not False
        or set(comparison["comparisons"]) != {"primary_vs_RI", "secondary_vs_R14"}
    ):
        raise ValueError("R15 COMPARISON roles/schema differ")
    for name, label in (("R3-RI", "primary_vs_RI"), ("R14-FRGA", "secondary_vs_R14")):
        reference, original = load_retained_reference(
            spec["retained_references"][name], name, spec["validation"], server=server
        )
        assert_paired_identity(reference, candidate, original, native)
        validate_comparison_contract(comparison["comparisons"][label], spec, original, native)
    return dict(
        status="validated_completed_result",
        spec_sha256=digest(spec),
        comparison_sha256=complete["comparison_sha256"],
        conditions=445,
        updates=0,
        zero_pkl=True,
        scientific_complete=False,
    )


if __name__ == "__main__":
    main()
