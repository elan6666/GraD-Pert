"""Sealed TRAIN40 two-native-branch gradient observation; never an optimizer update.

Run this published analysis file with the original dc5233e native src first on
PYTHONPATH. ``seal`` is CPU/metadata-only apart from loading the frozen bank;
``run`` owns one admitted GPU and expands a two-complete-group pilot only after
its immutable input, state, read, numeric, and resource contracts pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

NATIVE_SHA = "dc5233ef1a243b2144107e5441a5a8db4f800fc6"
CP_SHA = "7ad95c52d04aa3accec6a6a82006f5d9c8111c2d602c058898b25706e99cb467"
SCHEMA = "response-anchor-train40-gradient-input-1"
PROTOCOL = dict(
    conditions=16,
    rows=40,
    queries=1000,
    allowed=4775,
    selection_seed=20261029,
    query_seed=20261029,
    pair_seed=1,
    pairing_epoch=1,
    aux_rng_epoch_labels=[0, 1, 2],
    pilot_conditions=2,
    pilot_replicates=1,
    precision="native_bfloat16_autocast",
    branches="native_paired_prediction_and_original_bank_anchor_different_inputs",
    updates=0,
    bootstrap_draws=1000,
    bootstrap_seed=20261029,
)
BUDGET_FIELDS = {
    "max_total_seconds",
    "max_peak_reserved_bytes",
    "max_cpu_rss_bytes",
    "min_free_cuda_bytes",
}
REQUIRED_OUTPUTS = {
    "SPEC.json",
    "METADATA.json",
    "SEALED.json",
    "RUN_STARTED.json",
    "PARAMETERS.json",
    "target_spectrum.json",
    "readout_spectrum.json",
    "PILOT.json",
    "cases.json",
    "geometry.json",
    "READ_SCOPE.json",
}
SERVER_PYTHON = "/data/yilangliu/GraD-Pert/source/.venv/bin/python"
CASE_SCALAR_DIR = "server-only/case-scalars"
CASE_ROW_FIELDS = {
    "condition_id",
    "bank_condition_index",
    "aux_rng_epoch_label",
    "pairing_epoch",
    "prediction_loss",
    "auxiliary_loss",
    "gradient",
    "support_parameter_indices",
    "modules",
    "decoder",
    "calls",
    "input_branches_differ",
    "historical_update_replay",
    "ssl_terms",
    "gradient_scaling",
    "timings",
    "derived_telemetry",
    "pilot_original_scalar_parity",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(binding, *, server=True):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("exact path/hash binding required")
    path = Path(binding["path"]).resolve(strict=True)
    if not path.is_file() or (server and not path.is_relative_to("/data/yilangliu")):
        raise ValueError("sealed server file required")
    if sha(path) != binding["sha256"]:
        raise ValueError("input bytes changed")
    return path


def array_sha(value):
    value = np.ascontiguousarray(value)
    return (
        digest(dict(shape=list(value.shape), dtype=str(value.dtype)))
        + ":"
        + hashlib.sha256(value.tobytes()).hexdigest()
    )


def analysis_module(name):
    """Load an analysis helper by this file's directory, never an old native script."""
    path = Path(__file__).resolve().with_name(name + ".py")
    module_name = "train40_analysis_" + name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def geometry_module(root=None):
    """Load NumPy analysis code without extending/replacing native gradpert."""
    analysis_root = Path(__file__).resolve().parents[2]
    if root is not None and Path(root).resolve() != analysis_root:
        raise ValueError("geometry dependency must belong to the exact analysis checkout")
    path = (analysis_root / "src/gradpert/evaluation/response_geometry.py").resolve(strict=True)
    if not path.is_relative_to(analysis_root):
        raise ValueError("geometry dependency escaped the exact analysis checkout")
    name = "train40_analysis_geometry_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # dataclass resolves its defining module through sys.modules during exec.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path or any(
        not callable(getattr(module, key, None))
        for key in ("WeightedRanks", "weighted_spearman", "_geometry", "_rsa")
    ):
        raise ImportError("analysis geometry dependency path/API differs")
    return module


def preload_analysis_dependencies(spec):
    module = geometry_module(spec["analysis_source"]["root"])
    return dict(
        response_geometry=dict(
            path=str(Path(module.__file__).resolve()), sha256=sha(module.__file__)
        )
    )


def case_scalar_path(number):
    if type(number) is not int or not 1 <= number <= 48:
        raise ValueError("case number must be in the sealed1..48 population")
    return f"{CASE_SCALAR_DIR}/{number:04d}.json"


def expected_outputs():
    return REQUIRED_OUTPUTS | {case_scalar_path(i) for i in range(1, 49)}


def persist_case_scalar(output, number, row, spec_digest):
    """Durable scalar-only case journal; never an activation/gradient resume cache."""
    if not isinstance(row, dict) or not set(row) <= CASE_ROW_FIELDS:
        raise ValueError("only declared scalar case-summary fields may be persisted")

    def scalar_tree(value):
        if isinstance(value, dict):
            return all(isinstance(k, str) and scalar_tree(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return all(v is None or isinstance(v, (bool, int, float, str)) for v in value)
        return value is None or isinstance(value, (bool, int, float, str))

    if not scalar_tree(row):
        raise ValueError("raw/latent/gradient arrays are forbidden in case scalar reports")
    record = dict(
        schema="response-anchor-train40-case-scalars-1",
        case_number=number,
        spec_digest=spec_digest,
        row=row,
    )
    serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
    path = Path(output) / case_scalar_path(number)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def geometry_after_scalar_aggregate(output, rows, metadata, arrays):
    """Geometry failure must not erase completed allowed scalar observations."""
    write_new(Path(output) / "cases.json", rows)
    return small_geometry(metadata, arrays)


def native_script(relative, root):
    path = Path(root) / relative
    name = "train40_native_" + path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("mixed native script imports")
    return module


def validate_spec(spec, *, server=True):
    fields = {
        "schema",
        "analysis_source",
        "endpoint",
        "data_root",
        "resource_policy",
        "budgets",
        "output",
        "protocol",
    }
    if set(spec) != fields or spec["schema"] != SCHEMA or spec["protocol"] != PROTOCOL:
        raise ValueError("frozen TRAIN40 protocol/schema differs")
    endpoint = spec["endpoint"]
    if (
        set(endpoint)
        != {"name", "source", "run_root", "run_manifest", "journal", "config", "checkpoint"}
        or endpoint["name"] != "R14-FRGA"
    ):
        raise ValueError("single retained R14 endpoint required")
    if endpoint["source"]["commit"] != NATIVE_SHA or endpoint["checkpoint"]["sha256"] != CP_SHA:
        raise ValueError("original R14 source/checkpoint identity differs")
    protected = [Path(spec["data_root"]).resolve(), Path(endpoint["run_root"]).resolve()]
    for source in (spec["analysis_source"], endpoint["source"]):
        if set(source) != {"root", "commit", "publication"}:
            raise ValueError("exact source/publication binding required")
        commit = source["commit"]
        if len(commit) != 40 or any(x not in "0123456789abcdef" for x in commit):
            raise ValueError("full source commit required")
        checked(source["publication"], server=server)
        protected.append(Path(source["root"]).resolve(strict=True))
    for key in ("run_manifest", "journal", "config", "checkpoint"):
        checked(endpoint[key], server=server)
    checked(spec["resource_policy"], server=server)
    budget = spec["budgets"]
    if set(budget) != BUDGET_FIELDS or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in budget.values()
    ):
        raise ValueError("explicit finite positive time/memory budgets required")
    output = Path(spec["output"]).resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output overlaps protected source/data/run")
    if server and any(not p.is_relative_to("/data/yilangliu") for p in [output, *protected]):
        raise ValueError("all scientific materialization stays on server")


def verify_sources(spec):
    postfit = analysis_module("response_anchor_postfit")
    for binding in (spec["endpoint"]["source"], spec["analysis_source"]):
        postfit.preimport_source(binding)
    native = postfit._source(spec["endpoint"]["source"])
    binding = spec["analysis_source"]
    stochastic = analysis_module("postfit_stochastic_audit")
    analysis = stochastic.verify_analysis_source(
        dict(
            root=binding["root"],
            commit=binding["commit"],
            publication=binding["publication"]["path"],
            publication_sha256=binding["publication"]["sha256"],
        )
    )
    if Path(__file__).resolve() != Path(binding["root"]).resolve() / (
        "scripts/v2/response_anchor_gradient_probe.py"
    ):
        raise ValueError("worker does not belong to the sealed analysis source")
    import gradpert

    if (
        not Path(gradpert.__file__)
        .resolve()
        .is_relative_to(Path(spec["endpoint"]["source"]["root"]).resolve() / "src")
    ):
        raise ValueError("native gradpert must come from exact original training checkout")
    return dict(native=native, analysis=analysis)


def load_config_identity(spec):
    from gradpert.config import load_experiment_config

    config = load_experiment_config(checked(spec["endpoint"]["config"]))
    identity = analysis_module("response_anchor_postfit")._bind_checkpoint(spec["endpoint"], config)
    return config, identity


def metadata_of(data):
    return dict(
        row_ids=data.row_ids,
        condition_ids=data.condition_ids,
        context_ids=data.context_ids,
        control_pools=data.control_pools,
        train_row_indices=data.train_row_indices,
        train_conditions=data.split.train_conditions,
        val_conditions=data.split.val_conditions,
        control_condition_id=data.split.control_condition_id,
    )


def recover_bank_slots(bank, metadata, *, make_plan, native_digest):
    """Recover IDs from metadata rules, never by matching expression vectors."""
    expected = bank.identity.get("metadata_sha256")
    if expected is None:
        return dict(status="unknown", reason="original_metadata_hash_unavailable"), None
    if expected != native_digest(metadata):
        raise ValueError("known bank metadata identity mismatch")
    plan = make_plan(metadata)
    entries = plan["splits"]["selected_train"]["eligible"]
    if [e["condition_id"] for e in entries] != list(bank.condition_ids):
        raise ValueError("bank eligible condition order differs from native metadata replay")
    ids, contexts = metadata["row_ids"], metadata["context_ids"]
    pools = {k: sorted(set(v["TA"]) | set(v["TB"])) for k, v in plan["controls"].items()}
    slots, offsets = [], [0]
    for entry in entries:
        p = entry["condition_id"]
        for k, n in sorted(Counter(contexts[i] for i in entry["rows"]).items()):
            pool = sorted(pools[k], key=lambda i: (native_digest([p, k, ids[i]]), i))
            if not pool:
                raise ValueError("original matched control pool empty")
            slots.extend(pool[j % len(pool)] for j in range(n))
        offsets.append(len(slots))
    unique = sorted(set(slots))
    lookup = {r: i for i, r in enumerate(unique)}
    indices = [lookup[r] for r in slots]
    if (
        not np.array_equal(offsets, bank.control_slot_offsets)
        or not np.array_equal(indices, bank.control_slot_indices)
        or len(unique) != bank.control_expression.shape[0]
    ):
        raise ValueError("known reconstructed slot/index/unique population mismatch")
    return dict(
        status="metadata_reconstructed_numeric_not_yet_verified",
        unique_row_ids_sha256=native_digest([ids[i] for i in unique]),
        all_slot_row_ids_sha256=native_digest([ids[i] for i in slots]),
    ), unique


def selected_groups(data, bank, *, pairer, native_digest):
    """One selected condition is one complete native forty-treated population."""
    members = {}
    for i in data.train_row_indices:
        members.setdefault(data.condition_ids[i], []).append(i)
    candidates = [
        (p, j)
        for j, p in enumerate(bank.condition_ids)
        if len(members.get(p, ())) == 40
        and int(bank.control_slot_offsets[j + 1] - bank.control_slot_offsets[j]) == 40
    ]
    candidates.sort(
        key=lambda x: (native_digest(["train40-diagnostic-condition-v1", 20261029, x[0]]), x[0])
    )
    if len(candidates) < 16:
        raise ValueError("fewer than sixteen genuine complete TRAIN40 conditions")
    result = []
    for p, index in candidates[:16]:
        rows = members[p]
        truth_ids = tuple(data.row_ids[i] for i in rows)
        if len(set(truth_ids)) != 40:
            raise ValueError("duplicate treated member")
        pairing = pairer.pair_epoch(
            epoch=1,
            perturbed_row_ids=truth_ids,
            context_ids=tuple(data.context_ids[i] for i in rows),
            control_pools=data.control_pools,
        )
        records = []
        for ti, ci in zip(rows, pairing.control_row_ids, strict=True):
            control_index = data._row_index[ci]
            if data.context_ids[ti] != data.context_ids[control_index]:
                raise ValueError("native pairing context mismatch")
            records.append(
                dict(
                    truth_row_id=data.row_ids[ti],
                    control_row_id=ci,
                    condition_id=p,
                    cell_line="Jurkat",
                    cell_type=str(data._adata.obs.iloc[ti]["cell_type"]),
                    context_id=str(data.context_ids[ti]),
                    batch=str(data.batch_ids[ti]),
                )
            )
        start, stop = map(int, bank.control_slot_offsets[index : index + 2])
        result.append(
            dict(
                condition_id=p,
                bank_condition_index=index,
                records=records,
                anchor_gene_index=int(bank.condition_anchor_gene_indices[index]),
                records_sha256=native_digest(records),
                bank_slot_indices=bank.control_slot_indices[start:stop].tolist(),
                bank_slots_sha256=native_digest(bank.control_slot_indices[start:stop].tolist()),
                view_seed=int(native_digest(["train40-main-graph-v1", p])[:15], 16),
            )
        )
    return result, len(candidates)


def bank_context(runtime, bank):
    """Bind original pure methods after checking all original constructor predicates.

    Evaluation purpose is retained; no Runtime.steps_per_epoch access and no
    groups/backward surface is installed. This is an observational context only.
    """
    from gradpert.training.v2.response_anchor_runtime import ResponseAnchorController

    data, identity = runtime.data, bank.identity
    if (
        identity.get("split_content_sha256") != data.split.split_content_sha256
        or (identity.get("selection") != runtime.identity.get("training_row_selection"))
        or identity.get("allowed_policy") != runtime.identity.get("training_expression_policy")
    ):
        raise ValueError("bank differs from original split/selection/allowed identity")
    if not set(bank.condition_ids) <= set(data.split.train_conditions):
        raise ValueError("bank contains non-TRAIN condition")
    steps = data.steps_per_epoch(batch_size=runtime.batch_size, max_unique_conditions=0)
    if steps != 150 or len(bank.condition_ids) != 1322:
        raise ValueError("original cap40 population/update budget differs")
    for g, p in zip(bank.condition_anchor_gene_indices, bank.condition_ids, strict=True):
        if not 0 <= int(g) < len(data.graph_gene_ids) or {data.graph_gene_ids[int(g)]} != (
            set(p.split("+")) - {data.split.control_condition_id}
        ):
            raise ValueError("bank single-target graph identity differs")
    context = SimpleNamespace(runtime=runtime, bank=bank, pair_mode=False)
    if hasattr(ResponseAnchorController, "native_group_forward"):
        context.native_group_forward = MethodType(
            ResponseAnchorController.native_group_forward, context
        )
    context.group_rng = MethodType(ResponseAnchorController.group_rng, context)
    context.group_loss = MethodType(ResponseAnchorController.group_loss, context)
    return context


def load_bank(runtime):
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.response_anchor_runtime import load_response_bank

    o = runtime.options
    bank = load_response_bank(
        o.response_anchor_path, o.response_anchor_sha256, runtime.allowed_expression_ids
    )
    data = runtime.data
    manifest = (
        sha256_json(data.manifest.payload())
        if hasattr(data.manifest, "payload")
        else (sha256_file(data.layout.canonical_manifest))
    )
    if (
        bank.arrays_sha256 != o.response_anchor_arrays_sha256
        or bank.identity.get("data_manifest") != manifest
    ):
        raise ValueError("bank canonical manifest/array identity differs")
    bank_context(runtime, bank)
    return bank


def make_metadata(runtime, bank, root):
    from gradpert.hashing import sha256_json
    from gradpert.training.controls import TrainingControlPairer

    rr = native_script("scripts/v2/response_repeatability.py", root)
    metadata = metadata_of(runtime.data)
    mapping, unique = recover_bank_slots(
        bank, metadata, make_plan=rr.make_plan, native_digest=sha256_json
    )
    groups, eligible = selected_groups(
        runtime.data, bank, pairer=TrainingControlPairer(run_seed=1), native_digest=sha256_json
    )
    allowed = np.asarray(runtime.allowed_expression_ids)
    if len(allowed) != 4775:
        raise ValueError("TRAIN expression axis is not original4775")
    queries = sorted(np.random.default_rng(20261029).choice(allowed, 1000, replace=False).tolist())
    if unique is not None:
        used = sorted({i for g in groups for i in g["bank_slot_indices"]})
        mapping["selected_bank_indices"] = used
        mapping["selected_control_row_ids"] = [runtime.data.row_ids[unique[i]] for i in used]
    return dict(
        schema="response-anchor-train40-metadata-1",
        groups=groups,
        eligible40=eligible,
        query_indices=queries,
        query_sha256=sha256_json(queries),
        allowed_axis_sha256=sha256_json(allowed.tolist()),
        bank_manifest_sha256=bank.manifest_sha256,
        bank_arrays_sha256=bank.arrays_sha256,
        bank_slot_identity=mapping,
        protocol=PROTOCOL,
        data_identity=runtime.identity,
    )


@contextmanager
def train_read_scope(runtime, metadata, bank):
    """Native allowed-first reads plus separately justified bank-control rows."""
    from gradpert.hashing import sha256_json

    medium = analysis_module("response_anchor_medium")
    records = [r for g in metadata["groups"] for r in g["records"]]
    data = runtime.data
    native_read = data._read_expression_indices
    receipt = {}
    mapping = metadata["bank_slot_identity"]
    extras = [data._row_index[x] for x in mapping.get("selected_control_row_ids", [])]
    if not set(extras) <= set(data.control_row_indices):
        raise ValueError("bank mapping contains noncontrol row")
    # configure_training_expression_columns exists in the frozen native version.
    if not callable(getattr(data, "configure_training_expression_columns", None)):
        raise ValueError("original allowed-first native reader unavailable")
    with medium.allowed_train_reads(runtime, records, controls_only=False, receipt=receipt):
        original_guard = data._read_expression_indices
        extra_calls = 0

        def guarded(indices):
            nonlocal extra_calls
            if not len(indices) or any(type(i) not in (int, np.int64, np.int32) for i in indices):
                raise ValueError("nonempty integer authorized row indices required")
            if extras and set(indices) <= set(extras):
                extra_calls += 1
                return native_read(indices)
            return original_guard(indices)

        with patch.object(data, "_read_expression_indices", guarded):
            if extras:
                actual = data._read_expression_indices(extras)[:, runtime.allowed_expression_ids]
                expected = bank.control_expression[mapping["selected_bank_indices"]]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError("known reconstructed bank control expression mismatch")
                receipt["bank_slot_row_ids_verified"] = True
                receipt["bank_selected_control_values_sha256"] = array_sha(actual)
            else:
                receipt["bank_slot_row_ids_verified"] = False
                receipt["bank_slot_row_ids_reason"] = mapping.get("reason", "unavailable")
            receipt["extra_control_row_ids_sha256"] = sha256_json(
                mapping.get("selected_control_row_ids", [])
            )
            yield receipt
        receipt["extra_control_read_calls"] = extra_calls


def spectrum(values):
    """Condition-axis spectra, no jitter, no per-cell/target matrix persistence."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite two-dimensional feature matrix required")
    n, d = x.shape
    centered = x - x.mean(0)
    total = float(np.mean(np.sum(x * x, axis=1)))
    # SVD bounds work by N as well as width and never forms a token-sized covariance.
    eig = np.linalg.svd(centered, compute_uv=False) ** 2 / n
    energy = float(eig.sum())
    valid = energy > max(total, 1e-30) * 1e-12
    probabilities = eig[eig > 0] / energy if valid else np.array([])
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if valid else 0.0
    ceiling = min(d, n - 1)
    norms = np.linalg.norm(x, axis=1)
    return dict(
        N=n,
        width=d,
        centered_rank_ceiling=ceiling,
        effective_rank=rank,
        rank_over_width=rank / d,
        rank_over_ceiling=rank / ceiling if ceiling else None,
        covariance_trace=energy,
        mean_squared_norm=float(np.square(x.mean(0)).sum()),
        mean_sample_squared_norm=total,
        relative_variance=energy / total if total > 0 else None,
        eigenvalues=eig.tolist(),
        top_energy={str(k): float(eig[:k].sum() / energy) if valid else None for k in (1, 5, 10)},
        row_norm_quantiles=np.quantile(norms, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        zero_norm_rows=int((norms == 0).sum()),
    )


def spectra(values):
    x = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1)
    keep = norm > 0
    return dict(
        raw=spectrum(x),
        direction=spectrum(x[keep] / norm[keep, None]) if keep.any() else None,
        direction_finite_N=int(keep.sum()),
    )


def readout_statistics(model):
    head = model.response_anchor_readout
    w, b = head.weight.detach().double().cpu().numpy(), head.bias.detach().double().cpu().numpy()
    if (
        w.shape != (256, 256)
        or b.shape != (256,)
        or not np.isfinite(w).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("retained finite Linear256 readout required")
    s = np.linalg.svd(w, compute_uv=False)
    energy = s * s
    q = energy[energy > 0] / energy.sum() if energy.sum() > 0 else np.array([])
    positive = s[s > (s[0] * 1e-12 if len(s) else 0)]
    return dict(
        singular_values=s.tolist(),
        squared_singular_energy_effective_rank=float(np.exp(-(q * np.log(q)).sum()))
        if len(q)
        else 0.0,
        weight_frobenius=float(np.linalg.norm(w)),
        bias_l2=float(np.linalg.norm(b)),
        condition_estimate=float(s[0] / positive[-1]) if len(positive) == 256 else None,
        condition_NA_reason=None if len(positive) == 256 else "rank_deficient_at_relative_1e-12",
        cutoff_relative=1e-12,
    )


def constant_state(runtime):
    from capacity_probe import state_fingerprint, validation_state

    from gradpert.modeling.v2.operators import ManifoldResidual

    result = analysis_module("response_anchor_parity").protected_state(
        runtime, validation_state=validation_state, fingerprint=state_fingerprint
    )
    # RNG and modes are intentionally scoped/restored, not immutable during train-mode observation.
    result["native"] = {
        k: v
        for k, v in result["native"].items()
        if k not in {"numpy_rng", "python_rng", "torch_rng", "module_modes"}
    }
    result["requires_grad"] = state_fingerprint(
        {n: p.requires_grad for n, p in runtime.objective.named_parameters()}
    )
    result["all_module_scalar_clocks"] = state_fingerprint(
        {
            name: {
                k: v
                for k, v in vars(module).items()
                if v is None or isinstance(v, (bool, int, float, str))
                if not (isinstance(module, ManifoldResidual) and k == "resolved_sinkhorn_backend")
            }
            for name, module in runtime.objective.named_modules()
        }
    )
    result["hooks"] = state_fingerprint(
        {
            name: {
                kind: [(key, id(value)) for key, value in getattr(module, kind).items()]
                for kind in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")
            }
            for name, module in runtime.objective.named_modules()
        }
    )
    return result


def derived_telemetry(runtime):
    from gradpert.modeling.v2.operators import ManifoldResidual

    return {
        name: module.resolved_sinkhorn_backend
        for name, module in runtime.objective.named_modules()
        if isinstance(module, ManifoldResidual)
    }


@contextmanager
def preserved_probe(runtime, native_preservation):
    """Supplement native restoration with the actual RelayDeltaAttention flag."""
    from capacity_probe import validation_state

    from gradpert.modeling.v2.operators import RelayDeltaAttention

    before, native_before = constant_state(runtime), validation_state(runtime)
    orders = [
        (m, m.randomize_order)
        for m in runtime.objective.modules()
        if isinstance(m, RelayDeltaAttention)
    ]
    try:
        with native_preservation(runtime):
            yield
    finally:
        for module, value in orders:
            module.randomize_order = value
        if constant_state(runtime) != before or validation_state(runtime) != native_before:
            raise RuntimeError("outer probe changed protected state, RNG, modes or unknown clocks")


@contextmanager
def forbidden_updates(runtime):
    """Allow counted autograd.grad only; all actual training entry points fail closed."""
    import torch

    from gradpert.training.v2 import checkpoint, engine, evaluation
    from gradpert.training.v2.response_anchor_runtime import ResponseAnchorController

    counts = Counter()
    original_grad = torch.autograd.grad

    def fail(name):
        def forbidden(*args, **kwargs):
            counts[name] += 1
            raise ValueError("forbidden probe operation: " + name)

        return forbidden

    def counted_grad(*args, **kwargs):
        counts["autograd_grad"] += 1
        return original_grad(*args, **kwargs)

    with ExitStack() as stack:
        handles = [
            m.register_forward_pre_hook(fail("teacher_forward"))
            for m in runtime.objective.teacher.modules()
        ]
        stack.callback(lambda: [h.remove() for h in handles])
        for owner, key, name in (
            (torch.Tensor, "backward", "tensor_backward"),
            (torch.autograd, "backward", "autograd_backward"),
            (runtime.optimizer, "step", "optimizer_step"),
            (runtime.optimizer, "zero_grad", "optimizer_zero_grad"),
            (runtime.objective, "commit_statistics", "statistics_commit"),
            (engine, "optimizer_step", "native_engine_step"),
            (runtime, "batches", "training_iterator"),
            (runtime, "validation_batches", "validation_iterator"),
            (ResponseAnchorController, "backward", "anchor_backward"),
            (ResponseAnchorController, "groups", "anchor_cursor"),
            (checkpoint, "save_checkpoint", "checkpoint_save"),
            (checkpoint, "load_checkpoint", "training_resume"),
            (evaluation, "evaluate", "native_evaluation"),
        ):
            stack.enter_context(patch.object(owner, key, fail(name)))
        stack.enter_context(patch.object(torch.autograd, "grad", counted_grad))
        yield counts


@contextmanager
def observe_case(runtime):
    """Original native outputs are returned untouched; references die with one case."""
    import torch

    model = runtime.objective.student
    original = model.encode_response
    capture = dict(phase=None, calls=Counter())

    def encode(*args, **kwargs):
        phase = capture["phase"]
        decode = kwargs.get("decode", True)
        if phase not in ("prediction", "auxiliary") or decode != (phase == "prediction"):
            raise ValueError("native prediction/auxiliary decode boundary changed")
        value = original(*args, **kwargs)
        capture["calls"][phase] += 1
        if capture["calls"][phase] != 1:
            raise ValueError("one original complete-group forward required per branch")
        capture[phase + "_cls"] = value["response_cls"].detach().float().mean(0).cpu().numpy()
        if phase == "prediction":
            capture["prediction"] = value["prediction"]
            capture["prediction_delta"] = value["delta"].detach().float().mean(0).cpu().numpy()
        return value

    def joint_hook(_module, args):
        if capture["phase"] != "prediction":
            raise ValueError("decoder forward on auxiliary path")
        capture["joint"] = args[0]

    def score_hook(_module, _args, value):
        capture["score"] = value.detach().float()

    def anchor_hook(_module, _args, value):
        if capture["phase"] != "auxiliary":
            raise ValueError("anchor readout used by prediction path")
        capture["auxiliary_readout"] = value.detach().float().cpu().numpy()

    if (
        not isinstance(model.prediction, torch.nn.Sequential)
        or len(model.prediction) != 3
        or getattr(model, "prior_readout", None) is not None
        or model.options.prediction_parameterization != "softplus"
    ):
        raise ValueError("retained pointwise native decoder required")
    with patch.object(model, "encode_response", encode):
        handles = [
            model.prediction[0].register_forward_pre_hook(joint_hook),
            model.prediction[2].register_forward_hook(score_hook),
            model.response_anchor_readout.register_forward_hook(anchor_hook),
        ]
        try:
            yield capture
        finally:
            for h in handles:
                h.remove()


def cpu_gradients(loss, parameters, *, to_cpu=True):
    import torch

    names, values = zip(*parameters, strict=True)
    gradients = torch.autograd.grad(
        loss, values, allow_unused=True, materialize_grads=False, retain_graph=False
    )
    return {
        n: None if g is None else (g.detach().float().cpu() if to_cpu else g.detach().float())
        for n, g in zip(names, gradients, strict=True)
    }


def native_reference_case(runtime, batch, context, group, label, prediction_losses):
    """Two original grad-enabled forwards, no observer/backward, pilot only."""
    before = constant_state(runtime)
    telemetry_before = derived_telemetry(runtime)
    with context.group_rng(group["bank_condition_index"], label):
        stream = prediction_losses(runtime, batch, bf16=True, world=1, global_rows=40)
        loss = next(stream)
        prediction = float(loss.detach())
        del loss
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise ValueError("reference prediction must be one complete40 forward")
    with context.group_rng(group["bank_condition_index"], label):
        loss = context.group_loss(batch, group["bank_condition_index"], label, bf16=True)
        auxiliary = float(loss.detach())
        del loss
    after = constant_state(runtime)
    if before != after or not np.isfinite([prediction, auxiliary]).all():
        raise RuntimeError("original pilot reference changed state or produced nonfinite loss")
    return dict(
        prediction_loss=prediction,
        auxiliary_loss=auxiliary,
        state_before_sha256=digest(before),
        state_after_sha256=digest(after),
        extra_native_forwards=dict(prediction=1, auxiliary=1),
        parameter_updates=0,
        derived_telemetry_before=telemetry_before,
        derived_telemetry_after=derived_telemetry(runtime),
    )


def scalar_parity(reference, observed, *, atol=2e-6):
    differences = {
        k: abs(reference[k] - observed[k]) for k in ("prediction_loss", "auxiliary_loss")
    }
    if not np.isfinite(list(differences.values())).all() or max(differences.values()) > atol:
        raise RuntimeError("pilot original/instrumented scalar-loss parity failed")
    return dict(
        passed=True,
        atol=atol,
        max_abs=max(differences.values()),
        absolute_differences=differences,
        reference=reference,
    )


def measure_case(runtime, batch, context, group, label, prediction_losses):
    import torch

    def synchronized_time():
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        return time.perf_counter()

    case_started = synchronized_time()
    telemetry_before = derived_telemetry(runtime)
    summary = analysis_module("gradient_pair_summary").summarize_gradient_pair
    parameters = [
        (n, p) for n, p in runtime.objective.student.named_parameters() if p.requires_grad
    ]
    audit_start = synchronized_time()
    before = constant_state(runtime)
    audit_seconds = synchronized_time() - audit_start
    pred_started = synchronized_time()
    with observe_case(runtime) as seen:
        seen["phase"] = "prediction"
        with context.group_rng(group["bank_condition_index"], label):
            losses = prediction_losses(runtime, batch, bf16=True, world=1, global_rows=40)
            loss = next(losses)
            try:
                next(losses)
            except StopIteration:
                pass
            else:
                raise ValueError("native prediction helper did not yield one complete40 loss")
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite native prediction loss")
            local = (
                torch.autograd.grad(seen["prediction"].sum(), seen["joint"], retain_graph=True)[0]
                .detach()
                .float()
            )
            width = local.shape[-1] // 2
            if local.shape[-1] != 512 or width != 256:
                raise ValueError("native decoder joint width differs")
            local_stats = {
                name: dict(
                    coordinate_mean_square=float(x.square().mean()),
                    squared_l2_per_output_mean=float(x.square().sum(-1).mean()),
                )
                for name, x in (
                    ("token", local[..., :width]),
                    ("direct_condition", local[..., width:]),
                )
            }
            score = seen["score"]
            slope = torch.sigmoid(score).flatten()
            score_stats = dict(
                score_quantiles=torch.quantile(
                    score.flatten(), score.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
                )
                .cpu()
                .tolist(),
                softplus_slope_quantiles=torch.quantile(
                    slope, slope.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
                )
                .cpu()
                .tolist(),
                local_sensitivity=local_stats,
            )
            pred_loss = float(loss.detach())
            pred = cpu_gradients(loss, parameters)
            prediction_delta = seen["prediction_delta"]
            pred_cls = seen["prediction_cls"]
            del local, slope, score, loss
            seen.pop("prediction")
            seen.pop("joint")
            seen.pop("score")
        pred_seconds = synchronized_time() - pred_started
        aux_started = synchronized_time()
        seen["phase"] = "auxiliary"
        with context.group_rng(group["bank_condition_index"], label):
            loss = context.group_loss(batch, group["bank_condition_index"], label, bf16=True)
            aux_loss = float(loss.detach())
            aux = cpu_gradients(loss, parameters, to_cpu=False)
        if not math.isfinite(aux_loss):
            raise FloatingPointError("nonfinite auxiliary loss")
        aux_seconds = synchronized_time() - aux_started
        summary_started = synchronized_time()
        pair = summary(pred, aux, auxiliary_scale=runtime.options.response_anchor_weight)
        modules = {}
        prefixes = set(
            (
                "graph",
                "expression",
                "cell",
                "response",
                "response_cls",
                "prediction",
                "response_anchor_readout",
            )
        )
        for name in pred:
            parts = name.split(".")
            if parts[0] in {"graph", "cell", "response"} and len(parts) > 2:
                prefixes.add(".".join(parts[:3] if parts[2].isdigit() else parts[:2]))
        for prefix in sorted(prefixes):
            names = [n for n in pred if n == prefix or n.startswith(prefix + ".")]
            if names:
                modules[prefix] = summary(
                    {n: pred[n] for n in names},
                    {n: aux[n] for n in names},
                    auxiliary_scale=runtime.options.response_anchor_weight,
                )
        support = {
            label: [i for i, name in enumerate(pred) if predicate(bank[name])]
            for label, bank, predicate in (
                ("prediction_connected", pred, lambda x: x is not None),
                ("auxiliary_connected", aux, lambda x: x is not None),
                ("prediction_zero", pred, lambda x: x is not None and not bool(x.any())),
                ("auxiliary_zero", aux, lambda x: x is not None and not bool(x.any())),
            )
        }
        del pred, aux
        result = dict(
            condition_id=group["condition_id"],
            bank_condition_index=group["bank_condition_index"],
            aux_rng_epoch_label=label,
            pairing_epoch=1,
            prediction_loss=pred_loss,
            auxiliary_loss=aux_loss,
            gradient=pair,
            support_parameter_indices=support,
            modules=modules,
            decoder=score_stats,
            calls=dict(seen["calls"]),
            input_branches_differ=True,
            historical_update_replay=False,
            ssl_terms="not_computed",
            gradient_scaling=dict(
                prediction="mean40x1000",
                auxiliary="native_complete_group_mean",
                lambda_scale=runtime.options.response_anchor_weight,
                original_training_coefficient_world2=runtime.options.response_anchor_weight
                * 2
                * 150
                / 1322,
                observed_distributed_reduce=False,
                clipping=False,
                optimizer_transform=False,
            ),
        )
        arrays = dict(
            pred_cls=pred_cls,
            aux_cls=seen["auxiliary_cls"],
            readout=seen["auxiliary_readout"],
            prediction_delta=prediction_delta,
            truth_delta=(batch.truth.float() - batch.control.float())
            .mean(0)
            .detach()
            .cpu()
            .numpy(),
        )
        summary_seconds = synchronized_time() - summary_started
    audit_start = synchronized_time()
    after = constant_state(runtime)
    audit_seconds += synchronized_time() - audit_start
    if after != before:
        raise RuntimeError("probe changed model/optimizer/buffer/gradient/pending state")
    result["timings"] = dict(
        prediction_forward_local_decoder_parameter_grad_seconds=pred_seconds,
        auxiliary_forward_parameter_grad_seconds=aux_seconds,
        cpu_summary_seconds=summary_seconds,
        state_audit_seconds=audit_seconds,
        total_case_seconds=synchronized_time() - case_started,
    )
    result["derived_telemetry"] = dict(before=telemetry_before, after=derived_telemetry(runtime))
    return result, arrays


def case_plan(metadata):
    if len(metadata["groups"]) != 16:
        raise ValueError("sealed sixteen-condition population required")
    pilot = [(i, 0) for i in range(2)]
    return pilot + [(i, e) for e in (0, 1, 2) for i in range(16) if (i, e) not in pilot]


def resource_gate(metrics, budgets, *, total_cases=48):
    for field, cap in (
        ("peak_reserved_bytes", "max_peak_reserved_bytes"),
        ("cpu_rss_bytes", "max_cpu_rss_bytes"),
        ("elapsed_seconds", "max_total_seconds"),
    ):
        if metrics[field] > budgets[cap]:
            raise RuntimeError("probe budget exceeded: " + cap)
    if metrics["free_cuda_bytes"] < budgets["min_free_cuda_bytes"]:
        raise RuntimeError("probe free-memory headroom below sealed budget")
    done = metrics["completed_cases"]
    case_time = metrics["case_seconds_sum"]
    if not math.isfinite(case_time) or case_time < 0 or case_time > metrics["elapsed_seconds"]:
        raise ValueError("finite case-only time within elapsed wall time required")
    projected = (
        metrics["elapsed_seconds"] + (total_cases - done) * case_time / done
        if done
        else float("inf")
    )
    if done == 2 and projected > budgets["max_total_seconds"]:
        raise RuntimeError("pilot projected full runtime exceeds sealed budget")
    return projected


def amplitude_rsa(truth, cls, queries, own, *, draws=1000):
    """Mask original genes first; report effect-size sensitivity, never a new target."""
    helper = geometry_module()
    WeightedRanks, weighted_spearman = helper.WeightedRanks, helper.weighted_spearman

    truth, cls = np.asarray(truth, float), np.asarray(cls, float)
    ii, jj = np.triu_indices(len(truth), 1)
    distance, direction, amplitude = [], [], []
    for i, j in zip(ii, jj, strict=True):
        keep = ~np.isin(queries, [*own[i], *own[j]])
        a, b = truth[i, keep], truth[j, keep]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        distance.append(float(np.square(a - b).sum()) if len(a) else np.nan)
        value = float(np.square(a / na - b / nb).sum()) if na and nb else np.nan
        direction.append(0.0 if value < 1e-24 else value)
        amplitude.append([na * na + nb * nb, abs(na * na - nb * nb)])
    distance, direction, amplitude = map(np.asarray, (distance, direction, amplitude))
    zdist = np.square(cls[ii] - cls[jj]).sum(-1)
    znorm = np.linalg.norm(cls, axis=1)
    zdir = np.full(len(ii), np.nan)
    validz = (znorm[ii] > 0) & (znorm[jj] > 0)
    zdir[validz] = np.square(
        cls[ii[validz]] / znorm[ii[validz], None] - cls[jj[validz]] / znorm[jj[validz], None]
    ).sum(-1)
    zdir[zdir < 1e-24] = 0.0  # Unit-direction roundoff, not representation jitter.

    def correlation(x, y, weight, controls=None):
        keep = np.isfinite(x) & np.isfinite(y) & (weight > 0)
        if controls is not None:
            keep &= np.isfinite(controls).all(-1)
        if keep.sum() < (4 if controls is not None else 2):
            return None
        x, y, w = x[keep], y[keep], weight[keep]
        if controls is None:
            return weighted_spearman(x, y, w)
        rx, ry = WeightedRanks(x).ranks(w), WeightedRanks(y).ranks(w)
        design = np.column_stack(
            [np.ones(len(x)), *[WeightedRanks(c[keep]).ranks(w) for c in controls.T]]
        )
        sw = np.sqrt(w)
        rx = rx - design @ np.linalg.lstsq(design * sw[:, None], rx * sw, rcond=None)[0]
        ry = ry - design @ np.linalg.lstsq(design * sw[:, None], ry * sw, rcond=None)[0]
        rx -= np.average(rx, weights=w)
        ry -= np.average(ry, weights=w)
        nx, ny = np.sqrt(np.dot(w, rx * rx)), np.sqrt(np.dot(w, ry * ry))
        if nx < 1e-10 or ny < 1e-10:
            return None
        return float(np.clip(np.dot(w, rx * ry) / (nx * ny), -1, 1))

    definitions = {
        "raw": (distance, zdist, None),
        "response_direction": (direction, zdist, None),
        "both_directions": (direction, zdir, None),
        "amplitude_controlled_rank_residual": (distance, zdist, amplitude),
    }
    result = {}
    weights = np.ones(len(ii))
    rng = np.random.default_rng(20261029)
    samples = {k: [] for k in definitions}
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, len(truth), len(truth)), minlength=len(truth))
        weight = counts[ii] * counts[jj]
        for name, (x, y, controls) in definitions.items():
            value = correlation(x, y, weight, controls)
            if value is not None:
                samples[name].append(value)
    for name, (x, y, controls) in definitions.items():
        values = samples[name]
        result[name] = dict(
            value=correlation(x, y, weights, controls),
            CI95=np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            valid_draws=len(values),
            invalid_draws=draws - len(values),
        )
    return dict(
        correlations=result,
        condition_N=len(truth),
        pair_N=len(ii),
        bootstrap_draws=draws,
        resampling_unit="original condition; pair multiplicity ci*cj",
        inference_scope="secondary fixed TRAIN16 query1000, not causal or biological replication",
    )


def small_geometry(metadata, arrays):
    """Same query axes, pair-union masks; condition pairs are not IID samples."""
    helper = geometry_module()
    _geometry, _rsa = helper._geometry, helper._rsa

    output = {}
    ids = [g["condition_id"] for g in metadata["groups"]]
    queries = metadata["query_indices"]
    for label in (0, 1, 2):
        rows = [arrays[(i, label)] for i in range(16)]
        truth, pred, cls = (
            np.stack([r[k] for r in rows]) for k in ("truth_delta", "prediction_delta", "aux_cls")
        )
        own = np.asarray([[g["anchor_gene_index"]] for g in metadata["groups"]], dtype=np.int64)
        geometry = _geometry(
            condition_ids=ids,
            allowed_gene_indices=queries,
            own_target_indices=own,
            prediction_delta=pred,
            truth_delta=truth,
            cls_means=cls,
        )
        output[str(label)] = dict(
            response_RSA=_rsa(geometry),
            amplitude_sensitivity=amplitude_rsa(
                truth, cls, queries, own, draws=PROTOCOL["bootstrap_draws"]
            ),
            prediction_condition_spectra=spectra(pred),
            prediction_cls_spectra=spectra(np.stack([r["pred_cls"] for r in rows])),
            auxiliary_cls_spectra=spectra(cls),
            readout_spectra=spectra(np.stack([r["readout"] for r in rows])),
            condition_N=16,
            pair_N=120,
            query_N=1000,
            scope="TRAIN subset/query geometry, not full4775/val endpoint",
        )
    return output


def seal(spec, *, spec_file):
    import torch

    from gradpert.training.v2.runtime import prepare_runtime

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("metadata sealing must be explicitly CPU-only")
    validate_spec(spec)
    sources = verify_sources(spec)
    dependencies = preload_analysis_dependencies(spec)
    config, identity = load_config_identity(spec)
    output = Path(spec["output"])
    if output.exists():
        raise ValueError("fresh probe output required")
    with prepare_runtime(
        config,
        data_root=Path(spec["data_root"]),
        run_seed=1,
        device=torch.device("cpu"),
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != identity["data"]:
            raise ValueError("original native runtime data identity differs")
        with patch.object(
            runtime.data,
            "_read_expression_indices",
            side_effect=ValueError("numerical H5AD read forbidden while sealing"),
        ):
            bank = load_bank(runtime)
            metadata = make_metadata(runtime, bank, spec["endpoint"]["source"]["root"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "SPEC.json", spec)
    write_new(output / "METADATA.json", metadata)
    result = dict(
        schema="response-anchor-train40-sealed-1",
        spec=spec_file,
        spec_digest=digest(spec),
        metadata_sha256=sha(output / "METADATA.json"),
        sources=sources,
        analysis_dependencies=dependencies,
        numerical_h5ad_reads=0,
        updates=0,
        scientific_complete=False,
    )
    write_new(output / "SEALED.json", result)
    return result


def validate_sealed(spec):
    validate_spec(spec)
    output = Path(spec["output"])
    sealed, stored = read(output / "SEALED.json"), read(output / "SPEC.json")
    if (
        stored != spec
        or sealed["spec_digest"] != digest(spec)
        or sha(output / "METADATA.json") != sealed["metadata_sha256"]
    ):
        raise ValueError("sealed spec/metadata bytes changed")
    return sealed, read(output / "METADATA.json")


def run(spec, *, owner_binding):
    started = time.monotonic()
    sealed, metadata = validate_sealed(spec)
    sources = verify_sources(spec)
    preload_analysis_dependencies(spec)
    import torch
    from run_group import lock
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import _move_batch_tree, prepare_runtime

    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("native allocator contract missing")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one isolated visible native GPU required")
    if any(
        (Path(spec["output"]) / name).exists()
        for name in ("PILOT.json", "COMPLETE.json", "FAILURE.json")
    ):
        raise ValueError("terminal or already-started probe cannot be silently rerun")
    if sources != sealed["sources"]:
        raise ValueError("published sources changed after seal")
    medium = analysis_module("response_anchor_medium")
    owner_receipt = read(checked(owner_binding))
    owner_pid = medium.verified_owner(owner_receipt)
    policy_doc = read(checked(spec["resource_policy"]))
    policy = policy_doc.get("shared_gpu_policy", policy_doc)
    config, identity = load_config_identity(spec)
    native_root = spec["endpoint"]["source"]["root"]
    calibration = native_script("scripts/v2/calibrate_response_anchor.py", native_root)
    cap = analysis_module("cap40_posthoc")
    output = Path(spec["output"])
    rows, arrays = [], {}
    resource_rows = []
    pilot_parities = []
    # An exclusive claim prevents a queued duplicate from contaminating a completed root.
    write_new(
        output / "RUN_STARTED.json",
        dict(pid=os.getpid(), spec_digest=digest(spec), owner=owner_receipt),
    )

    def admitted():
        if medium.verified_owner(owner_receipt) != owner_pid:
            raise ValueError("resource owner changed")
        observed = snapshot(owner_pid)
        admit(policy, observed)
        resource_rows.append(digest(observed))

    try:
        with lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock"):
            admitted()
            torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
            torch.set_num_threads(1)
            with prepare_runtime(
                config,
                data_root=Path(spec["data_root"]),
                run_seed=1,
                device=torch.device("cuda:0"),
                purpose="evaluation",
            ) as runtime:
                if runtime.identity != identity["data"] or runtime.options.microbatch != 80:
                    raise ValueError("original R14 data/microbatch differs")
                load_evaluation_checkpoint(
                    checked(spec["endpoint"]["checkpoint"]),
                    runtime.objective,
                    training_identity=identity,
                    checkpoint_sha256=CP_SHA,
                )
                write_new(
                    output / "PARAMETERS.json",
                    dict(
                        parameters=[
                            dict(name=n, shape=list(p.shape), numel=p.numel())
                            for n, p in runtime.objective.student.named_parameters()
                            if p.requires_grad
                        ],
                        optimizer_routes=runtime.optimizer.routes,
                        support_indices_reference="ordered parameters in this file",
                    ),
                )
                bank = load_bank(runtime)
                bank_bindings = [
                    (bank.manifest_path, bank.manifest_sha256),
                    (bank.arrays_path, bank.arrays_sha256),
                ]
                canonical = runtime.data.layout.canonical_adata
                canonical_stat = (canonical.stat().st_size, canonical.stat().st_mtime_ns)
                current = make_metadata(runtime, bank, native_root)
                if digest(current) != digest(metadata):
                    raise ValueError("server metadata replay differs from seal")
                context = bank_context(runtime, bank)
                target_selected = bank.condition_targets[
                    [g["bank_condition_index"] for g in metadata["groups"]]
                ]
                write_new(
                    output / "target_spectrum.json",
                    dict(
                        fullTRAIN=spectra(bank.condition_targets),
                        selected16=spectra(target_selected),
                        original_target_transform="frozen4775_projection256_TRAIN_RMS_no_refit",
                    ),
                )
                write_new(
                    output / "readout_spectrum.json", readout_statistics(runtime.objective.student)
                )
                torch.cuda.reset_peak_memory_stats()
                # Restore ephemeral state; check scientific bytes before restoration as well.
                with (
                    preserved_probe(runtime, calibration.preserved_calibration),
                    forbidden_updates(runtime) as forbidden,
                ):
                    state0 = constant_state(runtime)
                    with train_read_scope(runtime, metadata, bank) as read_receipt:
                        for number, (i, label) in enumerate(case_plan(metadata), 1):
                            loop_started = time.monotonic()
                            admitted()
                            group = metadata["groups"][i]
                            batch_cpu = cap.assemble(
                                runtime,
                                group["records"],
                                metadata["query_indices"],
                                group["view_seed"],
                                deep=False,
                            )
                            batch = _move_batch_tree(batch_cpu, runtime.device)
                            queries = batch.graph.ids[batch.query_positions].detach().cpu().tolist()
                            if queries != metadata["query_indices"] or len(batch.control) != 40:
                                raise ValueError("native complete40/query assembly changed")
                            reference_seconds = 0.0
                            reference = None
                            if number <= 2:
                                reference_started = time.monotonic()
                                reference = native_reference_case(
                                    runtime,
                                    batch,
                                    context,
                                    group,
                                    label,
                                    calibration.prediction_losses,
                                )
                                torch.cuda.synchronize()
                                reference_seconds = time.monotonic() - reference_started
                            row, values = measure_case(
                                runtime, batch, context, group, label, calibration.prediction_losses
                            )
                            if reference is not None:
                                parity = scalar_parity(reference, row)
                                pilot_parities.append(parity)
                                row["pilot_original_scalar_parity"] = parity
                            rows.append(row)
                            arrays[(i, label)] = values
                            del batch, batch_cpu
                            torch.cuda.synchronize()
                            free, _ = torch.cuda.mem_get_info()
                            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            # Linux reports KiB, macOS bytes (formal execution is Linux server).
                            outer_audit_started = time.monotonic()
                            if (
                                forbidden["autograd_grad"] != number * 3
                                or any(
                                    value
                                    for key, value in forbidden.items()
                                    if key != "autograd_grad"
                                )
                                or constant_state(runtime) != state0
                            ):
                                raise RuntimeError("zero-update/call-count/state contract failed")
                            row["timings"]["outer_state_audit_seconds"] = (
                                time.monotonic() - outer_audit_started
                            )
                            row["timings"]["complete_loop_case_seconds"] = (
                                time.monotonic() - loop_started - reference_seconds
                            )
                            row["timings"]["pilot_reference_seconds"] = reference_seconds
                            persist_case_scalar(output, number, row, digest(spec))
                            metrics = dict(
                                completed_cases=number,
                                elapsed_seconds=time.monotonic() - started,
                                case_seconds_sum=sum(
                                    r["timings"]["complete_loop_case_seconds"] for r in rows
                                ),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                cpu_rss_bytes=int(rss * (1024 if sys.platform != "darwin" else 1)),
                                free_cuda_bytes=free,
                            )
                            projected = resource_gate(metrics, spec["budgets"])
                            if number == 2:
                                write_new(
                                    output / "PILOT.json",
                                    dict(
                                        status="passed",
                                        complete_groups=2,
                                        rows_per_group=40,
                                        replicates=1,
                                        spec_digest=digest(spec),
                                        metrics=metrics,
                                        projected_total_seconds=projected,
                                        state_unchanged=True,
                                        original_scalar_parity=pilot_parities,
                                        extra_original_forwards=dict(prediction=2, auxiliary=2),
                                        forbidden_calls=dict(forbidden),
                                        remaining_cases=46,
                                        updates=0,
                                    ),
                                )
                        if len(rows) != 48 or len(arrays) != 48:
                            raise RuntimeError("full sealed case coverage incomplete")
                        geometry = geometry_after_scalar_aggregate(output, rows, metadata, arrays)
                    if constant_state(runtime) != state0:
                        raise RuntimeError("protected scientific state changed")
                    if canonical_stat != (canonical.stat().st_size, canonical.stat().st_mtime_ns):
                        raise RuntimeError("canonical scientific input stat changed")
                # The native context independently checks outer RNG/mode/grad restoration.
                final_read = copy.deepcopy(read_receipt)
            admitted()
            if verify_sources(spec) != sources or validate_sealed(spec)[0] != sealed:
                raise ValueError("final source/spec/input closure changed")
            if any(sha(path) != expected for path, expected in bank_bindings):
                raise ValueError("frozen bank bytes changed")
            checked(owner_binding)
            if list(output.rglob("*.pkl")):
                raise ValueError("whole probe root must remain zero-PKL")
            write_new(output / "geometry.json", geometry)
            write_new(output / "READ_SCOPE.json", final_read)
            result = dict(
                schema="response-anchor-train40-gradient-complete-1",
                status="completed",
                spec_digest=digest(spec),
                metadata_sha256=sealed["metadata_sha256"],
                cases=48,
                complete_conditions=16,
                rows_per_condition=40,
                replicates=3,
                source=sources,
                checkpoint_sha256=CP_SHA,
                owner=owner_receipt,
                resource_snapshot_sha256=resource_rows,
                metrics=metrics,
                forbidden_calls=dict(forbidden),
                native_forward_totals=dict(
                    prediction=50,
                    auxiliary=50,
                    formal_cases_each_branch=48,
                    extra_pilot_original_each_branch=2,
                ),
                updates=0,
                teacher_updates=0,
                center_updates=0,
                state_unchanged=True,
                zero_pkl=True,
                scientific_complete=False,
                interpretation=(
                    "Different native inputs at one checkpoint; no historical optimizer replay."
                ),
            )
            result["output_sha256"] = {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            }
            if set(result["output_sha256"]) != expected_outputs():
                raise ValueError("unexpected/missing successful output closure")
            write_new(output / "COMPLETE.json", result)
            return result
    except BaseException as error:
        if not (output / "FAILURE.json").exists():
            write_new(
                output / "FAILURE.json",
                dict(
                    status="failed",
                    error_type=type(error).__name__,
                    message=str(error),
                    completed_cases=len(rows),
                    persisted_scalar_case_files=sorted(
                        str(p.relative_to(output))
                        for p in (output / CASE_SCALAR_DIR).glob("*.json")
                    ),
                    updates=0,
                    scientific_complete=False,
                ),
            )
        raise


def validate_result(spec):
    sealed, metadata = validate_sealed(spec)
    output = Path(spec["output"])
    result = read(output / "COMPLETE.json")
    pilot = read(output / "PILOT.json")
    rows = read(output / "cases.json")
    actual_files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual_files - {"COMPLETE.json"} != set(result.get("output_sha256", {})) or (
        set(result.get("output_sha256", {})) != expected_outputs()
    ):
        raise ValueError("complete output file set differs from exact required hash closure")
    expected = [(metadata["groups"][i]["condition_id"], e) for i, e in case_plan(metadata)]
    actual = [(r["condition_id"], r["aux_rng_epoch_label"]) for r in rows]
    if actual != expected or len(set(actual)) != 48 or pilot.get("status") != "passed":
        raise ValueError("sealed pilot/full ordered coverage differs")
    for number, row in enumerate(rows, 1):
        record = read(output / case_scalar_path(number))
        if (
            set(record) != {"schema", "case_number", "spec_digest", "row"}
            or record["schema"] != "response-anchor-train40-case-scalars-1"
            or record["case_number"] != number
            or record["spec_digest"] != digest(spec)
            or record["row"] != row
        ):
            raise ValueError("per-case scalar report differs from its exact aggregate row")
    if (
        pilot.get("complete_groups"),
        pilot.get("rows_per_group"),
        pilot.get("replicates"),
        pilot.get("remaining_cases"),
    ) != (2, 40, 1, 46):
        raise ValueError("pilot must cover two complete40 groups before expanding")
    parities = pilot.get("original_scalar_parity", [])
    if len(parities) != 2 or any(
        p.get("passed") is not True
        or p.get("atol") != 2e-6
        or not math.isfinite(p.get("max_abs", float("inf")))
        or not 0 <= p["max_abs"] <= 2e-6
        or p["reference"].get("state_before_sha256") != p["reference"].get("state_after_sha256")
        or p["reference"].get("extra_native_forwards") != dict(prediction=1, auxiliary=1)
        for p in parities
    ):
        raise ValueError("real native pilot scalar/state parity evidence missing or failed")
    if (
        result.get("spec_digest") != digest(spec)
        or result.get("metadata_sha256") != sealed["metadata_sha256"]
    ):
        raise ValueError("terminal immutable identity differs")
    if (
        result.get("updates") != 0
        or result.get("state_unchanged") is not True
        or result.get("scientific_complete") is not False
    ):
        raise ValueError("terminal zero-update contract failed")
    if result.get("native_forward_totals") != dict(
        prediction=50,
        auxiliary=50,
        formal_cases_each_branch=48,
        extra_pilot_original_each_branch=2,
    ):
        raise ValueError("extra pilot original forwards not truthfully counted")
    if result.get("forbidden_calls", {}).get("autograd_grad") != 144 or any(
        v for k, v in result["forbidden_calls"].items() if k != "autograd_grad"
    ):
        raise ValueError("terminal forbidden operation/call count differs")
    if (output / "FAILURE.json").exists() or list(output.rglob("*.pkl")):
        raise ValueError("failure marker or PKL in successful probe root")
    for name, value in result["output_sha256"].items():
        if name not in expected_outputs() or sha(output / name) != value:
            raise ValueError("terminal output bytes changed")
    return dict(status="validated", cases=48, updates=0, scientific_complete=False)


def dispatch(spec, *, spec_file, gpu):
    """One long-lived native owner, one child, external logs; never retries/kills."""
    validate_sealed(spec)
    sources = verify_sources(spec)
    preload_analysis_dependencies(spec)
    if gpu not in ("0", "1") or os.path.abspath(sys.executable) != SERVER_PYTHON:
        raise ValueError("dispatch requires the original server Python and one explicit GPU")
    output = Path(spec["output"])
    controller = output.with_name(output.name + "-dispatch")
    controller.mkdir(parents=False, exist_ok=False)
    try:
        return _dispatch_owned(spec, spec_file, gpu, controller, sources)
    except BaseException as error:
        if not (controller / "FAILURE.json").exists():
            write_new(
                controller / "FAILURE.json",
                dict(status="failed", error_type=type(error).__name__, automatic_retry=False),
            )
        raise


def validated_child_identity(observed, *, pid, command, parent_pid):
    fields = ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
    expected_hash = hashlib.sha256(b"\0".join(os.fsencode(a) for a in command) + b"\0").hexdigest()
    if (
        not isinstance(observed, dict)
        or any(k not in observed for k in fields)
        or observed["pid"] != pid
        or observed["uid"] != os.getuid()
        or not isinstance(observed["start_ticks"], str)
        or not observed["start_ticks"].isdigit()
        or observed["command_sha256"] != expected_hash
        or observed["script_basename"] != Path(__file__).name
        or observed.get("ppid") != parent_pid
    ):
        raise ValueError("live child PID/UID/start/raw-command/script/parent identity mismatch")
    return {k: observed[k] for k in fields}


def _dispatch_owned(spec, spec_file, gpu, controller, sources):
    from shared_gpu import process_info

    output = Path(spec["output"])
    fields = ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
    owner = None
    for _ in range(20):
        info = process_info(os.getpid())
        if info is not None and all(k in info for k in fields):
            owner = {k: info[k] for k in fields}
            break
        time.sleep(0.1)
    if owner is None:
        write_new(
            controller / "FAILURE.json", dict(status="failed", reason="owner_identity_unavailable")
        )
        raise RuntimeError("bounded dispatcher owner identity unavailable")
    owner_path = controller / "OWNER.json"
    write_new(owner_path, dict(schema="response-anchor-analysis-owner-v1", process=owner))
    command = [
        SERVER_PYTHON,
        str(Path(__file__).resolve()),
        "run",
        "--spec",
        spec_file["path"],
        "--spec-sha256",
        spec_file["sha256"],
        "--owner-receipt",
        str(owner_path),
        "--owner-sha256",
        sha(owner_path),
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(spec["endpoint"]["source"]["root"]) / "src"),
                spec["endpoint"]["source"]["root"],
            ]
        ),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    with (
        (controller / "stdout.log").open("xb") as stdout,
        (controller / "stderr.log").open("xb") as stderr,
    ):
        child = subprocess.Popen(
            command,
            cwd=spec["analysis_source"]["root"],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )
        child_identity = None
        identity_error, observations = None, []
        try:
            for _ in range(20):
                observed = process_info(child.pid)
                observations.append(observed)
                if observed is not None:
                    try:
                        child_identity = validated_child_identity(
                            observed, pid=child.pid, command=command, parent_pid=os.getpid()
                        )
                    except ValueError as error:
                        identity_error = str(error)
                    break
                if child.poll() is not None:
                    break
                time.sleep(0.1)
            if child_identity is None:
                identity_error = identity_error or "live child identity unavailable within bound"
            write_new(
                controller / "LAUNCH.json",
                dict(
                    command=command,
                    pid=child.pid,
                    owner=owner,
                    child_identity=child_identity,
                    child_parent_pid=os.getpid(),
                    observations=observations,
                    identity_error=identity_error,
                    bounded_child_observation=True,
                    sources=sources,
                ),
            )
        finally:
            # Drain exactly the child we spawned even if telemetry/receipt writing raises.
            code = child.wait()
    if identity_error:
        write_new(
            controller / "FAILURE.json",
            dict(
                status="failed",
                returncode=code,
                reason=identity_error,
                automatic_retry=False,
                child_drained=True,
            ),
        )
        raise RuntimeError("native probe child identity failed; child drained without retry")
    if code:
        write_new(
            controller / "FAILURE.json",
            dict(
                status="failed",
                returncode=code,
                native_failure_marker=(output / "FAILURE.json").exists(),
                automatic_retry=False,
            ),
        )
        raise RuntimeError("native probe child failed; evidence retained without retry")
    validation = validate_result(spec)
    result = dict(
        status="completed",
        returncode=0,
        native_result_validation=validation,
        owner=owner,
        child_identity=child_identity,
        sources=sources,
        output=str(output),
        logs_outside_native_result=True,
        scientific_complete=False,
    )
    write_new(controller / "COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "validate", "run", "dispatch", "validate-result"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--owner-receipt", type=Path)
    parser.add_argument("--owner-sha256")
    parser.add_argument("--gpu", choices=("0", "1"))
    args = parser.parse_args()
    spec_file = dict(path=str(args.spec.resolve()), sha256=args.spec_sha256)
    spec = read(checked(spec_file))
    if args.mode == "seal":
        result = seal(spec, spec_file=spec_file)
    elif args.mode == "dispatch":
        if args.gpu is None:
            parser.error("dispatch requires --gpu")
        result = dispatch(spec, spec_file=spec_file, gpu=args.gpu)
    elif args.mode == "validate":
        result = validate_sealed(spec)[0]
    elif args.mode == "validate-result":
        result = validate_result(spec)
    else:
        if args.owner_receipt is None or args.owner_sha256 is None:
            parser.error("run requires existing sealed owner receipt and hash")
        result = run(
            spec,
            owner_binding=dict(path=str(args.owner_receipt.resolve()), sha256=args.owner_sha256),
        )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
