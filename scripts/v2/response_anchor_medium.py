# ruff: noqa: RUF001
"""Direct sealed medium/crossed diagnostics; no family registration or fit path.

Run as a file from the published analysis checkout, with the endpoint's original
native src first on PYTHONPATH. Only synthetic CPU tests run locally. Native
medium/collector kernels and checkpoint validators remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

COUNTS = dict(
    conditions=1335,
    rows=5317,
    deep_conditions=128,
    deep_rows=506,
    gradient_conditions=32,
    gradient_rows=125,
)
SCHEMA = "response-anchor-direct-medium-spec-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def checked(binding: dict[str, Any], *, server: bool = True) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError("exact file path/hash binding required")
    path = Path(binding["path"]).resolve(strict=True)
    if server and not path.is_relative_to("/data/yilangliu"):
        raise ValueError("scientific input must stay on server")
    if sha(path) != binding["sha256"]:
        raise ValueError("sealed file bytes changed")
    return path


def identity_diff(before: dict, after: dict, prefix: str = "") -> list[dict]:
    changes = []
    for key in sorted(before.keys() | after.keys()):
        path = prefix + "/" + key
        a, b = key in before, key in after
        if a and b and isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.extend(identity_diff(before[key], after[key], path))
        elif a != b or before.get(key) != after.get(key):
            changes.append(
                dict(
                    path=path,
                    before=dict(present=a, **({"value": before[key]} if a else {})),
                    after=dict(present=b, **({"value": after[key]} if b else {})),
                )
            )
    return changes


def bind_parent(parent: dict, identity: dict, approved: list[dict]) -> dict:
    """Allow only explicitly sealed protocol deltas; physical identities stay exact."""
    changes = identity_diff(parent["data"], identity["data"])
    if changes != approved or any(
        not (
            r["path"] == "/response_anchor"
            or r["path"].startswith("/response_anchor/")
            or r["path"].startswith("/loss_protocol/")
        )
        for r in changes
    ):
        raise ValueError("unapproved parent data/protocol difference")
    result = copy.deepcopy(parent)
    result["data"] = copy.deepcopy(identity["data"])
    result["training_protocol_binding"] = dict(
        training_run_id=identity["run_id"],
        changed_fields=changes,
        sampling_changed=False,
        original_data_sha256=digest(parent["data"]),
        training_data_sha256=digest(identity["data"]),
        interpretation="protocol descriptor binding only; not endpoint numerical equivalence",
    )
    return result


def population(schedule: list[dict]) -> str:
    # Exact native cap40_posthoc.population_identity serialization is reused at runtime.
    from cap40_posthoc import population_identity

    return population_identity(schedule)


def validate_panel(panel: dict, parent: dict) -> None:
    from cap40_posthoc_stats import digest_json

    if (
        panel.get("conditions"),
        panel.get("rows"),
        panel.get("controls_per_context"),
        panel.get("condition_context_groups"),
    ) != (32, 128, 4, 32):
        raise ValueError("only the existing 32x4 crossed128 panel is allowed")
    if panel.get("panel_sha256") != digest_json(panel["schedule"]):
        raise ValueError("panel ordered population changed")
    contexts: dict[tuple, list] = {}
    train = {g["condition_id"] for g in parent["schedule"]}
    for group in panel["schedule"]:
        context = tuple(group["context"])
        if len(context) != 4 or group["condition_id"] not in train:
            raise ValueError("panel context/condition left parent")
        contexts.setdefault(context, []).append(group)
        if group["queries"] != parent["schedule"][0]["queries"] or len(group["rows"]) != 4:
            raise ValueError("panel query/order/row budget changed")
    if len(contexts) != 16:
        raise ValueError("crossed panel requires16 distinct full context quadruples")
    for groups in contexts.values():
        if len(groups) != 2 or len({g["condition_id"] for g in groups}) != 2:
            raise ValueError("crossed context must have two distinct conditions")
        controls = [[r["control_row_id"] for r in g["rows"]] for g in groups]
        if controls[0] != controls[1] or len(set(controls[0])) != 4:
            raise ValueError("crossed conditions must share four exact ordered controls")


@contextmanager
def allowed_train_reads(
    runtime: Any, records: list[dict], *, controls_only: bool, receipt: dict
) -> Iterator[None]:
    """Instance-only allowed-column-first adapter, with strict selected-row whitelist.

    Older native sources lack configure_training_expression_columns. The fallback
    uses consecutive selected row slices and one fancy allowed-column axis before
    .X, preserving the original global zero placeholder axis. Physical HDF5 bytes
    are unmeasured; no claim of zero physical I/O of excluded storage bytes.
    """
    context_batches: dict[tuple, set] = {}
    for row in records:
        key = tuple(row.get(k, "") for k in ("cell_line", "cell_type", "context_id"))
        context_batches.setdefault(key, set()).add(row.get("batch", ""))
    if any(len(batches) != 1 for batches in context_batches.values()):
        raise ValueError("native grouped-bank triple key would merge distinct batches")
    data = runtime.data
    allowed = np.asarray(runtime.allowed_expression_ids)
    n = data.manifest.n_expression_genes
    if (
        allowed.ndim != 1
        or allowed.dtype.kind not in "iu"
        or not len(allowed)
        or (allowed < 0).any()
        or (allowed >= n).any()
        or (np.diff(allowed) <= 0).any()
    ):
        raise ValueError("invalid allowed expression axis")
    train, controls = set(data.train_row_indices), set(data.control_row_indices)
    whitelist = set()
    for row in records:
        ti, ci = data._row_index[row["truth_row_id"]], data._row_index[row["control_row_id"]]
        if ti not in train or ci not in controls:
            raise ValueError("medium/panel row left selected TRAIN/control population")
        if (
            str(data.condition_ids[ti]) != row["condition_id"]
            or str(data.context_ids[ti]) != row["context_id"]
            or str(data.context_ids[ci]) != row["context_id"]
        ):
            raise ValueError("condition/control context membership mismatch")
        if (
            str(data.batch_ids[ti]) != row["batch"]
            or str(data.batch_ids[ci]) != row["batch"]
            or str(data._adata.obs.iloc[ti]["cell_type"]) != row["cell_type"]
            or str(data._adata.obs.iloc[ci]["cell_type"]) != row["cell_type"]
            or row["cell_line"] != "Jurkat"
        ):
            raise ValueError("full cell-line/type/context/batch metadata differs")
        whitelist.add(ci)
        if not controls_only:
            whitelist.add(ti)
    if (
        getattr(data, "_control_expression_cache", None) is not None
        or getattr(data, "_training_expression_read_started", False)
        or getattr(data.pipeline_stats, "control_cache_requested", False)
        or getattr(data.pipeline_stats, "yielded_batches", 0)
    ):
        raise ValueError("allowed-first boundary must precede every read/cache")
    original = data._read_expression_indices
    native = getattr(data, "configure_training_expression_columns", None)
    if native is not None:
        native(allowed)
    receipt.update(
        method="native_configure" if native else "old_source_allowed_slice_adapter",
        allowed_axis_sha256=digest(allowed.tolist()),
        allowed_count=len(allowed),
        calls=0,
        rows=0,
        controls_only=controls_only,
        whitelist_sha256=digest(sorted(whitelist)),
        physical_bytes_measured=False,
    )

    def guarded(indices):
        indices = list(indices)
        if not indices or any(type(i) not in (int, np.int64, np.int32) for i in indices):
            raise ValueError("nonempty integer row indices required")
        if not set(indices) <= whitelist:
            raise ValueError("diagnostic read outside exact selected row whitelist")
        receipt["calls"] += 1
        receipt["rows"] += len(indices)
        if native is not None:
            return original(indices)
        rows, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
        values = np.empty((len(rows), len(allowed)), dtype=np.float32)
        start = 0
        while start < len(rows):
            stop = start + 1
            while stop < len(rows) and rows[stop] == rows[stop - 1] + 1 and stop - start < 2048:
                stop += 1
            selection = data._adata[slice(int(rows[start]), int(rows[stop - 1]) + 1), allowed]
            matrix = selection.X
            block = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            if block.shape != (stop - start, len(allowed)) or not np.isfinite(block).all():
                raise ValueError("invalid allowed numeric block")
            values[start:stop] = block
            start = stop
        result = np.zeros((len(indices), n), dtype=np.float32)
        result[:, allowed] = values[inverse]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite expression conversion")
        return result

    with patch.object(data, "_read_expression_indices", guarded):
        yield


@contextmanager
def diagnostic_adapter(
    cap: Any,
    panel_module: Any,
    records: list[dict],
    *,
    controls_only: bool,
    read_receipts: list,
    frozen_panel: dict | None = None,
    parent_binding: dict | None = None,
    expected_axis: dict | None = None,
    expected_views: list[dict] | None = None,
) -> Iterator[None]:
    """Replace endpoint binding hooks only, preserving native collectors/validators."""
    original_args = cap.runtime_args

    def runtime_args(args):
        torch, config, identity, prepare, root = original_args(args)

        @contextmanager
        def guarded_prepare(*a, **kw):
            if kw.get("purpose") != "evaluation":
                raise ValueError("medium worker must use evaluation runtime")
            with prepare(*a, **kw) as runtime:
                axis = np.asarray(runtime.allowed_expression_ids)
                if expected_axis is not None and (
                    len(axis) != expected_axis["count"]
                    or digest(axis.tolist()) != expected_axis["sha256"]
                ):
                    raise ValueError("actual allowed axis differs before any numerical read")
                receipt = {}
                with allowed_train_reads(
                    runtime, records, controls_only=controls_only, receipt=receipt
                ):
                    yield runtime
                read_receipts.append(receipt)

        return torch, config, identity, guarded_prepare, root

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(cap, "runtime_args", runtime_args))
        if panel_module is not None:
            stack.enter_context(patch.object(panel_module, "runtime_args", runtime_args))

            def bind(parent, identity, resource, args):
                rows = resource.get("rows", [])
                if (
                    len(rows) != 1
                    or resource["source_commit"] != args.method_sha
                    or rows[0]["plan"]["run_id"] != identity["run_id"]
                    or Path(rows[0]["plan"]["run_root"]).resolve() != args.run_root.resolve()
                    or Path(rows[0]["plan"]["config"]).resolve() != args.config.resolve()
                    or rows[0]["plan"]["config_sha256"] != sha(args.config)
                    or parent["data"] != identity["data"]
                ):
                    raise ValueError("single endpoint resource/config/parent binding differs")
                return copy.deepcopy(parent_binding)

            def panel_schedule(parent, *, seed, conditions, controls):
                if (seed, conditions, controls) != (frozen_panel["seed"], 32, 4):
                    raise ValueError("new panel sampling is not authorized")
                validate_panel(frozen_panel, parent)
                return copy.deepcopy(frozen_panel)

            stack.enter_context(patch.object(panel_module, "panel_parent_binding", bind))
            stack.enter_context(patch.object(panel_module, "panel_schedule", panel_schedule))
            if expected_views is not None:
                original_views = panel_module.MatchedCellViews

                class FrozenMatchedViews(original_views):
                    def __init__(self):
                        super().__init__()
                        self.cursor = 0

                    def bind(self, batch, records):
                        result, receipt = super().bind(batch, records)
                        actual = dict(condition_id=records[0]["condition_id"], **receipt)
                        if self.cursor >= len(expected_views) or matched_signature(actual) != (
                            matched_signature(expected_views[self.cursor])
                        ):
                            raise ValueError(
                                "frozen matched crop/mask/order changed before collection"
                            )
                        self.cursor += 1
                        return result, receipt

                stack.enter_context(
                    patch.object(panel_module, "MatchedCellViews", FrozenMatchedViews)
                )
        yield


def matched_signature(row: dict) -> tuple:
    return tuple(
        row[k] if k != "context" else tuple(row[k])
        for k in (
            "condition_id",
            "context",
            "matched_cell_views_sha256",
            "query_sha256",
            "controls_sha256",
        )
    )


def verified_owner(receipt: dict, *, lookup=None, parents=None, pid=None, uid=None) -> int:
    from shared_gpu import descendant, parent_map, process_info

    lookup = process_info if lookup is None else lookup
    parents = parent_map() if parents is None else parents
    pid, uid = os.getpid() if pid is None else pid, os.getuid() if uid is None else uid
    expected = receipt.get("process", {})
    fields = {"pid", "uid", "start_ticks", "command_sha256", "script_basename"}
    if (
        receipt.get("schema") != "response-anchor-analysis-owner-v1"
        or set(expected) != fields
        or type(expected["pid"]) is not int
        or expected["pid"] <= 0
        or expected["uid"] != uid
    ):
        raise ValueError("invalid sealed resource owner")
    actual = lookup(expected["pid"])
    if (
        actual is None
        or any(actual.get(k) != expected[k] for k in fields)
        or not descendant(pid, expected["pid"], parents)
    ):
        raise ValueError("resource owner PID/start/command/UID/ancestry changed")
    return expected["pid"]


@contextmanager
def resource_snapshots(receipt: dict) -> Iterator[None]:
    """Classify sibling workers under one exact live dispatcher, never relax admit."""
    import shared_gpu

    original = shared_gpu.snapshot

    def snapshot(_requested_pid):
        owner = verified_owner(receipt)
        observed = original(owner)
        if verified_owner(receipt) != owner:
            raise ValueError("resource owner changed during telemetry")
        return observed

    with patch.object(shared_gpu, "snapshot", snapshot):
        yield


def validate_scopes(deep: list[dict], *, anchor: bool) -> None:
    if not anchor:
        return
    for group in deep:
        scope = group.get("diagnostic_objective_scope", {})
        if scope.get("unavailable_terms", {}).get("response_anchor") != (
            "requires_complete_train_condition_bank"
        ) or "response_anchor" in group.get("loss_terms", {}):
            raise ValueError("partial anchor objective must be unavailable, never zero")


def validate_spec(spec: dict, *, server: bool = True) -> dict:
    if (
        set(spec)
        != {
            "schema",
            "name",
            "analysis_source",
            "native_source",
            "run_root",
            "run_manifest",
            "resolved_config",
            "config",
            "journal",
            "checkpoint",
            "data_root",
            "medium",
            "panel",
            "allowed_axis",
            "resource_policy",
            "resource_owner",
            "output",
            "gpu",
        }
        or spec["schema"] != SCHEMA
    ):
        raise ValueError("exact sealed medium SPEC schema required")
    for key, fields in (
        ("medium", {"parent", "performance", "population_sha256", "identity_changes"}),
        ("panel", {"population", "matched_views"}),
        ("checkpoint", {"role", "epoch", "sha256"}),
        ("allowed_axis", {"count", "sha256"}),
        ("native_source", {"root", "commit", "publication", "publication_sha256"}),
        ("analysis_source", {"root", "commit", "publication", "publication_sha256"}),
    ):
        if not isinstance(spec[key], dict) or set(spec[key]) != fields:
            raise ValueError("unexpected or missing sealed binding fields: " + key)
    if spec["name"] not in ("R14-FRGA", "R3-RI") or spec["gpu"] not in ("0", "1"):
        raise ValueError("unknown endpoint/GPU")
    if spec["checkpoint"].get("role") != "last" or spec["checkpoint"].get("epoch") != 2:
        raise ValueError("only retained epoch2 last is authorized")
    files = [
        spec[k]
        for k in (
            "run_manifest",
            "resolved_config",
            "config",
            "journal",
            "resource_policy",
            "resource_owner",
        )
    ]
    files += [spec["medium"][k] for k in ("parent", "performance")]
    files += [spec["panel"][k] for k in ("population", "matched_views")]
    for binding in files:
        checked(binding, server=server)
    parent = read(spec["medium"]["parent"]["path"])
    from cap40_posthoc import reuse_medium_schedule

    reuse_medium_schedule(parent)
    if any(parent.get(k) != v for k, v in COUNTS.items()) or (
        population(parent["schedule"]) != spec["medium"]["population_sha256"]
    ):
        raise ValueError("fixed medium population/counts differ")
    validate_panel(read(spec["panel"]["population"]["path"]), parent)
    identity = read(spec["run_manifest"]["path"])
    if (
        identity["source"]["commit"] != spec["native_source"]["commit"]
        or identity["source"].get("dirty") is not False
    ):
        raise ValueError("native evaluation source must be original clean training source")
    if spec["name"] == "R3-RI":
        if (
            spec["native_source"]["commit"] != "f0cb20ebb2efab749970f2bf5d205ed2f358592c"
            or spec["checkpoint"]["sha256"]
            != "96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4"
            or spec["config"]["sha256"]
            != "65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7"
            or "response_anchor" in identity["data"]
        ):
            raise ValueError("R3-RI must be the exact original retained source/config/epoch2 CP")
    elif "response_anchor" not in identity["data"]:
        raise ValueError("R14-FRGA must bind its enabled training anchor descriptor")
    bound = bind_parent(parent, identity, spec["medium"]["identity_changes"])
    from gradpert.hashing import sha256_json

    if (
        sha256_json(read(spec["resolved_config"]["path"])) != identity["resolved_config_sha256"]
        or identity.get("config_sha256", spec["config"]["sha256"]) != spec["config"]["sha256"]
    ):
        raise ValueError("resolved/raw config identity differs")
    run = Path(spec["run_root"]).resolve()
    if any(
        Path(spec[key]["path"]).resolve() != run / name
        for key, name in [
            ("run_manifest", "run_manifest.json"),
            ("resolved_config", "resolved_config.json"),
            ("journal", "fit/epoch_state.json"),
        ]
    ):
        raise ValueError("run/journal metadata must belong to exact endpoint")
    output = Path(spec["output"]).resolve()
    protected = [
        Path(spec["data_root"]).resolve(),
        run,
        Path(spec["native_source"]["root"]).resolve(),
        Path(spec["analysis_source"]["root"]).resolve(),
    ]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output overlaps source/data/retained run")
    if server and any(not p.is_relative_to("/data/yilangliu") for p in [run, output, *protected]):
        raise ValueError("server-only endpoint materialization")
    if spec["allowed_axis"].get("count") != 4775:
        raise ValueError("fixed4775 allowed expression boundary required")
    return bound


def export_tables(analysis: Path, output: Path) -> None:
    """Aggregate-only tables; no fabricated training/test or saved-probe endpoint."""
    from cap40_posthoc import csv_write
    from cap40_posthoc_report import flatten

    output.mkdir(exist_ok=False)
    summary, deep = read(analysis / "summary.json"), read(analysis / "deep.json")
    for key in ("global_features", "global_heads", "metrics", "cls_decomposition", "centers"):
        csv_write(
            output / (key + ".csv"),
            [dict(name=k, **flatten(v)) for k, v in summary.get(key, {}).items()],
        )
    for key in ("pairs", "same_gene", "cross_view", "per_cell", "mask_eligibility"):
        csv_write(
            output / (key + ".csv"),
            [dict(condition_id=g["condition_id"], **r) for g in deep for r in g.get(key, [])],
        )
    csv_write(
        output / "frozen_losses.csv",
        [
            dict(
                condition_id=g["condition_id"],
                joint=g["joint"],
                **g["loss_terms"],
                **{
                    name: None
                    for name in g.get("diagnostic_objective_scope", {}).get("unavailable_terms", {})
                },
                **flatten(g.get("diagnostic_objective_scope", {})),
            )
            for g in deep
        ],
    )
    csv_write(output / "gradients.csv", [flatten(g) for g in read(analysis / "gradients.json")])
    output.joinpath("SCOPE_CN.md").write_text(
        "# 冻结中等端点诊断\n\n1335条件/5317行；深度128条件/506行；梯度32条件/125行。"
        "crossed128是另一轴，不能混用rank ceiling。此处joint为部分冻结SSL/预测探针，"
        "response_anchor缺完整TRAIN条件bank时记NA，不是训练joint或完整目标梯度。"
        "SSL1 ssl1_cls为condition e_p，不是图CLS；SSL1 node未使用时保留缺失，不造0熵。"
        "原GenePT/图/Cell/Response特征及H/JS/CE/KL保留原生键。"
        "不生成test、saved训练小探针、历史端点等价或科学成功声明。\n"
    )


def execute(spec: dict, mode: str, *, spec_path: Path, spec_sha256: str) -> dict:
    import cap40_posthoc as cap
    import run_representation_panel as panel_module
    from checkpoint_endpoint import select_checkpoint
    from postfit_spectral_audit import verify_source
    from postfit_stochastic_audit import verify_analysis_source
    from run_fgpo_medium import validate_analysis_spec

    from gradpert.data._io import atomic_json

    if sha(spec_path) != spec_sha256 or read(spec_path) != spec:
        raise ValueError("SPEC bytes changed")
    bound = validate_spec(spec)
    validate_analysis_spec(
        dict(
            parent_manifest=spec["medium"]["parent"]["path"],
            parent_manifest_sha256=spec["medium"]["parent"]["sha256"],
            performance_receipt=spec["medium"]["performance"]["path"],
            performance_receipt_sha256=spec["medium"]["performance"]["sha256"],
        )
    )
    native_source = verify_source(spec["native_source"])
    analysis_source = verify_analysis_source(spec["analysis_source"])
    if Path(__file__).resolve().parents[2] != Path(spec["analysis_source"]["root"]).resolve():
        raise ValueError("worker script must belong to exact analysis checkout")
    run, output = Path(spec["run_root"]), Path(spec["output"])
    checkpoint, endpoint = select_checkpoint(run, "last")
    if checkpoint["sha256"] != spec["checkpoint"]["sha256"] or endpoint["epoch"] != 2:
        raise ValueError("retained epoch2 checkpoint differs")
    config = Path(spec["config"]["path"])
    identity = read(spec["run_manifest"]["path"])
    common = dict(
        run_root=run,
        config=config,
        data_root=Path(spec["data_root"]),
        method_sha=spec["native_source"]["commit"],
        checkpoint_role="last",
    )
    # Native import-root/config/CP validators are never replaced.
    _, _, _, _, native_root = cap.runtime_args(SimpleNamespace(**common))
    if native_root.resolve() != Path(spec["native_source"]["root"]).resolve():
        raise ValueError("actual native package root differs")
    if mode == "seal":
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "SPEC.json", spec)
        atomic_json(output / "bound-parent.json", bound)
        policy_doc = read(spec["resource_policy"]["path"])
        policy = policy_doc.get("shared_gpu_policy", policy_doc)
        resource = dict(
            schema="response-anchor-one-endpoint-resource-v1",
            source_commit=spec["native_source"]["commit"],
            shared_gpu_policy=policy,
            rows=[
                dict(
                    name=spec["name"],
                    plan=dict(
                        run_id=identity["run_id"],
                        run_root=str(run),
                        config=str(config),
                        config_sha256=sha(config),
                    ),
                )
            ],
        )
        atomic_json(output / "RESOURCE_BINDING.json", resource)
        receipts = []
        records = [r for g in bound["schedule"] for r in g["rows"]]
        with diagnostic_adapter(
            cap,
            None,
            records,
            controls_only=False,
            read_receipts=receipts,
            expected_axis=spec["allowed_axis"],
        ):
            cap.seal_medium(
                SimpleNamespace(
                    **common,
                    output=output / "manifest",
                    parent_manifest=output / "bound-parent.json",
                    parent_manifest_sha256=sha(output / "bound-parent.json"),
                    reuse_medium_schedule=True,
                )
            )
        manifest = read(output / "manifest/manifest.json")
        if population(manifest["schedule"]) != spec["medium"]["population_sha256"]:
            raise ValueError("reseal changed fixed population")
        result = dict(
            status="sealed",
            spec_sha256=spec_sha256,
            manifest_sha256=sha(output / "manifest/manifest.json"),
            binding_sha256=sha(output / "RESOURCE_BINDING.json"),
            parent_sha256=sha(output / "bound-parent.json"),
            read_receipts=receipts,
            native_evaluation_source=native_source,
            training_source=identity["source"],
            analysis_adapter_source=analysis_source,
            checkpoint=endpoint,
            updates=0,
            scientific_complete=False,
        )
        if (
            validate_spec(spec) != bound
            or sha(spec_path) != spec_sha256
            or verify_source(spec["native_source"]) != native_source
            or verify_analysis_source(spec["analysis_source"]) != analysis_source
        ):
            raise ValueError("seal inputs/source changed")
        atomic_json(output / "SEALED.json", result)
        return result
    sealed = read(output / "SEALED.json")
    if (
        sealed["spec_sha256"] != spec_sha256
        or read(output / "SPEC.json") != spec
        or sha(output / "manifest/manifest.json") != sealed["manifest_sha256"]
        or sha(output / "RESOURCE_BINDING.json") != sealed["binding_sha256"]
        or sha(output / "bound-parent.json") != sealed["parent_sha256"]
    ):
        raise ValueError("sealed endpoint metadata changed")
    if mode == "validate":
        result = read(output / "COMPLETE.json")
        if result["spec_sha256"] != spec_sha256 or result["updates"] != 0:
            raise ValueError("terminal endpoint identity differs")
        medium = read(output / "analysis/summary.json")
        panel_result = read(output / "panel/COMPLETE.json")
        phase = read(output / "MEDIUM_COMPLETE.json")
        if (
            phase.get("counts") != COUNTS
            or phase.get("parity", {}).get("passed") is not True
            or phase["parity"].get("atol") != 1e-8
            or not 0 <= phase["parity"].get("max_abs", float("inf")) <= 1e-8
        ):
            raise ValueError("medium phase/parity failed")
        if any(medium.get(k) != v for k, v in COUNTS.items()):
            raise ValueError("terminal medium count axis differs")
        for row in (medium, panel_result):
            if (
                row.get("updates") != 0
                or row.get("state_before") != row.get("state_after")
                or row.get("checkpoint_sha256") != spec["checkpoint"]["sha256"]
                or row.get("method_sha") != spec["native_source"]["commit"]
                or row.get("config_sha256") != spec["config"]["sha256"]
            ):
                raise ValueError("native terminal state/source/config/CP differs")
        if panel_result.get("panel_rows") != 128 or panel_result.get("state_unchanged") is not True:
            raise ValueError("terminal crossed population/state differs")
        validate_scopes(
            read(output / "analysis/deep.json"), anchor="response_anchor" in identity["data"]
        )
        expected_names = {
            str(p.relative_to(output))
            for p in output.rglob("*")
            if p.is_file() and p != output / "COMPLETE.json"
        }
        if set(result["outputs"]) != expected_names:
            raise ValueError("terminal closure omits or adds artifacts")
        for name, entry in result["outputs"].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output.resolve()) or sha(path) != entry["sha256"]:
                raise ValueError("terminal output bytes changed")
        return result
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True" or (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != spec["gpu"]
    ):
        raise ValueError("allocator/determinism/physical GPU binding differs")
    from run_group import lock

    with (
        lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{spec['gpu']}.lock"),
        resource_snapshots(read(spec["resource_owner"]["path"])),
    ):
        receipts = []
        if mode == "medium":
            if (output / "MEDIUM_COMPLETE.json").exists():
                raise ValueError("completed medium cannot be overwritten")
            records = [r for g in bound["schedule"] for r in g["rows"]]
            args = dict(
                **common,
                manifest=output / "manifest/manifest.json",
                manifest_sha256=sealed["manifest_sha256"],
                resource_queue=output / "RESOURCE_BINDING.json",
                arm=spec["name"],
                deterministic=True,
                checkpoint_conditions=100,
                profile=False,
                resume_from=None,
                resume_from_sha256=None,
                peer_receipt=None,
                owner_receipt=None,
                seed=bound["seed"],
            )
            with diagnostic_adapter(
                cap,
                None,
                records,
                controls_only=False,
                read_receipts=receipts,
                expected_axis=spec["allowed_axis"],
            ):
                for label, backend in [
                    ("reference-probe", "reference"),
                    ("fast-probe", "fast"),
                    ("analysis", "fast"),
                ]:
                    if label == "analysis":
                        from profile_medium_observer import compare

                        compare(output / "reference-probe", output / "fast-probe")
                    cap.run(
                        SimpleNamespace(
                            **args,
                            output=output / label,
                            stats_backend=backend,
                            limit=0 if label == "analysis" else 3,
                            gate_indices="" if label == "analysis" else "0,1,15",
                        )
                    )
            from profile_medium_observer import compare

            parity = compare(output / "reference-probe", output / "fast-probe")
            summary = read(output / "analysis/summary.json")
            if any(summary.get(k) != v for k, v in COUNTS.items()):
                raise ValueError("full medium coverage differs")
            anchor = "response_anchor" in identity["data"]
            validate_scopes(read(output / "analysis/deep.json"), anchor=anchor)
            validate_scopes(read(output / "analysis/gradients.json"), anchor=anchor)
            export_tables(output / "analysis", output / "tables")
            result = dict(
                status="completed",
                updates=0,
                counts=COUNTS,
                parity=parity,
                reads=receipts,
                spec_sha256=spec_sha256,
                diagnostic_population_sha256=spec["medium"]["population_sha256"],
            )
            phase_receipt = output / "MEDIUM_COMPLETE.json"
        elif mode == "panel":
            frozen_panel = read(spec["panel"]["population"]["path"])
            records = [r for g in frozen_panel["schedule"] for r in g["rows"]]
            with diagnostic_adapter(
                cap,
                panel_module,
                records,
                controls_only=True,
                read_receipts=receipts,
                frozen_panel=frozen_panel,
                parent_binding=bound["training_protocol_binding"],
                expected_axis=spec["allowed_axis"],
                expected_views=read(spec["panel"]["matched_views"]["path"]),
            ):
                result = panel_module.execute(
                    SimpleNamespace(
                        **common,
                        output=output / "panel",
                        parent_manifest=output / "bound-parent.json",
                        parent_manifest_sha256=sealed["parent_sha256"],
                        resource_queue=output / "RESOURCE_BINDING.json",
                        resource_queue_sha256=sealed["binding_sha256"],
                        seed=frozen_panel["seed"],
                        conditions=32,
                        controls=4,
                    )
                )
            actual = read(output / "panel/matched_expression_views.json")
            expected = read(spec["panel"]["matched_views"]["path"])

            # Native signatures may differ when graph sampling consumes RNG; the
            # matched recipe actually used must remain exact per four-part context.
            if list(map(matched_signature, actual)) != list(map(matched_signature, expected)):
                raise ValueError(
                    "actual crossed input/view/order signatures differ from frozen recipe"
                )
            validate_scopes(
                result["diagnostic_objective_scopes"], anchor="response_anchor" in identity["data"]
            )
            result = dict(
                status="completed",
                updates=0,
                spec_sha256=spec_sha256,
                reads=receipts,
                panel_rows=128,
                matched_views_exact=True,
            )
            phase_receipt = output / "PANEL_COMPLETE.json"
        else:
            raise ValueError("unknown worker mode")
    for receipt in receipts:
        if (
            receipt["allowed_axis_sha256"] != spec["allowed_axis"]["sha256"]
            or receipt["allowed_count"] != 4775
        ):
            raise ValueError("actual native allowed axis differs")
    if (
        sha(spec_path) != spec_sha256
        or validate_spec(spec) != bound
        or (
            verify_source(spec["native_source"]) != native_source
            or verify_analysis_source(spec["analysis_source"]) != analysis_source
        )
    ):
        raise ValueError("endpoint inputs/source changed")
    atomic_json(phase_receipt, result)
    if (output / "MEDIUM_COMPLETE.json").exists() and (output / "PANEL_COMPLETE.json").exists():
        if list(output.rglob("*.pkl")):
            raise ValueError("zero-PKL postcondition failed")
        result = dict(
            status="completed",
            updates=0,
            scientific_complete=False,
            spec_sha256=spec_sha256,
            training_source=identity["source"],
            native_evaluation_source=native_source,
            analysis_adapter_source=analysis_source,
            endpoint=endpoint,
            counts=COUNTS,
            panel_rows=128,
            outputs={
                str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        )
        atomic_json(output / "COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "medium", "panel", "validate"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    try:
        execute(spec, args.mode, spec_path=args.spec, spec_sha256=args.spec_sha256)
    except Exception as error:
        output = Path(spec.get("output", "/invalid"))
        if (
            output.is_relative_to("/data/yilangliu")
            and (output / "SPEC.json").is_file()
            and not (output / "COMPLETE.json").exists()
            and (read(output / "SPEC.json") == spec)
        ):
            from gradpert.data._io import atomic_json

            atomic_json(
                output / f"FAILURE_{args.mode}_{time.time_ns()}.json",
                dict(status="failed", error=repr(error), updates=0, scientific_complete=False),
            )
        raise


if __name__ == "__main__":
    main()
