"""Dynamic last300 TRAIN population and strict read-only diagnostic primitives.

Native algorithms are reused with dynamic populations and explicit source,
resource-owner, state-transaction and finite-schedule closure. No legacy Jurkat
whole-run controller or its fixed population labels are invoked.
"""

from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.dataset_transfer_validation import (
    binding,
    checked,
    protected_roots,
    training_terminal,
    validate_output_path,
)
from gradpert.hashing import sha256_file, sha256_json

SCHEMA = "dataset-transfer300-medium-population-v1"


@contextmanager
def _scripts() -> Iterator[tuple[Any, Any, Any]]:
    """Reuse published algorithms, rejecting modules imported from another tree."""
    source = Path(__file__).resolve().parents[3] / "scripts/v2"
    old = sys.path[:]
    sys.path.insert(0, str(source))
    try:
        modules = tuple(
            importlib.import_module(name)
            for name in ("cap40_posthoc", "representation_panel", "grouped_view_diagnostics")
        )
        if any(Path(m.__file__).resolve().parent != source for m in modules):
            raise ValueError("diagnostic statistics imports differ from this adapter source")
        for name in (
            "mfu_probe",
            "shared_gpu",
            "run_group",
            "run_dataset_transfer_300",
            "response_stage_diagnostics",
            "prototype_pair_diagnostics",
            "run_representation_panel",
            "update_parity",
        ):
            dependency = importlib.import_module(name)
            if Path(dependency.__file__).resolve().parent != source:
                raise ValueError(
                    "diagnostic dependency import differs from adapter source: " + name
                )
        yield modules
    finally:
        sys.path[:] = old


def records_from_data(data: Any) -> list[dict[str, Any]]:
    """Existing sampler identities only; no expression values or fake cell lines."""
    records = []
    observed = set()
    train_ids, control_ids = set(data.train_row_indices), set(data.control_row_indices)
    obs = data._adata.obs

    def field(index: int, name: str) -> str | None:
        if name not in obs.columns:
            return None
        value = obs.iloc[index][name]
        return None if value is None or str(value) in ("", "nan", "None", "<NA>") else str(value)

    for spec in data.training_batch_identity_specs(epoch=0, batch_size=64, max_unique_conditions=0):
        for truth, control, condition in zip(
            spec.perturbed_row_ids, spec.control_row_ids, spec.condition_ids, strict=True
        ):
            ti, ci = data._row_index[truth], data._row_index[control]
            if ti in observed or ti not in train_ids or ci not in control_ids:
                raise ValueError("population row is duplicated or outside TRAIN/control identities")
            observed.add(ti)
            if data.condition_ids[ti] != condition or data.context_ids[ti] != data.context_ids[ci]:
                raise ValueError("population condition or compatible control context differs")
            records.append(
                dict(
                    truth_row_id=truth,
                    control_row_id=control,
                    condition_id=condition,
                    context_id=str(data.context_ids[ti]),
                    batch=str(data.batch_ids[ti]),
                    cell_line=field(ti, "cell_line"),
                    cell_type=field(ti, "cell_type"),
                    control_batch=str(data.batch_ids[ci]),
                    control_cell_line=field(ci, "cell_line"),
                    control_cell_type=field(ci, "cell_type"),
                )
            )
    if observed != train_ids:
        raise ValueError("metadata population does not cover all canonical TRAIN rows")
    return records


def population(
    records: list[dict[str, Any]],
    *,
    expression_genes: int,
    allowed: tuple[int, ...],
    query_count: int,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Seal dynamic condition budgets with the existing outcome-blind algorithms."""
    if (
        expression_genes not in (5000, 5045)
        or not allowed
        or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in allowed
        )
        or tuple(sorted(set(allowed))) != allowed
        or min(allowed) < 0
        or max(allowed) >= expression_genes
        or type(query_count) is not int
        or not 2 <= query_count <= len(allowed)
        or type(seed) is not int
    ):
        raise ValueError("invalid frozen expression axis/query budget")
    groups: dict[str, list] = defaultdict(list)
    seen = set()
    for row in records:
        if (
            not isinstance(row.get("context_id"), str)
            or not row["context_id"]
            or not isinstance(row.get("condition_id"), str)
            or not row["condition_id"]
            or row.get("truth_row_id") in seen
            or not row.get("truth_row_id")
            or not row.get("control_row_id")
        ):
            raise ValueError("population requires unique truth IDs and explicit condition/context")
        seen.add(row["truth_row_id"])
        groups[row["condition_id"]].append(deepcopy(row))
    if not groups:
        raise ValueError("empty canonical TRAIN population")
    with _scripts() as (cap, panel, _):
        queries = cap.query_panels(np.asarray(allowed), query_count, seed)
        parent = dict(
            seed=seed,
            schedule=[
                dict(
                    index=i,
                    condition_id=condition,
                    rows=sorted(rows, key=lambda row: row["truth_row_id"]),
                    queries=queries[i % len(queries)],
                    view_seed=cap.seeded_int(condition, seed) % 2**32,
                )
                for i, (condition, rows) in enumerate(sorted(groups.items()))
            ],
        )
        schedule = cap.medium_schedule(
            parent,
            cells=4,
            deep_conditions=min(128, len(groups)),
            gradient_conditions=min(32, len(groups)),
        )
        medium = dict(
            schema=SCHEMA, analysis_scope="dataset_transfer300_medium", seed=seed, schedule=schedule
        )
        # The legacy sampler's scope discriminator is passed only in memory;
        # no artifact claims a Jurkat/cap40 population.
        panel_parent = {**medium, "analysis_scope": "medium_cap40_stratified", "schedule": []}
        excluded = []
        for group in schedule:
            rows = []
            for row in group["rows"]:
                fields = ("cell_line", "cell_type", "batch")
                complete = all(row.get(k) is not None for k in fields)
                compatible = all(row.get(k) == row.get("control_" + k) for k in fields)
                if complete and compatible:
                    rows.append(row)
                else:
                    excluded.append(
                        dict(
                            truth_row_id=row["truth_row_id"],
                            reason="missing_or_incompatible_context_metadata",
                        )
                    )
            if rows:
                panel_parent["schedule"].append({**group, "rows": rows})
        try:
            if len({g["condition_id"] for g in panel_parent["schedule"]}) < 2:
                raise ValueError("fewer than two eligible conditions")
            crossed = panel.panel_schedule(panel_parent, seed=seed, conditions=32, controls=4)
            crossed.update(status="sealed_metadata_only", excluded_rows=excluded)
        except ValueError as error:
            crossed = dict(
                status="not_applicable",
                reason=str(error),
                excluded_rows=excluded,
                controls_per_context=4,
                conditions=0,
                rows=0,
                schedule=[],
            )
        counts = dict(
            conditions=len(schedule),
            rows=sum(len(g["rows"]) for g in schedule),
            deep_conditions=sum(bool(g["deep_rows"]) for g in schedule),
            deep_rows=sum(len(g["deep_rows"]) for g in schedule),
            gradient_conditions=sum(bool(g["gradient_rows"]) for g in schedule),
            gradient_rows=sum(len(g["gradient_rows"]) for g in schedule),
        )
        medium.update(
            status="sealed_metadata_only",
            canonical_train_rows=len(records),
            selected_independent_controls=len(
                {r["control_row_id"] for g in schedule for r in g["rows"]}
            ),
            canonical_train_identity_sha256=sha256_json(records),
            **counts,
            allowed_expression_ids=list(allowed),
            query_panels=queries,
            expression_genes=expression_genes,
            diagnostic_population_sha256=cap.population_identity(schedule),
            crossed=crossed,
            selection_use="none",
            population_scope="canonical TRAIN, not proof every row was seen in300 updates",
            collection_status="not_executed_no_statistics_or_parity_proof",
        )
        return medium


def validate_data_metadata(data: Any, identity: dict[str, Any]) -> tuple[int, ...]:
    from gradpert.training.expression_policy import expression_policy

    expected = identity["data"]
    if (
        data.manifest.canonical_adata_sha256 != expected["canonical_sha256"]
        or data.split.split_content_sha256 != expected["split_sha256"]
        or sha256_json(list(data.expression_gene_ids)) != expected["expression_gene_order_sha256"]
        or sha256_json(list(data.graph_gene_ids)) != expected["graph_gene_order_sha256"]
    ):
        raise ValueError("medium canonical/split/expression/graph identity differs")
    allowed, policy = expression_policy(
        tuple(data.expression_gene_ids),
        tuple(data.split.test_conditions),
        data.split.control_condition_id,
        enabled=True,
    )
    if (
        policy != expected["training_expression_policy"]
        or policy["allowed_expression_gene_ids_sha256"]
        != expected["effective_training_expression_ids_sha256"]
    ):
        raise ValueError("medium frozen exclusion policy differs")
    return tuple(int(i) for i in allowed)


def seal_metadata(
    *,
    training_plan: dict[str, str],
    data: Any,
    output: Path,
    query_count: int,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Metadata-only stage; caller owns the separately opened native data instance."""
    training = read_json(checked(training_plan))
    terminal = training_terminal(training)
    identity = terminal["identity"]
    destination = validate_output_path(output, protected_roots(training, identity))
    if destination.exists():
        raise FileExistsError("medium metadata output must be fresh")
    config = __import__(
        "gradpert.config", fromlist=["load_experiment_config"]
    ).load_experiment_config(training["config"])
    if (
        data.manifest.dataset_id != config.dataset_id
        or data.manifest.protocol_id != config.data.protocol_id
        or query_count != int(config.model.parameters["query_count"].value)
    ):
        raise ValueError("medium dataset/query config differs from training")
    allowed = validate_data_metadata(data, identity)
    result = population(
        records_from_data(data),
        expression_genes=len(data.expression_gene_ids),
        allowed=allowed,
        query_count=query_count,
        seed=seed,
    )
    result.update(
        training_plan=training_plan,
        training_identity=identity,
        last=terminal["last"],
        terminal_inputs=[
            binding(Path(training["run_root"]) / name)
            for name in (
                "run_manifest.json",
                "fit/epoch_state.json",
                "fit/history.json",
                "TRAIN_COMPLETE.json",
            )
        ],
    )
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "POPULATION_SEALED.json", result)
    return result


@contextmanager
def strict_train_reads(
    data: Any,
    records: list[dict[str, Any]],
    allowed: tuple[int, ...],
    *,
    controls_only: bool = False,
) -> Iterator[dict[str, Any]]:
    """Fresh diagnostic instance only; no sparse value read before axis filtering."""
    if (
        getattr(data, "_training_expression_read_started", False)
        or getattr(data, "_control_expression_cache", None) is not None
        or getattr(data, "_training_expression_columns", None) is not None
        or getattr(data.pipeline_stats, "control_cache_requested", False)
        or getattr(data.pipeline_stats, "yielded_batches", 0)
    ):
        raise ValueError("strict medium reader must precede every read/cache/iteration")
    from gradpert.training.expression_policy import expression_policy

    expected, _ = expression_policy(
        tuple(data.expression_gene_ids),
        tuple(data.split.test_conditions),
        data.split.control_condition_id,
        enabled=True,
    )
    if tuple(int(i) for i in expected) != allowed:
        raise ValueError("medium read axis differs from frozen dataset expression exclusions")
    whitelist = set()
    train_ids, control_ids = set(data.train_row_indices), set(data.control_row_indices)
    for row in records:
        ti, ci = data._row_index[row["truth_row_id"]], data._row_index[row["control_row_id"]]
        if (
            ti not in train_ids
            or ci not in control_ids
            or str(data.condition_ids[ti]) != row["condition_id"]
            or str(data.context_ids[ti]) != row["context_id"]
            or data.context_ids[ti] != data.context_ids[ci]
        ):
            raise ValueError("medium requested row outside exact TRAIN/context whitelist")
        whitelist.add(ci)
        if not controls_only:
            whitelist.add(ti)
    if not whitelist:
        raise ValueError("strict medium read whitelist is empty")
    from gradpert.training.expression_io import read_expression_selection

    receipt = dict(
        allowed_expression_ids_sha256=sha256_json(list(allowed)),
        whitelist_sha256=sha256_json(sorted(whitelist)),
        calls=0,
        rows=0,
        controls_only=controls_only,
    )

    def read(indices):
        values = np.asarray(indices)
        if (
            values.ndim != 1
            or not len(values)
            or values.dtype.kind not in "iu"
            or not set(map(int, values)) <= whitelist
        ):
            raise ValueError("medium attempted a forbidden numerical row read")
        data._training_expression_read_started = True
        unique, inverse = np.unique(values, return_inverse=True)
        selected = read_expression_selection(data._adata.file["X"], unique, allowed)
        result = np.zeros((len(values), data.manifest.n_expression_genes), dtype=np.float32)
        result[:, list(allowed)] = selected[inverse]
        receipt["calls"] += 1
        receipt["rows"] += len(values)
        return result

    # This dedicated data instance is not reusable as an unrestricted val reader.
    data.configure_training_expression_columns(allowed)
    with (
        patch.object(data, "_read_expression_indices", read),
        patch.object(data, "_dataset_transfer_strict_guard", True, create=True),
    ):
        yield receipt


def assemble_diagnostic(
    runtime: Any, group: dict[str, Any], *, deep: bool, controls_only: bool = False
) -> Any:
    """Reuse native view/mask construction under an already installed strict guard."""
    if not getattr(runtime.data, "_dataset_transfer_strict_guard", False):
        raise ValueError("diagnostic assembly requires the strict reader boundary")
    records = group["rows"] if controls_only or not deep else group["deep_rows"]
    with _scripts() as (cap, panel, _):
        if controls_only:
            return panel.assemble_control_only(
                runtime, records, group["queries"], group["view_seed"], deep=deep
            )
        return cap.assemble(runtime, records, group["queries"], group["view_seed"], deep=deep)


def collect_batch_statistics(
    objective: Any,
    batch: Any,
    records: list[dict[str, Any]],
    *,
    seed: int,
    state_probe: Callable[[], Any],
    bf16: bool = True,
) -> dict[str, Any]:
    """Read-only single-batch primitive, not full medium/crossed completion.

    Caller supplies the native full-state/RNG/optimizer signature and owns the
    terminal/source/resource contract. A future controller must aggregate sealed
    batches and prove full population closure before emitting COMPLETE.
    """
    import torch

    from gradpert.training.v2.diagnostic_objective import diagnostic_objective

    if objective.pending:
        raise ValueError("pending teacher statistics at diagnostic entry")
    if len(records) != len(batch.control):
        raise ValueError("batch and sealed records differ")
    before = state_probe()
    modes = [(module, module.training) for module in objective.modules()]
    randomization = [
        (module, module.randomize_relay_order)
        for module in objective.modules()
        if hasattr(module, "randomize_relay_order")
    ]
    numpy_state = np.random.get_state()
    import random

    python_state = random.getstate()
    device = batch.control.device
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with _scripts() as (cap, _, grouped):
        collector = cap.Collector(batch, records, seed, stats_backend="reference")
        installed = False
        try:
            with torch.random.fork_rng(devices=devices):
                objective.eval()
                torch.manual_seed(seed)
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16),
                ):
                    _, baseline, scope = diagnostic_objective(objective, batch)
                baseline = {k: v.detach().clone() for k, v in baseline.items()}
                objective.pending.clear()
                torch.manual_seed(seed)
                installed = True
                _install_collector(collector, objective)
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16),
                ):
                    _, observed, observed_scope = diagnostic_objective(objective, batch)
                    result = collector.result()
                    bank = grouped.GroupedViewBank()
                    collector.add_to_view_bank(bank)
                    view_result = bank.result()
                if (
                    scope != observed_scope
                    or baseline.keys() != observed.keys()
                    or any(not torch.equal(baseline[k], observed[k]) for k in baseline)
                ):
                    raise ValueError("single-batch observer off/on loss-term parity failed")
                result.update(
                    status="batch_statistics_only",
                    grouped_views=view_result,
                    diagnostic_scope=scope,
                    observer_loss_term_parity=True,
                    parity_scope="loss terms only; prediction/token array parity pending",
                    full_run_complete=False,
                    gradient_statistics=None,
                    gradient_status="not_collected_by_this_primitive",
                )
        finally:
            if installed:
                _close_collector(collector)
            objective.pending.clear()
            for module, mode in modes:
                module.training = mode
            for module, value in randomization:
                module.randomize_relay_order = value
            np.random.set_state(numpy_state)
            random.setstate(python_state)
    if state_probe() != before:
        raise ValueError("native full state/RNG changed during batch collection")
    return result


EXECUTION_SCHEMA = "dataset-transfer300-medium-execution-v1"


class _DiagnosticCursor:
    """No training iterator is opened; transaction API still seals an empty cursor."""

    def state_dict(self):
        return {}

    def load_state_dict(self, value):
        if value != {}:
            raise ValueError("unexpected diagnostic cursor state")

    def close(self):
        pass


@contextmanager
def state_transaction(runtime: Any) -> Iterator[dict[str, Any]]:
    """Reuse native complete state snapshots and unconditional failure restoration."""
    with _scripts():
        transaction = importlib.import_module("mfu_probe")
        cursor = _DiagnosticCursor()
        saved = transaction.capture_state(runtime, cursor)
        before = transaction.digest(saved)
        proof = dict(before=before, after=None, restored=False)
        try:
            yield proof
        finally:
            transaction.restore_state(runtime, cursor, saved)
            after = transaction.digest(transaction.capture_state(runtime, cursor))
            proof.update(after=after, restored=after == before)
            if after != before:
                raise ValueError("full diagnostic state restoration failed")


def _normalized_feature(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    ceiling = result.get("rank_ceiling")
    rank = result.get("effective_rank")
    result["normalized_effective_rank"] = rank / ceiling if rank is not None and ceiling else None
    return result


def _guarded_objective(runtime: Any):
    """Reject all optimizer/EMA/center update entrypoints while diagnostics run."""
    from contextlib import ExitStack

    stack = ExitStack()

    def forbidden(*args, **kwargs):
        raise ValueError("diagnostic attempted optimizer/EMA/center mutation")

    stack.enter_context(patch.object(runtime.optimizer, "step", forbidden))
    stack.enter_context(patch.object(runtime.objective, "commit_statistics", forbidden))
    return stack


def _install_collector(collector: Any, objective: Any) -> None:
    collector._adapter_overrides = [
        (model, "encode_response" in model.__dict__, model.__dict__.get("encode_response"))
        for model in (objective.student, objective.teacher)
    ]
    from gradpert.training.v2 import objective as objective_module

    collector._adapter_ce = (objective_module, objective_module.cross_entropy_terms)
    collector._adapter_hooks = [
        (hooks, hooks.copy())
        for module in objective.modules()
        for name, hooks in vars(module).items()
        if "hook" in name and isinstance(hooks, dict)
    ]
    collector.install(objective)


def _close_collector(collector: Any) -> None:
    try:
        if hasattr(collector, "module") and hasattr(collector, "original_ce"):
            collector.close()
    finally:
        for hooks, original in getattr(collector, "_adapter_hooks", []):
            hooks.clear()
            hooks.update(original)
        if hasattr(collector, "_adapter_ce"):
            module, original = collector._adapter_ce
            module.cross_entropy_terms = original
        for model, existed, value in getattr(collector, "_adapter_overrides", []):
            if existed:
                model.encode_response = value
            else:
                model.__dict__.pop("encode_response", None)


def _used_array_parity(
    runtime: Any, batch: Any, records: list[dict], cached: Any, seed: int, bf16: bool
) -> dict[str, Any]:
    import torch

    with _scripts() as (cap, _, _):
        digest = importlib.import_module("update_parity").tree_digest
        outputs = []
        for observed in (False, True):
            with state_transaction(runtime):
                runtime.objective.eval()
                torch.manual_seed(seed)
                collector = cap.Collector(batch, records, seed, stats_backend="reference")
                try:
                    if observed:
                        _install_collector(collector, runtime.objective)
                    mapper = None
                    if observed:
                        mapper = runtime.objective.student.graph.register_forward_pre_hook(
                            lambda module, inputs, collector=collector: (
                                collector.graph_view_names.setdefault(id(inputs[0]), "main")
                                and None
                            )
                        )
                    try:
                        with (
                            torch.no_grad(),
                            torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16),
                        ):
                            fresh_graph = cap.static_graph(runtime)
                            response = cap.main_response(runtime, batch, cached)
                    finally:
                        if mapper is not None:
                            mapper.remove()
                    graph_tokens = (
                        fresh_graph["prefix"] if isinstance(fresh_graph, dict) else fresh_graph
                    )
                    reference_graph = cached["prefix"] if isinstance(cached, dict) else cached
                    if not torch.equal(graph_tokens, reference_graph):
                        raise ValueError(
                            "used cached graph differs from observer graph recomputation"
                        )
                    response = {**response, "used_graph_tokens": graph_tokens}
                    required = {"prediction", "response_cls", "response_tokens"}
                    if not required.issubset(response):
                        raise ValueError("used-path parity lacks prediction/CLS/token arrays")
                    outputs.append(
                        {key: value.detach().cpu().clone() for key, value in response.items()}
                    )
                finally:
                    if observed:
                        _close_collector(collector)
        if outputs[0].keys() != outputs[1].keys() or any(
            not torch.equal(outputs[0][key], outputs[1][key]) for key in outputs[0]
        ):
            raise ValueError("used prediction/token array observer parity failed")
        return dict(
            used_path_passed=True,
            observer_off_sha256=digest(outputs[0]),
            observer_on_sha256=digest(outputs[1]),
            max_abs=0.0,
            atol=0,
            rtol=0,
            shapes={k: list(v.shape) for k, v in outputs[0].items()},
            cell_chunk_used=False,
            cell_chunk_status="not_attempted",
        )


def _statistical_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Existing banks key by context_id; include batch without changing native inputs."""
    return [
        {
            **row,
            "source_context_id": row["context_id"],
            "context_id": "batch_context:"
            + sha256_json([row.get(k) for k in ("cell_line", "cell_type", "context_id", "batch")]),
        }
        for row in records
    ]


def _collect_runtime(
    runtime: Any,
    manifest: dict[str, Any],
    *,
    phase: str,
    resource_check: Callable[[], None],
    bf16: bool = True,
    assemble: Callable = assemble_diagnostic,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Full finite schedule aggregation, CPU-testable with a native synthetic runtime."""
    import torch

    from gradpert.training.v2.diagnostic_objective import diagnostic_objective
    from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
    from gradpert.training.v2.process_diagnostics import evaluate_graph_gradient_diagnostics
    from gradpert.training.v2.runtime import _move_batch_tree

    if phase not in ("medium", "crossed"):
        raise ValueError("unknown dynamic diagnostic phase")
    panel = phase == "crossed"
    selected = manifest["crossed"]["schedule"] if panel else manifest["schedule"]
    if panel and manifest["crossed"]["status"] == "not_applicable":
        return dict(
            status="not_applicable",
            reason=manifest["crossed"]["reason"],
            conditions=0,
            rows=0,
            updates=0,
        )
    if not selected:
        raise ValueError("diagnostic schedule is empty")
    records = [row for group in selected for row in group["rows"]]
    allowed = tuple(manifest["allowed_expression_ids"])
    with _scripts() as (cap, panel_api, grouped):
        panel_collect = importlib.import_module("run_representation_panel")
        prototype = importlib.import_module("prototype_pair_diagnostics")
        stages_api = importlib.import_module("response_stage_diagnostics")
        digest = importlib.import_module("update_parity").tree_digest
        views, main_views = grouped.GroupedViewBank(), grouped.GroupedViewBank()
        prototypes = prototype.PrototypePairBank()
        matched = panel_api.MatchedCellViews()
        matched_dummy = panel_api.MatchedCellViews()
        recipes = []
        all_features, all_heads = {}, {}
        per_condition, gradients, losses, view_receipts = [], [], [], []
        loss_pair_rows = []
        stage_rows = {
            name: [] for name in ("cell_statistics", "group_statistics", "pair_statistics")
        }
        counts = dict(
            conditions=0,
            rows=0,
            deep_conditions=0,
            deep_rows=0,
            gradient_conditions=0,
            gradient_rows=0,
        )
        expected = (
            dict(
                conditions=len(selected),
                rows=sum(len(g["rows"]) for g in selected),
                deep_conditions=len(selected),
                deep_rows=sum(len(g["rows"]) for g in selected),
                gradient_conditions=0,
                gradient_rows=0,
            )
            if panel
            else {key: manifest[key] for key in counts}
        )
        with strict_train_reads(
            runtime.data, records, allowed, controls_only=panel
        ) as read_receipt:
            with state_transaction(runtime) as closure, _guarded_objective(runtime):
                runtime.objective.eval()
                torch.manual_seed(manifest["seed"])
                with (
                    torch.no_grad(),
                    torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16),
                ):
                    cached = cap.static_graph(runtime)
                    if not panel:
                        sources = {
                            "raw_genept": runtime.objective.student.graph.embedding.weight,
                            "projected_prior": runtime.objective.student.graph.identities(),
                            "graph_full": cached
                            if not isinstance(cached, dict)
                            else cached["prefix"],
                        }
                        for label, values in sources.items():
                            moments = cap.Moments(values.shape[-1])
                            moments.add(cap.npval(values))
                            all_features["student." + label + ".all_graph_genes"] = moments
                parity = None
                for group in selected:
                    resource_check()
                    batch = _move_batch_tree(
                        assemble(runtime, group, deep=False, controls_only=panel), runtime.device
                    )
                    if batch.graph.ids[batch.query_positions].tolist() != group["queries"]:
                        raise ValueError("actual query axis differs from sealed population")
                    dummy = cap.assemble(
                        runtime,
                        group["rows"],
                        group["queries"],
                        group["view_seed"],
                        deep=False,
                        dummy=True,
                    )
                    if digest(cap.recipe(dummy)) != digest(cap.recipe(batch)):
                        raise ValueError(
                            "actual main recipe differs from metadata-only native recipe"
                        )
                    recipe_entry = dict(
                        condition_id=group["condition_id"], main_sha256=digest(cap.recipe(dummy))
                    )
                    recipes.append(recipe_entry)
                    if parity is None:
                        with (
                            state_transaction(runtime),
                            torch.no_grad(),
                            torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16),
                        ):
                            native_cache_parity = cap.verify_cache(runtime, batch, cached)
                        parity = _used_array_parity(
                            runtime, batch, group["rows"], cached, group["view_seed"], bf16
                        )
                        parity["native_cache_parity"] = native_cache_parity
                    with (
                        torch.no_grad(),
                        torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16),
                    ):
                        with stages_api.ResponseStageObserver(
                            runtime.objective.student.response,
                            gene_ids=batch.graph.ids[batch.query_positions],
                            records=_statistical_records(group["rows"]),
                            seed=manifest["seed"],
                        ) as observer:
                            response = cap.main_response(runtime, batch, cached)
                        if panel:
                            for key, values in observer.result().items():
                                if key in stage_rows:
                                    stage_rows[key].extend(values)
                            for name in ("control_cls", "response_cls", "delta"):
                                main_views.add(
                                    name,
                                    "student",
                                    "main",
                                    _statistical_records(group["rows"]),
                                    cap.npval(response[name]),
                                )
                        else:
                            per_condition.extend(cap.output_rows(response, batch, group["rows"]))
                            for name in ("control_cls", "response_cls", "response_tokens"):
                                values = cap.npval(response[name])
                                if values.ndim == 3:
                                    values = values.reshape(-1, values.shape[-1])
                                moments = all_features.setdefault(
                                    "student.population_main." + name, cap.Moments(values.shape[-1])
                                )
                                moments.add(values)
                    counts["conditions"] += 1
                    counts["rows"] += len(group["rows"])
                    deep_records = group["rows"] if panel else group["deep_rows"]
                    if deep_records:
                        deep = _move_batch_tree(
                            assemble(runtime, group, deep=True, controls_only=panel), runtime.device
                        )
                        dummy_deep = cap.assemble(
                            runtime,
                            deep_records,
                            group["queries"],
                            group["view_seed"],
                            deep=True,
                            dummy=True,
                        )
                        if panel:
                            deep, receipt = matched.bind(deep, group["rows"])
                            dummy_deep, _ = matched_dummy.bind(dummy_deep, group["rows"])
                            view_receipts.append({"condition_id": group["condition_id"], **receipt})
                        if digest(cap.recipe(dummy_deep)) != digest(cap.recipe(deep)):
                            raise ValueError(
                                "deep recipe differs from frozen metadata/matched view recipe"
                            )
                        recipe_entry["deep_sha256"] = digest(cap.recipe(dummy_deep))
                        collector = cap.Collector(
                            deep,
                            _statistical_records(deep_records),
                            manifest["seed"],
                            stats_backend="reference",
                        )
                        try:
                            _install_collector(collector, runtime.objective)
                            with (
                                torch.no_grad(),
                                torch.autocast(
                                    runtime.device.type, dtype=torch.bfloat16, enabled=bf16
                                ),
                            ):
                                _, terms, scope = diagnostic_objective(runtime.objective, deep)
                                collector.complete_feature_views()
                                collector.add_to_view_bank(views)
                                loss_pair_rows.extend(
                                    dict(
                                        condition_id=group["condition_id"],
                                        cohort="matched_crossed" if panel else "deep",
                                        cells=len(deep_records),
                                        **deepcopy(pair),
                                    )
                                    for pair in collector.pairs
                                )
                                if panel:
                                    panel_collect.add_prototype_rows(collector, prototypes)
                                else:
                                    for key, moments in collector.features.items():
                                        if moments is not None:
                                            all_features.setdefault(
                                                key, cap.Moments(len(moments.mean))
                                            ).merge(
                                                moments.n,
                                                moments.mean,
                                                moments.m2,
                                                moments.norm_sum,
                                            )
                                    for key, dist in collector.distribution_objects.items():
                                        total = all_heads.setdefault(
                                            key, cap.Distribution(len(dist.sum))
                                        )
                                        total.n += dist.n
                                        total.sum += dist.sum
                                        total.hits += dist.hits
                                        total.entropy_sum += dist.entropy_sum
                                        total.top1_sum += dist.top1_sum
                                        total.logit_std_sum += dist.logit_std_sum
                                    losses.append(
                                        dict(
                                            condition_id=group["condition_id"],
                                            scope=scope,
                                            terms={k: float(v) for k, v in terms.items()},
                                        )
                                    )
                        finally:
                            _close_collector(collector)
                            runtime.objective.pending.clear()
                        counts["deep_conditions"] += 1
                        counts["deep_rows"] += len(deep_records)
                    if not panel and group["gradient_rows"]:
                        resource_check()
                        gradient_group = {**group, "deep_rows": group["gradient_rows"]}
                        grad_batch = _move_batch_tree(
                            assemble(runtime, gradient_group, deep=True, controls_only=False),
                            runtime.device,
                        )
                        # Native APIs use autograd.grad and their own read-only scopes.
                        with state_transaction(runtime) as gradient_closure:
                            graph = evaluate_graph_gradient_diagnostics(
                                runtime.objective, grad_batch, bf16=bf16
                            )
                            basal = evaluate_loss_diagnostics(
                                runtime.objective, grad_batch, bf16=bf16
                            )
                        gradients.append(
                            dict(
                                condition_id=group["condition_id"],
                                cells=len(group["gradient_rows"]),
                                graph=graph,
                                basal=basal,
                                state=gradient_closure,
                            )
                        )
                        counts["gradient_conditions"] += 1
                        counts["gradient_rows"] += len(group["gradient_rows"])
                    runtime.objective.pending.clear()
                    if on_progress is not None:
                        on_progress(
                            dict(
                                phase=phase,
                                completed=counts.copy(),
                                expected=expected,
                                latest_condition=group["condition_id"],
                                model_updates=0,
                            )
                        )
                if counts != expected or parity is None or not parity["used_path_passed"]:
                    raise ValueError("finite diagnostic schedule did not close all sealed cohorts")
                result = dict(
                    status="completed",
                    phase=phase,
                    counts=counts,
                    population_sha256=manifest["diagnostic_population_sha256"],
                    parity=parity,
                    features={k: _normalized_feature(v.result()) for k, v in all_features.items()},
                    heads={k: v.result() for k, v in all_heads.items()},
                    grouped_views=views.result(),
                    main_crossed=main_views.result() if panel else None,
                    prototype_pairs=panel_collect.prototype_result(prototypes, runtime.objective)
                    if panel
                    else None,
                    matched_views=view_receipts,
                    response_stages=stage_rows if panel else None,
                    condition_outputs=per_condition,
                    losses=losses,
                    loss_pairs=loss_pair_rows,
                    gradients=gradients,
                    read_receipt=read_receipt,
                    recipe_receipts=recipes,
                    feature_scopes={
                        "student.population_main.*": "all main schedule inputs",
                        "student.main.*": "deep cohort only",
                        "student.raw_genept.*": "frozen prior identity, not post-projection",
                    },
                    updates=0,
                    state=closure,
                    scope="fixed TRAIN diagnostic population, no test scoring",
                    arithmetic=dict(
                        outer_bfloat16=bf16,
                        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                    ),
                    statistical_contexts=[
                        {
                            "context_id": row["context_id"],
                            "original_context_id": row["source_context_id"],
                            "batch": row.get("batch"),
                            "cell_line": row.get("cell_line"),
                            "cell_type": row.get("cell_type"),
                        }
                        for row in _statistical_records(records)
                    ],
                    limits=[
                        "Rows/views/controls are correlated; not independent biological replicates",
                        "Medium marginals and crossed inputs have distinct populations",
                    ],
                    missing={
                        "per_cell_gene_rank": "not_collected; pooled tokens are correlated",
                        "prototype_LL": "not_applicable; Teacher has global views only",
                    },
                    active_heads={
                        head: bool(coefficient and weight)
                        for head, coefficient, weight in (
                            (
                                "ssl1_cls",
                                runtime.objective.lambda1,
                                runtime.objective.weights[0][0],
                            ),
                            (
                                "ssl1_node",
                                runtime.objective.lambda1,
                                runtime.objective.weights[0][1],
                            ),
                            (
                                "ssl2_cls",
                                runtime.objective.lambda2,
                                runtime.objective.weights[1][0],
                            ),
                            (
                                "ssl2_node",
                                runtime.objective.lambda2,
                                runtime.objective.weights[1][1],
                            ),
                        )
                    },
                )
            if not closure["restored"]:
                raise ValueError("diagnostic runtime state closure missing")
        result["result_sha256"] = digest(result)
        return result


def build_execution_spec(
    *,
    training_plan: dict[str, str],
    population_binding: dict[str, str],
    runtime: dict[str, str],
    output: Path,
    family_root: Path,
    gpu: str,
    compatibility: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind published analysis code and the original native training contract."""
    from gradpert.config import load_experiment_config
    from gradpert.execution.dataset_transfer_validation import _compatibility
    from gradpert.execution.identity import inspect_source_identity

    if gpu not in ("0", "1"):
        raise ValueError("medium requires one existing physical GPU selector0/1")
    training = read_json(checked(training_plan))
    terminal = training_terminal(training)
    frozen = read_json(checked(population_binding))
    if (
        frozen.get("schema") != SCHEMA
        or frozen.get("training_plan") != training_plan
        or frozen.get("training_identity") != terminal["identity"]
        or frozen.get("last") != terminal["last"]
        or frozen.get("collection_status") != "not_executed_no_statistics_or_parity_proof"
    ):
        raise ValueError("sealed medium population differs from original last300")
    for item in frozen["terminal_inputs"]:
        checked(item)
    runtime_config = read_json(checked(runtime))
    if Path(runtime_config["data_root"]).resolve() != Path(training["data_root"]).resolve():
        raise ValueError("medium runtime data differs from training")
    config = load_experiment_config(training["config"])
    source = inspect_source_identity(
        Path(__file__).resolve().parents[3],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=runtime_config["publication_receipt"],
        expected_publication_receipt_sha256=runtime_config["publication_sha256"],
    ).payload()
    _compatibility(
        dict(training_identity=terminal["identity"], evaluation_source=source), compatibility
    )
    # Policy is inherited byte-for-byte from the sealed TRAIN runtime; never a new allowance.
    train_runtime = Path(training["runtime"])
    if sha256_file(train_runtime) != training["runtime_sha256"]:
        raise ValueError("original training runtime changed")
    original_runtime = read_json(train_runtime)
    policy = original_runtime["shared_gpu_policy"]
    with _scripts():
        importlib.import_module("shared_gpu").validate_policy(policy)
    devices = policy.get("devices", [policy])
    selected = [device for device in devices if str(device["gpu_index"]) == gpu]
    if len(selected) != 1:
        raise ValueError("medium GPU differs from original resource policy")
    selected_device = {key: selected[0][key] for key in ("gpu_index", "gpu_uuid")}
    roots = protected_roots(training, terminal["identity"])
    destination = validate_output_path(output, roots)
    from gradpert.execution.v2_checkpoint_eval import SERVER_ROOT

    runs = Path(original_runtime["runs_root"]).resolve()
    family = family_root.resolve()
    if (
        family != runs.parent
        or family == SERVER_ROOT.resolve()
        or not family.is_relative_to(SERVER_ROOT.resolve())
        or roots[1] == runs
        or not roots[1].is_relative_to(runs)
    ):
        raise ValueError("medium family must be the original sealed runs_root parent")
    if family == destination or not destination.is_relative_to(family):
        raise ValueError("medium family root must strictly contain its output")
    return dict(
        schema=EXECUTION_SCHEMA,
        training_plan=training_plan,
        population=population_binding,
        runtime=runtime,
        original_runtime=binding(train_runtime),
        analysis_source=source,
        compatibility=compatibility,
        output=str(destination),
        family_root=str(family),
        gpu=gpu,
        selected_device=selected_device,
        checkpoint_sha256=terminal["last"]["sha256"],
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        training_run_id=training["run_id"],
        training_source=terminal["identity"]["source"],
        policy=policy,
        config_sha256=training["config_sha256"],
        precision="as_trained_outer_bfloat16_native_autocast_heads",
        scientific_complete=False,
    )


def validate_execution_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("unknown medium execution schema")
    expected = build_execution_spec(
        training_plan=spec["training_plan"],
        population_binding=spec["population"],
        runtime=spec["runtime"],
        output=Path(spec["output"]),
        family_root=Path(spec["family_root"]),
        gpu=spec["gpu"],
        compatibility=spec["compatibility"],
    )
    if spec != expected:
        raise ValueError("medium execution source/inputs changed")


def validate_completion(spec: dict[str, Any], receipt: dict[str, Any]) -> None:
    validate_execution_spec(spec)
    frozen = read_json(checked(spec["population"]))
    expected = {
        key: frozen[key]
        for key in (
            "conditions",
            "rows",
            "deep_conditions",
            "deep_rows",
            "gradient_conditions",
            "gradient_rows",
        )
    }
    if (
        receipt.get("status") != "completed"
        or receipt.get("updates") != 0
        or receipt.get("spec_sha256") != sha256_json(spec)
        or receipt.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or receipt.get("training_source") != spec["training_source"]
        or receipt.get("analysis_source") != spec["analysis_source"]
        or receipt.get("config_sha256") != spec["config_sha256"]
        or receipt.get("dataset_id") != spec["dataset_id"]
        or receipt.get("training_run_id") != spec["training_run_id"]
        or receipt["medium"]["counts"] != expected
        or receipt["medium"]["population_sha256"] != frozen["diagnostic_population_sha256"]
        or not receipt["medium"]["state"]["restored"]
        or receipt["medium"]["state"]["before"] != receipt["medium"]["state"]["after"]
        or not receipt["medium"]["parity"]["used_path_passed"]
        or receipt["medium"]["parity"]["native_cache_parity"]["used_path_passed"] is not True
    ):
        raise ValueError("medium lacks actual full-population parity/state closure")
    gradients = receipt["medium"]["gradients"]
    if (
        len(gradients) != expected["gradient_conditions"]
        or sum(row["cells"] for row in gradients) != expected["gradient_rows"]
        or any(
            row["state"]["restored"] is not True or row["state"]["before"] != row["state"]["after"]
            for row in gradients
        )
        or len(receipt["medium"]["recipe_receipts"]) != expected["conditions"]
        or receipt["medium"].get("updates") != 0
    ):
        raise ValueError("gradient cohort/recipe transaction closure missing")
    crossed = receipt["crossed"]
    if frozen["crossed"]["status"] == "not_applicable":
        if crossed.get("status") != "not_applicable":
            raise ValueError("crossed support NA was relabeled")
    elif (
        crossed.get("status") != "completed"
        or crossed["counts"]
        != dict(
            conditions=len(frozen["crossed"]["schedule"]),
            rows=frozen["crossed"]["rows"],
            deep_conditions=len(frozen["crossed"]["schedule"]),
            deep_rows=frozen["crossed"]["rows"],
            gradient_conditions=0,
            gradient_rows=0,
        )
        or crossed.get("updates") != 0
        or len(crossed["recipe_receipts"]) != len(frozen["crossed"]["schedule"])
        or not crossed["state"]["restored"]
        or crossed["state"]["before"] != crossed["state"]["after"]
        or not crossed["parity"]["used_path_passed"]
        or crossed["parity"]["native_cache_parity"]["used_path_passed"] is not True
    ):
        raise ValueError("crossed population/state/parity closure missing")
    if any(Path(spec["family_root"]).rglob("*.pkl")):
        raise ValueError("whole successful medium family must contain zero PKL")
    training = read_json(checked(spec["training_plan"]))
    if any(Path(training["run_root"]).rglob("*.pkl")):
        raise ValueError("original successful training root must contain zero PKL")


def _run_worker(worker_binding: dict[str, str]) -> None:
    import os

    import torch

    from gradpert.config import load_experiment_config
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import prepare_runtime

    worker = read_json(checked(worker_binding))
    spec = read_json(checked(worker["spec"]))
    validate_execution_spec(spec)
    owner = read_json(checked(worker["owner"]))
    if (
        owner["spec_sha256"] != sha256_json(spec)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != spec["gpu"]
        or os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("medium worker owner/environment differs")
    with _scripts():
        shared = importlib.import_module("shared_gpu")
        mfu = importlib.import_module("mfu_probe")

        def guard():
            checked(worker_binding)
            checked(worker["spec"])
            checked(worker["owner"])
            if shared.process_info(owner["process"]["pid"]) != owner[
                "process"
            ] or not shared.descendant(os.getpid(), owner["process"]["pid"], shared.parent_map()):
                raise ValueError("medium live resource owner identity changed")
            mfu.verify_lease_records(
                Path("/proc/locks").read_text(),
                owner["process"]["pid"],
                [mfu.LEASE_ROOT / f"v2-gpu-{spec['gpu']}.lock"],
            )
            shared.admit(spec["policy"], shared.snapshot(owner["process"]["pid"]))

        guard()
        training = read_json(checked(spec["training_plan"]))
        terminal = training_terminal(training)
        frozen = read_json(checked(spec["population"]))
        config = load_experiment_config(training["config"])
        checkpoint = Path(training["run_root"]) / "fit" / terminal["last"]["file"]
        torch.cuda.set_per_process_memory_fraction(spec["policy"]["max_own_memory_fraction"], 0)
        results = {}
        environment = (
            __import__("gradpert.execution.identity", fromlist=["inspect_environment"])
            .inspect_environment(spec["analysis_source"]["repository_root"], device_name="cuda:0")
            .payload()
        )
        for phase in ("medium", "crossed"):
            guard()
            if phase == "crossed" and frozen["crossed"]["status"] == "not_applicable":
                results[phase] = dict(
                    status="not_applicable", reason=frozen["crossed"]["reason"], updates=0
                )
                continue
            with prepare_runtime(
                config,
                data_root=Path(training["data_root"]),
                run_seed=training["seed"],
                device=torch.device("cuda:0"),
                purpose="evaluation",
            ) as runtime:
                if runtime.identity != terminal["identity"]["data"]:
                    raise ValueError("diagnostic runtime data identity changed")
                allowed = validate_data_metadata(runtime.data, terminal["identity"])
                expected_population = population(
                    records_from_data(runtime.data),
                    expression_genes=len(runtime.data.expression_gene_ids),
                    allowed=allowed,
                    query_count=runtime.options.query_count,
                    seed=frozen["seed"],
                )
                if any(frozen.get(key) != value for key, value in expected_population.items()):
                    raise ValueError("runtime metadata selection differs from sealed population")
                load_evaluation_checkpoint(
                    checkpoint,
                    runtime.objective,
                    training_identity=terminal["identity"],
                    checkpoint_sha256=spec["checkpoint_sha256"],
                )
                results[phase] = _collect_runtime(
                    runtime,
                    frozen,
                    phase=phase,
                    resource_check=guard,
                    on_progress=lambda progress: atomic_json(
                        Path(spec["output"]) / "progress.json", progress
                    ),
                )
                atomic_json(Path(spec["output"]) / f"{phase}.json", results[phase])
        guard()
        receipt = dict(
            status="completed",
            updates=0,
            spec_sha256=sha256_json(spec),
            checkpoint_sha256=spec["checkpoint_sha256"],
            training_source=spec["training_source"],
            dataset_id=spec["dataset_id"],
            protocol_id=spec["protocol_id"],
            training_run_id=spec["training_run_id"],
            config_sha256=spec["config_sha256"],
            analysis_source=spec["analysis_source"],
            environment=environment,
            **results,
            scientific_complete=False,
        )
        validate_completion(spec, receipt)
        atomic_json(Path(spec["output"]) / "WORKER_COMPLETE.json", receipt)


def execute_medium(*, spec_binding: dict[str, str]) -> dict[str, Any]:
    """Resource-owned CPU controller; only explicit invocation can dispatch a worker."""
    import os
    import sys
    import time
    from contextlib import ExitStack

    spec = read_json(checked(spec_binding))
    validate_execution_spec(spec)
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError("medium execution requires a fresh independent output")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("medium controller allocator/cuBLAS environment missing")
    with _scripts():
        runner = importlib.import_module("run_dataset_transfer_300")
        shared = importlib.import_module("shared_gpu")
        lock = importlib.import_module("run_group").lock
        with ExitStack() as stack:
            stack.enter_context(
                lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{spec['gpu']}.lock"))
            )
            shared.admit(spec["policy"], shared.snapshot(os.getpid()))
            owner = shared.process_info(os.getpid())
            if owner is None:
                raise ValueError("medium controller identity unavailable")
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(
                output / "OWNER.json",
                dict(process=owner, spec_sha256=sha256_json(spec), started_unix=time.time()),
            )
            atomic_json(
                output / "WORKER_BINDING.json",
                dict(spec=spec_binding, owner=binding(output / "OWNER.json")),
            )
            worker = binding(output / "WORKER_BINDING.json")
            command = [
                sys.executable,
                "-m",
                "gradpert.execution.dataset_transfer_medium",
                "--worker-binding",
                worker["path"],
                "--worker-binding-sha256",
                worker["sha256"],
            ]
            source = Path(__file__).resolve().parents[3]
            # native_run supplies the original continuous15s admission/identity/termination loop.
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=spec["gpu"],
                PYTHONPATH=__import__("os").pathsep.join(
                    [str(source / "src"), str(source), str(source / "scripts/v2")]
                ),
                PYTHONDONTWRITEBYTECODE="1",
            )
            try:
                with patch.dict(os.environ, env, clear=True):
                    runner.native_run(command, output / "worker.log", spec["policy"])
                receipt = read_json(output / "WORKER_COMPLETE.json")
                validate_completion(spec, receipt)
                atomic_json(output / "COMPLETE.json", receipt)
                return receipt
            except BaseException as error:
                atomic_json(
                    output / "FAILURE.json",
                    dict(type=type(error).__name__, error=str(error), scientific_complete=False),
                )
                raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--spec", type=Path)
    choice.add_argument("--worker-binding", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--worker-binding-sha256")
    args = parser.parse_args()
    if args.worker_binding:
        if not args.worker_binding_sha256:
            parser.error("worker binding requires an explicit hash")
        _run_worker(dict(path=str(args.worker_binding), sha256=args.worker_binding_sha256))
    else:
        if not args.sha256:
            parser.error("execution spec requires an explicit hash")
        execute_medium(spec_binding=dict(path=str(args.spec), sha256=args.sha256))


if __name__ == "__main__":
    main()
