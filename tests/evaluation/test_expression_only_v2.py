"""Expression-only evaluation must avoid DEG access and preserve Pearson definitions."""

from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize


def populations():
    rng = np.random.default_rng(83)
    return dict(
        condition_id="g0",
        prediction=rng.normal(size=(300, 6)),
        input_control=rng.normal(size=(300, 6)),
        truth=rng.normal(size=(11, 6)),
        metric_control_pool_mean=rng.normal(size=6),
        systema_reference=rng.normal(size=6),
        de_gene_indices=[1, 3],
    )


class NoDegAccess:
    def __iter__(self):
        raise AssertionError("DEG indices must not be inspected")

    def __getitem__(self, key):
        raise AssertionError("DEG condition metadata must not be inspected")

    def get(self, key, default=None):
        raise AssertionError("DEG availability must not be inspected")


@pytest.mark.parametrize("selected", [None, (0, 2, 4), (1,)])
def test_expression_only_matches_legacy_all_scopes_and_does_not_read_deg(selected):
    args = populations()
    legacy = compute_condition_metrics_v2(**args, gene_indices=selected)
    result = compute_condition_metrics_v2(
        **{**args, "de_gene_indices": NoDegAccess()},
        gene_indices=selected,
        include_deg=False,
    )
    assert result.results == tuple(r for r in legacy.results if r.metric_id.endswith("_all"))
    assert len(result.results) == 3
    assert len(macro_summarize([result])) == 3


def test_expression_only_pearsons_equal_independent_population_mean_calculation():
    args = populations()
    result = compute_condition_metrics_v2(**args, include_deg=False)
    predicted = args["prediction"].mean(axis=0)
    truth = args["truth"].mean(axis=0)
    for entry, baseline in zip(
        result.results,
        [
            args["input_control"].mean(axis=0),
            args["metric_control_pool_mean"],
            args["systema_reference"],
        ],
        strict=True,
    ):
        assert entry.value == pytest.approx(
            np.corrcoef(predicted - baseline, truth - baseline)[0, 1]
        )
        assert entry.gene_count == 6


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_deg_switch_rejects_non_boolean_values(invalid):
    with pytest.raises(ValueError, match="explicit boolean"):
        compute_condition_metrics_v2(**populations(), include_deg=invalid)


def test_checkpoint_scope_fails_before_source_or_cuda_access():
    from argparse import Namespace

    from gradpert.execution.v2_checkpoint_eval import (
        evaluation_metric_scope,
        execute_evaluation_plan,
        resolve_evaluation_plan,
    )

    assert evaluation_metric_scope({}) == "all_and_deg"
    assert evaluation_metric_scope({"metric_scope": "expression_only"}) == "expression_only"
    for invalid in [None, False, 0, [], {}, "all", "deg"]:
        with pytest.raises(ValueError, match="metric scope"):
            execute_evaluation_plan({"metric_scope": invalid})
        with pytest.raises(ValueError, match="metric scope"):
            resolve_evaluation_plan(Namespace(metric_scope=invalid))


def test_evaluation_never_reads_deg_metadata_and_preserves_empty_group(monkeypatch):
    torch = pytest.importorskip("torch")
    from gradpert.training.v2 import evaluation as module

    args = populations()
    index = SimpleNamespace(gene_ids=tuple(f"g{i}" for i in range(6)), n_nodes=6)
    control_ids = tuple(f"c{i}" for i in range(300))
    data = SimpleNamespace(
        split_name="val",
        expression_gene_ids=index.gene_ids,
        split=SimpleNamespace(
            val_conditions=["g0"], train_conditions=["g1"], control_condition_id="ctrl"
        ),
        control_manifest=SimpleNamespace(
            split_name="val",
            draws=[SimpleNamespace(condition_id="g0", ordered_row_ids=control_ids)],
        ),
        load_control_rows=lambda ids: SimpleNamespace(
            ordered_row_ids=ids, expression=args["input_control"]
        ),
        load_truth_rows=lambda condition: SimpleNamespace(
            ordered_row_ids=tuple(f"t{i}" for i in range(11)), expression=args["truth"]
        ),
        control_manifest_file_sha256="controls",
    )
    reference = SimpleNamespace(
        manifest=SimpleNamespace(
            schema_version="evaluation-state-v2",
            condition_ids=["g0"],
            systema_reference_condition_ids=["g1"],
            de_gene_indices=NoDegAccess(),
            de_unavailable_reasons=NoDegAccess(),
        ),
        metric_control_means=[args["metric_control_pool_mean"]],
        systema_reference=args["systema_reference"],
        manifest_file_sha256="reference",
    )
    calls = []
    monkeypatch.setattr(module, "prepare_evaluation_graph", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "predict_controls", lambda *a, **k: calls.append(1) or args["prediction"]
    )
    result = module.evaluate(
        SimpleNamespace(),
        index,
        data,
        reference,
        expected_split="val",
        device=torch.device("cpu"),
        cell_batch=2,
        query_count=6,
        metric_gene_groups={
            "seen_expression": (0, 1, 2),
            "unseen_expression": (3, 4, 5),
            "empty": (),
        },
        include_deg_metrics=False,
    )
    assert calls == [1]
    assert result["metric_scope"] == "expression_only"
    assert len(result["metrics"]) == 3
    assert all(m["metric_id"].endswith("_all") for m in result["metrics"])
    assert len(result["metric_gene_groups"]["seen_expression"]["metrics"]) == 3
    assert result["metric_gene_groups"]["empty"]["unavailable_reason"] == "empty_expression_group"
    assert module.merge_evaluation_shards([result], ordered_conditions=("g0",)) == result
    with pytest.raises(ValueError, match="metric_scope"):
        module.merge_evaluation_shards(
            [result, {**result, "metric_scope": "all_and_deg"}], ordered_conditions=("g0",)
        )
