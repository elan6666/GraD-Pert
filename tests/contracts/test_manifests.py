from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from gradpert.contracts import (
    CanonicalDataManifest,
    ControlDraw,
    EvaluationControlManifest,
    EvaluationStateManifest,
    SplitManifest,
)
from gradpert.hashing import sha256_json

HASH = "a" * 64
COMMIT = "b" * 40


def _split_payload() -> dict[str, Any]:
    content: dict[str, object] = {
        "dataset_id": "replogle_k562_essential",
        "protocol_id": "within_cell_unseen_single",
        "policy_id": "grouped_0.5625_0.1875_0.25",
        "split_seed": 42,
        "control_condition_id": "ctrl",
        "train_conditions": ["A", "B"],
        "val_conditions": ["C"],
        "test_conditions": ["D"],
    }
    return {
        "schema_version": "split-manifest-v1",
        **content,
        "split_content_sha256": sha256_json(content),
    }


def test_split_manifest_hashes_exact_partition() -> None:
    manifest = SplitManifest.model_validate(_split_payload())
    assert manifest.split_seed == 42
    assert manifest.split_content_sha256 == sha256_json(manifest.content_payload())


@pytest.mark.parametrize(
    "mutation",
    [
        {"val_conditions": ["A"]},
        {"train_conditions": ["ctrl", "A"]},
        {"train_conditions": ["A", "A"]},
        {"split_content_sha256": HASH},
    ],
)
def test_split_manifest_rejects_leakage_duplicates_and_stale_hash(
    mutation: dict[str, Any],
) -> None:
    payload = _split_payload()
    payload.update(mutation)
    with pytest.raises(ValidationError):
        SplitManifest.model_validate(payload)


def test_control_draw_allows_replacement_but_requires_exact_order_hash() -> None:
    row_ids = [f"ctrl-{index % 7}" for index in range(300)]
    context_ids = ["K562::batch-1"] * 300
    draw = ControlDraw(
        condition_id="A",
        context_policy="truth_cell_context_resampling",
        source_pool_sha256=HASH,
        ordered_context_ids=context_ids,
        ordered_context_ids_sha256=sha256_json(context_ids),
        ordered_row_ids=row_ids,
        ordered_row_ids_sha256=sha256_json(row_ids),
    )
    manifest = EvaluationControlManifest(
        schema_version="evaluation-controls-v1",
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        split_name="test",
        split_content_sha256=HASH,
        evaluation_seed=20260824,
        rng="numpy_pcg64",
        sample_with_replacement=True,
        context_policy="truth_cell_context_resampling",
        n_controls_per_condition=300,
        draws=[draw],
    )
    assert len(set(manifest.draws[0].ordered_row_ids)) == 7


def test_control_draw_rejects_not_300() -> None:
    row_ids = ["ctrl-0"] * 299
    context_ids = ["K562::batch-1"] * 299
    with pytest.raises(ValidationError, match="exactly 300"):
        ControlDraw(
            condition_id="A",
            context_policy="truth_cell_context_resampling",
            source_pool_sha256=HASH,
            ordered_context_ids=context_ids,
            ordered_context_ids_sha256=sha256_json(context_ids),
            ordered_row_ids=row_ids,
            ordered_row_ids_sha256=sha256_json(row_ids),
        )


def test_canonical_manifest_binds_data_split_controls_and_both_gene_axes() -> None:
    manifest = CanonicalDataManifest(
        schema_version="canonical-data-v1",
        dataset_id="norman",
        protocol_id="norman_combo_seen2",
        state="canonical_ready",
        canonical_adata_path="data/norman/norman_combo_seen2/canonical/adata.h5ad",
        canonical_adata_sha256=HASH,
        source_manifest_sha256=HASH,
        preprocessing_manifest_sha256=HASH,
        qc_manifest_sha256=HASH,
        split_manifest_sha256=HASH,
        split_content_sha256=HASH,
        evaluation_controls_sha256=HASH,
        expression_gene_order_sha256=HASH,
        graph_gene_order_sha256=HASH,
        observation_order_sha256=HASH,
        n_cells=10,
        n_expression_genes=5,
        n_graph_genes=6,
        n_conditions=4,
        n_controls=2,
    )
    assert manifest.n_graph_genes == 6

    with pytest.raises(ValidationError, match="cannot be smaller"):
        CanonicalDataManifest.model_validate({**manifest.model_dump(), "n_expression_genes": 7})


def test_evaluation_state_records_unrankable_conditions_without_dropping_them() -> None:
    condition_ids = ["A", "B"]
    de = {"A": [0, 2], "B": []}
    unavailable = {"B": "insufficient_truth_cells_for_t_test:n=1"}
    manifest = EvaluationStateManifest(
        schema_version="evaluation-state-v1",
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        canonical_data_sha256=HASH,
        split_content_sha256=HASH,
        expression_gene_order_sha256=HASH,
        condition_ids=condition_ids,
        condition_ids_sha256=sha256_json(condition_ids),
        de_gene_indices=de,
        de_gene_indices_sha256=sha256_json(de),
        top_de_gene_indices=de,
        top_de_gene_indices_sha256=sha256_json(de),
        de_unavailable_reasons=unavailable,
        de_unavailable_reasons_sha256=sha256_json(unavailable),
        de_method="scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets",
        de_reference="ctrl",
        de_source_commit=COMMIT,
        systema_reference_condition_ids=["train-A", "A"],
        systema_reference_condition_ids_sha256=sha256_json(["train-A", "A"]),
        arrays_path="data/state_arrays.npz",
        arrays_sha256=HASH,
        systema_reference_content_sha256=HASH,
        metric_control_means_content_sha256=HASH,
    )

    assert manifest.de_gene_indices["B"] == []
    with pytest.raises(ValidationError, match="availability and reason disagree"):
        EvaluationStateManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "de_unavailable_reasons": {},
                "de_unavailable_reasons_sha256": sha256_json({}),
            }
        )
