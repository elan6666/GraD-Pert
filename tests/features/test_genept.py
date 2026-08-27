from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from gradpert.features import (
    GENEPT_EMB_B_SHA256,
    GenePTArtifact,
    MissingGenePTTargetsError,
    build_genept_coverage_plan,
    build_ordered_genept_matrix,
    genept,
    verify_genept_emb_b,
)


def _artifact(embeddings: dict[str, list[float]]) -> GenePTArtifact:
    validated = genept._validate_embedding_object(
        embeddings,
        expected_entry_count=len(embeddings),
        expected_width=3,
    )
    return GenePTArtifact(
        source_path=Path("emb_b.pickle"),
        source_sha256="a" * 64,
        source_size_bytes=123,
        embeddings=validated,
        entry_count=len(validated),
        embedding_width=3,
    )


def test_unapproved_pickle_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tampered.pickle"
    source.write_bytes(pickle.dumps({"TP53": [1.0, 2.0, 3.0]}))
    called = False

    def forbidden_load(handle: object) -> object:
        del handle
        nonlocal called
        called = True
        raise AssertionError("unapproved pickle must not be deserialized")

    monkeypatch.setattr(genept.pickle, "load", forbidden_load)
    with pytest.raises(ValueError, match="refusing to deserialize") as error:
        verify_genept_emb_b(source)
    assert GENEPT_EMB_B_SHA256 in str(error.value)
    assert not called


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([1.0], "exact dict"),
        ({1: [1.0]}, "keys"),
        ({" TP53": [1.0]}, "keys"),
        ({"TP53": (1.0,)}, "exact list"),
        ({"TP53": [1]}, "finite exact float"),
        ({"TP53": [float("nan")]}, "finite exact float"),
        ({"TP53": [1.0, 2.0]}, "width"),
    ],
)
def test_embedding_schema_fails_closed(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        genept._validate_embedding_object(
            payload,
            expected_entry_count=1,
            expected_width=1,
        )


def test_exact_case_matching_never_rescues_a_missing_gene() -> None:
    artifact = _artifact({"TP53": [1.0, 2.0, 3.0], "Tp53": [4.0, 5.0, 6.0]})
    plan = build_genept_coverage_plan(
        artifact,
        ordered_graph_gene_ids=("TP53", "tp53", "Tp53"),
        perturbation_target_gene_ids=("TP53",),
    )
    assert plan.retained_graph_gene_ids == ("TP53", "Tp53")
    assert plan.removed_non_target_gene_ids == ("tp53",)
    assert plan.to_receipt()["identifier_matching"] == "exact_case_sensitive"


def test_missing_perturbation_target_raises_before_matrix_construction() -> None:
    artifact = _artifact({"NON_TARGET": [1.0, 2.0, 3.0]})
    with pytest.raises(MissingGenePTTargetsError) as error:
        build_genept_coverage_plan(
            artifact,
            ordered_graph_gene_ids=("NON_TARGET", "PERT"),
            perturbation_target_gene_ids=("PERT",),
        )
    assert error.value.missing_target_gene_ids == ("PERT",)


def test_missing_non_targets_are_removed_without_reordering_retained_genes() -> None:
    artifact = _artifact(
        {
            "A": [1.0, 2.0, 3.0],
            "PERT": [4.0, 5.0, 6.0],
            "C": [7.0, 8.0, 9.0],
        }
    )
    plan = build_genept_coverage_plan(
        artifact,
        ordered_graph_gene_ids=("A", "MISSING_1", "PERT", "MISSING_2", "C"),
        perturbation_target_gene_ids=("PERT",),
    )
    assert plan.retained_graph_gene_ids == ("A", "PERT", "C")
    assert plan.removed_non_target_gene_ids == ("MISSING_1", "MISSING_2")
    receipt = plan.to_receipt()
    assert receipt["missing_non_target_policy"] == "remove_preserving_canonical_order"
    assert receipt["missing_perturbation_target_policy"] == ("fail_before_model_construction")
    assert receipt["fill_policy"] == "forbidden"


def test_targets_must_belong_to_the_requested_graph_axis() -> None:
    artifact = _artifact({"A": [1.0, 2.0, 3.0], "PERT": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="requested graph axis"):
        build_genept_coverage_plan(
            artifact,
            ordered_graph_gene_ids=("A",),
            perturbation_target_gene_ids=("PERT",),
        )


def test_ordered_matrix_and_hash_are_deterministic_and_order_sensitive() -> None:
    artifact = _artifact(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
        }
    )
    first_plan = build_genept_coverage_plan(
        artifact,
        ordered_graph_gene_ids=("B", "A"),
        perturbation_target_gene_ids=("A",),
    )
    first = build_ordered_genept_matrix(artifact, first_plan)
    repeated = build_ordered_genept_matrix(artifact, first_plan)
    np.testing.assert_array_equal(
        first.values,
        np.asarray([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], dtype=np.float32),
    )
    assert first.values.dtype == np.float32
    assert not first.values.flags.writeable
    assert first.matrix_sha256 == repeated.matrix_sha256

    reverse_plan = build_genept_coverage_plan(
        artifact,
        ordered_graph_gene_ids=("A", "B"),
        perturbation_target_gene_ids=("A",),
    )
    reverse = build_ordered_genept_matrix(artifact, reverse_plan)
    assert reverse.matrix_sha256 != first.matrix_sha256
    assert reverse.gene_order_sha256 != first.gene_order_sha256


def test_duplicate_graph_or_target_ids_are_rejected() -> None:
    artifact = _artifact({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="ordered graph gene IDs must be unique"):
        build_genept_coverage_plan(
            artifact,
            ordered_graph_gene_ids=("A", "A"),
            perturbation_target_gene_ids=("A",),
        )
    with pytest.raises(ValueError, match="perturbation target gene IDs must be unique"):
        build_genept_coverage_plan(
            artifact,
            ordered_graph_gene_ids=("A",),
            perturbation_target_gene_ids=("A", "A"),
        )
