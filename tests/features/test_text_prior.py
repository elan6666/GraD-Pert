import hashlib

import numpy as np
import pytest

from gradpert.features import verify_text_prior_npz


def _write_prior(tmp_path, *, genes, vectors, model="doubao-embedding-vision"):
    path = tmp_path / "prior.npz"
    np.savez_compressed(
        path,
        genes=np.asarray(genes),
        vectors=np.asarray(vectors, dtype=np.float32),
        model=np.asarray(model),
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_text_prior_npz_selects_runtime_order_and_receipts_extras(tmp_path):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA", "genea", "EXTRA"],
        vectors=[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
    )
    artifact = verify_text_prior_npz(
        path,
        expected_sha256=digest,
        expected_gene_ids=("genea", "TP53", "GeneA"),
        perturbation_target_gene_ids=("GeneA",),
        expected_source_gene_count=4,
        expected_embedding_width=2,
    )
    assert artifact.embedding_width == 2
    assert artifact.model == "doubao-embedding-vision"
    assert artifact.gene_ids == ("genea", "TP53", "GeneA")
    assert artifact.values.tolist() == [[3.0, 3.5], [1.0, 1.5], [2.0, 2.5]]
    assert artifact.extra_source_gene_count == 1
    assert artifact.extra_source_gene_ids == ("EXTRA",)
    assert artifact.perturbation_target_gene_ids == ("GeneA",)
    assert (
        artifact.selected_matrix_sha256
        == hashlib.sha256(
            np.asarray([[3.0, 3.5], [1.0, 1.5], [2.0, 2.5]], dtype=np.float32).tobytes()
        ).hexdigest()
    )
    assert artifact.zero_vector_gene_ids == ()


def test_verify_text_prior_npz_rejects_missing_runtime_gene(tmp_path):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA"],
        vectors=[[1.0, 1.5], [2.0, 2.5]],
    )
    with pytest.raises(ValueError, match="missing runtime graph gene"):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53", "genea"),
            expected_source_gene_count=2,
            expected_embedding_width=2,
        )


def test_verify_text_prior_npz_rejects_missing_target_before_runtime_selection(tmp_path):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA"],
        vectors=[[1.0, 1.5], [2.0, 2.5]],
    )
    with pytest.raises(ValueError, match="missing perturbation target"):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53", "MISSING"),
            perturbation_target_gene_ids=("MISSING",),
            expected_source_gene_count=2,
            expected_embedding_width=2,
        )


def test_verify_text_prior_npz_rejects_duplicate_source_labels(tmp_path):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "TP53"],
        vectors=[[1.0, 1.5], [2.0, 2.5]],
    )
    with pytest.raises(ValueError, match="exact duplicates"):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53",),
            expected_source_gene_count=2,
            expected_embedding_width=2,
        )


def test_verify_text_prior_npz_preserves_legacy_exact_axis_and_zero_receipt(tmp_path):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA"],
        vectors=[[0.0, 0.0], [2.0, 2.5]],
        model="legacy-model",
    )
    artifact = verify_text_prior_npz(
        path,
        expected_sha256=digest,
        expected_gene_ids=("TP53", "GeneA"),
    )
    assert artifact.model == "legacy-model"
    assert artifact.zero_vector_gene_ids == ("TP53",)
    assert artifact.extra_source_gene_ids == ()
    with pytest.raises(ValueError, match="gene axis differs"):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("GeneA", "TP53"),
        )


def test_verify_text_prior_npz_rejects_wrong_sha_and_schema(tmp_path):
    path, _digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA"],
        vectors=[[1.0, 1.5], [2.0, 2.5]],
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_text_prior_npz(
            path,
            expected_sha256="0" * 64,
            expected_gene_ids=("TP53", "GeneA"),
        )

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        genes=np.asarray(["TP53", "GeneA"]),
        vectors=np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32),
        model=np.asarray("doubao-embedding-vision"),
        unexpected=np.asarray(1),
    )
    invalid_digest = hashlib.sha256(invalid.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly genes, vectors, and model"):
        verify_text_prior_npz(
            invalid,
            expected_sha256=invalid_digest,
            expected_gene_ids=("TP53", "GeneA"),
        )


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.0, 0.0], [2.0, 2.5]], "zero vectors"),
        ([[float("nan"), 1.0], [2.0, 2.5]], "non-finite"),
    ],
)
def test_verify_text_prior_npz_rejects_invalid_rows(tmp_path, vectors, message):
    path, digest = _write_prior(tmp_path, genes=["TP53", "GeneA"], vectors=vectors)
    with pytest.raises(ValueError, match=message):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53", "GeneA"),
            expected_source_gene_count=2,
            expected_embedding_width=2,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_model": "wrong-model"}, "model differs"),
        ({"expected_source_gene_count": 3}, "source gene count differs"),
        ({"expected_embedding_width": 3}, "matrix shape"),
    ],
)
def test_verify_text_prior_npz_rejects_wrong_contract(tmp_path, override, message):
    path, digest = _write_prior(
        tmp_path,
        genes=["TP53", "GeneA"],
        vectors=[[1.0, 1.5], [2.0, 2.5]],
    )
    arguments = {
        "expected_model": "doubao-embedding-vision",
        "expected_source_gene_count": 2,
        "expected_embedding_width": 2,
        **override,
    }
    with pytest.raises(ValueError, match=message):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53", "GeneA"),
            **arguments,
        )
