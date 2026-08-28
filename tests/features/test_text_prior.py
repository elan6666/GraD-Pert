import hashlib

import numpy as np
import pytest

from gradpert.features import verify_text_prior_npz


def test_verify_text_prior_npz_requires_exact_axis_and_hash(tmp_path):
    path = tmp_path / "prior.npz"
    np.savez_compressed(
        path,
        genes=np.asarray(["C12orf45", "TP53"]),
        vectors=np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        model=np.asarray("test-prior"),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = verify_text_prior_npz(
        path,
        expected_sha256=digest,
        expected_gene_ids=("C12orf45", "TP53"),
    )
    assert artifact.embedding_width == 2
    assert artifact.model == "test-prior"
    with pytest.raises(ValueError, match="gene axis"):
        verify_text_prior_npz(
            path,
            expected_sha256=digest,
            expected_gene_ids=("TP53", "C12orf45"),
        )
