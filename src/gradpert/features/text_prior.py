"""Verification for exact-axis NPZ text priors used by vNext ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gradpert.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class TextPriorArtifact:
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    gene_ids: tuple[str, ...]
    values: np.ndarray[Any, Any]
    model: str
    embedding_width: int
    gene_order_sha256: str


def verify_text_prior_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_gene_ids: tuple[str, ...],
) -> TextPriorArtifact:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"text-prior artifact is missing: {source}")
    observed_sha256 = sha256_file(source, chunk_size=8 * 1024 * 1024)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "text-prior artifact SHA-256 differs: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != {"genes", "vectors", "model"}:
            raise ValueError("text-prior NPZ must contain exactly genes, vectors, and model")
        gene_ids = tuple(str(value) for value in payload["genes"].tolist())
        values = np.asarray(payload["vectors"], dtype=np.float32)
        model = str(payload["model"].item())
    if gene_ids != expected_gene_ids:
        raise ValueError("text-prior gene axis differs from the runtime graph axis")
    if values.ndim != 2 or values.shape[0] != len(gene_ids) or values.shape[1] < 1:
        raise ValueError("text-prior matrix shape is invalid")
    if not np.isfinite(values).all():
        raise ValueError("text-prior matrix contains non-finite values")
    values = np.ascontiguousarray(values, dtype=np.float32)
    values.setflags(write=False)
    return TextPriorArtifact(
        source_path=source.resolve(),
        source_sha256=observed_sha256,
        source_size_bytes=source.stat().st_size,
        gene_ids=gene_ids,
        values=values,
        model=model,
        embedding_width=int(values.shape[1]),
        gene_order_sha256=sha256_json(list(gene_ids)),
    )
