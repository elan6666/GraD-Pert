"""Verification for exact-axis NPZ text priors used by vNext ablations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gradpert.hashing import sha256_file, sha256_json

GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256 = (
    "34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318"
)
GENEPT_SEED_GO_PROTEIN_PATHWAY_MODEL = "doubao-embedding-vision"
GENEPT_SEED_GO_PROTEIN_PATHWAY_GENE_COUNT = 17_730
GENEPT_SEED_GO_PROTEIN_PATHWAY_WIDTH = 2_048


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
    zero_vector_gene_ids: tuple[str, ...]
    source_gene_count: int
    source_gene_order_sha256: str
    extra_source_gene_count: int
    extra_source_gene_ids: tuple[str, ...]
    extra_source_gene_ids_sha256: str
    perturbation_target_gene_ids: tuple[str, ...]
    perturbation_target_gene_ids_sha256: str
    selected_matrix_sha256: str


def verify_text_prior_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_gene_ids: tuple[str, ...],
    perturbation_target_gene_ids: tuple[str, ...] = (),
    expected_model: str | None = None,
    expected_source_gene_count: int | None = None,
    expected_embedding_width: int | None = None,
) -> TextPriorArtifact:
    """Verify a sealed superset and select the runtime graph axis in exact order."""

    locked_master = expected_sha256 == GENEPT_SEED_GO_PROTEIN_PATHWAY_SHA256
    if locked_master:
        locked_contract = (
            ("model", expected_model, GENEPT_SEED_GO_PROTEIN_PATHWAY_MODEL),
            (
                "source gene count",
                expected_source_gene_count,
                GENEPT_SEED_GO_PROTEIN_PATHWAY_GENE_COUNT,
            ),
            ("embedding width", expected_embedding_width, GENEPT_SEED_GO_PROTEIN_PATHWAY_WIDTH),
        )
        for label, supplied, locked in locked_contract:
            if supplied is not None and supplied != locked:
                raise ValueError(
                    f"sealed Seed-GO-ProteinPathway {label} cannot override {locked!r}"
                )
        expected_model = GENEPT_SEED_GO_PROTEIN_PATHWAY_MODEL
        expected_source_gene_count = GENEPT_SEED_GO_PROTEIN_PATHWAY_GENE_COUNT
        expected_embedding_width = GENEPT_SEED_GO_PROTEIN_PATHWAY_WIDTH
    strict_superset = locked_master or any(
        value is not None
        for value in (expected_model, expected_source_gene_count, expected_embedding_width)
    )

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
        raw_gene_ids = payload["genes"]
        raw_model = payload["model"]
        if raw_gene_ids.ndim != 1:
            raise ValueError("text-prior genes must be a one-dimensional string array")
        gene_values = raw_gene_ids.tolist()
        if any(not isinstance(value, str) or not value for value in gene_values):
            raise ValueError("text-prior genes must contain non-empty exact string labels")
        source_gene_ids = tuple(gene_values)
        values = np.asarray(payload["vectors"], dtype=np.float32)
        if raw_model.shape != () or not isinstance(raw_model.item(), str):
            raise ValueError("text-prior model must be one scalar string")
        model = raw_model.item()
    if expected_model is not None and model != expected_model:
        raise ValueError(f"text-prior model differs: expected {expected_model}, observed {model}")
    if (
        expected_source_gene_count is not None
        and len(source_gene_ids) != expected_source_gene_count
    ):
        raise ValueError(
            "text-prior source gene count differs: "
            f"expected {expected_source_gene_count}, observed {len(source_gene_ids)}"
        )
    if len(set(source_gene_ids)) != len(source_gene_ids):
        raise ValueError("text-prior source gene labels contain exact duplicates")
    if values.ndim != 2 or values.shape[0] != len(source_gene_ids) or values.shape[1] < 1:
        raise ValueError("text-prior matrix shape is invalid")
    if expected_embedding_width is not None and values.shape[1] != expected_embedding_width:
        raise ValueError("text-prior matrix shape is invalid")
    if not np.isfinite(values).all():
        raise ValueError("text-prior matrix contains non-finite values")
    zero_vector_gene_ids = tuple(
        gene_id for gene_id, row in zip(source_gene_ids, values, strict=True) if not np.any(row)
    )
    if strict_superset and zero_vector_gene_ids:
        raise ValueError("text-prior source contains zero vectors")

    if len(set(expected_gene_ids)) != len(expected_gene_ids) or any(
        not isinstance(gene_id, str) or not gene_id for gene_id in expected_gene_ids
    ):
        raise ValueError("runtime graph gene IDs must be unique non-empty exact strings")
    if len(set(perturbation_target_gene_ids)) != len(perturbation_target_gene_ids) or any(
        not isinstance(gene_id, str) or not gene_id for gene_id in perturbation_target_gene_ids
    ):
        raise ValueError("perturbation target gene IDs must be unique non-empty exact strings")
    runtime_gene_set = set(expected_gene_ids)
    targets_outside_runtime = tuple(
        gene_id for gene_id in perturbation_target_gene_ids if gene_id not in runtime_gene_set
    )
    if targets_outside_runtime:
        raise ValueError("perturbation targets are absent from the runtime graph axis")

    if not strict_superset and source_gene_ids != expected_gene_ids:
        raise ValueError("text-prior gene axis differs from the runtime graph axis")

    source_index = {gene_id: index for index, gene_id in enumerate(source_gene_ids)}
    missing_targets = tuple(
        gene_id for gene_id in perturbation_target_gene_ids if gene_id not in source_index
    )
    if missing_targets:
        raise ValueError("text-prior source is missing perturbation target gene labels")
    missing_runtime = tuple(gene_id for gene_id in expected_gene_ids if gene_id not in source_index)
    if missing_runtime:
        raise ValueError("text-prior source is missing runtime graph gene labels")

    selected_indices = np.asarray(
        [source_index[gene_id] for gene_id in expected_gene_ids], dtype=np.int64
    )
    selected_values = np.ascontiguousarray(values[selected_indices], dtype=np.float32)
    selected_values.setflags(write=False)
    extra_source_gene_ids = tuple(
        gene_id for gene_id in source_gene_ids if gene_id not in runtime_gene_set
    )
    return TextPriorArtifact(
        source_path=source.resolve(),
        source_sha256=observed_sha256,
        source_size_bytes=source.stat().st_size,
        gene_ids=expected_gene_ids,
        values=selected_values,
        model=model,
        embedding_width=int(selected_values.shape[1]),
        gene_order_sha256=sha256_json(list(expected_gene_ids)),
        zero_vector_gene_ids=zero_vector_gene_ids,
        source_gene_count=len(source_gene_ids),
        source_gene_order_sha256=sha256_json(list(source_gene_ids)),
        extra_source_gene_count=len(extra_source_gene_ids),
        extra_source_gene_ids=extra_source_gene_ids,
        extra_source_gene_ids_sha256=sha256_json(list(extra_source_gene_ids)),
        perturbation_target_gene_ids=perturbation_target_gene_ids,
        perturbation_target_gene_ids_sha256=sha256_json(list(perturbation_target_gene_ids)),
        selected_matrix_sha256=hashlib.sha256(selected_values.tobytes(order="C")).hexdigest(),
    )
