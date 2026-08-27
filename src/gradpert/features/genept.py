"""Fail-closed verification and ordering for the frozen GenePT ``emb_b`` artifact."""

from __future__ import annotations

import hashlib
import math
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from gradpert.hashing import canonical_json_bytes, sha256_file, sha256_json

GENEPT_EMB_B_SHA256 = "fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862"
GENEPT_EMB_B_ENTRY_COUNT = 93_800
GENEPT_EMB_B_WIDTH = 1_536


class MissingGenePTTargetsError(ValueError):
    """Raised before model construction when a perturbation target lacks GenePT."""

    def __init__(self, missing_target_gene_ids: Sequence[str]) -> None:
        self.missing_target_gene_ids = tuple(missing_target_gene_ids)
        joined = ", ".join(self.missing_target_gene_ids)
        super().__init__(f"perturbation targets are absent from frozen GenePT emb_b: {joined}")


@dataclass(frozen=True)
class GenePTArtifact:
    """Verified frozen embedding artifact without identifier normalization."""

    source_path: Path
    source_sha256: str
    source_size_bytes: int
    embeddings: Mapping[str, list[float]]
    entry_count: int
    embedding_width: int


@dataclass(frozen=True)
class GenePTCoveragePlan:
    """Exact-case filtering plan for one ordered runtime graph axis."""

    schema_version: str
    source_sha256: str
    input_graph_gene_ids: tuple[str, ...]
    perturbation_target_gene_ids: tuple[str, ...]
    retained_graph_gene_ids: tuple[str, ...]
    removed_non_target_gene_ids: tuple[str, ...]
    input_graph_gene_order_sha256: str
    perturbation_target_gene_ids_sha256: str
    retained_graph_gene_order_sha256: str
    removed_non_target_gene_ids_sha256: str

    def to_receipt(self) -> dict[str, object]:
        """Return a JSON-safe receipt retaining exact removal order."""

        return {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "input_graph_gene_count": len(self.input_graph_gene_ids),
            "input_graph_gene_order_sha256": self.input_graph_gene_order_sha256,
            "perturbation_target_gene_count": len(self.perturbation_target_gene_ids),
            "perturbation_target_gene_ids": list(self.perturbation_target_gene_ids),
            "perturbation_target_gene_ids_sha256": self.perturbation_target_gene_ids_sha256,
            "retained_graph_gene_count": len(self.retained_graph_gene_ids),
            "retained_graph_gene_ids": list(self.retained_graph_gene_ids),
            "retained_graph_gene_order_sha256": self.retained_graph_gene_order_sha256,
            "removed_non_target_gene_count": len(self.removed_non_target_gene_ids),
            "removed_non_target_gene_ids": list(self.removed_non_target_gene_ids),
            "removed_non_target_gene_ids_sha256": self.removed_non_target_gene_ids_sha256,
            "identifier_matching": "exact_case_sensitive",
            "missing_non_target_policy": "remove_preserving_canonical_order",
            "missing_perturbation_target_policy": "fail_before_model_construction",
            "fill_policy": "forbidden",
        }


@dataclass(frozen=True)
class GenePTOrderedMatrix:
    """Ordered float32 embedding matrix and hashes for later model integration."""

    gene_ids: tuple[str, ...]
    values: np.ndarray[Any, Any]
    gene_order_sha256: str
    matrix_sha256: str
    source_sha256: str


def _validate_embedding_object(
    value: object,
    *,
    expected_entry_count: int,
    expected_width: int,
) -> dict[str, list[float]]:
    if type(value) is not dict:
        raise ValueError("GenePT emb_b must be an exact dict[str, list[float]]")
    raw = cast(dict[object, object], value)
    if len(raw) != expected_entry_count:
        raise ValueError(
            "GenePT emb_b entry count differs: "
            f"expected {expected_entry_count}, observed {len(raw)}"
        )
    for key, vector in raw.items():
        if type(key) is not str or not key or key != key.strip():
            raise ValueError("GenePT emb_b keys must be non-empty exact strings")
        if type(vector) is not list:
            raise ValueError(f"GenePT embedding for {key!r} must be an exact list[float]")
        vector_list = cast(list[object], vector)
        if len(vector_list) != expected_width:
            raise ValueError(
                f"GenePT embedding width for {key!r} differs: "
                f"expected {expected_width}, observed {len(vector_list)}"
            )
        for index, item in enumerate(vector_list):
            if type(item) is not float or not math.isfinite(item):
                raise ValueError(f"GenePT embedding {key!r}[{index}] must be a finite exact float")
    return cast(dict[str, list[float]], raw)


def verify_genept_emb_b(path: str | Path) -> GenePTArtifact:
    """Verify the exact approved artifact before deserializing its pickle payload."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"GenePT emb_b artifact is missing: {source}")
    observed_sha256 = sha256_file(source, chunk_size=8 * 1024 * 1024)
    if observed_sha256 != GENEPT_EMB_B_SHA256:
        raise ValueError(
            "GenePT emb_b SHA-256 differs; refusing to deserialize unapproved pickle: "
            f"expected {GENEPT_EMB_B_SHA256}, observed {observed_sha256}"
        )
    with source.open("rb") as handle:
        payload = pickle.load(handle)
        if handle.read(1):
            raise ValueError("GenePT emb_b pickle contains trailing bytes")
    embeddings = _validate_embedding_object(
        payload,
        expected_entry_count=GENEPT_EMB_B_ENTRY_COUNT,
        expected_width=GENEPT_EMB_B_WIDTH,
    )
    return GenePTArtifact(
        source_path=source.resolve(),
        source_sha256=observed_sha256,
        source_size_bytes=source.stat().st_size,
        embeddings=embeddings,
        entry_count=len(embeddings),
        embedding_width=GENEPT_EMB_B_WIDTH,
    )


def _require_unique_gene_ids(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    ordered = tuple(values)
    if not ordered or any(type(value) is not str or not value for value in ordered):
        raise ValueError(f"{label} must contain non-empty exact strings")
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"{label} must be unique")
    return ordered


def build_genept_coverage_plan(
    artifact: GenePTArtifact,
    *,
    ordered_graph_gene_ids: Iterable[str],
    perturbation_target_gene_ids: Iterable[str],
) -> GenePTCoveragePlan:
    """Filter missing non-targets and reject every missing perturbation target."""

    graph_gene_ids = _require_unique_gene_ids(
        ordered_graph_gene_ids,
        label="ordered graph gene IDs",
    )
    targets = tuple(
        sorted(
            _require_unique_gene_ids(
                perturbation_target_gene_ids,
                label="perturbation target gene IDs",
            )
        )
    )
    graph_set = set(graph_gene_ids)
    targets_outside_graph = tuple(target for target in targets if target not in graph_set)
    if targets_outside_graph:
        joined = ", ".join(targets_outside_graph)
        raise ValueError(f"perturbation targets are absent from requested graph axis: {joined}")
    missing_targets = tuple(target for target in targets if target not in artifact.embeddings)
    if missing_targets:
        raise MissingGenePTTargetsError(missing_targets)
    retained = tuple(gene_id for gene_id in graph_gene_ids if gene_id in artifact.embeddings)
    removed = tuple(gene_id for gene_id in graph_gene_ids if gene_id not in artifact.embeddings)
    return GenePTCoveragePlan(
        schema_version="genept-emb-b-coverage-v1",
        source_sha256=artifact.source_sha256,
        input_graph_gene_ids=graph_gene_ids,
        perturbation_target_gene_ids=targets,
        retained_graph_gene_ids=retained,
        removed_non_target_gene_ids=removed,
        input_graph_gene_order_sha256=sha256_json(list(graph_gene_ids)),
        perturbation_target_gene_ids_sha256=sha256_json(list(targets)),
        retained_graph_gene_order_sha256=sha256_json(list(retained)),
        removed_non_target_gene_ids_sha256=sha256_json(list(removed)),
    )


def build_ordered_genept_matrix(
    artifact: GenePTArtifact,
    plan: GenePTCoveragePlan,
) -> GenePTOrderedMatrix:
    """Build a deterministic read-only float32 matrix in the planned graph order."""

    if plan.source_sha256 != artifact.source_sha256:
        raise ValueError("GenePT coverage plan and artifact SHA-256 differ")
    values = np.empty(
        (len(plan.retained_graph_gene_ids), artifact.embedding_width),
        dtype=np.float32,
    )
    for row, gene_id in enumerate(plan.retained_graph_gene_ids):
        vector = artifact.embeddings.get(gene_id)
        if vector is None:
            raise ValueError(f"planned GenePT gene disappeared from artifact: {gene_id}")
        values[row] = np.asarray(vector, dtype=np.float32)
    values = np.ascontiguousarray(values, dtype=np.float32)
    descriptor = {
        "schema_version": "genept-emb-b-ordered-matrix-v1",
        "source_sha256": artifact.source_sha256,
        "gene_order_sha256": plan.retained_graph_gene_order_sha256,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "dtype": values.dtype.str,
        "byte_order": "little" if values.dtype.byteorder in {"<", "="} else "big",
        "memory_order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(descriptor))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    values.setflags(write=False)
    return GenePTOrderedMatrix(
        gene_ids=plan.retained_graph_gene_ids,
        values=values,
        gene_order_sha256=plan.retained_graph_gene_order_sha256,
        matrix_sha256=digest.hexdigest(),
        source_sha256=artifact.source_sha256,
    )
