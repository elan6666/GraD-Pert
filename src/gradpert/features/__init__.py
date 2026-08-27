"""Audited external feature inputs for native GraD-Pert models."""

from gradpert.features.genept import (
    GENEPT_EMB_B_ENTRY_COUNT,
    GENEPT_EMB_B_SHA256,
    GENEPT_EMB_B_WIDTH,
    GenePTArtifact,
    GenePTCoveragePlan,
    GenePTOrderedMatrix,
    MissingGenePTTargetsError,
    build_genept_coverage_plan,
    build_ordered_genept_matrix,
    verify_genept_emb_b,
)

__all__ = [
    "GENEPT_EMB_B_ENTRY_COUNT",
    "GENEPT_EMB_B_SHA256",
    "GENEPT_EMB_B_WIDTH",
    "GenePTArtifact",
    "GenePTCoveragePlan",
    "GenePTOrderedMatrix",
    "MissingGenePTTargetsError",
    "build_genept_coverage_plan",
    "build_ordered_genept_matrix",
    "verify_genept_emb_b",
]
