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
from gradpert.features.text_prior import (
    GENEPT_PROTEIN_REACTOME_SIGNOR_GENE_COUNT,
    GENEPT_PROTEIN_REACTOME_SIGNOR_MODEL,
    GENEPT_PROTEIN_REACTOME_SIGNOR_SHA256,
    GENEPT_PROTEIN_REACTOME_SIGNOR_WIDTH,
    TextPriorArtifact,
    verify_text_prior_npz,
)

__all__ = [
    "GENEPT_EMB_B_ENTRY_COUNT",
    "GENEPT_EMB_B_SHA256",
    "GENEPT_EMB_B_WIDTH",
    "GENEPT_PROTEIN_REACTOME_SIGNOR_GENE_COUNT",
    "GENEPT_PROTEIN_REACTOME_SIGNOR_MODEL",
    "GENEPT_PROTEIN_REACTOME_SIGNOR_SHA256",
    "GENEPT_PROTEIN_REACTOME_SIGNOR_WIDTH",
    "GenePTArtifact",
    "GenePTCoveragePlan",
    "GenePTOrderedMatrix",
    "MissingGenePTTargetsError",
    "TextPriorArtifact",
    "build_genept_coverage_plan",
    "build_ordered_genept_matrix",
    "verify_genept_emb_b",
    "verify_text_prior_npz",
]
