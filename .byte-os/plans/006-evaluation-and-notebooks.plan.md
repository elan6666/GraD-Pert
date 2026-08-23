---
id: 006
title: Shared Evaluation, Artifacts, and Notebook Handoff
status: pending
wave: 3
updated_at: 2026-08-24T01:20:00Z
owner_role: Evaluation Engineer
depends_on: [002, 003, 004, 005]
start_directory: src/gradpert/evaluation
context_files: [AGENTS.md, docs/design/DATA_AND_EVALUATION.md, docs/provenance/REFERENCE_ALIGNMENT.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Seal prediction artifacts, join truth only in the evaluator, compute frozen
metrics, export small summaries and read-only notebook examples.

# OKR Link

KR4 metric union/availability; KR5 reproducible artifacts/notebooks.

# Scope

Artifact IO/sealing, evaluator join, three Pearson metrics plus applicable
registry, macro aggregation, CSV/JSON/Parquet/PKL/H5AD, notebook provenance.

# Non-Goals

No model training/splitting in evaluator or notebooks; no local big artifact sync.

# Steps

## Step 1: Artifact storage and trust

- Purpose: make outputs independently verifiable.
- Actions: trusted PKL writer/loader with checksum/schema; array sidecars if
  needed; atomic sealing; pointer manifests; prediction truth-absence check.
- Files/modules: `src/gradpert/artifacts`, tests.
- Expected output: condition-keyed round trips and corruption rejection.
- Verification: checksum/schema/path traversal/untrusted input tests.
- Subagent: none.

## Step 2: Implement metric registry

- Purpose: preserve exact distinct semantics.
- Actions: three Pearson families, MSE/R2/DE variants and applicability,
  undefined reasons, condition-equal macro, source/version metadata.
- Files/modules: `src/gradpert/evaluation`, fixtures.
- Expected output: deterministic per-condition and summary tables.
- Verification: frozen TxPert/TriShift golden vectors and edge cases.
- Subagent: none.

## Step 3: Truth join and export

- Purpose: enforce test access boundary and reusable outputs.
- Actions: join canonical truth/control/DE, validate every hash, write EvaluationBundle,
  summaries, coverage/failures/run manifests and recompute command.
- Files/modules: evaluator CLI/services.
- Expected output: metrics reproduced without model environment.
- Verification: sealed bundle recomputation byte/numeric equivalence.
- Subagent: none.

## Step 4: Notebook and small-result interface

- Purpose: support analysis without hidden computation.
- Actions: one artifact-inspection notebook and one comparison notebook, paired
  with scripts/tests; local pointer/small file loading and missing-state display.
- Files/modules: `notebooks`, analysis helpers, docs.
- Expected output: deterministic reports from frozen data only.
- Verification: execute notebooks on synthetic small bundle and inspect outputs.
- Subagent: none.

# Dependencies

All prediction runners and canonical data contracts.

# Scoped Commands

- Test: `python -m pytest -q tests/artifacts tests/evaluation tests/notebooks`
- Lint: `python -m ruff check src/gradpert/artifacts src/gradpert/evaluation notebooks tests`
- Typecheck: `python -m mypy src/gradpert/artifacts src/gradpert/evaluation`
- Build: `gradpert smoke evaluation`; notebook execution command.

# AGENTS.md Context

- Root context; no local module file unless plan 001 adds one.
- Safe boundary: runner artifacts have no truth; notebooks cannot mutate canon.

# Subagent Plan

- Primary agent implementation; review later.

# Code Change Guardrails

No NaN-to-zero coercion, metric aliasing, unsafe arbitrary PKL, or test-derived
data entering runner/training.

# Acceptance Criteria

Golden metrics pass; artifact recomputation matches; notebook is read-only;
availability reasons and macro denominators are explicit.

# Verification

Record fixture hashes and notebook execution receipt.

# Experiment Or Measurement

Metric drift against frozen implementations equals zero within tolerance.

# Risks

Definition ambiguity; resolve only from frozen code/provenance registry.

# Notes

Large evaluator artifacts stay server-only.

