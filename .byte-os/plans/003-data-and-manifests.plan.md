---
id: 003
title: Five-Dataset Preparation and Fairness Manifests
status: pending
wave: 2
updated_at: 2026-08-24T01:20:00Z
owner_role: Data Engineer
depends_on: [002]
start_directory: .
context_files: [AGENTS.md, docs/design/DATA_AND_EVALUATION.md, .byte-os/RESEARCH.md]
agents_context_stack: [AGENTS.md]
subagent_policy: read_only_exploration
---

# Goal

Implement idempotent server-side acquisition, canonicalization, QC, split,
control sampling, GEARS conversion, and readiness for all five datasets.

# OKR Link

KR2 all five canonical-ready; KR6 server receipts and server-only assets.

# Scope

Source registry/checksums/licenses, download resume, four within-cell pipelines,
Norman audit, QC, canonical splits, 300-control manifests, GEARS custom views.

# Non-Goals

No model training and no cross-cell feature construction.

# Steps

## Step 1: Freeze data sources

- Purpose: eliminate moving/ambiguous inputs.
- Actions: resolve official source URL, filename, checksum, size, license, and
  raw/processed semantics for all five; block unresolved sources explicitly.
- Files or modules: `registry/datasets`, provenance/data docs.
- Expected output: versioned source registry with no placeholder accepted as ready.
- Step verification: HEAD/checksum/license metadata probes and registry tests.
- Subagent: read_only_exploration allowed for source evidence only.

## Step 2: Implement preparation/QC

- Purpose: produce consistent canonical expression and metadata.
- Actions: resumable download, safe extraction, target-signal filtering,
  normalize/log/HVG, label/gene validation, expression-scale audits, QC receipts.
- Files or modules: `src/gradpert/data`, `tests/data`.
- Expected output: idempotent `prepare/status/verify` commands.
- Step verification: synthetic/raw/processed fixtures and corrupt archive tests.
- Subagent: none.

## Step 3: Freeze split and evaluation controls

- Purpose: guarantee model-independent fairness.
- Actions: implement 0.5625/0.1875/0.25 condition split excluding control,
  Norman official split import, hash/overlap gates, exact per-condition 300 row
  sampling with replacement and stable seed derivation.
- Files or modules: split/control services and manifests.
- Expected output: canonical IDs/hashes reused by all runners.
- Step verification: deterministic/golden/overlap/with-replacement tests.
- Subagent: none.

## Step 4: Build GEARS-compatible views and server readiness

- Purpose: use identical canonical biology across runners.
- Actions: build custom AnnData paths, inject exact split, report condition/gene
  drops, compare Replogle published assets, seal five readiness receipts.
- Files or modules: data adapters, `benchmarks/gears/data_adapter.py`.
- Expected output: five `canonical_ready` datasets and GEARS coverage reports.
- Step verification: server QC and hash summary; zero unexplained condition loss.
- Subagent: none.

# Dependencies

Plan 002 contracts. Server/network for formal materialization.

# Scoped Commands

- Test: `python -m pytest -q tests/data`
- Lint: `python -m ruff check src/gradpert/data tests/data benchmarks/gears/data_adapter.py`
- Typecheck: `python -m mypy src/gradpert/data`
- Build: `gradpert data prepare --all`; `gradpert data verify --all`

# AGENTS.md Context

- Root context: `AGENTS.md`
- Module context: benchmark local context if created.
- Scoped command source: `CODEBASE_MAP.md`.
- Safe edit boundaries: server data outside Git; no upstream package call in native downloader.
- Missing or stale notes: update exact server paths after materialization.

# Subagent Plan

- Exploration: source evidence only, read-only.
- Implementation: primary agent owns all data code.
- Review: plan 008.
- Isolation: no model code edits.
- Handoff: readiness/index receipts to plans 005/007.

# Code Change Guardrails

Never substitute cross-cell cache for independent within-cell data. Never mark
unknown scale/checksum/license ready. Safe archive extraction only.

# Acceptance Criteria

All five have canonical files, hashes, QC, split and control manifests; exact
condition/gene coverage is known and train gates pass.

# Verification

Local synthetic suite plus server receipt/checksum verification.

# Experiment Or Measurement

Report cell/gene/condition/control counts, retained/dropped targets and memory.

# Risks

Public source availability and license/checksum gaps may hard-block individual
datasets; preserve blocked state without substitution.

# Notes

Formal data remains server-only.

