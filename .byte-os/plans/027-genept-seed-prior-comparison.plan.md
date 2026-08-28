---
id: 027
title: GenePT-Seed exact-axis prior comparison
status: in_progress
wave: 27
updated_at: 2026-08-28T21:35:00+08:00
owner_role: Research Engineer
depends_on: [026]
start_directory: src/gradpert
context_files: [AGENTS.md, docs/experiments/GENEPT_SEED_PRIOR_COMPARISON.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Compare latest GenePT, GenePT-Seed, and GenePT-Seed+GO-EXP as exact-axis text
priors in one frozen Nadig Jurkat B2-vNext coordinate.

# OKR Link

Extends the vNext GenePT ablation from unavailable `emb_b` targets to audited
full-axis NPZ priors without changing the model/training/evaluation contract.

# Scope

Three self-contained configs, exact artifact/hash/axis checks, zero-row audit,
seed 1, ten epochs, and the existing metrics-only evaluator.

# Non-Goals

No new trainer, no five-dataset sweep, no multi-seed inference, no test tuning,
and no claim that the 60 zero rows in latest GenePT are official embeddings.

# Steps

## Step 1: Harden exact-axis prior receipts

- Purpose: expose missing official prior rows explicitly.
- Actions: record exact all-zero gene IDs and residual policy in run receipts.
- Files or modules: `src/gradpert/features/text_prior.py`,
  `src/gradpert/execution/native.py`, focused tests.
- Expected output: exact-axis receipt includes zero-row count and IDs.
- Step verification: focused text-prior tests, Ruff, format, and mypy.
- Subagent: none.

## Step 2: Generate three self-contained configs

- Purpose: freeze the only intended experimental difference.
- Actions: render configs with pinned prior paths and SHA-256 values while
  preserving the A0 graph, model, split, losses, seed, budget, and evaluator.
- Files or modules: `scripts/ablations/`, `configs/experiments/`, experiment doc.
- Expected output: three independently loadable configs and a comparison manifest.
- Step verification: config loader tests and cross-config diff audit.
- Subagent: none.

## Step 3: Execute and compare on the server

- Purpose: obtain matched downstream evidence.
- Actions: synchronize one clean commit, run the standard pilot entrypoint,
  validate run roots, and stage only small metrics/receipts.
- Files or modules: server run roots and compact synced evidence.
- Expected output: three complete metrics-only result bundles.
- Step verification: exact-commit gates, strict artifact checks, and fairness hash audit.
- Subagent: none.

# Dependencies

Pinned 2,809-row prior artifacts from `/data/yilangliu/GenePT-Seed`.

# Scoped Commands

- Test: `uv run --no-sync python -m pytest -q tests/features/test_text_prior.py tests/config`
- Lint: `uv run --no-sync ruff check src/gradpert scripts/ablations tests/features`
- Typecheck: server `python -m mypy src`
- Build: server `python -m build --no-isolation`

# AGENTS.md Context

- Root context: `AGENTS.md`.
- Module context: none; root and harness cover active modules.
- Safe edit boundaries: features, native receipt, experiment configs/scripts/docs/tests.

# Subagent Plan

Unavailable in this environment; execution remains sequential because config,
artifact, and credential boundaries overlap.

# Code Change Guardrails

Keep the common training protocol byte-identical; no normalization or fallback
may be applied to only one prior condition.

# Acceptance Criteria

All three runs pass exact-axis/hash checks and share canonical data, split,
graph, seed, training, evaluator, and control-manifest identities.

# Verification

Focused/full code gates, server artifact audit, cross-condition receipt diff,
and current review.

# Experiment Or Measurement

Report TxPert macro Pearson delta, TriShift Pearson delta, Systema Pearson,
runtime, zero-prior rows, and exact artifact hashes.

# Risks

One seed cannot support uncertainty claims; differing embedding widths remain
part of the embedding-model comparison and are absorbed by the same projection.
