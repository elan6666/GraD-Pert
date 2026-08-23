---
id: 004
title: Native GraD-Pert B2 Model and Training
status: pending
wave: 2
updated_at: 2026-08-24T01:20:00Z
owner_role: ML Engineer
depends_on: [001, 002]
start_directory: src/gradpert/modeling
context_files: [AGENTS.md, docs/design/GRADPERT_V1.md, docs/provenance/REFERENCE_ALIGNMENT.md]
agents_context_stack: [AGENTS.md, src/gradpert/modeling/AGENTS.md]
subagent_policy: none
---

# Goal

Implement the standalone B2 graph/view/loss/prediction stack, resumable trainer,
and synthetic end-to-end evidence without importing upstream models.

# OKR Link

KR1 local verification; KR3 complete model/EMA/center/resume/prediction.

# Scope

Graph pruning/views, adaptive encoder, projector, losses, teacher state, basal/
decoder, optimizer/early stop/checkpoint, training control pairing, CLI smoke.

# Non-Goals

No B3, ablation flag, alternate backbone/decoder, or real formal training.

# Steps

## Step 1: Graph and view system

- Purpose: implement the exact active topology/view semantics.
- Actions: deterministic Top-20, self-loops, DropEdge, global masks, eight
  RingInduced locals, anchor mask selection and structured view statistics.
- Files or modules: `src/gradpert/graphs`, modeling views, tests.
- Expected output: deterministic seeded GraphViewBatch.
- Step verification: source-separated Top-K and view golden tests.
- Subagent: none.

## Step 2: Encoder, prediction, and projector

- Purpose: materialize the active architecture and gradient routes.
- Actions: shared embeddings, two four-layer GATv2 towers, node-adaptive fusion,
  additive basal decoder, projector, teacher clone/no-grad.
- Files or modules: `src/gradpert/modeling`.
- Expected output: typed shape-stable modules under native names.
- Step verification: shape/init/additivity/forbidden-name/gradient tests.
- Subagent: none.

## Step 3: Loss and state updates

- Purpose: reproduce verified centered/EMA mechanics in graph semantics.
- Actions: 18 condition pairs, masked-node objective, embedding-spread loss,
  independent centers, step schedules, optimizer→EMA→center order.
- Files or modules: modeling losses/state/training.
- Expected output: finite loss breakdown and exact state lifecycle.
- Step verification: numerical, gradient, center, EMA/order tests.
- Subagent: none.

## Step 4: Trainer, resume, and inference

- Purpose: run B2 reliably.
- Actions: control pairing, AdamW config, max 200/patience 10, best/last
  checkpoint, deterministic condition inference with manifest controls, receipts.
- Files or modules: `src/gradpert/training`, CLI, tests.
- Expected output: synthetic train-resume-predict flow.
- Step verification: uninterrupted/resumed equivalence and end-to-end smoke.
- Subagent: none.

# Dependencies

Torch/PyG for full tests; lightweight modules should import safely when optional
dependencies are absent.

# Scoped Commands

- Test: `python -m pytest -q tests/graphs tests/modeling tests/training`
- Lint: `python -m ruff check src/gradpert/graphs src/gradpert/modeling src/gradpert/training`
- Typecheck: `python -m mypy src/gradpert/graphs src/gradpert/modeling src/gradpert/training`
- Build: `gradpert smoke model`

# AGENTS.md Context

- Root context: `AGENTS.md`
- Module context: `src/gradpert/modeling/AGENTS.md`
- Scoped command source: module context and `CODEBASE_MAP.md`.
- Safe edit boundaries: native code only; upstream checkout read-only.
- Missing notes: none after plan 001.

# Subagent Plan

- All roles: primary agent; final review isolated in plan 008.

# Code Change Guardrails

No upstream imports/classes, no hidden ablation branches, no test tuning, no
silent K/batch fallback.

# Acceptance Criteria

All specified unit/golden/gradient/resume/synthetic tests pass and public module
names are GraD-Pert-native.

# Verification

Capture CPU/local dependency constraints separately from server GPU evidence.

# Experiment Or Measurement

Synthetic loss/gradient/entropy and deterministic-output health report.

# Risks

PyG availability and memory; server plan owns formal fit.

# Notes

The active spec is authoritative over historical alternatives.

