---
id: 008
title: Quality Review, Three Iterations, and Delivery
status: pending
wave: 5
updated_at: 2026-08-24T17:56:09+08:00
owner_role: Release Engineer
depends_on: [007]
start_directory: .
context_files: [AGENTS.md, .byte-os/PRODUCT_SPEC.md, .byte-os/TECH_SPEC.md, .byte-os/HARNESS.md]
agents_context_stack: [AGENTS.md]
subagent_policy: read_only_exploration
---

# Goal

Audit implementation/results against the frozen contract, repair evidence-led
issues through three iterations, obtain a fresh ship review, and hand off.

# OKR Link

All KRs and delivery evidence.

# Scope

Full review, code/data/result reproducibility audits, three required improvement
iterations, regression checks, docs, delivery record and exact next commands.

# Non-Goals

No new model/dataset/ablation scope or performance-driven redesign.

# Steps

## Step 1: Full review

- Purpose: find correctness/fairness/reproducibility gaps.
- Actions: inspect diff, configs, boundaries, tests, server receipts, data lineage,
  result recomputation and local sync tree; write prioritized review verdict.
- Files/modules: `.byte-os/reviews`, read-only repository/server evidence.
- Expected output: current verdict and actionable findings.
- Verification: every finding has file/evidence/reproduction.
- Subagent: read_only_exploration allowed.

## Step 2: Complete three evidence-led iterations

- Purpose: improve reliability/usability without scope drift.
- Actions: choose highest-value bounded repair per iteration, implement, run
  scoped/full regression, record before/after evidence in iterations 1–3.
- Files/modules: issue-dependent, plus `.byte-os/iterations` and BUILD_LOG.
- Expected output: three completed iteration records.
- Verification: each proves a measured improvement and no regression.
- Subagent: none unless a repair plan explicitly assigns one.

## Step 3: Fresh review and delivery

- Purpose: ensure final state, not stale pre-repair state, is shippable.
- Actions: rerun full review, require ship verdict, update README/docs/harness,
  build artifacts, final server/source/result pointer audit, DELIVERY.md.
- Files/modules: reviews, docs, `.byte-os/DELIVERY.md`, final commit/push/sync.
- Expected output: complete truthful handoff.
- Verification: clean build/test/lint/typecheck, three-way commit, artifact index.
- Subagent: read_only_exploration allowed for independent review.

# Dependencies

All build/server plans and accessible evidence.

# Scoped Commands

- Test: full `python -m pytest -q`
- Lint: Ruff format/check
- Typecheck: mypy
- Build: wheel/sdist, CLI smoke, server/source/artifact audits

# AGENTS.md Context

- Root plus any affected module context.
- Safe boundary: review agents read-only; no deferred scope promotion.

# Subagent Plan

- Exploration/review: independent read-only review allowed.
- Implementation: primary agent owns repairs.
- Isolation: no concurrent overlapping edits.
- Handoff: fresh review must inspect post-iteration commit.

# Code Change Guardrails

No greenwashing missing formal results, no stale receipts, no deletion of user
data, and no performance tuning based on test.

# Acceptance Criteria

Initial review resolved, three iteration records complete, fresh review `ship`,
all checks current, delivery record and small-result pointers available.

# Verification

Record exact commands, commits, server paths/hashes, skipped/blocked items.

# Experiment Or Measurement

Compare before/after evidence for each iteration and final metric recomputation.

# Risks

Long formal jobs or upstream data blocks can prevent a truthful ship verdict;
do not mark delivery complete while required scope remains unresolved.

# Notes

User acceptance can replace the default iteration count only if explicit.

Three evidence-led iteration records now exist. The current pre-server review
is `block`, not ship: it resolved official test-reader lifetime, development
commit binding, matrix/sync implementation and stale documentation, while
retaining the missing formal runs/source parity/catalog/final review as release
blockers. This plan remains pending until the post-result fresh review ships.
