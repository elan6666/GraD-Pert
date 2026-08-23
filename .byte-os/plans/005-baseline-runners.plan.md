---
id: 005
title: Nonlearned and Isolated Learned Baseline Runners
status: pending
wave: 3
updated_at: 2026-08-24T01:20:00Z
owner_role: Benchmark Engineer
depends_on: [002, 003]
start_directory: benchmarks
context_files: [AGENTS.md, docs/design/DATA_AND_EVALUATION.md, docs/provenance/REFERENCE_ALIGNMENT.md]
agents_context_stack: [AGENTS.md, benchmarks/AGENTS.md]
subagent_policy: none
---

# Goal

Produce truth-free common-protocol predictions for three nonlearned baselines,
GEARS, and public TxPert on each canonical dataset.

# OKR Link

KR4 all models enter evaluator; KR2 exact condition parity.

# Scope

Train-only delta models, isolated environment locks, data/split adapters,
val-only training, 300-control per-row inference, prediction artifacts/receipts.

# Non-Goals

No upstream source modification, native-package import of upstream models, or
paper-best/private-graph claim.

# Steps

## Step 1: Implement nonlearned runners

- Purpose: establish transparent floors.
- Actions: matched controls, global training delta, general/additive train
  delta; capabilities and train-only assertions; five configs each.
- Files/modules: `src/gradpert/baselines`, tests.
- Expected output: `[300,G]` prediction-only artifacts.
- Verification: hand-computed synthetic conditions including Norman doubles.
- Subagent: none.

## Step 2: Pin external environments and contracts

- Purpose: isolate license/dependency behavior.
- Actions: exact commits, environment locks, preflight, subprocess JSON
  protocol, no-truth input schema, consumed split/coverage receipts.
- Files/modules: `benchmarks/gears`, `benchmarks/txpert`, tests.
- Expected output: reproducible isolated shells.
- Verification: commit/environment/schema preflight-only tests.
- Subagent: none.

## Step 3: Build GEARS runner

- Purpose: train official architecture on exact data/splits and preserve 300 rows.
- Actions: custom data, exact split injection, max 200/patience 10 common val
  selection, per-control forward, condition/gene/hash verification.
- Files/modules: `benchmarks/gears`.
- Expected output: five-dataset capable prediction artifacts.
- Verification: tiny fixture, upstream-symbol golden behavior, no `.predict()` mean path.
- Subagent: none.

## Step 4: Build TxPert runner

- Purpose: benchmark frozen public code without test leakage.
- Actions: exact split PKLs, custom train entry/subclass disabling test-on-val,
  official/preregistered hyperparameter receipts, public graph labels, shared
  control inference, prediction adapter.
- Files/modules: `benchmarks/txpert`.
- Expected output: five-dataset capable public-code benchmark artifacts.
- Verification: validation cannot access test fixture; split/shape/gene hashes.
- Subagent: none.

# Dependencies

Canonical datasets/manifests and server CUDA environments.

# Scoped Commands

- Test: `python -m pytest -q tests/baselines tests/benchmarks`
- Lint: `python -m ruff check src/gradpert/baselines benchmarks tests/baselines tests/benchmarks`
- Typecheck: native baselines only.
- Build: each runner `--preflight-only`, then tiny `--smoke`.

# AGENTS.md Context

- Root and `benchmarks/AGENTS.md`.
- Safe boundaries: upstream names allowed only here; no source edits; no truth.

# Subagent Plan

- Primary implementation; isolated review later.

# Code Change Guardrails

Do not reuse TriShift monkeypatches, call GEARS mean-only predict, or claim exact
TxPert paper reproduction.

# Acceptance Criteria

All five configs accepted; actual split IDs/hashes match; `[300,G]` predictions;
test truth inaccessible; complete upstream/environment receipts.

# Verification

Tiny local/server smoke and common artifact validation.

# Experiment Or Measurement

Condition/gene coverage, params, runtime/memory, early-stop epoch.

# Risks

Official TxPert training recipe gaps and dependency conflicts.

# Notes

External runner Python imports are permitted only inside their isolated envs.

