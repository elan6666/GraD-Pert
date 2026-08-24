---
id: 007
title: Server Synchronization and Five-Dataset Experiment Matrix
status: in_progress
wave: 4
updated_at: 2026-08-24T18:42:00+08:00
owner_role: Research Operations Engineer
depends_on: [003, 004, 005, 006]
start_directory: scripts/server
context_files: [AGENTS.md, docs/design/SERVER_EXECUTION.md, .byte-os/ENGINEERING_RULES.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Run formal compute on the server from the exact GitHub commit, retain large
artifacts there, and bring back only verified small results/pointers.

# OKR Link

KR2–KR6 formal evidence and source/artifact topology.

# Scope

Server clone/envs/preflight, five data gates, K-head fit, every learned
model/dataset one-epoch gate, GraD-Pert-only full runs, nonlearned/evaluation,
and small result sync.

# Non-Goals

No local formal compute, force sync, large download to Mac, or unplanned sweep.

# Steps

## Step 1: Establish exact source and environments

- Purpose: make server execution attributable.
- Actions: first commit/push, non-destructive server clone at resolved path,
  three-way SHA/clean preflight, native/GEARS/TxPert env locks, CUDA smoke.
- Files/modules: server scripts, environment locks, receipts.
- Expected output: repeatable clean server preflight.
- Verification: SHA equality, import/CUDA/PyG tests, no active collision.
- Subagent: none.

## Step 2: Materialize and verify data

- Purpose: finish all five readiness gates where sources permit.
- Actions: run plan 003 commands, checksum/QC/split/control/coverage, store large
  assets only on server, sync small readiness summaries.
- Files/modules: server data root and small local results.
- Expected output: five canonical-ready receipts or exact hard upstream blocker.
- Verification: `data verify --all` and manifest hashes.
- Subagent: none.

## Step 3: Select K-head and run all one-epoch gates

- Purpose: prevent expensive invalid matrix launch.
- Actions: worst-case full-step candidates under 85% usable memory; freeze one;
  run GraD-Pert and the official GEARS/TxPert packages for exactly one epoch on
  each of five datasets; seal/recompute predictions and verify hashes.
- Files/modules: server run roots, fit registry, receipts.
- Expected output: global K and 15 passed learned integration receipts.
- Verification: memory, gradient, early-stop, artifact and metric health gates.
- Subagent: none.

## Step 4: Run formal matrix

- Purpose: fulfill five-dataset comparison.
- Actions: GraD-Pert × five datasets × seeds 1–4 at max 200/patience 10, three
  nonlearned models per dataset, common inference/evaluation; do not launch
  full GEARS/TxPert runs; resume idempotently and never overwrite sealed runs.
- Files/modules: server-only artifacts and small summaries.
- Expected output: complete run/metric index with explicit failures.
- Verification: split/control/config/code hashes equal and recomputation passes.
- Subagent: none.

## Step 5: Sync small evidence

- Purpose: support local review without moving large artifacts.
- Actions: dry-run allowlist, inspect paths/bytes, copy JSON/CSV/TXT/MD only,
  create pointers with server checksums and reproduction commands.
- Files/modules: `results/` small files/pointers.
- Expected output: auditable local result index, no large files.
- Verification: extension/size/tree audit and pointer resolution check.
- Subagent: none.

# Dependencies

All local implementation and server/network availability.

# Scoped Commands

- Test: `python -m pytest -q tests/server`
- Lint: `python -m ruff check scripts/server tests/server`
- Typecheck: N/A
- Build: server preflight, matrix/status, sync-small dry-run.

# AGENTS.md Context

- Root context and server design.
- Safe boundary: explicit paths, no broad destructive operations, no secrets.

# Subagent Plan

- Primary agent only for remote mutation/launch; review reads receipts.

# Code Change Guardrails

Abort on commit/config/hash mismatch, dirty tree, unknown output collision, failed
data gate, unexplained condition drop, or memory threshold violation.

# Acceptance Criteria

Formal server receipts and complete or honestly failed matrix; no large local
artifacts; exact source/pointer evidence.

# Verification

Current GPU/disk/job audit before every launch and final sealed index audit.

# Experiment Or Measurement

Training/validation histories, epochs stopped, peak memory/runtime, per-condition
metrics, four-seed mean/std and availability.

# Risks

Long runtime, upstream downloads, environment incompatibility, and public recipe
gaps. Resume and failure receipts are mandatory.

# Notes

Do not call a launched job complete until artifacts and metrics are verified.

Local execution-layer evidence is complete: the exact 15/15/20 task phases,
dry-run and deliberate execution surfaces, completion identity validation,
15-smoke full-run dependency gate, cross-model fairness hashes, and sealed
small-file staging/verification are implemented and tested. Plan status remains
pending until the actual server matrix and result synchronization complete.
