---
id: 002
title: Contracts, Config Matrix, and Artifact Schemas
status: complete
wave: 1
updated_at: 2026-08-24T01:39:02Z
owner_role: Research Platform Engineer
depends_on: [001]
start_directory: .
context_files: [AGENTS.md, .byte-os/ENGINEERING_RULES.md, docs/design/DATA_AND_EVALUATION.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Implement strict, hashable experiment/data/run contracts and the complete 30-
file self-contained configuration matrix.

# OKR Link

KR2 split/data gates; KR4 common evaluator provenance; KR7 config completeness.

# Scope

Pydantic/dataclass schemas, canonical JSON/hash utilities, YAML restrictions,
five dataset specs, six model IDs, 30 resolved configs, artifact/run manifests.

# Non-Goals

No real data, learned training, or metrics beyond schema fixtures.

# Steps

## Step 1: Implement strict schemas and hashing

- Purpose: make all downstream equality checks executable.
- Actions: add IDs/enums, provenance values, canonical JSON SHA-256, data/split/
  control/config/run/prediction/evaluation/pointer manifests, finite-value rules.
- Files or modules: `src/gradpert/contracts`, `src/gradpert/config`.
- Expected output: round-trippable typed contracts with stable hashes.
- Step verification: schema and hash golden tests.
- Subagent: none.

## Step 2: Enforce self-contained YAML

- Purpose: prevent hidden global config behavior.
- Actions: reject YAML aliases/anchors/merge/defaults/unknown/missing fields;
  attach source provenance per hyperparameter; store exact bytes and hash.
- Files or modules: config loader/validator, `tests/contracts`.
- Expected output: fail-closed config validation.
- Step verification: malicious/incomplete/inherited config fixtures fail.
- Subagent: none.

## Step 3: Materialize the matrix

- Purpose: one auditable file for every model/dataset pair.
- Actions: write five configs for each of six models; include exact split/run/
  eval seeds, per-model smoke/full execution policy, official or preregistered provenance, server
  artifact policy, metric registry version.
- Files or modules: `configs/experiments/**`.
- Expected output: exactly 30 resolved YAML files.
- Step verification: `gradpert config verify --all` asserts count and matrix.
- Subagent: none.

# Dependencies

Plan 001 package shell.

# Scoped Commands

- Test: `python -m pytest -q tests/contracts tests/config`
- Lint: `python -m ruff check src/gradpert/contracts src/gradpert/config tests/contracts tests/config`
- Typecheck: `python -m mypy src/gradpert/contracts src/gradpert/config`
- Build: `gradpert config verify --all --json`

# AGENTS.md Context

- Root context: `AGENTS.md`
- Module context: none unless created in foundation.
- Scoped command source: `CODEBASE_MAP.md`.
- Safe edit boundaries: no global config or implicit defaults.
- Missing or stale AGENTS.md notes: none.

# Subagent Plan

- Exploration/implementation/review subagents: none in this plan.
- Isolation boundaries: contracts/configs only.
- Merge notes: all 30 files land atomically with completeness test.

# Code Change Guardrails

Do not make optional fields a path for hidden experimental behavior. Every
training value must be present and provenance-labeled.

# Acceptance Criteria

Stable hashes, strict parser, 30 complete configs, no upstream runtime import,
and full negative-test coverage.

# Verification

Save config matrix report and hash index.

# Experiment Or Measurement

None.

# Risks

YAML parsers can resolve anchors before validation; scan tokens/nodes first.

# Notes

`project_preregistered` is honest metadata, not a validation error.

Completed locally with 30/30 independently resolved YAML files, strict
composition/path/protocol rejection, hash-linked manifest contracts, and
execution-policy tests. Every learned pair declares a one-epoch gate; only
GraD-Pert declares full max-200/patience-10 execution. External configs pin
frozen official packages/configuration and are smoke-only. The formal server
run will seal exact commit/environment/data hashes into run manifests.
