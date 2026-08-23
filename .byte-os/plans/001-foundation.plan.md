---
id: 001
title: Repository and Package Foundation
status: in_progress
wave: 1
updated_at: 2026-08-24T01:20:00Z
owner_role: Tech Lead
depends_on: []
start_directory: .
context_files: [AGENTS.md, CLAUDE.md, .byte-os/TECH_SPEC.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Create the Git-backed, installable, strict Python project shell and verified
developer harness without changing frozen upstream evidence.

# OKR Link

KR1 clean install/test/build; KR6 three-way source parity foundation.

# Scope

Git initialization/remote, `pyproject.toml`, package/CLI shell, test layout,
tool configs, CI, module context, license/citation/readme, source-integrity tests.

# Non-Goals

No model, data download, baseline execution, or formal server job.

# Steps

## Step 1: Initialize source control safely

- Purpose: make GitHub the canonical source plane.
- Actions: initialize `main`, add the verified empty remote, audit nested repos,
  ensure reference/large paths are ignored, and create no submodule accidentally.
- Files or modules: `.git`, `.gitignore`, `.gitmodules` only if intentionally needed.
- Expected output: local repository with correct origin and cleanly enumerable files.
- Step verification: `git remote -v`; `git status --short`; `git check-ignore` fixtures.
- Subagent: none.

## Step 2: Create package and verification shell

- Purpose: establish one reproducible entrypoint.
- Actions: create strict `pyproject.toml`, `src/gradpert`, CLI, version module,
  tests/fixtures, Ruff/mypy/pytest/build settings, and deterministic JSON helper.
- Files or modules: `pyproject.toml`, `src/gradpert`, `tests`, `README.md`.
- Expected output: editable install and buildable wheel/sdist.
- Step verification: install, `pytest`, Ruff, mypy, build, CLI help.
- Subagent: none.

## Step 3: Complete module harness

- Purpose: keep future edits scoped and safe.
- Actions: add local context for modeling and benchmarks; update map/harness/audit
  with real commands; add static tests protecting frozen repos and native names.
- Files or modules: module `AGENTS.md`/`CLAUDE.md`, `.byte-os/*`, `tests/contracts`.
- Expected output: ready navigation and enforceable source boundaries.
- Step verification: harness audit and static contract tests.
- Subagent: none.

# Dependencies

Python runtime and Git/GitHub authentication already available.

# Scoped Commands

- Test: `python -m pytest -q tests/contracts`
- Lint: `python -m ruff check .`
- Typecheck: `python -m mypy src`
- Build: `python -m build`

# AGENTS.md Context

- Root context: `AGENTS.md`
- Module context: created by this plan where it reduces navigation cost.
- Scoped command source: `.byte-os/HARNESS.md`
- Safe edit boundaries: do not modify nested upstream checkout or binary paths.
- Missing or stale AGENTS.md notes: add real module commands after scaffolding.

# Subagent Plan

- Exploration subagents: none.
- Implementation subagents: none.
- Review subagents: later plan 008.
- Isolation boundaries: primary agent owns all foundation files.
- Merge or handoff notes: commit only after full foundation checks.

# Code Change Guardrails

No force push, no rewriting nested reference history, no dependency download
into Git, no upstream-named native model classes.

# Acceptance Criteria

- Clean local build/test/lint/typecheck.
- Correct empty GitHub remote configured and first commit pushed without force.
- Root/module context and ignored-path tests pass.

# Verification

Capture commands, versions, commit, and hashes in BUILD_LOG.

# Experiment Or Measurement

None.

# Risks

Nested `.git` trees can be accidentally committed as gitlinks; explicitly audit.

# Notes

Remote is currently verified empty, but recheck immediately before first push.
