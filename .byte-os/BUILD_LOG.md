# Build Log

## 2026-08-24 — plan 001 started

- Initialized root Git repository on `main` and configured
  `https://github.com/elan6666/GraD-Pert.git` as `origin` after verifying the
  remote was empty.
- Excluded the two local upstream checkouts and temporary PDF render from the
  project tracking surface.
- Added Python package/build/test/lint/typecheck shell, read-only doctor CLI,
  canonical hash helper, source-boundary tests, CI, and local module context.
- Local verification:
  - editable install with `--no-build-isolation`: passed;
  - pytest: 8 passed;
  - Ruff format check: passed;
  - Ruff lint: passed;
  - wheel build: passed, SHA-256
    `64ff0a2ff57827e2a9c60d766daca782a39ce8659b74ff51e5da44a972695c19`;
  - CLI version/doctor JSON: passed;
  - mypy: not run locally because the tool download did not complete; no pass
    is claimed. CI/server verification remains required.
- Git staging audit confirmed no `TxPert/`, `tmp/`, data, run, checkpoint, or
  artifact path is included.
