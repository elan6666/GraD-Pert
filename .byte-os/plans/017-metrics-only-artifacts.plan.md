# Plan 017 — Metrics-only default and optional single result PKL

## Objective

Make persistent large prediction output opt-in. Every run always preserves the
best checkpoint, exact ordered 300-control IDs, truth IDs, hashes, inference
recipe, and small metrics. Explicit full export writes one `result.pkl` with a
shared deduplicated control-expression pool.

## Locked behavior

- Every self-contained experiment config explicitly selects `metrics_only` or
  `single_pkl`; all 30 configs default to `metrics_only`.
- Prediction hashes and ordered control IDs are sealed before Truth access.
- `metrics_only` leaves no persistent PKL.
- The zero-PKL postcondition scans the entire successful run root, including
  official-runner checkpoint and adapter directories.
- `single_pkl` is the only persistent PKL and contains exact predictions,
  shared unique control rows, per-condition ordered 300 control indices/IDs,
  Truth, DE state, metrics, and provenance.
- Repeated control draws and their order remain reconstructable exactly.
- Existing historical `evaluation-pkl-v1` remains readable; new materialization
  uses `result-pkl-v1`.
- Training, model, split, control draws, metrics, and update semantics do not change.

## Verification

- [x] Config validation covers the explicit mode in all 30 files.
- [x] Artifact tests prove shared-row deduplication, ordered reconstruction,
  duplicate draws, content-hash validation, and legacy loading.
- [x] Execution tests prove default runs retain recipes/small metrics but no PKL,
  while opt-in runs retain exactly `artifacts/result.pkl`.
- [x] Execution tests reject framework-created PKL outside the result directory;
  GEARS explicitly retains only `model.pt` after hashing and removing its
  reproducible `config.pkl` and `custom_split.pkl` metadata.
- [x] Local pytest: 178 passed, 9 honest environment/receipt skips.
- [x] Local Ruff check/format, compileall, diff check, and package build passed.
- [x] The synchronized server checkout reached the isolated wheel/sdist build
  through a `set -e` pytest/Ruff/format/strict-mypy gate at `e68712e`.
- [ ] Repeat the GEARS K562 hard gate in a fresh namespace with the verified
  loopback SOCKS variables inherited by the runner; the first `e68712e` launch
  stopped before model construction when `git ls-remote` timed out.

## Execution boundary

This source/config change invalidates any new formal lineage. Do not resume
server training until a clean synchronized commit and fresh execution contract
are established. Historical outputs remain immutable.

## Server launch evidence

- The first `e68712e` hard-gate attempt never entered training and persisted no
  PKL. Its exact namespace and launch log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-e687-git-ls-remote-timeout`;
  the log SHA-256 is
  `b10de4fa65993e1b49dd16e09bcd16fbaa3808b7bc7ae4c04dc7aa8a18b2ea89`.
- The observed failure was the formal publication check timing out on direct
  GitHub access. After renewing the loopback reverse SOCKS forward, the same
  bounded `git ls-remote --exit-code origin refs/heads/main` check returned the
  exact public commit `e68712e7f3bb6b95250302f1b38f1b35c23544f9`.
- The next formal namespace must encode the verified proxy environment in its
  immutable launch contract and child-process environment; it must not reuse
  the failed namespace or artifacts.
