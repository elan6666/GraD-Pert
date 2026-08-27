---
id: 017
status: complete
wave: 1
depends_on: []
updated_at: 2026-08-28T03:15:00+08:00
---

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
- [x] Repeat the GEARS K562 hard gate in a fresh namespace with the verified
  loopback SOCKS variables inherited by the runner; the first `e68712e` launch
  stopped before model construction when `git ls-remote` timed out.
- [x] The subsequent `e867c69` GEARS K562 hard gate passed one epoch, exact
  fairness and metric checks, retained only `model.pt`, and left zero PKL.
- [x] Repair the observed TxPert split-cache boundary: after official fitting,
  hash-check and remove only `splits/train_test_split.pkl` and
  `splits/subgroup.pkl`, receipt the cleanup, then repeat the exact external
  matrix in a fresh synchronized lineage.
- [x] The `2bf2771` external matrix completed all five GEARS and all five TxPert
  one-epoch coordinates with both queues returning zero, exact fairness and
  three-metric contracts, selected checkpoints, and zero persistent PKL across
  the complete formal run root.
- [x] The small-result allowlist includes resolved YAML configs; a real formal
  staging dry run exposed and now regression-tests this previously missing
  extension.
- [x] Real ordered-ID manifests require at most 6.42 MiB per file and the two
  sealed lineage selections require 100.26 and 118.10 MiB total. Defaults are
  bounded at 8 MiB per file and 128 MiB total, with the per-file rejection
  boundary regression-tested.

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
- The `e867c69` external queue then proved TxPert's adapter cache also persisted
  two reconstructible split PKLs. The whole-run postcondition rejected RPE1
  after evaluation, the other queue was stopped, and the exact failed/interrupted
  evidence was preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260826-e867-txpert-split-pkl`.
- Commit `2bf2771` closed that boundary. Its TxPert RPE1 hard gate and remaining
  nine-task queues completed exactly ten external one-epoch coordinates under
  `/data/yilangliu/GraD-Pert/runs/formal-v3-zero-pkl-2bf2`; both queues returned
  zero and the complete run root contained zero PKL after success.
- The user-authorized historical cleanup removed 126 non-active experiment PKLs
  (381,290,534,496 bytes; 355.10 GiB) from old run/superseded roots while
  excluding the active formal root and all data/environment caches. The sealed
  server receipt is
  `/data/yilangliu/GraD-Pert/receipts/pkl-cleanup-20260826/cleanup-receipt.json`
  with SHA-256
  `321318f97be9088f9a2f001bf6208e75fe29c10be8f7cb656cc945b4a7252217`.
- The repaired small-result stages were executed and verified on both server and
  Mac: 195 external files and 325 retained-c240 files, with file-list SHA-256
  `2dd0b89886e28c503369981585d4500e3dd62a87dd327ac20a831bd7ce1ec8a7`
  and `925009e128bec63e7c5bfb20ed6bb6c54054f655bd30e55f9329d2540b05bed0`.
  The combined audit passed exact 30 coordinates and all five cross-model
  fairness identities; all 15 learned checkpoint hashes were revalidated.
- The 20 retained c240 coordinates honestly lack the newer
  `inference_recipe.json` because they predate this artifact policy. Their
  remaining small evidence and five learned checkpoints are preserved, but the
  missing recipe is a documented historical limitation rather than backfilled.
