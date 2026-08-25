---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: building
current_workflow: byte-build
next_workflow: byte-review
review_verdict: block
iteration_count: 3
harness_status: ready
hard_blocked: false
updated_at: 2026-08-25T16:52:00+08:00
---

# Status

- Current execution policy supersedes the earlier full-run plan: all learned
  coordinates stop after exactly one epoch, and no 100-epoch/full task may be
  launched. Goal mode is paused while the server matrix is running; it resumes
  only after all 30 one-epoch/nonlearned coordinates pass strict validation.
- Formal-v2 at clean commit `c240157` has valid manifests for all five
  GraD-Pert runs, all 15 nonlearned runs, and GEARS K562. GEARS RPE1 and Jurkat
  completed their optimization epoch but failed inside the frozen official
  `compute_metrics`: their source AnnData lacks DE rankings, the current
  `skip_calc_de=True` path emits a one-index sentinel, and SciPy Pearson rejects
  vectors shorter than two. Both queues stopped fail-closed with no full task.
- Plan 009 is the bounded repair: run the frozen official GEARS DE-ranking path
  on the already test-free train+validation AnnData, preserve the failed c240
  evidence, and establish a fresh benchmark-runner lineage. The existing c240
  GraD-Pert Nadig Jurkat run remains the immutable optimization B0 and will not
  be rerun.
- First repair commit `dc8e24a` passed all server quality gates and its GEARS
  K562 run completed, but RPE1/Jurkat then exposed two/six singleton
  train+validation conditions for which frozen Scanpy refuses a t-test. The
  follow-up keeps every shared condition, uses the official ranking unchanged
  for rankable groups, and supplies singleton groups a stable full-gene order
  only for GEARS' internal one-epoch bookkeeping. Shared evaluation continues
  to mark singleton DE metrics unavailable.
- Follow-up commit `5f82d73` proved the singleton ranking maps themselves are
  complete, but also exposed ordering inside the frozen API: with
  `skip_calc_de=True`, `new_data_process` builds its PyG cache before the
  adapter attaches those maps. RPE1/Jurkat consequently retained one-index
  pre-ranking graphs and failed after the epoch; K562 passed only via its older
  correctly ranked cache. That lineage and all three involved cache directories
  are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-5f82-pre-ranking-graphs`.
  Plan 009 now attaches rankings before the single frozen official graph build
  and verifies exactly 20 DE indices per retained condition before fit.
- Commit `dddc767` validated that repair on GEARS K562, RPE1, Jurkat, and
  Norman. The first TxPert task then exposed a separate storage compatibility
  issue: frozen Anndata 0.11.4 cannot read Anndata 0.13.2's explicit null
  encoding at `/uns/log1p/base`. The stopped lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-dddc-txpert-anndata-null`.
  Plan 010 removes only that adapter-copy key when its value is null; a real
  server cross-version fixture proves the sanitized file preserves expression
  values and loads successfully in the frozen environment.
- The `c127380` hard gate showed the exception actually occurs earlier, while
  frozen Anndata opens the immutable canonical H5AD before an adapter copy is
  built. That stopped lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-c127-canonical-null-reader`.
  The follow-up registers the inspected Anndata 0.13.2 `null` 0.1.0 read
  semantics (`return None`) in the frozen runner's in-memory registry before
  `CanonicalTrainingData`, then retains the adapter-copy write sanitation. It
  does not edit the canonical file or frozen environment.
- Commit `9207e8c` passed that canonical-read boundary and reached the frozen
  official TxPert data module, then failed on its first CUDA `torch.cat`. The
  isolated official environment uses Torch 2.6/cu124, whose wheel supports only
  through `sm_90`, while both server RTX 5090 GPUs require `sm_120`. The failed
  root and log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-9207-txpert-sm120`.
  Plan 011 keeps GraD-Pert and GEARS environments unchanged and builds a new
  TxPert-only Torch 2.7/cu128 environment from the frozen official lock plus a
  narrow committed CUDA override. The runner will reject the environment unless
  exact versions, CUDA 12.8, `sm_120`, a core CUDA kernel, and a PyG extension
  kernel all pass before data/model work.
- 2026-08-25 execution override: GraD-Pert full runs now use `max_epochs=100`
  and train/evaluation batch size 256. The ad6 full gate/final watcher were
  stopped before any full task launched. Completed ad6 smoke/nonlearned outputs
  remain evidence but the native capacity gate and learned smoke matrix must be
  rerun at the next clean commit before full training.
- Performance work in progress: batch disconnected graph views per encoder
  call and reuse the two actual branch-backward traversals for gradient
  diagnostics, without changing losses, view identities, gradient ownership,
  optimizer→Teacher EMA→center order, splits, or evaluation.
- First batch-256 capacity attempt at `4d68da40` exhausted all frozen prototype
  candidates but emitted no structured failure receipt. The log is retained;
  the gate is being repaired to atomically persist every rejected candidate and
  terminal failure before any memory-policy change is considered.
- The repaired receipt at `3e366fd` proved allocator reservation fragmentation:
  at 8,192 prototypes K562 peaked at 18.17GB allocated but 27.39GB reserved,
  just above the 26.71GB threshold. The next evidence-led iteration freezes
  expandable segments and caps the speed-first candidate set at 16,384/8,192.
- The same-code 64-vs-256 comparison is complete and selects batch 256. Both
  passed five datasets × 128 real steps at 16,384 prototypes. Batch 256 reduced
  estimated epoch time by at least 3.223×, retained at least 24.55% headroom
  below the 85%-of-initial-free threshold, and raised cell throughput by
  3.84--4.04×. The hash-pinned decision is tracked at
  `registry/capacity/batch_comparison.development.json` and uses no test metric.

- Goal mode: paused during current server training; no active Codex goal.
- Project state: root Git repository and standalone package shell exist; plans
  002 and 003 are complete, and native model plan 004 is active.
- Discussion source: Codex task `01a00ab3-9864-7032-98d9-45f6d0016838`, `TxPert/grad-pert`, and `.byte-os/DISCUSSION.md`.
- Confirmed scope: all five datasets through `canonical_ready`, training, evaluation, and manifests.
- Training route: one B2 configuration; no B3 and no ablation matrix.
- Benchmarks: GEARS, TxPert, matched-control mean, global train delta, and general train delta with Norman additive seen singles.
- Fairness: one canonical condition split/evaluation manifest; paired run seeds and shared 300-control evaluation draws.
- Metric contract: frozen union with three distinct Pearson headlines: TxPert macro delta, TriShift delta, Systema Pearson.
- Artifact contract: versioned condition-level PKL plus Parquet/JSON/H5AD; notebooks consume artifacts only.
- Execution policy: all 15 learned model/dataset integrations stop after exactly
  one epoch under the current user amendment; no model continues to a full run.
- Server topology: formal compute only on `/data/yilangliu`; local/GitHub/server must share one clean commit; large artifacts remain server-only.
- Historical batch-64 hardware decision: two RTX 5090 with 32607 MiB each; the sustained global
  gate selected `K_head=16384`. Candidate 65,536 failed on K562 at step 34,
  32,768 failed on Jurkat at step 20, and 16,384 completed 128 real steps on
  each dataset. Its worst peak reserved memory was 24,226,299,904 bytes on
  Jurkat, below the 28,168,037,990-byte threshold. This receipt does not qualify
  the new batch-256 implementation; a fresh capacity receipt is required.
- Harness: ready; Claude context ready; Codex context ready; `AGENTS.md` ready.
- Config contract: exactly 30 self-contained model/dataset YAML files verified with 30 unique byte hashes; no global config/default/merge/inheritance path.
- Evidence iteration 1 migrated all five splits to the frozen
  official GEARS default-graph representability intersection without
  reshuffling retained conditions. Data/hash verification, graph rebuild and
  evaluator-state rebuild passed. The sustained `K_head` refit is complete and
  the five native configs now freeze 16,384.
- Publication gate: cleared. Explicit approval was received on 2026-08-24, the
  complete source and CI history was pushed to public `origin/main`, and local
  `HEAD` and the remote ref were byte-identical at `5a9109f` immediately before
  this status update. No CI file was removed and no history was rewritten.
- Server source gate: restored and verified. The non-Git storage root remains
  untouched; public `main` was cloned to the new clean checkout
  `/data/yilangliu/GraD-Pert/source`, whose HEAD matches local/GitHub
  `a8a7247c3ac54e570f67cf0b392397e9309abb66` before the current launcher fix.
- Data state: all five official sources remain sealed and all five datasets are
  `canonical_ready` under `datasets-v2`. The canonical H5AD hashes did not
  change; split/control hashes did. Server verification passed for all five,
  and prior split/evaluation/graph receipts are recoverable under
  `/data/yilangliu/GraD-Pert/superseded/20260824-gears-default-intersection`.
- Local prepared/result hygiene: prior datasets-v1 and seed-0 small artifacts
  are preserved under explicit `superseded/` roots. Current v2 receipt chains
  are marked pending sync rather than being inferred from terminal output.
- Local verification: 137 tests pass and 9 optional Torch/PyG/anndata or
  pending-v2-receipt tests skip; all 30 self-contained configs, Ruff
  lint/format, the 20-module available-dependency strict mypy surface, and the
  offline wheel/sdist build pass. Full local strict mypy cannot complete because
  the intentionally lightweight local environment has no Torch/PyG; server
  policy/config/type gates passed before the latest resume/launcher additions,
  so a fresh complete server regression remains required after synchronization.
- Evidence iteration 2 closed the server-operations implementation gap. The
  dry-run-first matrix now freezes 15 learned smoke, 15 paired nonlearned and
  20 native full tasks. Full runs are machine-gated on exact one-epoch training
  receipts, checkpoints, no test Truth during fit, shared commit/config
  identities, and per-dataset equality of canonical-data/split/300-control
  hashes across all three learned models.
- Small-file synchronization now has a sealed staging layer: only allowlisted,
  size-bounded, non-symlink files from named `small_results` or one explicit
  receipt root can enter a new staging tree, and the transferred tree must
  reverify with no missing, extra or changed files.
- Evidence iteration 3 strengthened the leakage and source boundaries. GEARS
  and TxPert now finish fit, checkpoint hashing and training receipt sealing
  before the canonical test reader exists; development Git worktrees reject a
  declared commit that differs from HEAD. The current pre-server review verdict
  is `block`, with exact unresolved formal-execution evidence listed under
  `.byte-os/reviews/2026-08-24-pre-server-review.md`.
- The final notebook surface is now executable rather than implicit. An
  explicit-source, dry-run-first builder seals a catalog only for the exact 45
  formal coordinates (all six models at paired seed 1 plus native seeds 2--4),
  one source commit, shared per-dataset fairness hashes, one native config per
  dataset, and exact three-metric schemas/denominators. The benchmark notebook
  was re-executed and uses the strict final loader; no formal catalog is claimed
  before the actual server results exist.
- TriShift architecture audit: source-level call chain recorded in
  `docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md`; the explicit hash-pinned
  ResultCatalog and evaluator-only condition bundle are implemented and tested,
  without newest-result discovery or runner-side Truth.
- Notebook handoff: the read-only benchmark notebook is generated and executed;
  it reports no result until an explicit hash-pinned catalog is synchronized.
- Fresh server regression: pass on the clean checkout with Torch 2.13 CUDA
  enabled on two RTX 5090 GPUs: 153 tests passed, three datasets-v2 receipt-sync
  tests skipped, Ruff passed, strict mypy passed on all 62 source files,
  isolated wheel/sdist build passed, all 30 configs verified, and Git remained
  clean. The earlier access-window notice is superseded by this live evidence.
- Fresh data gate: all five server datasets passed full `data verify --all` as
  `canonical_ready`; their canonical-data, split and ordered 300-control hashes
  were observed directly from the storage root.
- Official-runner gate: all five GEARS and all five TxPert configs passed the
  guarded official-checkout preflight at commits `f374e43` and `08d82ee`.
  The two GEARS graph resources were hard-linked into the dedicated official
  cache only after their hashes matched all five registry entries.
- The first exact 15-task smoke dry run selected no execution and exposed a
  launcher bug before training: resolving interpreter symlinks collapsed every
  isolated Python path to `/usr/bin/python3.12`. The launcher now validates but
  preserves each virtualenv path, with a subprocess regression test.
- The repaired launcher passed the fresh server gate at `1412939`: 154 tests,
  three honest pending-receipt skips, Ruff, strict mypy on 62 source files,
  isolated build, all 30 configs, clean tree and a 15-task no-write dry run.
- The first deliberate K562 smoke attempt failed before training because the
  server's direct `git ls-remote` path to GitHub timed out. Its empty run,
  failed matrix receipt and log are retained for superseding; no GPU work or
  scientific artifact was produced.
- A loopback-only remote SOCKS forward on the SSH ControlMaster restored live
  GitHub verification and returned public `main=1412939`. Git identity commands
  now also disable interactive prompts and fail closed after 30 seconds.
- Next action: publish/synchronize the bounded Git identity fix, archive the
  failed empty attempt, repeat all gates at the final frozen commit, then rerun
  the first exact one-epoch GraD-Pert smoke through the verified SSH tunnel.
- Commit `28e859a` passed the complete server regression and source-parity gate.
  All 15 nonlearned runs completed and their run identities, formal lifecycle,
  three-metric schemas/denominators, task receipts, and per-dataset fairness
  hashes passed strict verification. These results remain evidence but must be
  superseded and rerun if the final source commit changes.
- The first real native K562 smoke completed all 1,346 training steps and wrote
  both checkpoints, then failed before its single test evaluation while loading
  `best.pt`: `torch.load(map_location=cuda)` had moved the saved CPU RNG tensor
  to CUDA, while `torch.set_rng_state` requires a CPU ByteTensor. GPU0 stopped
  fail-closed; RPE1 was deliberately interrupted before it could encounter the
  same deterministic failure. Both incomplete runs, logs and receipts are
  recoverably archived under
  `/data/yilangliu/GraD-Pert/superseded/20260824-checkpoint-rng-device-failure`.
- A bounded repair now validates RNG tensor types and moves CPU and CUDA RNG
  states back to CPU before PyTorch restoration. A CUDA checkpoint regression
  reproduces the server path. Local Ruff/format/diff checks pass; the local
  environment has no Torch, so server CUDA regression and a fresh full gate are
  required before the next frozen formal launch.
