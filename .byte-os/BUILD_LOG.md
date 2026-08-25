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

## 2026-08-24 — plan 002 completed locally

- Added strict experiment config schemas and a fail-closed YAML loader that
  rejects anchors, aliases, merge keys, `defaults`, unknown fields, identity
  mismatches, and frozen-protocol drift.
- Added exactly 30 complete configs: six model IDs by five dataset IDs. Each
  file independently carries data, model, training, evaluation, artifact, and
  per-value provenance fields; no global experiment config exists.
- Froze a one-epoch integration gate for every learned model/dataset pair.
  GraD-Pert alone then permits full max-200/patience-10 training; GEARS and
  TxPert are `smoke_only` with actual `max_epochs=1`, no orchestrator early
  stopping, and shared smoke seed 1. Their model/optimizer values remain tied
  to frozen official code/configs.
- Added strict hash-linked source, canonical-data, split, 300-control,
  prediction, evaluation, run, and server-pointer manifest contracts.
- Verification: 24 tests passed; Ruff passed; config CLI reported 30 entries,
  30 unique config SHA-256 values, six models, and five datasets.

## 2026-08-24 — plans 003–006 implementation checkpoint

- Implemented five source registries, observed-to-canonical metadata mapping,
  resumable/checksummed acquisition, safe ZIP extraction, within-cell
  canonical preprocessing, condition split, and deterministic exact 300-row
  control manifests.
- Implemented deterministic dual-source Top-20 graph pruning and the frozen
  prediction/global/RingInduced view contracts, plus the native encoder,
  projector, additive predictor, losses, centers, and EMA primitives.
- Implemented matched-control, global train-delta, and condition-aware/additive
  train-delta baselines without validation/test truth input.
- Implemented three separate headline Pearson formulas with undefined-value
  reasons and equal-condition macro summaries.
- Audited TriShift commit `87ac2c51c3c266391093f71a8bce2e6beaa81518`.
  Retained thin runner/core, condition-keyed artifact, result-adapter, and
  notebook-consumer layers; rejected global config inheritance, newest-result
  discovery, truth-in-runner metrics, and runtime GEARS model/loss mutation.
- Added a trusted-root/hash-gated, atomic, condition-keyed prediction PKL that
  enforces `[300,G]`, exact row IDs/gene hashes, and recursive Truth absence.
- Added the isolated official-checkout preflight and a GEARS official API
  adapter that calls `PertData`, `GEARS.model_initialize/train/save_model`,
  and the official model forward while preserving all 300 controls. No GEARS
  model/loss is implemented locally. TxPert configs now pin the frozen official
  YAML and its SHA-256.
- Server source state: K562 and RPE1 sources are checksum-verified; Nadig HepG2
  and Jurkat official downloads are active and growing; Norman official source
  lineage remains unresolved. No formal preprocessing/training has started.
- Latest local verification: 72 tests passed, 2 Torch/PyG suites skipped only
  because those optional dependencies are absent locally; Ruff lint and format
  checks passed. Server dependency tests remain required.

## 2026-08-24 — TriShift architecture hardening checkpoint

- Followed the concrete TriShift config/runner/artifact/analysis/notebook call
  chain beyond the initial layout audit.
- Retained its useful exact-config snapshots, run metadata, per-epoch loss
  tables, small-summary/rich-artifact split, and single analysis adapter.
- Added `ResultCatalogManifest` and a typed `ResultCatalog` that pins the exact
  run manifest, server pointer, and metrics file by SHA-256. It never searches
  directories or selects by mtime.
- Catalog loading rejects untrusted-root escape, symlinks, changed files,
  unevaluated runs, multiple test evaluations, and run/pointer identity drift.
- Targeted verification: 16 artifact/manifest tests passed; strict mypy and
  Ruff passed for the touched surfaces.
- Full local regression after the catalog change: 76 passed, 2 optional
  Torch/PyG suites skipped; Ruff and the dependency-available strict mypy
  surface passed.
- HepG2 reached the exact official byte size but did not match the preregistered
  SHA-256, so it remains an unsealed `.part`. HDF5 open and independent first/
  last 1 MiB NCBI range comparisons passed; the registry hash is not changed
  until an independent full-stream checksum settles the discrepancy.
- Replaced snapshot-style official imports with a context-managed official
  module session that remains active through lazy imports and API calls.
- Added the narrow public-package API for graph/model construction, one-epoch
  Lightning fit, checkpoint save, and exact 300-control official forward. The
  fit disables validation because the frozen public validation hook also reads
  test; it does not reproduce model, loss, optimizer, or the training loop.
- Full local regression after these runner changes: 78 passed, 2 optional
  Torch/PyG suites skipped; Ruff passed.
- Implemented the evaluator-only condition-keyed `EvaluationBundle`. It copies
  sealed prediction/control arrays, joins all truth rows and references, stores
  DE/Top-DE inputs and Systema state, and recomputes all three headline metrics
  on every trusted load.
- Refactored atomic/hash-gated pickle IO into one artifact core used by both
  prediction and evaluation bundles. Full local regression: 81 passed, 2
  optional Torch/PyG suites skipped; Ruff and strict available-dependency mypy
  passed.
- Jurkat also completed at the exact NCBI byte size but disagreed with its
  preregistered SHA-256; HDF5 and independent first/last 1 MiB official range
  checks passed. Background full-stream hash audits (no duplicate file write)
  are running for both Nadig sources before any registry change or seal.
- Added deterministic small-result exports: per-condition CSV, summary CSV/JSON,
  availability JSON, and a file-hash manifest. These contain no arrays or Truth
  and are the only notebook-facing synchronized metric surface.
- Added and executed `notebooks/benchmark_results.ipynb` using nbformat/nbclient.
  It consumes only an explicit ResultCatalog, checks paired data/split/control
  hashes before comparisons, and currently renders the honest unavailable state.
- Full local regression: 83 passed, 2 optional Torch/PyG suites skipped; Ruff,
  notebook structure/execution checks, and strict available-dependency mypy
  passed.

## 2026-08-24 — plan 003 completed on the server

- Sealed all five official sources and materialized five independent canonical
  datasets: Replogle K562 essential, Replogle RPE1 essential, Nadig Jurkat,
  Nadig HepG2 and Norman.
- The complete server verifier recomputed every source/canonical H5AD hash,
  preprocessing/QC/split link, ordered expression/graph gene axis and combined
  validation/test control-manifest hash and passed for all five datasets.
- Canonical counts are recorded in `registry/prepared/README.md`; the local
  receipt mirror is about 1.9 MB. H5AD files and full ordered 300-control
  manifests remain server-only.
- Corrected the repository ignore rules to anchor large-data directories at
  repository root. This prevents `src/gradpert/data`, `src/gradpert/graphs`,
  `src/gradpert/artifacts`, their tests and `benchmarks/txpert` from being
  silently omitted from version control.
- Added a local test that verifies all available links in the five small
  receipt chains. Focused local verification: 27 passed, with two optional
  Torch/PyG modules skipped locally.
- Hardened native training receipts to reject duplicate/discontinuous step and
  validation rows after resume. The sealed test set is now claimed durably
  before its callback can access Truth, so a crash fails closed instead of
  allowing a second evaluation.

## 2026-08-24 — evidence iteration 1: shared representability and sustained capacity

- The first real official GEARS K562 integration reached its frozen data APIs,
  built 820 condition graphs, and then exposed five targets absent from the
  official default perturbation graph. Frozen source inspection confirmed that
  GEARS filters such conditions and cannot predict them.
- Added `datasets-v2`: first create the original seeded/source split, then
  remove the exact official-default-graph unsupported conditions from every
  model's train/validation/test partitions without reshuffling retained IDs.
  Each dataset registry pins exclusions, GEARS commit and both resource hashes.
- Backed up all prior receipts and recoverably moved old graph/evaluator state
  to the dated server superseded root. Refreshed all five split and exact
  300-control hash chains without changing canonical H5AD content hashes.
- Full server data verification passed. Rebuilt all five independent graph
  receipts; all retained candidate targets have at least one GO/STRING source.
  Rebuilt all five evaluator states against the new condition IDs.
- The one-step 65,536 prototype receipt was invalidated by sustained K562
  training: 31,688/32,607 MiB (97.2%) was externally observed before a safe
  SIGINT. The incomplete run and 122 logged steps were preserved. The capacity
  gate now requires 128 consecutive real steps per dataset and records a
  per-step reservation trajectory; its replacement run is active.
- Added source-tree-bound native checkpoint schema v2 and exact epoch-boundary
  resume. Resume now rejects changed config, environment, source tree, data,
  split or run metadata.
- Added an official-runner server launcher that dry-runs exact argv and
  PYTHONPATH by default, preventing isolated environments from losing access to
  the local adapter package while official model imports remain guarded.
- The replacement capacity run completed. Candidate 65,536 failed on K562 at
  step 34 with 30,641,487,872 reserved bytes; 32,768 passed K562/RPE1 but
  failed on Jurkat at step 20 with 28,651,290,624 bytes. Candidate 16,384
  completed 128 steps on all five datasets; its maximum was 24,226,299,904
  bytes on Jurkat against a 28,168,037,990-byte threshold. All five native
  configs now freeze `prototype_count=16384`.
- Archived the stale local datasets-v1 receipt mirror and seed-0 nonlearned
  results under explicit superseded roots. Added a local v2 sync-state receipt
  and a regression test proving that each frozen exclusion set reproduces the
  verified v2 partition counts and split hashes without reshuffling.
- Unified nonlearned execution with the shared seed-1 contract and added a
  full-suite-safe runner test. GEARS and TxPert now write a small one-epoch
  training receipt binding official API, configured optimizer values,
  checkpoint hash and test-isolation evidence.
- Re-read TriShift's runner/result-adapter paths. Retained its thin
  entry/core/artifact/notebook separation while keeping GraD-Pert's stricter
  self-contained config and hash-pinned catalog contracts. Corrected the active
  graph node-loss wording to the original shared-global `1 / |M|` formula.
- New server launches and small-file synchronization are blocked by the
  account/tool usage window until 2026-08-27 14:11 CST. No alternate access
  path is used; the already-running capacity session was only collected to
  normal completion.
- Re-ran the local handoff gates after the runner receipt, seed, datasets-v2,
  and TriShift architecture-alignment changes: 117 tests passed and 9 optional
  dependency/current-receipt tests skipped; all 30 standalone configs verified;
  Ruff lint and format checks passed; offline wheel and sdist builds passed.
  Local strict mypy remains unavailable for Torch/PyG modules, so the current
  source still requires the fresh server regression after synchronization.
- Removed a misleading condition-indexed interface from the masked-node loss.
  The public signature, train-step call, and test now state the implemented
  design directly: one batch-shared masked global and a uniform mean over its
  masked nodes. Added the TriShift architecture audit to the root navigation.

## 2026-08-24 — evidence iteration 2: executable matrix and sealed small sync

- Added a tested experiment-matrix core plus dry-run-first server entrypoint.
  It materializes exactly 15 learned seed-1 smokes, 15 seed-1 nonlearned runs,
  and 20 GraD-Pert full runs (five datasets x seeds 1--4) from the 30 exact
  standalone configs.
- Full execution is formal-only and requires all 15 learned smoke manifests and
  training receipts to prove one epoch, matching checkpoint/commit/config,
  evaluator completion, no test Truth during fit, and identical canonical-data,
  split and 300-control hashes across learned models for each dataset.
- Added a native training receipt matching the official-runner surface. Existing
  outputs are skipped only after exact receipt validation; incomplete output
  fails closed, and resume is explicit and limited to native full runs.
- Added allowlisted small-file staging with default `small_results` selection
  and explicit-root mode for dataset/QC/capacity receipts. It rejects symlinks,
  path escape, binary extensions, per-file/total size excess, mutation during
  copy, and missing/extra/changed files after transfer.
- Updated stale server execution, harness, codebase-map and technical-spec
  documentation to the implemented command surfaces.
- Verification: 131 tests passed and 9 optional dependency/current receipt tests
  skipped; Ruff lint/format passed; all 30 configs have unique hashes; strict
  mypy passed on 19 non-Torch modules; offline wheel/sdist build passed.

## 2026-08-24 — evidence iteration 3: Truth lifetime and snapshot identity

- Review found that both official runners excluded test rows from fit inputs but
  constructed the canonical test reader before fit. Moved test-reader creation
  after official fit, checkpoint hash and training receipt in both runners.
- Added a parameterized source-lifecycle contract proving the ordering for
  GEARS and TxPert. Prediction remains inside the guarded official-module
  session so lazy official imports remain bound to the frozen checkout.
- Development source inspection now rejects a declared snapshot commit that
  differs from an actual Git worktree HEAD; added a real Git fixture regression.
- Updated foundation-era harness/AGENTS/OKR statements and recorded three
  iteration files plus a truthful pre-server `block` review.
- Verification: 134 tests passed, 9 honest skips; Ruff lint/format passed;
  focused official/execution/server tests passed; no forbidden native imports,
  credential pattern or large tracked result was found. Server regression and
  formal experiment receipts remain required.
- Added the final formal ResultCatalog planner/sealer. It consumes only an
  explicit 45-entry source spec, defaults to no-write planning, validates exact
  coordinates/source/fairness/native-config/metric-schema/denominator gates,
  and writes a trusted SHA sidecar only with `--execute`.
- Re-executed `notebooks/benchmark_results.ipynb` through nbclient. It now calls
  the strict final loader and continues to render an honest unavailable state
  while the formal catalog is absent.
- Final local handoff regression: 137 tests passed and 9 honest skips; Ruff,
  strict mypy over 20 available-dependency modules, offline build, Byte state,
  notebook structure/execution and Git diff checks passed.

## 2026-08-24 — authorized publication preflight

- Received explicit approval to publish the current repository, completed the
  credential/large-file review, preserved hash-bound CRLF audit CSVs through
  path-scoped Git attributes, and created snapshot commit `06fb363`.
- GitHub rejected the first push before creating `main`: the available HTTPS
  OAuth credential can write repository content but lacks the additional
  `workflow` scope required to publish `.github/workflows/ci.yml`. SSH is not
  configured and the GitHub CLI token is expired; no CI file was removed and no
  history was rewritten.
- CI preflight found and repaired two latent failures before retry: the project
  requires Python 3.12 but the workflow included 3.10, and the workflow ran the
  full suite/mypy without installing the data/model/analysis extras. The revised
  job uses Python 3.12, a hash-pinned official uv setup action plus uv 0.12.5,
  `uv sync --locked --all-extras --dev`, hash-pinned checkout, read-only
  repository permission, concurrency cancellation, and a 30-minute timeout.
- After the user completed GitHub authentication, the same HTTPS Git credential
  successfully created public `origin/main`. A read-only `ls-remote` check
  proved both local `HEAD` and the remote ref were
  `5a9109f47470a3b85bcb937097fc092d79226ae3` before this status-only follow-up.

## 2026-08-24 — live server recovery and clean regression

- Re-tested the connection instead of relying on the earlier cooldown notice:
  port 22, BatchMode SSH, `/data/yilangliu`, disk, and both RTX 5090 GPUs all
  passed. The prior 2026-08-27 access estimate is superseded.
- Preserved the existing non-Git storage root and `dev-snapshots/current` by
  cloning public `main` into the new `/data/yilangliu/GraD-Pert/source`
  checkout. Local, GitHub, and server matched clean commit `1ec6bd6`.
- Installed all locked extras in `source/.venv`; Torch 2.13 CUDA detected both
  GPUs. Full server verification reached 153 passed and three honest pending
  datasets-v2 receipt-sync skips; Ruff, all 62 source files under strict mypy,
  isolated wheel/sdist build, all 30 configs, diff check, and Git clean passed.
- An initial `build --no-isolation` attempt correctly failed because `wheel`
  is a build-system dependency rather than a runtime/dev dependency. Re-running
  the repository's specified isolated `python -m build` command passed.

## 2026-08-24 — formal smoke dry-run preflight

- Verified all five server datasets with the full canonical data verifier; all
  are `canonical_ready` with explicit canonical-data, split and ordered
  300-control manifest hashes.
- Verified the five GEARS and five TxPert configurations through the guarded
  official-checkout preflight. Both detached upstream worktrees are clean and
  match their frozen commits.
- Created the dedicated GEARS official cache root and hard-linked the two
  already-present resources only after their SHA-256 values matched every
  dataset registry entry.
- The 15-task formal smoke planning command wrote no run and selected no task,
  but revealed that `Path.resolve()` erased all three virtualenv identities by
  converting their Python symlinks to `/usr/bin/python3.12`.
- Replaced interpreter canonicalization with existence/execute validation that
  preserves the requested virtualenv path. Added a subprocess regression test;
  the focused launcher/matrix suite passes 8 tests and Ruff passes.
- Synchronized commit `1412939` and reran the complete server gate: 154 tests
  passed, three pending-receipt tests skipped, Ruff and strict mypy passed,
  isolated wheel/sdist build passed, all 30 configs verified and the tree stayed
  clean. The repaired dry run preserved all three virtualenv interpreters.
- The first selected K562 task reached the formal source gate but direct server
  `git ls-remote` failed before any GPU work or receipt files inside the run.
  The matrix correctly wrote `state=failed`, `returncode=1`.
- Established an SSH remote dynamic SOCKS endpoint on server loopback only;
  through it, the same server checkout observed GitHub `main=1412939` live.
  Hardened internal Git calls with `GIT_TERMINAL_PROMPT=0` and a 30-second
  timeout; 13 focused identity/matrix/launcher tests and Ruff pass locally.

## 2026-08-24 — first formal matrix evidence and CUDA checkpoint repair

- Froze local, public GitHub and server source at `28e859a`; the server passed
  156 tests with three honest datasets-v2 receipt-sync skips, Ruff, format,
  strict mypy on all 62 source files, and a clean worktree.
- Completed all 15 nonlearned runs at that commit. Repository validators proved
  five runs for each baseline, exactly one formal test evaluation per run, all
  three metric rows with valid denominators, 15 successful task receipts, and
  identical canonical/split/300-control identities within each dataset.
- K562 completed all 1,346 one-epoch training steps and sealed `best.pt`,
  `last.pt`, validation and training receipts, then failed before prediction at
  `torch.set_rng_state`: loading the checkpoint with a CUDA map location had
  also moved its saved CPU RNG ByteTensor to CUDA. The queue then rejected the
  incomplete run root and stopped as designed.
- Interrupted RPE1 before the same deterministic post-training failure and
  archived both incomplete runs, logs and receipts without deletion under the
  server's dated superseded root.
- The repair keeps CUDA-mapped model/optimizer tensors intact while explicitly
  validating and moving saved CPU/CUDA RNG tensors to CPU for PyTorch's RNG
  restoration APIs. Added a CUDA checkpoint load regression; local Ruff,
  format and diff checks pass, while the Torch-free local test is honestly
  skipped pending the server regression.

## 2026-08-25 — formal-v2 GEARS DE-ranking repair

- At clean commit `c240157`, all five GraD-Pert runs, all 15 nonlearned runs,
  and GEARS K562 had valid manifests. GEARS RPE1 and Jurkat completed one
  optimization epoch and then failed in the frozen official
  `gears.inference.compute_metrics` because their adapted H5AD files lacked
  `rank_genes_groups_cov_all`; the upstream fallback emitted one sentinel DE
  index and SciPy rejected length-one Pearson inputs.
- Changed only the isolated GEARS adapter to call the frozen official
  `PertData.new_data_process` DE-ranking path on the already test-free
  train+validation AnnData. The official model, loss, optimizer, split, and
  common 300-control evaluation remain unchanged. Adapter receipts now state
  the DE-ranking truth scope.
- Added a focused API regression asserting that official DE calculation is
  enabled and the canonical test loader is still removed before fit. All ten
  local benchmark tests pass; Ruff lint/format and diff checks pass.
- The ordinary local Python environment lacks notebook, mypy, and build
  dependencies. A frozen uv sync was attempted but stopped after prolonged
  package-mirror retries; no dependency file changed. Full pytest, strict mypy,
  build, and the real GEARS integration remain pending in the existing server
  environments.
- Commit `dc8e24a` subsequently passed the server gate with 165 tests, three
  honest skips, Ruff lint/format, strict mypy over 73 files, isolated build and
  a clean tree. GEARS K562 completed under the new ranking path. RPE1 and
  Jurkat then failed before training because two and six retained conditions
  respectively have one cell, and frozen Scanpy refuses a t-test for them.
- The follow-up still calls the frozen official condition-name, rank-by-covariate
  and dropout/nonzero functions. It excludes only singleton groups from the
  undefined t-test, never from the shared data/split, and fills their official
  internal metric indices with the stable full gene order. The common evaluator
  remains authoritative and already labels singleton DE metrics unavailable.
- Commit `5f82d73` then completed GEARS K562 but RPE1/Jurkat again failed in
  frozen `compute_metrics` with length-one DE vectors. Direct inspection of the
  sealed H5ADs found complete 6,386/6,506-gene singleton fallback rankings and
  20/20 top-gene membership, ruling out ranking corruption. Frozen source
  inspection showed `new_data_process(skip_calc_de=True)` creates and persists
  PyG graphs before the adapter attaches rankings, leaving its one-index
  missing-DE sentinel in the cache. The new repair calls frozen `get_DE_genes`
  for condition names, attaches rankings, and only then calls frozen
  `PertData.new_data_process` once. It gates every retained condition on 20 DE
  indices without a second expensive graph build or any local index mutation.

## 2026-08-25 — TxPert frozen-Anndata null compatibility

- Clean commit `dddc767` completed GEARS K562, RPE1, Jurkat, and Norman, proving
  the ranking-before-graph repair. TxPert RPE1 then failed before model creation
  because frozen Anndata 0.11.4 has no reader for Anndata 0.13.2's explicit
  `null` encoding at `/uns/log1p/base`; the other queue was stopped and the
  entire lineage preserved before any mixed-commit continuation.
- A server synthetic cross-version check reproduced the failure with
  `base=None`. Removing only that key made the frozen reader load the same
  `[1.0, 2.0]` expression row and `(1, 2)` shape. The adapter now applies that
  exact transformation only to its test-free cache copy, records the removed
  path, and preserves non-null bases. Six focused TxPert/API/lifecycle tests,
  Ruff, format, and diff checks pass locally.
- The `c127380` hard gate failed before `_write_official_cache`: the full stack
  showed `CanonicalTrainingData` opening the immutable canonical H5AD in frozen
  Anndata first. Inspection of Anndata 0.13.2 identified its exact registered
  reader for HDF5 `IOSpec("null", "0.1.0")`: return `None`. The follow-up
  registers that same reader in the frozen process before canonical loading,
  records the action, and still strips the null from the later adapter copy.
  No H5AD source or environment package is changed; seven focused tests pass.
