# Build Log

## 2026-09-02 — plan 034 Protein+Reactome+SIGNOR binding and local gates

- Verified on the server that the requested `Protein+Reactome+SIGNOR`
  condition is GenePT-Seed profile `protein-pathway`, historically named
  `Seed-GO-ProteinPathway`. The canonical aligned NPZ and unaligned NPZ are
  byte-identical at SHA-256
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
- Bound the user-facing provenance name without copying or regenerating the
  17,730 by 2,048 `doubao-embedding-vision` vectors. Backward-compatible
  constants retain the upstream artifact label, and a focused test prevents
  either name from drifting to another SHA/model/shape.
- Re-rendered the 25-row config matrix; no generated YAML or matrix content
  changed, proving that the requested prior was already the E-row artifact.
- Added Plan 034 with hard module order E, then D, then L, then M and at most
  two rows active inside one module.
- Local verification passed: 579 tests with two honest skips (frozen TxPert
  evidence absent and CUDA unavailable), Ruff, format on 259 files, strict
  mypy on 76 source files, and isolated wheel/sdist build.

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

## 2026-08-25 — TxPert RTX 5090 runtime repair

- The `9207e8c` RPE1 hard gate successfully read the canonical H5AD using the
  exact null reader, then stopped in the official data module at its first CUDA
  `torch.cat`: Torch 2.6/cu124 advertises only `sm_50` through `sm_90`, while
  the RTX 5090 reports capability `sm_120`.
- Preserved that complete failed root and log under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-9207-txpert-sm120`.
- Added a TxPert-only CUDA runtime contract that remains hash-bound to the
  official commit and lock, replaces only Torch/PyG CUDA distributions with
  matching 2.7/cu128 builds, and leaves the GraD-Pert and GEARS environments
  unchanged.
- Added a dry-run-first server environment builder and fail-closed runner gate
  for exact module versions, CUDA 12.8, active/wheel `sm_120`, a core CUDA
  kernel, and a PyG extension kernel. Focused contract/filter/runtime tests are
  in place. Local verification passed 155 tests with nine honest missing
  Torch/Anndata or pending-receipt skips, Ruff lint/format, isolated build, and
  Git diff checks. Full mypy and CUDA execution remain assigned to the server
  environment because the lightweight Mac environment intentionally lacks
  Torch/PyG.
- The first server regression ran 173 tests but one contract unit test assumed
  the local historical `TxPert/official-repo` path existed inside the clean
  server checkout. Formal execution correctly uses the separate frozen
  `/upstreams/txpert` path. The unit test now uses a temporary hash-bound lock
  fixture; production lock validation remains unchanged.
- The first dry-run-approved environment build stopped during dependency
  resolution before installation: Torch 2.7 requires `sympy>=1.13.3`, while
  the official Torch 2.6 lock pins 1.13.1. Its partial 84 KiB environment and
  full log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-txpert-env-10a-sympy`. The
  hardware override now replaces that one transitive pin with exact
  `sympy==1.13.3`; no unconstrained dependency is introduced.

## 2026-08-26 — Nadig Jurkat B1/B2/B3 speed-pilot delivery

- B1 graph-only executed at clean `0a4d339`: one epoch/582 steps, 2,798 graph
  nodes, 89,561 nonself edges, 5,000 expression/output/evaluation genes, zero
  PKL, one retained checkpoint, 844.180 s training wall.
- B2 systems-only executed and strictly validated at clean `2e30fb5`: one
  epoch/582 steps, full 6,506-node/222,654-edge graph, all seven systems groups
  active, zero PKL, one retained checkpoint, 718.681 s training wall. Its
  validation receipt passed 41 checks.
- B3 combined implementation commit `44ae7ff` passed 191 local tests with 9
  honest dependency/receipt skips, Ruff, format, and isolated build. The exact
  synchronized clean server passed 214 tests with 3 honest skips, Ruff,
  format, strict mypy on 66 source files, and isolated build.
- B3 formal run completed one epoch/582 steps, matched B0/B1/B2 canonical,
  split, ordered 300-control and truth identities, used no test truth during
  fit, evaluated once, retained only `best.pt` SHA-256
  `b4bf958f9197f999eae81cca0dfec101f618c56e902a01fb7d02a3ef8ac4cdcc`,
  and left zero PKL. Strict validation passed 49/49 checks; receipt SHA-256
  `65cf90788dc6a213148b35de0685cb1216d50d127a19d21b1cd175c6801c4274`.
- Actual one-epoch walls were B1 844.180 s, B2 718.681 s, B3 507.718 s. B3
  delivered 252.63 cells/s, a 1.663x speedup over B1 and 1.416x over B2.
  Comparison receipt SHA-256
  `72f4d1b8a3c6bffc246ea3d58bd92b4ad191e9a3b4fdb7f9306456cecfdf11cf`.
- The reviewed local evidence allowlist contains five server-origin files,
  totals 56 KiB, and includes no scientific arrays, checkpoint, log, H5AD, or
  PKL. All three metrics are retained only as non-decisional one-epoch
  evidence.
- Final delivery gates passed 191 local tests with 9 honest missing-dependency/
  prepared-receipt skips, Ruff check, Ruff format check, isolated wheel/sdist
  build, and evidence-hash verification. Local strict mypy remains unavailable
  because this lightweight venv lacks Torch/PyG; the exact B3 server execution
  commit passed strict mypy on all 66 source files.

## 2026-08-26 — B0 metrics-only timing rerun preflight

- The user explicitly authorized one new B0 performance coordinate to remove
  persistent-PKL output from its timing. Historical c240 B0 remains untouched.
- Added a self-contained pilot config with canonical full graph,
  `systems_optimizations: disabled`, no systems helper flags, batch 256,
  16,384 prototypes, one-epoch smoke policy, and `result_mode: metrics_only`.
- Targeted config/artifact tests passed 20 checks. Full local gates passed 192
  tests with 9 honest missing-dependency/prepared-receipt skips, Ruff check,
  Ruff format check, and isolated wheel/sdist build. Local strict mypy remains
  unavailable because the lightweight Mac venv lacks Torch/PyG. Exact clean
  server commit `7bed1f0` passed 215 tests with 3 honest skips, Ruff, format,
  strict mypy on 66 source files, isolated build, and clean-tree verification
  before launch.

## 2026-08-26 — B0 metrics-only timing rerun execution and comparison

- The fresh namespace `/data/yilangliu/GraD-Pert/runs/pilot-b0-metrics-only-7bed`
  executed exactly once on GPU0 at source commit `7bed1f0`. It completed one
  epoch/582 optimizer steps over 128,266 cells, used no canonical test truth
  during fit, performed one test evaluation, retained only `best.pt` SHA-256
  `ad77678e62da0c73faef0c9beee08d1e6deb33f26a9ab4862423c7f43fb7e5a1`,
  and left zero persistent PKL/work directories.
- Runtime evidence recorded the canonical full 6,506-node/222,654-edge graph,
  5,000 expression/output/evaluation genes, all seven systems groups disabled,
  2,951.487 s actual training wall, 0.1972 steps/s, 43.46 cells/s, 18.53 GiB
  peak allocated GPU memory, and 25.12 GiB peak CPU RAM. The historical c240
  B0 manifest remained byte-identical.
- The independent strict validator passed 44/44 checks. Contract SHA-256 is
  `61efdb8fd51769914f60dbbe3883860282934027fd0d2037fd7cf86951df3244`;
  validator SHA-256 is
  `d3143fd7cda3b854064565549d9b969adfa473aa6d215cd4c14612d82361800a`;
  PASS receipt SHA-256 is
  `acc60de269b85e16b6164a1bd4035acc869ca2a62b183d32f09d457d18e63920`.
- The rebuilt four-variant comparison selects B3. Direct factor speedups are
  B0→B1 graph-only 3.496x, B0→B2 systems-only 4.107x, B1→B3 systems on the
  reduced graph 1.663x, and B2→B3 graph reduction with systems 1.416x.
  Combined B0→B3 speedup is 5.813x. Comparison receipt SHA-256 is
  `d4da6aac3a71cf3fcf2aba645d1c423fe1a4f52ae593a49e0f0361b0a20defe1`.
- Only reviewed small JSON/Python evidence was transferred locally; checkpoint,
  scientific matrices, H5AD, launch log, and all PKL remain server-side or
  absent. Metrics are non-decisional and no one-epoch effect-equivalence claim
  is made.
- Final local delivery gates passed 192 tests with 9 honest
  dependency/prepared-receipt skips, Ruff check, Ruff format check, isolated
  wheel/sdist build, evidence-bundle policy audit, and diff check. Local strict
  mypy remains dependency-limited because the Mac venv lacks Torch/PyG; the
  exact public delivery commit therefore requires the full server mypy gate.
- Public delivery commit `167e31a` synchronized exactly to the clean server.
  Server gates passed 215 tests with 3 honest prepared-receipt skips, Ruff,
  format, strict mypy on 66 source files, isolated wheel/sdist build, and
  clean-tree verification. The initial isolated-build attempt inherited the
  loopback SOCKS variables and failed because its temporary venv lacked SOCKS
  support; after the already-completed public identity check, rerunning only
  the isolated build without proxy variables succeeded.

## 2026-08-26 — B2 exact 10-epoch pilot preparation

- Added a separate self-contained Nadig Jurkat B2 10-epoch config using the
  canonical full graph, all seven systems groups, seed 1, batch 256, 16,384
  prototypes, `metrics_only`, and zero-persistent-PKL policy.
- Added a bounded `fixed_epoch_pilot` config policy and `gradpert model pilot`
  lifecycle. It requires exactly 10 configured epochs and seed 1, validates
  every epoch, ignores early-stop termination, and does not relax existing
  one-epoch smoke or 100-epoch formal constraints.
- Targeted config tests passed 18 checks with one honest missing-Torch skip.
  Full local pytest passed 193 tests with 9 honest dependency/receipt skips;
  Ruff, format, isolated wheel/sdist build, and diff check passed. The exact
  synchronized server commit still requires Torch-backed tests and strict mypy
  before launch.
# 2026-08-26 — Plan 021 B2 default and 200-epoch full budget

- Changed all five formal `gradpert_b2` configs to the canonical full graph,
  all seven semantics-preserving systems groups, `max_epochs=200`,
  validation-only `patience=10`, batch 256, 16,384 prototypes, expandable
  allocator, and `metrics_only` zero-PKL output.
- Updated config/trainer guards and matrix tests from the superseded 100-epoch
  ceiling to 200 while preserving exact fixed-10-epoch pilot support.
- Added a synthetic full-run regression proving an initially improved metric
  followed by 10 equal validations stops after epoch 11 rather than consuming
  the 200-epoch ceiling.
- Added generic full-run timing fields and a formal validation-selection label
  to the native performance receipt while retaining legacy fields for sealed
  pilot compatibility.
- Focused local verification: 19 passed, one honest Torch/PyG dependency skip;
  Ruff and format passed. Full local pytest stopped during collection because
  `nbformat` is absent, and local build is unavailable because the `build`
  module is not installed. Exact-commit server pytest/Ruff/format/strict
  mypy/build remains required before launch.
- The user then requested formal "4 split" training. The execution scope is
  therefore Nadig Jurkat run seeds 1--4 on the existing frozen canonical split,
  distributed as two ordered GPU queues; no new split manifests are generated.
- First exact-commit server pytest exposed that the new default validator made
  sealed historical B0/B1 performance configs unreadable. The repair permits
  their explicitly labeled `performance_pilot_variant` metadata to retain the
  historical 100-epoch schedule while enforcing 200 epochs and all-seven B2
  systems only for current formal configs. No historical run or receipt is
  modified.

## 2026-08-28 — B2-vNext integrated build and development gates

- Implemented the A0 default and all 22 preregistered Nadig Jurkat ablation
  configs through one strict native architecture object and the existing
  `gradpert model pilot` lifecycle. No ablation-specific trainer/model main was
  introduced.
- Added TxPert-aligned full-cell-line pre-split HVG512 materialization after
  weak-signal filtering, exact target union/order receipts, deterministic
  Fanout views, native single/multi graph encoders, D1/D2 decoders and four
  exact GenePT emb_b routes.
- Expression-scale handling is explicit: RPE1/Jurkat/HepG2 require audited raw
  integer counts before normalize-4000/log1p/Top5000; pinned K562 and Norman
  processed matrices preserve their original X values and axes. A real Norman
  server audit observed exact before/after sparse matrix equality.
- Closed review findings: independent Transformer views no longer share
  training BatchNorm statistics; W1/W2/W3/WS freeze one full-topology STRING
  mapping; resident sparse prediction unions are cached; D2/GenePT routes have
  gradient/EMA/checkpoint/resume tests; manifests reject duplicate/shuffled/
  extra axes; the launcher validates every row before creating a process.
- Real development Jurkat materialization completed on 238,977 filtered cells
  and 2,393 conditions: 512 direct HVGs, 2,372 candidate targets, exact union of
  2,809 graph genes, 51,495 STRING and 38,287 GO nonself edges.
- Local gates: 253 passed with 11 honest missing-dependency/pending-receipt
  skips, Ruff, format, diff check, focused mypy and isolated build passed.
- Server development gates with Torch/PyG/anndata: targeted 82 passed; full
  315 passed with only three intentionally pending formal-receipt skips; Ruff,
  format, strict mypy on 72 source files and isolated wheel/sdist build passed.
- Two read-only reviews reported no remaining model-side P0/P1 blocker. Formal
  readiness still requires one clean public commit, corrected exact-commit
  dataset/HVG/GenePT receipts and the one-epoch A0 CUDA hard gate.

## 2026-08-28 — B2-vNext formal data and graph receipts

- Published and synchronized clean implementation commit `a942114`; exact
  server gates passed 314 tests with four honest dependency/evidence skips,
  Ruff, format, strict mypy on 72 source files and isolated build.
- Materialized a fresh non-destructive five-dataset root at
  `/data/yilangliu/GraD-Pert/data-vnext-a942114`. Individual full-hash reloads
  passed and all five split hashes equal the frozen datasets-v2 contract.
- Verified source-aware preprocessing: K562 and Norman preserve their official
  processed expression axes; RPE1, Jurkat and HepG2 use raw integer counts,
  weak-signal filtering, normalize-total 4,000, log1p and Seurat Top-5000.
- Built and re-verified all five STRING/GO graph receipts and evaluator states.
  An initial graph command named the wrong clean checkout and failed before
  creating any graph directory; its exact log is preserved under
  `superseded/20260828-vnext-a942-graph-wrong-checkout`, and the corrected
  TxPert-checkout run passed.
- The formal Jurkat vNext graph has 512 direct HVGs, 2,372 modeled targets,
  2,809 union genes, 51,495 STRING and 38,287 GO nonself edges. Its topology
  SHA-256 is `ba22af6e9e9a558533aaae850f619840ea2d717310eb3362a52476c3c1ea9128`.
- Exact GenePT `emb_b` verification passed 93,800 entries x 1,536 dimensions,
  but 17 modeled perturbation targets are absent. The preflight wrote an
  unavailable receipt and created no GenePT graph/model/training root.
- Mirrored 79 reviewed small receipts (about 62 MB) with inventory SHA-256
  `228b8d9c124b8a8324bc475f0d62b2435918c10e30f275702c9f350fb930bd80`.
  No H5AD, NPZ, PKL, checkpoint or evaluator array was transferred. The real
  ordered-control receipts reach 9.64 MB, so the bounded small-sync per-file
  limit is raised from 8 to 16 MiB while the total limit remains 128 MiB.

## 2026-08-29 — Successor ratio/H/L matrix and measured-performance harness

- Replaced the superseded fixed local coordinates with the preregistered
  successor A0/H/L matrix. A0 is HVG512+targets, RingInduced, exact local node
  ratio `1/2`, eight locals and mask-view ratio `0/1`. H1/H2/H3 request
  HVG1024/2048/5000 through the same pre-split materializer; L1--L5 each differ
  directly from A0 by one requested local factor.
- Implemented exact rational parsing, floor-derived local budgets, integral
  mask-view counts, generalized 4/8-local consistency terms and generic
  512/1024/2048/5000 graph manifests. The sealed 2,809-node A0 resolves to
  budget 1,404 with remainder 1.
- Added a pre-model all-condition anchor-capacity gate, a single resolved
  contract shared with the step engine, model/engine architecture identity
  checks, and a compact streaming `local_view_realization.json` aggregate bound
  into training and performance receipts. Historical architecture-omitted
  capacity probes retain fixed-512/four-mask behavior.
- Hardened schema-v2 launch validation: exact matrix ID/25-row variant set,
  canonical config path, resolved variant identity, semantic factor and
  recomputed A0 parameter diffs are required. Forged declarations, config
  relabeling and rehashed multi-factor drift fail before a process is created;
  intentional 22-row schema-v1 compatibility remains separate.
- Added a bounded training-only A0 measurement harness. It hash-pins the config,
  graph/source/protocol/split identity, prevents real validation/test data
  construction and caching, stops in the Nth completed-step call, records
  p50/p90/p95/p99 and profiler traces, enforces idle physical GPU/RAM/disk and
  `max(4 GiB, 15% VRAM)` capacity predicates, and always attempts an atomic
  primary/teardown failure receipt.
- Added existing-H destination lineage checks and a production four-axis audit
  requiring common materialization identity, target coverage, nested HVG/graph
  sets, ordered ranking consistency and per-axis artifact/topology hashes.
- Primary-agent local integration: 324 passed with 10 honest skips when
  notebook tests are excluded; three Torch suites relevant to the new runtime
  contract are among the local dependency skips. Ruff, format on 151 files,
  compileall and diff-check passed. Full local pytest is blocked at collection
  only because `nbformat` is absent; local Torch/anndata/mypy/build are also
  unavailable. Exact-commit server pytest/Ruff/format/strict-mypy/build and
  CUDA integration remain mandatory before Plan 029 is complete.

## 2026-08-29 — Queue-scoped publication receipt repair

- The 8221 Local Graph GPU1 queue completed L3 exactly, then L4 failed before
  model construction when its second live `git ls-remote` could no longer
  reach GitHub through the expired loopback proxy. The frozen failure receipt
  returned 1. GPU0 was stopped per contract with A0 at 3,422/5,820 steps; L2
  had not started. The whole lineage remained zero-PKL and source-clean.
- Plan 027 adds one live queue-preflight publication receipt rather than
  weakening publication identity. The receipt is immutable and externally
  hash-pinned. Every row rechecks local HEAD, clean tree, origin and source-tree
  hash, and rejects a missing pair, changed receipt, stale commit or stale tree
  before model construction.
- Local verification passed 262 tests with eight honest missing-dependency
  skips, full Ruff and format, isolated wheel/sdist build, focused strict mypy,
  diff check, and a real temporary Git/CLI receipt creation-and-consumption
  workflow. Exact-environment full mypy and Torch-backed tests remain server
  gates.

## 2026-08-30 — Exact-effect sparse-union performance acceptance

- Measured real A0 profiles selected sparse-union preparation after Student
  locals accounted for 68--70% of step wall and cProfile attributed 52.829 s
  cumulative to 342 union builds. The real first-batch union microbenchmark
  improved from 8,276.7 to 2,310.5 ms with exact tensor parity.
- At clean commit `7332cc1`, deterministic CUDA reference/optimized execution
  matched every non-timing metric, view/union identity, gradient, model,
  Teacher, optimizer, center and RNG state.
- Serial single-GPU ABBA used five warmups and twenty measured steps per arm.
  Reference p50 was 11,166.911/11,082.219 ms; optimized p50 was
  4,842.376/4,890.634 ms. The paired ratio was 0.437470 (56.253% reduction,
  2.286x), both p90 comparisons improved, peak allocated/reserved GPU memory
  changed by only +0.017/+0.041%, and all arms retained zero PKL with no
  validation/test access. Plan 028 is complete; its figures describe the
  historical eight-local coordinate.

## 2026-08-31 — four-local successor matrix migration started

- Stopped and preserved the active `f1c14d8` formal runs. A0 ended at 1,094
  steps and H3 at 1,874 steps; neither is scientific completion. No L row
  launched, GPUs were released, zero persistent PKL remained, and the obsolete
  monitor automation was removed.
- Rebased the successor scientific matrix on four RingInduced locals. L2 is
  now the eight-local single-factor row; proportional L4/L5 masks resolve to
  two and one masked views. Matrix identity advances to v3, while schema v2 is
  retained because the receipt structure is unchanged.
- Regenerated all 25 self-contained ablation configs and the three GenePT-Seed
  prior comparison configs. At migration start, fresh gates remained pending;
  the local results below now close the available local surface, while exact-
  environment server gates remain pending before any CUDA launch.
- Focused config/execution/performance gates pass 113 tests. The complete
  dependency-available local surface passes 437 tests with 10 honest skips for
  absent Torch/anndata and frozen-reference dependencies. Ruff, format on 160
  files, compileall, deterministic regeneration and diff checks pass. The
  notebook collection remains unrun locally because `nbformat` is absent;
  full Torch/notebook/mypy/build gates remain required on the server.

## 2026-08-31 — private Trackio scalar dashboard integration

- Added the optional `tracking` dependency group pinned to Trackio 0.37 and
  Hugging Face Hub 1.x, plus an out-of-process formal-only sidecar and private
  Space preflight. Native training and every performance script remain free of
  Trackio imports.
- The sidecar tails stable snapshots of only `train_steps.csv`,
  `validation.csv`, `run_meta.json` and the pre-test training receipt. It maps
  loss components, distillation/gradient health, stage timing, throughput and
  validation scalars; it never opens test summaries or uploads artifacts.
- Trackio's all-device background GPU monitor is disabled. The sidecar samples
  exactly one configured GPU every 30 seconds, keeps CPU telemetry isolated in
  its own process and preserves the native 64-step receipt buffer.
- Fresh-lineage locking, finite/contiguous receipt gates, exact source/run
  identity, private-Space verification, fsync-plus-atomic state and explicit
  provisional/non-authoritative receipt fields prevent dashboard state from
  being confused with scientific evidence. Twenty-one focused tests pass; the
  full locked dev/data/model/tracking local surface passes 544 tests with 2 honest
  skips, Ruff and format on 255 files, strict mypy on all 76 source files, and
  an isolated wheel/sdist build. Fresh server synchronization, exact-commit
  gates and server login/Space creation remain pending. The private destination is
  bound to Space `elan68681/grad-pert-vnext-ablations` and Bucket
  `elan68681/grad-pert-vnext-ablations-bucket`; both require server-side Hub
  write authentication before activation.
- A final privacy review found that a default server `umask` could leave the
  local Trackio store traversable by other users. The sidecar now applies
  `0077` for its complete lifetime, verifies the store as `0700`, restores the
  caller's prior mask, and tests owner-only store/database/state/receipt/lock
  permissions.

## 2026-08-31 — four-local RingInduced index accepted

- The four-local checkpoint count-two candidate passed deterministic exactness
  but was rejected after same-GPU ABBA measured a 1.026487 ratio (2.65%
  slower). Formal default therefore remains checkpoint count four.
- A new real Python profile attributed 5.173 seconds across 192 RingInduced
  builds to repeated base-edge/self-loop and incident-node scans. The selected
  implementation builds one immutable source-aware incoming-edge index per
  topology and preserves ordered edges, weights, self-loop insertion, warnings
  and complete views.
- On the sealed 2,809-node Jurkat topology, exact 32-view construction improved
  from median 983.661 to 833.647 ms (15.251%, 150.014 ms). Deterministic CUDA
  reference/indexed runs matched all non-timing metrics, views, CPU/CUDA RNG,
  losses, every gradient, Student/Teacher, optimizer, centers and predictions.
- Serial same-GPU ABBA reduced paired median step wall by 15.116% and 452.540
  ms (ratio 0.848836); both p90 pairs improved, GPU memory was identical, and
  all arms retained zero retry/OOM, zero PKL and no validation/test access.
- The user authorized formal A/H row-level use of both GPUs after performance
  delivery. Private Trackio Space creation remains externally blocked by a
  truthful `402 Payment Required`; the private Bucket and owner keychain
  authentication are ready, and no public fallback is authorized.

## 2026-08-31 — private Bucket Trackio fallback

- Added an explicit `local_private_bucket_archive` sidecar mode for the
  account-plan case where a private Trackio Space cannot be created. It keeps
  the same scalar allowlist and owner-only local store, skips Space creation,
  and receipts that no remote Space sync occurred.
- The real Trackio integration smoke recorded two train points and one
  validation point, exposed the local project through `trackio list`, and
  uploaded the 172,032-byte SQLite store to the private Bucket prefix
  `smoke/local-only-77645d6`. The remote object hash is
  `2be6f2decd5d442c7dc4fb2af565e11f554e236c4d1ae86985b172a3c0f0e4c2`.
  No public Space, test metric, prediction, data or checkpoint was uploaded.

## 2026-09-01 — four-local A/H formal completion

- Exact source `845c10a` completed A0/H1/H2/H3. A strict independent replay
  validated 5,820 contiguous steps, ten validations, no test truth during fit,
  one evaluation from hash-matched `best.pt`, exact three finite metrics, the
  four-local ratio contract, CPU-vectorized sparse union and clean source
  identity for every row.
- All rows share canonical data SHA `f051343c...e845861`, split SHA
  `ecb2099c...63cdd0`, condition-order SHA `3cfe9206...338b15`, ordered
  300-control-row SHA `de102e67...dff0b8` and truth-row SHA
  `e410ef47...3c4ef`. The full run root contains zero PKL and each row retains
  only `best.pt`.
- H point estimates are mixed: larger axes do not improve all three metrics
  consistently, so the review retains A0 and makes no equivalence claim.
- Trackio delivery remained non-authoritative. A0/H3 local stores completed,
  while the expired reverse SOCKS endpoint prevented private Bucket archival
  and blocked H1/H2 Trackio startup. No scientific run was replayed or relabeled.
- Post-result local gates pass 578 tests with two honest skips (frozen upstream
  evidence absent and CUDA unavailable), Ruff, format on 258 files, strict mypy
  on 76 source files, JSON/diff checks and an isolated wheel/sdist build.
