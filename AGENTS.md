# AGENTS.md

## Project Purpose

Build a standalone GraD-Pert research package and a fair five-dataset benchmark
for GraD-Pert, GEARS, TxPert, and nonlearned baselines.

## Start Here

- Current state: `.byte-os/STATUS.md`
- Active product contract: `.byte-os/PRODUCT_SPEC.md`
- Engineering rules: `.byte-os/ENGINEERING_RULES.md`
- Codebase map: `.byte-os/CODEBASE_MAP.md`
- Harness: `.byte-os/HARNESS.md`
- Plans: `.byte-os/plans/`
- Active design: `docs/design/GRADPERT_V1.md`
- Reference alignment: `docs/provenance/REFERENCE_ALIGNMENT.md`
- Experiment architecture: `docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md`

## Non-Negotiable Rules

- Native distribution/import root is `gradpert`; native APIs, classes,
  checkpoints, and logs use GraD-Pert domain names.
- Never frame GraD-Pert product prose or code comments as a renamed/borrowed
  model. Preserve truthful citations, licenses, benchmark IDs, and provenance.
- `src/gradpert/modeling/**` must not import or call TxPert, `gspp`, GEARS,
  DINO, or an upstream checkout. No native upstream-named model classes.
- Inspect frozen official commit/file/function/tensor/update behavior before
  alignment. Never guess or copy-paste/rebrand upstream code.
- v1 is B2 only: from-scratch joint prediction plus graph self-distillation.
  No B3, ablations, cross-cell task, extra learned model, or test-set tuning.
- Five datasets only: Replogle K562/RPE1 essential, Nadig Jurkat/HepG2, Norman.
- Keep observed upstream metadata columns separate from canonical columns. A
  blocked or unaudited source cannot be relabeled ready, and no cross-cell cache
  may substitute for an independent within-cell dataset.
- All models consume the exact same canonical condition split and 300-control
  evaluation manifests; equality means IDs and hashes, not merely equal seeds.
- Apply the frozen official GEARS default-graph representability intersection
  to every model's train/validation/test partitions. Preserve the source split
  assignment and order of retained conditions; never let one runner silently
  drop unsupported conditions on its own.
- Configs are self-contained files at
  `configs/experiments/<model_id>/<dataset_id>.yaml`. No global experiment
  config, hidden defaults chain, or config inheritance.
- Every GraD-Pert, GEARS, and TxPert dataset integration must first pass an
  exactly one-epoch server smoke. Only GraD-Pert proceeds to a full run with
  `max_epochs=200` and validation-only early stopping `patience=10`; GEARS and
  TxPert remain smoke-only until the user changes the execution policy.
- GEARS and TxPert models are never reimplemented in this repository. Their
  isolated runners import and call the frozen official checkout/package with
  official model/training configuration; local code is limited to canonical
  data/split adaptation, leakage guards, prediction export, and receipts.
- LR, batch size, optimizer, weight decay, and architecture values start from
  each model/dataset's frozen official config; preserve genuine dataset
  differences and label missing values `project_preregistered`.
- Formal data preparation, graphs, GPU fitting, training, inference, and metric
  materialization run only on `/data/yilangliu`.
- Local, GitHub `elan6666/GraD-Pert`, and server source must be the same clean
  commit before formal work. Abort on mismatch.
- Datasets, H5AD, PKL predictions, checkpoints, weights, and per-cell matrices
  stay on the server. Pull only dry-run-reviewed small result/receipt files.

## Repository Map

- `src/gradpert/`: standalone package; no upstream runtime dependency.
- `configs/experiments/`: self-contained model-by-dataset configs.
- `benchmarks/`: isolated external-model runners and environment locks.
- `tests/`: local synthetic/unit/contract tests.
- `docs/design/`: active v1 specification; authoritative over historical notes.
- `docs/provenance/`: frozen reference and license registry.
- `TxPert/`: historical design and read-only reference evidence; not product code.
- `.byte-os/`: product state, decisions, plans, reviews, and delivery evidence.

## Global Commands

- Test: `python -m pytest -q`
- Lint: `python -m ruff check .`
- Format check: `python -m ruff format --check .`
- Typecheck: `python -m mypy src`
- Build: `python -m build`

## Safe Edit Boundaries

- Prefer: `src/`, `tests/`, `configs/`, `benchmarks/`, `docs/`, `.byte-os/`.
- Avoid editing: `TxPert/official-repo/**` and all frozen upstream evidence.
- Never touch/commit: `data/`, `artifacts/`, `runs/`, `checkpoints/`, `.server/`
  or ignored binary/scientific artifacts.

## Navigation

- Search with `rg`; list files with `rg --files --hidden`.
- Start from the relevant plan, then nearest module `AGENTS.md`, then symbols.
- Treat `TxPert/official-repo.incomplete-archive` as noisy/incomplete evidence.

## Subagents

- Exploration is read-only with explicit paths and a written handoff.
- Implementation agents require disjoint directory ownership from a plan.
- Review agents do not modify code unless reassigned to a repair plan.

## Maintenance

- Last reviewed: 2026-08-24
- Next review: after v1 server smoke or 2026-11-24
- DRI: repository owner
