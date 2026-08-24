# Server Execution and Synchronization

## Source topology

```text
local working copy -> Git commit -> GitHub main -> server clean checkout
```

Canonical remote: `https://github.com/elan6666/GraD-Pert.git`.
Canonical server storage root: `/data/yilangliu/GraD-Pert`. Because that root
already contains data, runs, environments, and frozen upstreams, the clean Git
checkout is `/data/yilangliu/GraD-Pert/source`; it must never be cloned over the
non-Git storage root or the preserved `dev-snapshots/current` tree.

Formal job preflight aborts unless:

- local HEAD equals GitHub `main` equals server HEAD;
- local and server tracked worktrees are clean;
- the per-model/per-dataset config hash matches the committed file;
- dataset, split, control, graph, and environment gates are ready;
- exact target output path is new or resumable by matching receipt;
- GPU, disk, and active-job state are recorded.

Git commands used by source identity are noninteractive and have a 30-second
timeout. A server without direct GitHub egress may use an operator-controlled
SSH remote dynamic forward bound only to server loopback; the formal check must
still execute `git ls-remote` against the configured GitHub `origin` and match
the exact public ref. A local tracking ref or copied bundle alone is not enough.

## Formal compute boundary

Server only:

- data download/extraction/canonicalization/QC;
- STRING/GO graph materialization;
- CUDA/PyG environment installation and locks;
- `K_head` 128-consecutive-step fit on every dataset;
- all learned training, inference, evaluator materialization, checkpoints;
- PKL/H5AD and per-cell outputs.

Execution policy is encoded independently in every model/dataset config:

- `gradpert_b2`: one-epoch integration gate, then full training up to 200
  epochs with validation-only patience 10;
- `gears` and `txpert_public`: exactly one training epoch on each dataset, using
  isolated frozen official packages/configuration; no full run in this phase;
- nonlearned models: inference/evaluation only.

An external smoke is not an import check: it must use the official model and
training API, consume the canonical split, write a checkpoint, preserve the
manifest-selected 300 control predictions per condition, and pass the common
artifact/evaluator validators.

The isolated GEARS/TxPert interpreters do not install the native project
package. Launch their thin runners from the repository root with
`PYTHONPATH=src:. <isolated-python> -m benchmarks.<model>.runner ...`; this
exposes only the local adapters/contracts while the learned model import still
comes from the frozen official checkout inside the guarded module session.
`scripts/server/run_official_smoke.py` constructs this environment explicitly,
prints the exact argv/PYTHONPATH as a dry run by default, and executes only when
passed `--execute`.

## Frozen matrix orchestration

`scripts/server/run_experiment_matrix.py` is the thin server entrypoint over
the tested `gradpert.execution.matrix` core. It verifies all 30 standalone
configs and materializes exactly these task sets:

- `smoke`: 3 learned models x 5 datasets x paired seed 1 = 15 tasks;
- `nonlearned`: 3 baselines x 5 datasets x paired seed 1 = 15 tasks;
- `full`: GraD-Pert x 5 datasets x seeds 1--4 = 20 tasks.

The command is a JSON dry run unless either one exact `--execute-task` or the
deliberate `--execute-all` flag is supplied. Execution also requires a receipt
root. Existing output is skipped only after its run ID, model, dataset, seed,
config hash, commit, formal status and one-time test lifecycle validate. An
incomplete directory fails closed; only native full runs accept the explicit
`--resume-native-full` path.

Interpreter arguments are validated for existence and execute permission while
preserving their lexical virtualenv paths. They must not be canonicalized to the
base interpreter behind a `venv/bin/python` symlink, because doing so discards
the native, GEARS, or TxPert environment selection.

The `full` phase is formal-only and is unavailable until all 15 smoke run
manifests plus training receipts prove one epoch, checkpoint identity, no test
Truth during fit, and equal canonical-data/split/300-control hashes for each
dataset across GraD-Pert, GEARS and TxPert.

Example planning invocation from the clean server checkout:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/server/run_experiment_matrix.py \
  --phase smoke \
  --project-root /data/yilangliu/GraD-Pert/source \
  --data-root /data/yilangliu/GraD-Pert/data \
  --runs-root /data/yilangliu/GraD-Pert/runs/formal-v1 \
  --native-python /data/yilangliu/GraD-Pert/source/.venv/bin/python \
  --gears-python /data/yilangliu/GraD-Pert/envs/gears/bin/python \
  --gears-checkout /data/yilangliu/GraD-Pert/upstreams/gears \
  --gears-data-root /data/yilangliu/GraD-Pert/data/gears-official \
  --txpert-python /data/yilangliu/GraD-Pert/envs/txpert/bin/python \
  --txpert-checkout /data/yilangliu/GraD-Pert/upstreams/txpert \
  --device cuda:0 --device cuda:1 \
  --expected-commit COMMIT_SHA --formal
```

Local only or allowed:

- source, docs, schemas, configs;
- unit/contract/static tests and synthetic CPU smoke;
- small server summaries/receipts/pointers.

## Source synchronization

- Initialize/fetch/push without force. Server obtains code by clone/fetch and
  fast-forward to an explicit commit.
- Do not rsync the repository as the source of truth.
- Formal launch takes `--expected-commit`; preflight records all three SHAs.
- A dirty local or server tree is a hard failure, not a warning.

## Result synchronization allowlist

Permitted by default: `.txt`, `.json`, `.jsonl`, `.csv`, small `.md`.
Plots require explicit path selection. Everything else is denied.

`scripts/server/stage_small_results.py` performs the enforced first half of the
transfer. Its default dry run discovers files only below named `small_results`
directories, prints every relative path/hash and total bytes, and rejects
symlinks, path traversal, forbidden extensions, files over 5 MiB, or a snapshot
over 100 MiB. `--execute` copies that immutable selection to a new empty staging
root outside the runs tree and writes `small-sync-manifest.json`.

The default `--scope run-small-results` cannot select anything outside a named
`small_results` directory. For dataset/QC/capacity receipt trees such as
`registry/prepared`, the operator must name that exact source root and pass
`--scope explicit-root`; the same suffix, size, symlink and hash gates remain
mandatory, so scientific/binary data still fails closed.

Only that reviewed staging root is transferred with checksum-preserving rsync.
After transfer, run the same script with `--verify --destination-root <local>`;
it rejects missing, extra, changed, linked, or forbidden files. Server pointers
inside the snapshot retain the server artifact paths and hashes; PKL, H5AD,
checkpoints and arrays never enter staging.
