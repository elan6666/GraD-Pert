# Server Execution and Synchronization

## Source topology

```text
local working copy -> Git commit -> GitHub main -> server clean checkout
```

Canonical remote: `https://github.com/elan6666/GraD-Pert.git`.
Canonical server root: `/data/yilangliu/GraD-Pert` unless a pre-existing path is
discovered before creation.

Formal job preflight aborts unless:

- local HEAD equals GitHub `main` equals server HEAD;
- local and server tracked worktrees are clean;
- the per-model/per-dataset config hash matches the committed file;
- dataset, split, control, graph, and environment gates are ready;
- exact target output path is new or resumable by matching receipt;
- GPU, disk, and active-job state are recorded.

## Formal compute boundary

Server only:

- data download/extraction/canonicalization/QC;
- STRING/GO graph materialization;
- CUDA/PyG environment installation and locks;
- `K_head` full-step fit;
- all learned training, inference, evaluator materialization, checkpoints;
- PKL/H5AD and per-cell outputs.

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

The sync command always performs checksum dry-run, prints the exact list and
total bytes, rejects symlinks/path traversal and any extension not in the
allowlist, then performs the transfer. It writes a local pointer containing the
server artifact root and checksums of server-only canonical files without
copying them.

