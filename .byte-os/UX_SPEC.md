# CLI and Analysis Experience

## Core journey

1. `gradpert doctor` reports local/server-capable dependencies without changing
   the machine.
2. `gradpert config verify --all` validates every model-by-dataset file and
   prints provenance and hashes.
3. On the server, `gradpert data prepare --all` downloads/resumes, verifies,
   canonicalizes, freezes splits/controls, and produces readiness receipts.
4. `gradpert data status --all` shows one row per dataset with the exact failing
   gate and next safe command.
5. `gradpert fit-head --config ...` selects the single global `K_head` through
   a memory-only full-step test.
6. `gradpert run --config ... --seed ...` performs preflight, train/resume,
   sealed prediction, then one test evaluation.
7. `gradpert evaluate --prediction ...` joins truth and recomputes metrics.
8. `gradpert results sync-small --run ... --dry-run` previews local transfer;
   an explicit second command copies only permitted files.
9. Notebooks consume a frozen result manifest/pointer and never train or split.

## Surfaces

- CLI only for v1; no web UI.
- Human-readable tables plus machine-readable JSON for doctor/status/preflight.
- Server run directory with receipts, logs, checkpoints, predictions, evaluation,
  summaries, and a sealed manifest.
- Local `results/` contains only small summaries/pointers.

## Navigation model

```text
doctor -> config verify -> data prepare/status -> fit-head -> run -> evaluate
       -> sync-small -> notebook/report
```

## States

- Empty: dataset/config/run not present; show exact create/prepare command.
- Loading: streamed stage name, item counts, bytes, epoch, ETA when defensible.
- Error: stable code, failed gate, evidence path, retry safety, no success claim.
- Success: status, hashes, server paths, selected checkpoint, small result paths.
- Blocked: upstream URL/license/semantic ambiguity or capacity mismatch; do not
  offer an unsafe automatic substitution.

## First-run experience

- `gradpert doctor --json` is non-mutating.
- `gradpert config verify --all` works before data exists.
- Data commands explain that large assets stay under the server data root.
- Every destructive/overwriting possibility is rejected; existing mismatched
  artifacts require a new versioned target rather than replacement.

