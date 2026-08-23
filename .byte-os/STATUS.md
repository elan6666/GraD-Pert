---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: planned
current_workflow: byte-plan
next_workflow: byte-build
review_verdict: none
iteration_count: 0
harness_status: ready
hard_blocked: false
updated_at: 2026-08-24T01:25:00Z
---

# Status

- Goal mode: on; Codex goal created for the end-to-end GraD-Pert delivery.
- Project state: research and navigation harness are ready; standalone implementation has not started; repository root is not yet a Git repository.
- Discussion source: Codex task `01a00ab3-9864-7032-98d9-45f6d0016838`, `TxPert/grad-pert`, and `.byte-os/DISCUSSION.md`.
- Confirmed scope: all five datasets through `canonical_ready`, training, evaluation, and manifests.
- Training route: one B2 configuration; no B3 and no ablation matrix.
- Benchmarks: GEARS, TxPert, matched-control mean, global train delta, and general train delta with Norman additive seen singles.
- Fairness: one canonical condition split/evaluation manifest; paired run seeds and shared 300-control evaluation draws.
- Metric contract: frozen union with three distinct Pearson headlines: TxPert macro delta, TriShift delta, Systema Pearson.
- Artifact contract: versioned condition-level PKL plus Parquet/JSON/H5AD; notebooks consume artifacts only.
- Training budget: learned models max 200 epochs, validation-only early stopping patience 10; per-model/per-dataset self-contained configs.
- Server topology: formal compute only on `/data/yilangliu`; local/GitHub/server must share one clean commit; large artifacts remain server-only.
- Hardware observation: two RTX 5090 with 32607 MiB each and about 32110 MiB free; recheck before fit and lock the largest `K_head` below 85% usable memory.
- Harness: ready; Claude context ready; Codex context ready; `AGENTS.md` ready.
- Current work: execute plan 001 repository/package foundation.
- Open data gates: independent within-cell source URLs/checksums for RPE1, Jurkat, and HepG2.
- Next recommended command: `byte-build`.
