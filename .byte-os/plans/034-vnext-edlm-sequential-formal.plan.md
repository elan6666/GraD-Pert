---
id: 034
status: active
wave: 7
depends_on: [033]
updated_at: 2026-09-02T00:00:00+08:00
---

# Plan 034 — four-local E, D, L, M sequential formal execution

## Objective

Run the remaining user-prioritized Nadig Jurkat vNext ablations in the strict
module order E, then D, then L, then M. A module barrier prevents any later
module from starting until every row in the preceding module is formally
complete and validated.

## Reference and prior identity

- Reuse completed A0 from source `845c10a` as descriptive cross-commit
  reference. Do not rerun or relabel it as a same-commit row.
- The new source commit may change only provenance naming, orchestration,
  tests, plans, and receipts before execution; the A0 model/config/data
  coordinate remains unchanged.
- Every E row uses GenePT-Seed `Protein+Reactome+SIGNOR`, whose upstream
  profile/artifact label is `protein-pathway` / `Seed-GO-ProteinPathway`.
  Canonical artifact:
  `/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz`.
- Artifact SHA-256:
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
  It contains 17,730 exact labels, 2,048-wide finite nonzero vectors and model
  `doubao-embedding-vision`. A fresh server preflight must prove complete
  Nadig Jurkat runtime-axis and perturbation-target coverage.

## Ordered modules

1. E: `e1_frozen_genept`, `e2_genept_id_residual`,
   `e3_genept_initialized`, `es_genept_shuffle`.
2. D: `d1_control_mlp`, `d2_control_transformer`.
3. L: `l1_fanout_ratio_half`, `l2_ring_half_count8`,
   `l3_ring_quarter`, `l4_ring_half_mask_half`,
   `l5_ring_half_mask_quarter`.
4. M: `m1_single_string_gat`, `m2_single_string_transformer`,
   `m4_adaptive_source_gat`.

Rows inside one module may use at most two independent physical GPUs. The
next module starts only after both queues stop normally and all row receipts
pass. A failure, CUDA OOM, source/input drift, missing target, persistent PKL,
or receipt mismatch stops the active peer and the entire successor sequence;
no repair or relaunch occurs in the same lineage.

## Scientific contract

- Dataset `nadig_jurkat`, frozen canonical split, seed 1.
- A0 coordinate: HVG512 plus targets, STRING+GO Exphormer-MG, four
  RingInduced locals at node ratio `1/2`, mask ratio `0/1`, additive decoder,
  learned-ID features and losses `1.0/0.8/0.4/0.1`.
- Every row differs from A0 only by its declared E, D, L, or M factor.
- Exactly 10 epochs, 5,820 ordered optimizer steps, ten validations, no early
  stopping, one test evaluation from `best.pt`, exact three headline metrics,
  `metrics_only`, zero persistent PKL and only `best.pt`.
- Test truth is unavailable during fit. Module/result comparisons remain
  single-seed descriptive evidence without equivalence or superiority claims.

## Systems and tracking

- Use exact-effect implementations `cpu_vectorized` sparse union and `indexed`
  RingInduced construction.
- Start no continuous goal execution during CUDA training. Monitor at most
  once every 30 minutes.
- Trackio is auxiliary and owner-only. It may mirror allowlisted train loss,
  validation loss and telemetry, but native receipts remain authoritative and
  tracking failure must not change scientific status or trigger replay.

## Gates and completion

Before launch, require a clean identical local/GitHub/server commit, exact
server pytest/Ruff/format/mypy/build gates, source publication receipt, fresh
hash-pinned contract/run/log roots, idle GPUs and adequate RAM/disk. Completion
requires all fourteen E/D/L/M rows to pass strict row and module validation,
small reviewed evidence, documentation updates, full final gates, clean
commit/push/server synchronization, and monitor deletion.
