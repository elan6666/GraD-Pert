---
id: 032
status: completed
wave: 2
depends_on: [029]
updated_at: 2026-08-31T06:30:00+08:00
---

# Plan 032 — Seed-GO-ProteinPathway prior migration

## Objective

Replace the obsolete frozen `emb_b` input in all four Nadig Jurkat E rows with
the user-selected `GenePT-Seed` `Seed-GO-ProteinPathway` prior while preserving
the registered feature-route interventions.

## Frozen source prior

- Project: `/data/yilangliu/GenePT-Seed`.
- Artifact:
  `/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz`.
- SHA-256:
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
- Model: `doubao-embedding-vision`; width 2,048; 17,730 exact case-sensitive
  labels; zero all-zero rows.
- Source vector audit proves exact graph-axis and perturbation-target coverage
  for all five GraD-Pert datasets. The GGI result is provenance for choosing
  this prior, not GraD-Pert prediction evidence.

## Live pre-implementation audit

The server artifact was re-read against the exact current HVG512-plus-targets
axis before code changes: source labels were unique `17,730/17,730`; all 2,809
runtime labels were present; runtime gene-order SHA-256 was
`25807825ded5a5b11bc9349c9fb098efbbf9b6e592738aa4d6932655dfc88de6`;
selected rows were finite, width 2,048 and zero-free; audited case pairs such as
`C12ORF57`/`C12orf57` and `C9ORF72`/`C9orf72` both remained present.

## Integration contract

- E1/E2/E3/ES select the exact runtime graph labels from the sealed superset
  and reorder them to runtime graph order before model construction.
- Exact-case labels take precedence and case-paired labels remain distinct.
- Extra source labels are ignored with count/order hashes in the receipt.
- A missing perturbation target, duplicate source label, non-finite row, wrong
  SHA/model/width or zero fill fails closed before model construction.
- A missing non-perturbation runtime gene is omitted in preserved canonical
  order and its ordered count/hash is receipted. The selected artifact covers
  the full current runtime axis, so this policy changes no registered E row.
- E rows use the same unfiltered HVG512-plus-targets topology as A0. H/L/A0 and
  every non-E scientific factor remain unchanged.

## Acceptance criteria

- The generator and schema-v2 matrix bind the exact artifact path/SHA and
  remove every old `emb_b` or GenePT-filtered-graph dependency from E rows.
- All four feature modes accept the same sealed NPZ identity and differ from
  A0 only by their declared feature-route and derived prior fields.
- Focused tests cover superset selection, runtime order, case pairs, duplicate
  labels, fail-closed missing perturbation targets, ordered omission of missing
  non-perturbation genes, non-finite values, wrong hashes and matrix tampering.
- The GenePT availability receipt preserves artifact path/SHA/schema/status,
  selected and extra label hashes, and target coverage. No receipt means
  blocked preflight, not a skip.
- `scripts/ablations/preflight_genept_seed.py` is the only successor preflight
  entrypoint. It verifies the sealed NPZ against the unchanged parent graph,
  preserves the parent topology hash, and writes the availability receipt
  before any E-row model is constructed.
- Each E-row launch-plan receipt binds the exact preflight path/SHA/schema,
  artifact SHA, status, and config artifact/runtime-graph identity. A legacy
  pickle availability receipt cannot authorize a successor NPZ row.
- Full repository gates pass before the changed matrix enters Plan 031.

## Non-goals

- No GraD-Pert training, validation or test evaluation in this plan.
- No comparison among old GenePT, Seed, GO-EXP, Protein or HPA priors.
- No zero fill, case folding, aliases, nearest-gene substitution or
  unreceipted graph pruning to manufacture coverage.
