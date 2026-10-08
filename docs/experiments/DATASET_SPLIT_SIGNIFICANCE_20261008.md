# Split-half significance and control sensitivity, 2026-10-08

Status: PROPOSED analysis; code locally verified, no new scientific result yet.
Authorization: user requested manuscript optimization after discussing dependent
resplits and shared control error. No model fitting or GPU work is authorized here.
Baseline: 21ab3cd1da71649ab41cec99c6045a00ece62004, verified clean and published.

## Frozen scope

Five canonical datasets and their existing test condition split only. Original
20 splits use seed 42 + repeat and their historical fixed 300-control manifests.
Retain original outputs; new results use a new server run ID. All expression and
profile matrices remain under /data/yilangliu. CPU only, nice 15, two threads.

## Control sensitivity

For every repeat, split each batch's control rows into disjoint A/B pools using
seed 100042 + repeat. For each condition, preserve the exact batch counts in its
frozen 300-control draw. Draw 300 with replacement from each corresponding pool
using seed 200042 + repeat, in sorted condition/batch order. Within draws,
duplicates are allowed; no row is shared between A and B. Compare shared-A and
shared-B Pearson averaged symmetrically against the symmetric mean of the two
A/B and B/A distinct-control Pearson values. Each arm uses 300 draw entries.
The frozen 300 score remains a separate historical reference. This measures
within-dataset sampling sensitivity, not independent-experiment uncertainty.
Pool disjointness does not make the finite-population samples biological repeats.

## Batch-matched conditional identity test

An audit before viewing the new scores found inadequate exact matches for whole
condition batch compositions, and <300 controls per batch in the four single-gene
datasets. Use a separate, explicitly restricted diagnostic rather than append
stars to the original whole-condition Pearson bars.

Primary sample: exactly 4 perturbation cells per condition-batch (2 per half),
>=20 control cells in that recorded batch, >=2 eligible test conditions of the
same target count in each batch block. Secondary sample: exactly 10 perturbation
cells (5 per half), fixed before results. Coverage is reported for both. This
compares equal sample sizes within precisely the same recorded batch. It trades
coverage and power; it does not estimate original full-condition reproducibility.
Each control half uses floor(batch control count / 2) disjoint rows; an odd final
control is omitted. Perturbation rows and control rows are disjoint within a
repeat, but repeat draws reuse the observed pools.

For each block, compute the full condition-by-condition Pearson matrix across
genes in each of 20 draws. Permute right-half CONDITION identities, never genes,
within batch and target count. Use one bijection jointly for all 20 draws of a
block. There are 9,999 randomizations, allowing fixed points. Average Pearson
over draws, then batches within condition, then conditions equally. This is a
conditional uniform-identity reference, not a claim that experimental labels
were randomized or biological effects replicate across cultures.

Report observed statistic, null mean, excess correlation, null 2.5/97.5 percentiles
(not confidence intervals), finite-randomization p=(1+exceedances)/10000. Apply
Holm correction within each prespecified five-dataset primary/secondary family;
Holm tolerates dependence among valid p values. No per-condition biological
p/q values or t-test across 20 draws will be manufactured. Batch metadata do not
establish independent biological replication; Norman has only a constant label.

## Acceptance

- Recompute historical frozen 300-control values, maximum absolute discrepancy
  <=1e-12; preserve all missing conditions/repeats.
- Exactly zero A/B row overlap, matched batch counts and equal draw counts.
- Duplication of a resplit cannot change the identity-randomization result.
- No condition identity crosses its recorded batch/target-count block.
- All five datasets complete with source/data/split/control hashes and zero PKL.
- Transfer only reviewed compressed scalar tables, summaries and figure assets.
- Update actual Methods, Results, Discussion, captions, tables and source bundle;
  compile and inspect the resulting PDF before reporting completion.
