# Remaining Nadig Jurkat ablations: observed results

Status: training and evaluation completed; final delivery review is pending.
These are seed-1, ten-epoch point estimates, not replicated superiority or
equivalence claims. No default is changed by this report.

## Observed metrics

| Row | TxPert macro delta | TriShift delta | Systema |
| --- | ---: | ---: | ---: |
| Retained A0 | 0.251406 | 0.245002 | 0.018933 |
| Retained M1 / W0 | 0.268334 | 0.306133 | 0.068080 |
| M2 | 0.247347 | 0.230472 | 0.023007 |
| M4 | 0.253538 | 0.241410 | 0.044929 |
| W1 | 0.274266 | 0.293444 | 0.050698 |
| W2 | 0.266592 | 0.262598 | 0.038396 |
| W3 | 0.269677 | 0.286494 | 0.039034 |
| WS | 0.261363 | 0.280979 | 0.023190 |
| O1 | 0.256727 | 0.259460 | 0.042624 |
| O2 | 0.247993 | 0.226892 | 0.026272 |
| O3 | 0.252665 | 0.234414 | 0.020035 |
| H4, native result | 0.247860 | 0.226320 | 0.044125 |

Each metric covers 592 total conditions; finite counts are 592, 590 and 590,
respectively. The missing-condition behavior is preserved rather than treating
unavailable values as zero.

## Interpretation boundaries

- W compares to retained M1/W0 (single-STRING GAT), not multi-source A0.
  W1 and W3 have higher TxPert point estimates but lower TriShift and Systema
  estimates than W0. None improves all three estimates over W0. The shuffled
  weight control WS is below W0 on all three estimates.
- O1 removes condition consistency, O2 masked-node loss, and O3 spread loss.
  O1 is above retained A0 on all three observed estimates; this one-seed,
  cross-commit comparison does not establish that removing the loss is better.
  O2 and O3 have mixed changes across metrics.
- M2 and M4 likewise do not improve all three estimates over A0.
- H4 uses the 9,853-gene candidate graph. Its native training/evaluation passed
  the core checks, but its old queue wrapper exited with RC80 because it looked
  for the training CSV at the wrong path. That failure remains preserved:
  native completion must not be represented as wrapper success.

## Provenance and checks

A0 is retained from `845c10a`; M1/W0 from its `formal-vnext-m-1bb0068-v1`
lineage; H4 from `2ca755f`. The new M2/M4/W/O rows use `51e922f`.
Cross-commit references are descriptive comparisons, not exact-code replicates.
The M2 singleton BatchNorm repair is part of the newer source lineage.

Read-only server checks verified 5,820 ordered optimizer steps, ten epochs and
validations, one best-checkpoint test evaluation, no canonical test truth during
fit, checkpoint hashes, best-only retention, metrics-only null PKL fields and
zero PKL across each successful run root. All ten recent rows share the exact
ordered 300-control/truth identities and canonical split with retained M1;
retained A0-to-M1 ordered identities were also checked directly.

The small [audit summary](../../.byte-os/evidence/vnext-performance/remaining-ablation-audit-20260907.json)
binds server manifest hashes and records the architecture checks performed.
All ten complete architecture payloads and hashes match the configuration
resolver. All 5,820 steps per row reproduce the configured four-term loss sum
within an absolute tolerance of 1e-4. Final repository gates and
publication/synchronization remain pending; this report is not final sign-off.
No data matrices, checkpoints or prediction PKLs were downloaded.
