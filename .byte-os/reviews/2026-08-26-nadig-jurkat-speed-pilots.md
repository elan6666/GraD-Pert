# Nadig Jurkat Speed Pilots Final Review — 2026-08-26

## Verdict

`pilot_ship`: B1, B2, and B3 are implemented, independently commit-pinned,
server-executed for exactly one epoch, strictly validated, and compared under
the frozen Nadig Jurkat contract. B3 is selected for speed. This verdict does
not claim one-epoch predictive-effect equivalence.

## Frozen contract

- Immutable B0 was referenced and never rerun. Its detailed timing fields are
  unavailable and remain an explicit limitation.
- Every executed pilot used Nadig Jurkat, seed 1, the same canonical split and
  ordered 300-control/truth IDs, batch size 256, 16,384 prototypes,
  `expandable_segments:True`, the same GPU class, exactly one epoch, and
  `metrics_only` with zero persistent PKL.
- Expression input, output, and evaluation axes remained 5,000 genes.
- B1 changed only the graph axis; B2 retained the full graph and enabled all
  seven systems groups; B3 combined them.

## Primary speed result

| Variant | Graph nodes | Nonself edges | Seven systems groups | Training wall | Actual cells/s | Peak GPU allocated |
|---|---:|---:|:---:|---:|---:|---:|
| B0 | 6,506 | 222,654 | no | unavailable | unavailable | unavailable |
| B1 graph-only | 2,798 | 89,561 | no | 844.180 s | 151.94 | 14.54 GiB |
| B2 systems-only | 6,506 | 222,654 | yes | 718.681 s | 178.47 | 18.54 GiB |
| B3 combined | 2,798 | 89,561 | yes | 507.718 s | 252.63 | 14.54 GiB |

- B1→B3 holds the reduced graph fixed and adds the seven systems groups: B3
  is 1.663x faster and reduces training wall by 39.86%.
- B2→B3 holds the systems groups active and reduces the graph: B3 is 1.416x
  faster and reduces training wall by 29.35%.
- B1→B2 is not a single-factor comparison. Contextually, B2 is 1.175x faster
  despite using the larger graph.

The speed decision uses `one_epoch_training_wall_ms`, a monotonic actual epoch
wall. Full-epoch steps/s and cells/s are recomputed from the same wall and all
582 steps/128,266 cells. The original warmup-excluded receipt throughput is
retained, but B2/B3's `measured_end_to_end_wall_ms` serially adds data-read and
GPU-step stages even though prefetch overlaps them; it is not used as actual
wall throughput.

## Runtime and equivalence evidence

- B3 completed 582 optimizer steps, no-test fit, one test evaluation, and
  retained only `best.pt` SHA-256
  `b4bf958f9197f999eae81cca0dfec101f618c56e902a01fb7d02a3ef8ac4cdcc`.
- B3's 49-check verifier proved exact commit/config/run identity, B0 fairness
  hashes, B1/B2 ordered control/truth IDs, 5,000 expression/output/evaluation
  genes, B1 graph hashes/counts, three-metric schema, and zero PKL.
- Control cache, prefetch, pinned memory, nonblocking transfer, resident graph
  tensors, validation cache, buffered logs, and single checkpoint
  serialization were active. The complete control cache served all 582
  batches, so the enabled merged-read fallback correctly recorded zero runtime
  batches. Reflink was unavailable and the validated byte-copy fallback was
  used.
- First-step perturbed/control row IDs and pre-transfer tensor hashes matched
  B2, all view/RNG/parameter hashes were present, losses were finite, and the
  update order was optimizer → Teacher EMA → center.
- Source gates: B3 passed 191 local tests/9 honest skips and 214 server tests/3
  honest skips, Ruff, format, server strict mypy on 66 files, isolated builds,
  and exact clean local/GitHub/server identity at `44ae7ff`.

## Non-decisional metrics

| Variant | TxPert macro Pearson delta | TriShift Pearson delta | Systema Pearson |
|---|---:|---:|---:|
| B1 | 0.154269 | 0.134412 | 0.016394 |
| B2 | 0.142702 | 0.118471 | 0.018789 |
| B3 | 0.152663 | 0.131387 | 0.008880 |

These values are recorded for traceability only. Exactly one epoch is
insufficient to conclude unchanged effect, so no threshold or scientific model
selection is applied here.

## Sealed evidence

- B3 contract SHA-256:
  `512f08c96ae0b4bac84afca96c88e89c05f0d802407e0831ad6b7d94cddbceb7`.
- Strict validator SHA-256:
  `4fe769f7aa4d0d7c7dc2807c551bbdf98d5456e9af98ba8d095b705023e955bf`.
- Strict PASS receipt SHA-256:
  `65cf90788dc6a213148b35de0685cb1216d50d127a19d21b1cd175c6801c4274`.
- Comparison receipt SHA-256:
  `72f4d1b8a3c6bffc246ea3d58bd92b4ad191e9a3b4fdb7f9306456cecfdf11cf`.
- Reviewed tracked bundle:
  `.byte-os/evidence/nadig-jurkat-speed-pilots/` (56 KiB). Checkpoints,
  scientific matrices, H5AD, large logs, and PKL remain absent from Git.

## Residual limitations

1. B0 has no comparable detailed timing receipt and was not rerun.
2. One epoch cannot establish effect equivalence; a longer controlled study is
   a separate future decision.
3. The warmup-excluded serial stage-sum throughput is not actual wall
   throughput under prefetch. Future instrumentation should record a monotonic
   measured-window wall directly; this does not change the completed epoch-wall
   comparison.
4. The merged HDF5 fallback was tested and enabled but not exercised in these
   complete-control-cache runs.

No blocker remains for delivery of the requested speed pilot.
