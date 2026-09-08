# B0 capacity series C1/C2/C3 (200 epoch ceiling)

Authorized2026-09-09. C numbers avoid collisions with historical B2/B3 model
versions. Three additional E3/B0 rows, not three B0/B1 pairs. Reference is the
running compact B0 at source224475e,6,338,568 total parameters,49.417367:1.

| Row | Graph depth/source | Projector hidden | Bottleneck | Complete parameters | Parameters/train cell |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 reference | 2 | 256 | 32 | 6,338,568 | 49.417367 |
| C1 | 3 | 512 | 32 | 8,359,944 | 65.176617 |
| C2 | 3 | 512 | 96 | 10,522,760 | 82.038576 |
| C3 | 4 | 768 | 96 | 12,839,048 | 100.097048 |

Counts include Student, Teacher and all heads, not just trainable parameters.
Denominator128,266 is training perturbed cells; no validation/test cells.
These are instantiated CPU parameter counts, not performance measurements.

## Fixed factors

Initial gene embedding128; each GATv2 layer two heads128; STRING+GO adaptive
fusion; node/condition/basal output64; basal/decoder hidden128; additive b+p;
16384 prototypes; E3 artifact SHA
34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318.
Same6506-node full graph,5000 expression genes,8Ring locals512 with4masked,
loss1/1/.1/.1,seed1,batch1024/eval256,no accumulation or DDP.
Same stepLR peak2e-4,16% warmup/cosine and EMA.994 toward1.
max200,validation-only patience10,one test evaluation of best.pt per lifecycle,
metrics_only/whole-root zero persistent PKL.200 is a ceiling, not guaranteed length.

Only capacity_profile (receipt label), graph depth, projector hidden and
bottleneck may differ from B0. No decoder/embedding changes. Some transitions
change more than one capacity dimension: this is a capacity-series experiment,
not independent depth-vs-width causal attribution. Cross-commit reference is
explicitly retained; no reference rerun or effect-equivalence claim.

## Queue and safety handoff

Publish a new clean source without modifying running224475e checkout.
GPU0: after running B0 complete/verified and GPU0 released, C1 then C3.
GPU1: after running B1 complete/verified and GPU1 released, C2.
Each C row gets a fresh one-epoch integration smoke before fresh full fit;
smoke metrics are non-decisional, not used to select configs. Existing runner
validates receipt identity, evaluated lifecycle,checkpoint and zeroPKL before
advancing. Failed predecessors block their own successor, never auto-retry.

Current GPUs each use about8.7GiB, but no real C-series concurrent peak/throughput
gate exists. Do not infer safe co-residency from that snapshot: queue one row/GPU.
Parallelize across GPUs when individually free. No changes to running B0/B1.

Use existing hourly monitor after preparation goal completes. Persist exact
source/publication/launcher hashes and queue dependencies in server contract.
On each check inspect current sessions/receipts/resources once; dispatch a ready
row only with no active goal, exact clean source, verified prerequisite, absent
new root/session/log and an idle physical GPU. No sleep-poll queue workers.
Record launch identity before returning. Notify only meaningful transitions,
failure,completion or required action. Two checks with no progress trigger
read-only stall analysis, not relaunch. Keep temporary evaluation PKL untouched.
At all-terminal validate shared data/split/control/truth identity, metrics_only,
best.pt/test-once and stop reason; disable monitor before analysis goal.

## Verification

CPU targeted suite22 passed: complete-model counts on6506/5000/16384,
fixed scientific-factor checks, baseline counts and both schedule/checkpoint
resume paths including all three new E3 models. Synthetic prior matrices in
unit tests do not replace production-prior identity checks. Full source gates
and publication are required before the prepared queue can dispatch.
