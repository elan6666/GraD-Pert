# B2-vNext Decoder Fusion and Perturbation-Width Factorial

## Objective

Measure two independent decoder factors on the frozen Nadig Jurkat A0 data,
graph, view, loss, training and evaluation contract:

1. decoder input fusion: direct concatenation versus concatenation plus a
   TriShift-aligned two-token Transformer interaction;
2. perturbation state width: 64 versus 256.

This module does not reinterpret the completed historical D1/D2 rows. It uses
fresh D3--D6 IDs and a new formal lineage.

## Frozen rows

| Row | Perturbation width | Decoder input |
|---|---:|---|
| D3 `d3_concat_p64` | 64 | `concat(b64, p64)` |
| D4 `d4_concat_transformer_p64` | 64 | `concat(b64, p64, T([b64, p64]))` |
| D5 `d5_concat_p256` | 256 | `concat(b64, p256)` |
| D6 `d6_concat_transformer_p256` | 256 | `concat(b64, p256, T([b64, Wp(p256)]))` |

At the sealed 2,809-node graph, 5,000-gene expression axis and 16,384
prototypes, the preflight parameter counts are:

| Row | Decoder input | Decoder params | Interaction params | Total params |
|---|---:|---:|---:|---:|
| D3 | 128 | 2,632,072 | 0 | 25,678,424 |
| D4 | 256 | 2,697,608 | 49,984 | 25,793,944 |
| D5 | 320 | 2,730,376 | 0 | 26,612,696 |
| D6 | 448 | 2,795,912 | 66,432 | 26,744,664 |

`b` is always the 64-wide basal/control representation. `T` receives exactly
two ordered tokens, control first and perturbation second. It is one 64-wide
pre-norm Transformer encoder layer with four heads, a 256-wide GELU FFN,
dropout zero, no position/type embeddings and a concatenated 128-wide readout.
For the 256-wide perturbation rows, `Wp` is a learned linear 256-to-64 token
projection. The raw 256-wide perturbation representation is still supplied to
the expression decoder; only the Transformer token is projected.

The expression decoder keeps the A0 operator order and hidden width:
`Linear(input,512) -> BatchNorm -> LeakyReLU -> Dropout(0.2) -> Linear(genes)`.
The shared DINO/iBOT projector accepts the selected perturbation width, so the
Teacher and Student remain structurally identical.

## Identifiable comparisons

- D3 versus D4: Transformer interaction effect at `p=64`.
- D5 versus D6: Transformer interaction effect at `p=256`.
- D3 versus D5: perturbation-width effect with direct concat.
- D4 versus D6: perturbation-width effect with Transformer concat.

D0/A0 may be shown as contextual additive evidence, but D5/D6 are not direct
single-factor comparisons to A0 because they change both fusion and width.

## Execution contract

Before formal training, all four rows require config/hash review, synthetic
shape/gradient/checkpoint tests, a bounded training-only capacity gate and a
fresh clean local/GitHub/server source identity. Formal rows retain seed 1,
batch 256, four RingInduced half-graph locals, 16,384 prototypes, exactly 10
epochs/5,820 ordered steps/10 validations, no early stopping, one test
evaluation from `best.pt`, three native metrics, `metrics_only` and zero
persistent PKL. Trackio was explicitly disabled for this lineage; native
receipts are authoritative.

## Formal results

Lineage `formal-vnext-decoder-75a2c2b-v2` completed all four rows at exact
source `75a2c2b`. Every row passed 5,820 ordered steps, ten validations, one
evaluation from `best.pt`, the exact three finite metrics, zero persistent PKL
and best-only checkpoint retention.

| Row | TxPert macro delta | TriShift delta | Systema Pearson | Mean step wall |
|---|---:|---:|---:|---:|
| D3 concat, p64 | 0.247715 | 0.247435 | 0.025971 | 2,599 ms |
| D4 concat+T, p64 | 0.209731 | 0.272646 | 0.003092 | 2,616 ms |
| D5 concat, p256 | 0.233277 | 0.197498 | 0.035529 | 2,573 ms |
| D6 concat+T, p256 | 0.221367 | 0.201207 | 0.008773 | 2,768 ms |

At p64, D4 minus D3 is `-0.037984/+0.025210/-0.022879` across the
TxPert/TriShift/Systema metrics. At p256, D6 minus D5 is
`-0.011910/+0.003709/-0.026756`. Width 256 versus 64 is also mixed under both
fusion routes. No factor improves all three metrics consistently.

The completed four-local additive A0 is contextual cross-commit evidence:
`0.251406/0.245002/0.018933`. D3 is closest on the TxPert primary point
estimate, while D4 leads TriShift and D5 leads Systema. Because these are
single-seed estimates, no equivalence or superiority claim is made and A0
remains the preregistered default. D5/D6 differ from A0 in both fusion and
width, so they are not direct single-factor comparisons to it.

Reviewed compact evidence is at
`.byte-os/evidence/vnext-performance/formal-decoder-75a2c2b.json`; large
scientific artifacts remain on `/data/yilangliu`.
