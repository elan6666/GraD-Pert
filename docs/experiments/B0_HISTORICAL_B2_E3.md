# B0: historical B2 plus E3

User specification, 2026-09-08. This new combination is not the historical
performance pilot B0. Preserve every old B0/B2 result and namespace.

Status: training authorized; launch preparation in progress. No capacity gate,
smoke or training launched yet.

Base: historical B2 at commit 6597804, canonical full Jurkat graph (recorded
6,506 nodes), STRING+GO independent adaptive-relation GATv2 towers, learned
128-wide gene inputs, 64-wide perturbation output and additive Decoder(b+p).
Keep Top20 per source, two globals, eight RingInduced locals with fixed
512-node caps, and four anchor-masked locals. Do not substitute current A0
graph size, ratio locals, four-local count or Exphormer-MG.

Changes authorized for this combination:

- E3: trainable GenePT-Seed initialized embeddings, Protein+Reactome+SIGNOR;
  use the existing sealed artifact and initialization implementation.
- Effective prediction/condition/masked-node/spread weights: 1/1/0.1/0.1.
- Training batch128; evaluation batch256 retained.
- Same scheduler as current A3: epoch-level sequential warmup/cosine restarts,
  min1e-6, first peak1e-4, warmup5, first cycle15, cycle_mult2, gamma0.9.
  Cycle lengths15/25/45/85. AdamW and weight decay0 remain unchanged.
- Maximum100 epochs, validation-only patience10, seed1. This is a ceiling,
  not a requirement to train past early stopping.

Self-contained configuration:
`configs/combinations/b0_historical_b2_e3_schedule_batch128/gradpert_b2/nadig_jurkat.yaml`.
The config uses the existing explicit combination training policy; its B0
identity is carried by this unique config and artifact root.

Before any future authorized launch: publish/gate clean source, verify exact
canonical split and ordered300-control/truth identities and prior coverage,
check capacity without reducing scientific settings, then complete a fresh
one-epoch integration smoke. Start full training from scratch in a fresh
root. Evaluate best.pt once; metrics_only, zero persistent PKL, best-only.
The user subsequently authorized training. Launch only after these gates;
never stop unrelated jobs to obtain capacity or select settings using test scores.
This changes several factors and is not a single-variable ablation of B2.

## Launch preparation

The first runtime audit found that the historical adaptive-relation GAT branch
ignored feature-mode selection. B0 now applies the existing E3 seeded random
projection (seed20260828) to initialize its 128-wide student embeddings, then
copies student state to Teacher through the existing initialization lifecycle.
Learned-ID behavior is unchanged; unsupported historical GAT prior modes fail
closed. A CPU server regression checks exact initialization, Teacher equality,
trainability and rejection of a missing prior matrix.

At preparation time both GPUs were occupied by DinoGenePT (about23--24GiB
each). No safe B0 shared-capacity result exists. Preserve those jobs and wait
for sufficient capacity; available memory is not a successful capacity gate.
