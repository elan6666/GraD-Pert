# A3: A0 plus E3 with user-selected learning rate and batch

User authorization (2026-09-07): start on GPU0 after A1 completes,
without waiting for A2 on GPU1. A4 is canceled.

A3 retains the recorded E3 architecture: A0's STRING+GO Exphormer-MG,
RingInduced four locals at one-half of the actual global node count,
HVG512 plus targets, additive decoder and 64-wide perturbation output.
The gene prior is the same hash-pinned Protein+Reactome+SIGNOR initialized
trainable embedding. No Fanout, single-source GAT, or D5 is added.

The self-contained config is
`configs/combinations/a3_a0_e3_lr5e5_batch128/gradpert_b2/nadig_jurkat.yaml`.
Training learning rate is explicitly 5e-5, training batch128, evaluation
batch256 unchanged, seed1, AdamW without weight decay or scheduler.
The user-selected rate is 20 times below A1/A2; this joint rate/batch
comparison is not a single-factor ablation and is not selected from test scores.

The combination budget is max100epochs with validation-only patience10,
preceded by exactly one epoch integration smoke. Full training starts from
scratch in a separate root after smoke acceptance. Derive steps per epoch
from the actual batch128 loader; never assume the batch256 count of582.
Preserve canonical split and ordered300-control/truth manifests, perform
one evaluation from best.pt, and retain zero PKL with metrics_only.

A2's active checkout is immutable. Publish a separate clean source checkout
and gate it before CUDA. Smoke and formal completion remain unclaimed until
their exact receipts pass verification.

## Learning-rate revision and stop (2026-09-07)

The user stopped A3 and changed its learning rate from 1e-7 to 1e-5.
The old config remains immutable historical evidence; it is superseded for
future A3 launches. The full run `a3-5dd683a-full-v1` was terminated by
targeted process-group SIGTERM, with 576 persisted steps (epoch0, last
global_step575). Its files remain on the server and are not a completed run.
The A3 process and GPU allocation were confirmed absent; GEARS, TxPert and
Scouter remained active. The revised A3 is configured but not relaunched.
A future launch requires a fresh clean source and run root, not in-place
learning-rate mutation or resumption of the old optimizer state.

### Latest revision: 5e-5

The user subsequently requested 5e-5 and immediate training. The 1e-5
attempt `a3-e4d680a-lr1e5-full-v1` is stopped and preserved, not completed.
The new configuration changes only the learning rate and artifact-root label;
batch128, model, split, seed, optimizer and max100/patience10 remain fixed.
Launch from scratch in a fresh published checkout and run root on GPU0,
alongside GEARS. TxPert and Scouter remain untouched on GPU1.
