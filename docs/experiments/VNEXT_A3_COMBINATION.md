# A3: A0 plus E3 with user-selected learning rate and batch

User authorization (2026-09-07): start on GPU0 after A1 completes,
without waiting for A2 on GPU1. A4 is canceled.

A3 retains the recorded E3 architecture: A0's STRING+GO Exphormer-MG,
RingInduced four locals at one-half of the actual global node count,
HVG512 plus targets, additive decoder and 64-wide perturbation output.
The gene prior is the same hash-pinned Protein+Reactome+SIGNOR initialized
trainable embedding. No Fanout, single-source GAT, or D5 is added.

The self-contained config is
`configs/combinations/a3_a0_e3_lr1e7_batch128/gradpert_b2/nadig_jurkat.yaml`.
Training learning rate is explicitly 1e-7, training batch128, evaluation
batch256 unchanged, seed1, AdamW without weight decay or scheduler.
The user-selected rate is 10,000 times below A1/A2; this joint rate/batch
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
