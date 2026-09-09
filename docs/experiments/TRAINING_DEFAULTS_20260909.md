# Forward defaults: batch512, step LR/EMA, loss1/1/1/0.1

User decision2026-09-09. Applies to future native GraD-Pert experiments unless
explicitly overridden. Historical configurations and results are immutable;
do not edit a completed run to imply it used this recipe. External baselines
retain their official hyperparameter policy. Existing configs are sealed
historical definitions, not the source of future training defaults.

These values must be written explicitly in each standalone experiment config;
this document is not a hidden runtime inheritance chain. Model-specific
architecture, view design and feature initialization are preserved separately.

| Setting | New default |
|---|---|
| Nominal train batch |512, one GPU/process, no gradient accumulation/DDP|
| Evaluation batch |256, unchanged|
| Optimizer / weight decay |AdamW /0, unchanged|
| Base LR |2e-4 at nominal global batch1024|
| Scaled peak LR |2e-4 sqrt(B/1024); at512,0.0001414213562373095|
| LR schedule |Per optimizer step;16% linear warmup from0, then cosine toward1e-6; no restarts|
| Teacher EMA |Cosine .994 toward1 over the same planned step budget|
| Prediction / condition / masked-node / spread |1 /1 /1 /0.1|
| Current run budget |Max100 epochs, validation patience10, seed1|
| Artifacts |metrics_only, zero persistent PKL, best.pt only|

The scalar `training.learning_rate` stores the scaled peak, not the base LR.
Batch size counts cells; the existing eight-condition cap remains. It does not
imply512 independent SSL conditions. Partial batches do not rescale LR again.
Early stopping remains enabled and may stop during warmup. Preserve centers,
temperatures, precision, split and ordered300-control/truth evaluation.

## Three prepared original-model experiments

| ID | Immutable base configuration | Main architecture |
|---|---|---|
|B0-original-step512|b0_historical_b2_e3_schedule_batch128|Historical full6506-node B2, STRING+GO adaptive GAT, E3|
|B1-original-step512|b1_historical_b2_schedule_batch128|Same historical B2, learned-ID instead of E3|
|A1-original-step512|a1_e3_l1_m1|HVG512+targets, single-STRING GAT, E3+Fanout|

Paths: configs/combinations/{b0,b1,a1}_original_step_batch512_loss111_epoch100/
gradpert_b2/nadig_jurkat.yaml. No compact capacity profile. B0/B1 retain
4-layer towers, embedding128, projector2048/256, basal/decoder512,8 fixed512
RingInduced locals with4 anchor-masked. A1 retains its own4 Fanout locals at
half global size and no local anchor masking. All original model parameters
except EMA start and the specified loss weights must match the base exactly.

This is a joint training-recipe change, not a one-variable ablation. These are
prepared configurations, not running jobs or completed scientific results.
Before CUDA: clean publication/synchronization, server gates, capacity checks,
fresh roots and a one-epoch integration smoke; no automatic retry on failure.
The earlier A0 and A1 batch1024 preparations were never launched and must not
be used by a stale runner or monitor.
