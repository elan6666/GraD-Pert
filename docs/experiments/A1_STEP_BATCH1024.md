# A1 with the current LR, EMA and batch recipe

**Superseded before launch:** use TRAINING_DEFAULTS_20260909.md and the
original-model batch512/loss111 configs. No batch1024 A1 CUDA was launched.

2026-09-09 user correction: A1, not A0. The prepared A0 recipe was never
launched; its source and CPU checks remain historical, not experiment results.

New ID: A1-step1024-100. Base: a1_e3_l1_m1. Keep E3 initialized GenePT,
Fanout four locals at half the actual graph node count, single-STRING GAT,
HVG512+targets, additive decoder and all original widths and losses unchanged.
Only Teacher EMA start .994, nominal training batch1024 and the step schedule
change: peak2e-4 at1024, sqrt(B/1024),16% linear warmup from0, cosine toward
1e-6, Teacher cosine .994 toward1. AdamW/weight decay0, eval256 unchanged.
Max100 epochs with existing validation patience10; seed1, no DDP/accumulation.
No compact B/C architecture changes. Early stopping may occur during warmup.

Use the same canonical Nadig Jurkat split and ordered300 control/truth recipe.
GenePT path and exact SHA are inherited from original A1. Native implementation
only, metrics_only, zero PKL and best.pt only. This changes three training
factors jointly and is not a single-factor LR ablation.

## Handoff

Full source gates and clean local/main/server commit before launch. Fresh root
a1-step1024-<commit>-v1, one idle physical GPU. Existing phase runner --row a1:
exactly one smoke epoch then fresh full after successful smoke. If smoke fails,
preserve evidence and do not lower batch or retry automatically. Allocator
expandable_segments:True and measured implementation switches are pinned in
the launcher. End preparation goal before CUDA. Hourly quiet monitor this one
row only: inspect phase/RC/source/config/steps/validations/zeroPKL/resources;
two unchanged snapshots trigger read-only diagnosis. On completion verify
test-once/best checkpoint/ordered controls and truth, then delete monitor.
Never launch the mistakenly prepared A0.
