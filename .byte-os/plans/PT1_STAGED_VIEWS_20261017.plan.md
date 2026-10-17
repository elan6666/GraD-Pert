# PT1: prioritized SSL pretraining then MSE finetuning

User approved the discussed two-stage mode and requested it run before DP2.
Preserve all completed runs and uncommitted singleton-DP2 preparation. Baseline
c5a8e598c4cd3dbc5e1b3b9697badaf73693d7d1 was clean/published before those edits;
live main still matches it before this change. No active GPU job/monitor.

PT1 keeps current B0 architecture/data/expression holdout: Jurkat cap40, seed1,
global320=micro80*2GPU*accum2, four total epochs/600 updates, no validation/best.
Epochs1-2: SSL1 condition1 + SSL2 DINO1/iBOT1, node/spread/KoLeo0, no perturbed
expression reading and no prediction-head forward/update. Epochs3-4: inherit
Student backbone/normalization/CLS, train all prediction-path parameters using
only row-mean prediction MSE. Preserve initial untouched prediction head.
Reset Muon/AdamW moments exactly once at transition, retain global step counter,
stop Teacher/center/temperature updates. Pretrain uses existing warmup16% cosine
over its own300 updates (1e-3 ->2e-4); finetuning constant2e-4/no new warmup is a
project setting. This is a composite training-mode ablation; only300 MSE steps,
versus600 for B0/D0. Do not claim matched supervised budget or causal isolation
of reset/schedule choices. Save pretrain checkpoint and phase-aware resume.

One published config/identity and four-epoch journal records both stages and
the exact boundary; interrupted stages resume with their own complete state.
Historical joint/default checkpoints and full-state continuation remain strict.
Final last18 uses frozen300 controls and all/seen/unseen x full/DEG. Same medium
population remains1335/5317, deep128/506, gradient32/125; no full-data expansion.

Extend future common diagnostics with SSL1 condition and SSL2 response-CLS
Global/Local feature matching: same row, same perturbation/different control,
different perturbation/different control, and different perturbation/same control
when available. Stratify cell line/type and context/batch; do not merge cross-cell
effects or count repeated controls/views as independent cells. Report GG/GL/LL,
raw/centered cosine, squared difference, true rows/controls/conditions/pairs,
per-view centered effective rank/relative variance, undefined groups explicitly.
Use server-only small feature banks from existing deep forwards; do not transfer
per-cell matrices. In MSE-only stage, read-only Student view probes are labeled
diagnostic-only, with no Teacher target/CE or center updates. Do not infer collapse
from raw cosine alone. Chinese template explains axes/units/N and links all data.

Acceptance: directed CPU output/gradient and nonzero-LR multistep tests; y-value
invariance and no decoder/target I/O in pretrain; first MSE update/reset; exact
resume before/after boundary; Teacher/centers fixed in finetune; grouped-view
identity/count fixtures and read-only parity. Publish scoped changes before GPU.
Fresh dual-GPU10-update engineering gate exercises5SSL+5MSE and exact checkpoint
restore, records shortened gate boundary separately from formal300. Eval128 on
both physical GPUs/reference128 remains unchanged. Main verifies gates, launches
formal PT1, verifies actual progress then hands off to designated20min supervisor.
On terminal return, main accepts artifacts and advances already authorized DP2,
then S12-L4 and remaining family; no reauthorization for unchanged dependencies.

Performance follow-up authorized: eliminate unused SSL decoder/concat and stage-specific
I/O/Teacher/view work, preserve exact nonzero-LR updates. Compare published idle-GPU
10-step unused-decoder reference with actual optimized gate; report both phase costs,
startup/data gaps and peak memory. CPU grouped-view pair-layout reuse has exact aggregate
parity; 506x256/4-view macOS ABBA component1.4981s->0.9794s (1.53x), synthetic not
end-to-end evidence. Server replication and GPU receipts remain required. Static main
graph query features are reused only for unchanged static-graph diagnostic views;
conditional graph paths remain per-control. No changes to model/math/masks/order/RNG.
