# R50 GLM-5 optimizer and schedule ablations

Decision: 2026-09-12. User authorizes exactly three new scientific rows:
optimizer only, schedule only, and their combination. DeepSeek optimizer,
Sinkhorn embedding updates, and DeepSeek LR rows are NOT queued.
Status: planned/queued for preparation; not implemented or launched.

## Queue position

Current sched512, batch1024 and batch128 must finish training and automatic
best/last tests first. Then complete the already authorized measured performance
engineering, exact-effect gates, three-epoch old/new comparison, and isolated
ABBA timing. Only after acceptance/publication prepare this GLM block, before
continuing the remaining R50 EMA/capacity/method stages. Do not alter active
checkouts, interrupt current runs, or launch when a GPU merely becomes free.
Existing heartbeat: grad-pert-r50; 30-minute short visible updates.

## Frozen common parent

Use the user-selected R50 batch512 E3 parent, fixed peak LR 0.001, AdamW,
weight decay 0, seed1, 50 epochs without early stopping, evaluation batch256.
Keep its original EMA .996-to-1 on the full 50-epoch step horizon, losses
1/.8/.4/.1, four half-graph RingInduced locals, HVG512+targets, STRING+GO
Exphormer-MG, Protein+Reactome+SIGNOR E3 initialization, model dimensions,
precision, split and ordered controls/truth unchanged. No batch warmup,
accumulation, DDP or new model architecture. Resolve all fields from the
actual parent config and seal hashes before launch; no hidden defaults.

The existing completed batch512 is the fixed-LR AdamW control, not sched512.
Reuse it only with verified complete receipts, exact scientific configuration,
and the accepted implementation equivalence evidence; label any cross-commit
comparison. It cannot supply isolated timing comparisons. If equivalence or
control evidence is insufficient, report and request a new control rather than
silently adding a fourth 50-epoch experiment.

## Exactly three new rows

| ID | Optimizer recipe | LR schedule | Direct comparisons |
|---|---|---|---|
| R50-G1 / glm5_muon_only | GLM-inspired Muon Split + auxiliary AdamW | constant .001 base LR | parent: optimizer recipe |
| R50-G2 / glm5_schedule_only | unchanged parent AdamW | linear warmup + cosine to .2 peak | parent: schedule |
| R50-G3 / glm5_muon_schedule | identical G1 optimizer recipe | identical G2 schedule | G1: schedule; G2: optimizer |

Report the combination against the parent as a two-factor change, not a
single-variable ablation. This block does not isolate head splitting from
vanilla Muon; do not claim that it does. No extra LR sweeps are authorized.

## GLM-inspired optimizer migration

Primary source: GLM-5 report https://arxiv.org/html/2602.15763v2,
section 2.1 (Muon Split) and appendix A (inherited GLM-4.5 training recipe).
GLM splits MLA Q/K/V up-projection matrices by attention head before
orthogonalization. Our graph encoder is not MLA: adapt only the actual Q/K/V
head structure; do not add MLA, change forward computation, or fabricate heads
for matrices that have no head axis.

Eligible hidden 2D linear weights (graph, projector MLP, basal and decoder
hidden layers) use Muon. Attention Q/K/V matrices use per-head Muon with the
actual axis/layout audited from native code. Gene embedding including E3,
final expression output, DINO weight-normalized prototype output, norm/bias/
scale parameters remain on unchanged auxiliary AdamW. Teacher uses EMA only.
Never select parameters solely using ndim==2. Seal a complete disjoint mapping
of names/shapes/roles/optimizer/head axes; every trainable parameter exactly once.

Before implementation freeze an official upstream algorithm reference and
commit, including the GLM-4.5 recipe inherited by GLM-5: momentum definition,
Nesterov, Newton-Schulz coefficients/count/precision, matrix orientation,
shape/RMS scaling, zero-gradient handling, LR correction and epsilon.
Missing publicly specified details must be labeled project_preregistered,
not invented as GLM defaults. Use a documented AdamW-compatible update-scale
variant so .001 is the common base LR; record its actual correction formula
and per-group effective LR. This is not evidence that .001 is optimal for Muon.
G1 and G3 must have byte-identical optimizer settings apart from schedule.
Preserve parent weight decay0 and auxiliary AdamW settings to avoid adding
regularization/beta changes. This is a GLM-inspired controlled transfer, not
a full GLM recipe reproduction. No Sinkhorn or embedding optimizer change.

## Explicit schedule migration

GLM-5 pretraining warms up to 2e-4 then cosine-decays to 4e-5 (20% of peak).
Its later mid-training linear decay is NOT transferred because we do not have
a corresponding separate data/context stage. Absolute LR, batch warmup and
language-token training durations are not copied.

Our registered schedule is peak .001, warmup 8/50 epochs, end .0002,
without restart or plateau. The 8-epoch warmup is project_preregistered,
shared with the existing R50 warmup design, not an official GLM duration.
Let S be verified steps/epoch, T=50*S, W=8*S and s in [0,T-1].
For s<W use .001*s/W. Otherwise use
.0002 + .5*(.001-.0002)*(1+cos(pi*(s-W)/(T-1-W))).
First used LR=0, at s=W LR=.001, last used LR=.0002.
Require T-1>W. Expected S335/T16750/W2680 must be checked against runtime,
not assumed. EMA has its own unchanged parent endpoint arithmetic.
Resume uses the same global optimizer-step index. Short smoke truncates
execution on this 50-epoch horizon; it does not compress the schedule.
Existing sched512 has a different floor (1e-6) and is not a duplicate of G2.

## Preparation and acceptance

Use the existing native training entrypoint with self-contained configs;
extend schema/optimizer checkpoint support in a new clean main publication.
Do not patch a running source tree. Preparation must implement and test
parameter routing, per-head reshape/inverse, full-matrix vs split reference,
finite/zero/rank-deficient gradients, schedule boundary values, one optimizer
step counter, EMA/center update order, and full mixed-optimizer save/resume.
Keep parent AdamW checkpoint compatibility. Log group update norms, attention
logit/entropy health, four losses, gradient norms, LR, teacher/center state,
optimizer/step wall, throughput and peak memory. No expensive extra full
diagnostic pass on every production step.

Muon intentionally changes training trajectories: do NOT demand Muon equals
AdamW outputs, gradients or weights, and do NOT describe it as exact-effect
performance optimization. Test implementation against its frozen algorithm
reference and resumed vs uninterrupted own trajectory. Existing performance
optimization still requires its separate strict old/new equality gates.

Full pytest/Ruff/format/mypy/build gates and exact clean local/GitHub/server
identity precede CUDA. Each new recipe/coordinate passes a fresh one-epoch
train/validation integration with no test access, then starts a fresh50 run.
Require finite state and capacity safety; a failed smoke blocks that row and
its dependent combination, preserving evidence without silent LR changes.
Before launch get_goal must show no active goal. Bounded preparation goal ends
before CUDA; pause this workflow monitor while that goal is active, restore
the same 30-minute monitor for background jobs. Every native CUDA process sets
PYTORCH_ALLOC_CONF=expandable_segments:True. Only bounded live resource/capacity
checks authorize concurrency; do not kill others or interpret concurrent wall
as isolated optimizer speed. Fresh roots, no automatic failed-root relaunch.

Full rows: 50 epochs, 50 validations, 50*S ordered optimizer steps; automatic
canonical best.pt AND true last.pt tests immediately after training subject
to memory safety (no whole-GPU-idle wait). Preserve both checkpoints and
deduplicate inference if identical by hash with explicit alias receipts.
Same three metrics and ordered 300-control/truth/split hashes, metrics_only,
no work directory and zero persistent PKL across each successful run root.

Compare best validation, final10 validation mean, best epoch, stability,
wall and peak memory. Parent selection uses validation only with the existing
R50 .002 practical tie rule; test best/last results are reporting only.
Report equal50epochs as equal exposure, not necessarily equal compute.
Measure time-to-matched-validation only with comparable hardware/load evidence.
Single seed1 is screening, not statistical significance. After all three
complete/review, freeze the selected recipe before resuming remaining R50.

## Algorithm audit in progress (2026-09-12)

Verified pre-change published baseline:
`6e53fc2dbcc76ed7e0adc1d3229d6d616ce72344`.
Muon author reference inspected at immutable commit
`f98f1cacc0263b04290753e32be8d498c1efc806`, file `muon.py`,
https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py.
This is an author algorithm reference, NOT GLM's released training code.
Its update uses momentum EMA beta .95, Nesterov interpolation, five
Newton-Schulz iterations with coefficients (3.4445, -4.7750, 2.0315),
BF16 internal matrices, Frobenius normalization plus 1e-7, and transpose
when rows exceed columns. Its shape correction is sqrt(max(1, rows/cols));
this is not yet the registered AdamW-compatible LR correction. Freeze that
variant separately before implementation. Auxiliary AdamW must retain our
parent defaults, not this reference's different defaults. The reference
mutates gradients and fills absent gradients with zeros; our handling must
be explicitly documented and tested, not accidentally inherited.

Native audit: `_SparseGraphTransformerLayer` in modeling/encoders.py owns
separate query/key/value Linear weights and reshapes their outputs into
(nodes, head_count, head_dim). Therefore head splitting uses contiguous
output-row blocks of weight[out,in]. Its separate edge projection and the
local GAT branch are not Q/K/V and must not be mislabeled as Muon Split.
Complete parameter routing, numeric variant freeze, implementation and
CUDA preflight are still pending; this document is not launch approval.
