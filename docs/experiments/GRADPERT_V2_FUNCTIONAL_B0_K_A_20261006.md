# Jurkat cap40: B0 / K / A functional ablations

Status: implementation and validation stage; no scientific result claimed here.
Authorization: user 2026-10-06, after accepting the preceding epoch10 evaluation.
Historical E23 source e27f49166dae688eb35444331a1ef1916238efe7 is immutable.
The clean pre-change source is 5d6db5735c1bcb3f6a62ab88dc40a67227026a2f,
which differs from E23 only in documentation. The published parent configuration
has SHA256 7214e3e382232ae8856998544514c54aecd048aeec2ab3cb8f329c7157c08df8.

## Frozen protocol

Five independent fresh models, seed1, six complete epochs, no validation during
training, final epoch6 frozen test only. Queue order B0 -> K1 -> K2 -> A1 -> A2.
There is no validation-selected best checkpoint; best is null, last is epoch6,
and selection_metric is final_epoch. Intermediate committed last checkpoints
remain resumable and are pruned after the next epoch commits. Training curves
record all saved components without inventing validation values.

All rows retain the same train-only cap40 selection (SHA256
 aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132),
canonical split, original 300-control manifests, six overall Pearson outputs
(three definitions, all/DEG) plus seen/unseen-expression groups. The preceding
run's expression exclusion=true is preserved explicitly. Test results do not
select settings. No H1, O, C, G or L rows are authorized by this queue.

The initial common batch is 272 = micro68 x accumulation2 x world2. Every row
must pass its exact clean-published-source/config 128-update capacity probe,
checkpoint reload and inference gate before any formal row starts. A failure
stops the queue with its evidence preserved; no silent per-arm batch fallback.
If a common smaller batch becomes necessary, regenerate/publish all five configs
under a new identity and recheck the five arms before training.

Optimizer GLM5MuonSplit_v2, LR 1e-3 -> 2e-4 warmup+cosine over this six-epoch
budget, weight decay0, mHC4, width256, four heads, FFN4d clipped SwiGLU,
projectors 256->2048->256->16384. SSL1 condition/node/spread=(1,1,0),
SSL2 DINO/iBOT/KoLeo=(1,1,0), lambda1=lambda2=1, row_mean with existing exceptions.
Teacher is isomorphic EMA; optimizer -> EMA/centers commit order is unchanged.

## Single-variable arms

| Arm | Difference from B0 | Preserved paths |
|---|---|---|
| B0 | Current full E23 method, new six-epoch no-validation budget | Graph3 KDA+1 sparse MLA; Cell2 KDA+1 MLA; Response2 self/cross KDA+1 self/cross MLA |
| K1 | All KDA write genes forward then reverse with carried state | Final query read; CLS writes once at the end; graph legal neighborhoods |
| K2 | Cell/Response self-KDA genes read updated prefix S_t | Graph and cross KDA final S; CLS final S; restore gene ID positions |
| A1 | Replace every graph/Cell/Response self/cross KDA and MLA with noncausal softmax attention | Graph masks/source inputs/gate/bias, all residuals/FFNs, three e_p injections and control cross |
| A2 | Same complete replacement scope, with ReLU retention core | Same task paths; legal graph source adaptation described below |

B0 writes S_0=0, A_t=D_t odot S_(t-1),
e_t=v_t-A_t^T k_t, S_t=A_t+beta_t k_t e_t^T.
All queries read S_final^T q_g after the tail CLS's single write. Graph layers
build independent S for each target's GO20/STRING20/three seeded Hamiltonian
rings/self union; graph has no CLS and outputs only that target's query.

K1 uses the same projected writes twice, with S_forward as reverse initial state.
K2 reads S_t^T q_t in the randomized order, then applies the inverse permutation
before output. It uses the existing chunk triangular solver, exposing its final
state for the one CLS write without rescanning genes. K2 is a method ablation,
not an equivalent speed optimization. Chunk sizes remain graph32 / sequence16,
graph target rows64 in all KDA configurations. No compiler option is introduced.

A1: O=softmax(QK^T/sqrt(d_h)+edge_bias)V, noncausal.
A2: Q'=ReLU(Q)/sqrt(d_h), K'=ReLU(K)/sqrt(d_h),
O=Q'(K'^T V), followed by non-affine scaled RMS normalization per head,
SiLU query output gating and W_O. Self and control cross share the same core;
cross Q comes from response, K/V from control genes, with no control CLS added.

The first three graph layers retain source embedding on legal neighbor inputs.
The fourth graph layer has no KV compression in either replacement. A1 preserves
gate K_ij=K_j odot (1+W_E s_ij) and bias b^T s_ij in softmax.
For A2's fourth layer, an explicitly local adaptation uses
w_ij = exp(b^T s_ij) ReLU(q_i)^T ReLU(K_ij)/d_h,
O_i=sum_(j in N(i)) w_ij V_j, then the retention head normalization/output gate.
This adapts edge-source information to unnormalized retention; it is not a
claim that CellFM supplies a graph attention or reproduces our edge mechanism.
Values, neighborhood limits and source bits remain aligned to gene IDs.

Randomized scan order, view distributions and separate view RNG remain frozen.
Architectures consume different operator/dropout random streams and have different
parameter counts; equal seed is not a claim of bit-identical random draws across
arms. Checkpoint recomputation within each arm must preserve output, gradient,
optimizer, EMA, centers and RNG. Softmax/retention are permutation-equivariant
cores in exact arithmetic; other stochastic operations still require verification.

## Primary reference audit for A2

CellFM: Zeng et al., Nature Communications (2025),
https://www.nature.com/articles/s41467-025-59926-5 . Official repository:
https://github.com/biomed-AI/CellFM , audited commit
72c9f4a9580a3716058c184900ed14a65151ed8f.
Source: retention.py, MHRetention / SRMSNorm:
https://github.com/biomed-AI/CellFM/blob/72c9f4a9580a3716058c184900ed14a65151ed8f/retention.py .
Observed: bias-free q/k/v/u, ReLU Q/K, both sqrt(head_dim) scales,
K^T V first, non-affine scaled RMS head norm, SiLU u gate and output projection.
The declared pre_norm is not invoked inside this core. Full ERetNet's FFN,
outer norms/residuals, official widths and MindSpore runtime are not imported.
Official code's CC BY-NC-ND license is recorded; implementation here is an
independent native equation adaptation, with no source copying or rebranding.

## Acceptance and operational boundary

Synthetic tests verify formulas/output gradients, graph illegality isolation,
K2 identity/prefix/CLS semantics, two nonzero-LR complete updates and checkpoint
recomputation parity (optimizer, Teacher, centers, RNG), plus interrupted-epoch
resume and final-only collector corruption gates. They do not prove scientific
model quality. CUDA capacity evidence precedes fresh training, not vice versa.

Use scripts/v2/generate_functional_group.py to regenerate self-contained configs
and their manifest, and scripts/v2/run_functional_group.py to seal exact source,
config/runtime and predecessor evaluation identity. Reuse the existing native
execution, GPU idle/lease gates and sequential queue loop. No external model
runtime or duplicate training main is added. Immutable server checkout, new
queue and run IDs, expandable_segments allocator and zero-PKL remain required.

A long queue transfers to the existing supervisor only after durable launch,
exact ACK and one active 20min monitor. Failure or terminal return pauses the
monitor and wakes main for independent acceptance and the next authorized step.

Local verification (2026-10-06): directed202 passed; broader590 passed, five
failures reproduced on clean pre-change source, eleven unavailable CUDA/Triton
skips. Final affected20 tests, Ruff, format and six-file mypy pass. Evidence:
.byte-os/coordination/receipts/functional-local-verification-20261006.json.
Published-source server CPU checks and exact128-update CUDA gates remain pending.
