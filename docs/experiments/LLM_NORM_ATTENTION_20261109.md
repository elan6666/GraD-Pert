# Residual-entrance normalization and terminal-attention comparison

The user authorized two new B0 R3-RI Jurkat TRAIN40 experiments, each from scratch
for two complete epochs (300 successful optimizer updates), on 2026-11-09.
Existing baseline and the earlier graph-only RMSNorm results are descriptive
references; do not launch another baseline. Both new arms enable the previously
user-selected relay validation reuse and KDA constant-cache execution defaults.

A (`A_norm_mla`) uses the `llm_pre` normalization layout and native terminal
compressed-KV attention, with direct Q. B (`B_norm_standard`) has the same layout
but full-width softmax terminal Q/K/V. B retains every earlier KDA operator,
including the existing randomized-order final-state read, and preserves graph
neighborhoods, edge masks, source key gates and source bias. It replaces the
cell final self read, response final self read and final cross read, and removes
the graph final sparse read's KV compression. Cross queries come from response;
keys/values come from control. Terminal token reads remain noncausal, including
existing CLS visibility and evaluation interventions. Graph attention never
becomes dense over all genes. Same width and heads do not imply equal parameter
counts: report actual model and optimizer-route inventories.

## Normalization contract

- Graph prior mapping and expression mapping have no terminal LayerNorm, and
  neither raw inputs nor the mapping MLP interiors get a new normalizer.
- All four graph attention and FFN entrances use RMSNorm, preserving the old
  epsilon and scale at each replaced LayerNorm and removing its beta.
- A new RMSNorm (eps1e-6, scale1) applies once to the final fourth graph read,
  not to its three-layer cache prefix. Teacher and student use the same rule.
- Cell/response entrance and final RMSNorm remain native. Each of the three
  response condition-injection residual branches gets RMSNorm on the concatenated
  2d input, before its MLP; the residual bypass remains unnormalized.
- Ordinary prediction and projector MLPs, concatenated normalized encoder outputs,
  scalar predictions and post-residual streams get no additional normalizers.
- Existing KDA Q/K L2 and per-head read RMSNorm, mHC routing RMS/Sinkhorn, SSL
  bottleneck L2, teacher centers, probabilities and temperatures remain unchanged.
- A retains compressed-KV latent RMSNorm. B removes that latent RMSNorm together
  with the removed bottleneck. Neither arm adds query compression or QK-Norm.

The layout is a native task adaptation, not an implementation or equivalence
claim for an external model. Frozen primary reference inspected:
[Moonshot model code, e6fa68d7125977edbcdf15feb46adfa9733f5da7](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e6fa68d7125977edbcdf15feb46adfa9733f5da7/modeling_kimi.py)
(`KimiDecoderLayer`, `KimiMLAAttention`, `KimiDeltaAttention`; published under MIT),
and [DeepSeek-V3 code, 9b4e9788e4a3a731f7567338ed15d3ec549ce03b](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py)
(`Block`, `Transformer`, `MLA`). Observed layouts are branch entrance RMSNorm,
encoder final RMSNorm, compressed-KV RMSNorm and operator-specific KDA norms.
Native counterparts are `norm_layout.configure_norm_layout`, graph entrances,
`TokenEncoder`/`RelayResponseEncoder` exits and native attention cores. Graph and
condition injection boundaries, unpaired expression prediction and mHC/SSL are
intentional task-specific differences. No external model code/runtime is imported.

## Training, evaluation and acceptance

All other parameters remain the parent recipe: width256, heads4, graph4,
KDA2, streams4, micro80 x accumulation4, effective batch320, seed1, GELU,
loss/optimizer/LR/EMA/center/dropout/precision/selection and frozen splits/rows.
GPU0 only, original selected-device shared-resource gates, lock, 90% memory
ceiling, 2GiB headroom, UUID/owner checks, expandable-segments allocator and
CUBLAS contract remain binding. Run sequentially. Do not silently reduce B's
batch/query workload on a capacity failure. Preserve failed attempts.

Before fitting: clean published identical local/GitHub/server source, synthetic
nonzero-LR update and exact optimizer/EMA/center/RNG checkpoint resume, module/
normalizer inventories, explicit softmax output/input-gradient checks, graph
neighborhood and CLS/cross-direction contracts, then native one-step integration.
The existing global `attention_replacement=softmax` is not this experiment:
it would remove KDA as well. Existing `ffn_input_norm=rmsnorm` is also not this
layout: it inserts extra normalizers in ordinary mappings/heads.

Select no checkpoint by Pearson. Evaluate fixed epoch2 last on the same frozen
445 conditions, 300 controls, query1000, all5000/seen4775/unseen225 expression
axes. Report three separate Pearson definitions, MSE, N/NA, per-epoch prediction/
joint/active SSL losses, full training time and peak memory. No DEG, test tuning,
external training or extra mechanism. B versus A changes the terminal bottleneck,
including its associated norm and parameter count; not a matched-capacity or
single-norm causal claim. Historical comparisons retain separate training/eval
commit/config/checkpoint identities. Large data/weights/results stay on server;
only reviewed small summaries are pulled with SHA checks.

## Mixed precision integration repair

The initial integration attempt (`cfc24349c2cd921f4bf36af4b1153a3e299694ff`)
failed before any successful update. Removing the mapping LayerNorm exposes
BF16 graph memory under autocast, while the learned mask parameter remains
FP32. The mask insertion now converts only its inserted view to the memory's
dtype/device, preserving the FP32 parameter and its gradient. Both relay and
static insertion sites follow this contract. Historical equal-dtype paths are
unchanged. A regression explicitly exercises BF16 mapped memory with an FP32
mask outside autocast, since autocast promotion rules differ between PyTorch
versions; both arms also exercise full BF16 updates and exact state/RNG resume.
The failed server directory is preserved; the repaired attempt uses a fresh
`llm-norm-standard-attention-20261109-a2` family, new plan/run identities, and a
new clean published source. Configurations and training budgets are unchanged.
