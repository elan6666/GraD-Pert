# v2 operator evidence and native implementation contract

Baseline for isolated implementation: published main
`19342809e0cacfe70196da32c356471f2f78d9e0` (verified GitHub 2026-09-21).
Root checkout has unrelated dirty edits; implementation workspace is
`/Users/elan/code/grad-pert-v2-build`. Existing v1 modules are preserved.

- KDA: inspected FLA `954438d1fcb5e1bb05c22f9908de9c5c2df74ae5`,
  `fla/ops/kda/naive.py::naive_recurrent_kda`, tensor order B,T,H,K;
  decay state per key channel, read old value after decay, beta-scaled rank-one
  error write, then query read. Source repository MIT. Native equation implementation
  does not import/copy the upstream module. Source:
  https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/naive.py
- mHC: primary paper arXiv:2512.24880v1, equations 3,7–9, inspected 2026-09-21.
  Dynamic per-token routing, sigmoid pre, twice-sigmoid post, 20 Sinkhorn iterations
  in float32; wrappers on attention and FFN separately. Implementation independent
  from upstream code. https://arxiv.org/html/2512.24880v1
- MLA: inspected DeepSeek-V3 `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`,
  `inference/model.py::MLA`, KV down-projection/norm/up-projection. Native v2 uses
  shared rank64 latent, full noncausal softmax; no RoPE, caching, tensor parallel,
  Q compression or upstream runtime. This is a task-specific MLA variant, not
  a claim of complete language-model parity. Source:
  https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py

Native initial choices: graph2 layers pre-LayerNorm; multi-source union edge with
summed trainable per-source/head biases initialized zero; edge weights used for
Top20 selection only. Graph memory is the adapted identity table at every layer.
Expr MLP d/GELU/d/LayerNorm; mean multi-target aggregation; response 2d→d;
output d/GELU/1; encoder RMSNorm, FFN GELU, mHC4 streams; final stream mean.
KDA no convolution/RoPE, sigmoid beta, normalized Q/K, negative-softplus channel
log decay with positive learned rate; native scalar recurrence is a correctness reference. A differentiable block triangular
solve (16-token blocks) now matches its output and all input gradients; neither
path is yet capacity-tested on CUDA.

Projector d→2048→GELU→256→L2 norm→16384 (bias-free prototypes); each of the four
heads independent; Teacher mirrors Student, teacher eval/no_grad. SSL1 distinct
condition mean, two globals plus four locals, no row-weighted duplication.
All adaptations are project-preregistered, not claimed official defaults.

Pending before formal release: optimized recurrence or measured viable throughput,
view builder and real-data lifecycle, optimizer/schedule provenance, version/CLI
integration, single-condition/deduplicated KoLeo semantics, scientific ablation
switches, all correctness and compatibility tests, real CUDA capacity receipts.

Numerical limitation observed in CPU tests: changing microbatch yields matching
loss/gradients within tolerance, but head-wise Muon can amplify floating-point
noise in near-null directions (one observed parameter difference 4.1e-5 at LR .001).
Do not claim bitwise accumulation equivalence; fix/report execution profiles.

## GLM-5.3-Flash-inspired v2 variant (2026-09-23)

The new `hybrid_sparse` variant retains three ordered KDA layers, then uses
content-indexed, noncausal low-rank attention in layer four. The independent
indexer scores gene queries against gene keys and selects `topk` **per gene and
per head**, reserving self and tail CLS inside that budget. Tail CLS reads all
genes. Gene order is not used by this fourth layer's selection, and the layer
is permutation equivariant over gene slots. The full four-layer encoder is not
permutation equivariant because KDA scans an explicit sequence order. There is
no causal mask, rotary position encoding, or contiguous index pooling. The
selected index score enters the attention logit with a learned bounded scale,
so the independent indexer receives training gradients despite hard top-k.
This score bias is a project adaptation, not an assertion of GLM parity.
Query-chunked gather limits transient KV memory; the indexer still calculates
all gene-pair scores and measured speedup is not assumed.

A small exploratory CPU forward check (PyTorch 2.12, one thread, eval/no-grad,
batch2, 256 genes + CLS, width64, 4 heads, rank16, top100, chunk8, three timed
repetitions after warmup) measured about 0.0006 s for dense MLA and 0.021 s
for indexed MLA. This is not a GPU throughput or 1000-gene result; it warns
that the native reference sparse gather must pass a separate CUDA capacity and
throughput check before sustained training.

`ffn_type=swiglu` uses separate bias-free gate, up, and down projections,
`SiLU(clamp(gate, max=10)) * clamp(up, -10, 10)` and a 4d hidden size.
Only the cell and response encoder FFNs change; the graph reader, expression,
prediction and projector nonlinearities remain as explicitly configured.
Historical `attention=hybrid`, `ffn_type=gelu` configurations remain readable.

The primary comparison is the official GLM-5.3-Flash config
(https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) and
Transformers GLM5-Next implementation
(https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py),
inspected on 2026-09-23 at Transformers main
`a008a653dee362f2f667738b51a31aa805e994d7`. Official GLM uses SiLU,
`swiglu_limit=10`, top-k 2048 and contiguous pool size 4 for causal language
tokens. GraD-Pert preregisters top500, with top100 as a separate ablation,
because it has 1000 unordered gene queries plus CLS. These values are task
adaptations, not official GLM defaults. No official source code is imported.
