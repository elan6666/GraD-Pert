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
