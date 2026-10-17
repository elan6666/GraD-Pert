# Whole-code audit and focused simplification

User asks for concise, modern and performant code alongside PT1. Inspect all native,
script and official-adapter Python AST/import/lint surfaces, then change measured hot
paths. Preserve v1, official model ownership, all active/history source and scientific
contracts. Do not delete historical runners solely for size or rebrand existing work.

Initial scope: native137files/35077lines, scripts139/41081, adapters33/6798; parser
accepts every file; Ruff src/scripts/benchmarks has zero findings. CLI --version cold
import is about0.20s on this Mac, without Torch import: not a current priority.
Only two >=13-line exact function-body duplication clusters found, in historical
checkpoint-evaluation CLI tails and performance-process helpers. Their provenance
and resource boundaries need preservation; avoid a broad mechanical rewrite.

Current high-value changes: explicit stage execution instead of repeatedly testing
zero loss weights, no-label data path, unused decoder bypass, common grouped-view
module with shared layouts/dot products and exact statistics. Appropriate behavior,
full-state multi-update, compatibility and read-only checks precede publication.

Next inspect static graph evaluation: same frozen graph is recomputed for every
condition. Candidate is evaluation-scoped full-graph reuse with no persisted cache,
no Teacher/training reuse, explicit conditional-graph exclusion and unchanged query/
control order. Validate outputs/RNG on different target sets and batches, then real
GPU2e-6 gate and timing (including cache construction); adopt only if worthwhile.
Do not infer a speedup from lint, fewer lines, or an isolated matrix operation.

Maintain one compact performance report and module map; make safe cleanup separately
traceable from PT1 semantics and measure before/after. Keep code-generation manifests
strict: reducing duplication must not relax exact family/order/source/resource guards.

Implemented separately from f73347ff989eabbdb9eb96741e926b60b5dc8481: evaluation-
scoped relay graph memory with model/index/version/dtype/device invalidation;
full-context zero-dropout topology packing reuse with exact RNG consumption and
no returned CPU alias. A scoped helper replaces repeated full graph forwards.
Directed85 checks passed. CPU3 bitgenerators plus varying views/dropout/targets/
query blocks/RNG/cache invalidation are covered. GPU/end-to-end claims await fresh
source-matched receipts. PT1 remains next formal job, DP2 queued after it.
