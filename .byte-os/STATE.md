# Current bounded stage: v2 GLM-5.3-Flash adaptation

Goal: publish a clean, validated native v2 variant with B1 GenePT-only PCA256
trainable seed table, clipped SwiGLU cell/response FFNs, and unordered-gene
content-indexed sparse MLA in fourth layer. The default new Jurkat config uses
Top500; Top100 is an independent comparison. Preserve v1 and frozen B0/B1 configs.
Do not resume stopped training in this stage.

Pre-change local/GitHub main: `03d038be562a9f5f8a3ddd8db1f86ac72441169d`.
Implementation checkout: `/Users/elan/code/grad-pert-v2-build`, not the dirty
root checkout. The full model remains order-sensitive due to three KDA layers;
only the fourth sparse attention is gene-permutation equivariant.

Acceptance: native implementation and explicit configs; relevant architecture,
config, gradient, permutation, optimizer-route and legacy tests; static checks;
source/provenance document; scoped commit pushed to main. No formal run claim.

Verification: 193 v2 tests passed in an isolated PyTorch 2.12/Python 3.10
environment; 7 run-group tests requiring Python 3.12+ passed separately under
Python 3.13. Ruff check and format check passed. An exploratory small CPU
forward found indexed MLA slower than dense MLA; CUDA capacity and throughput
are not established and must be checked before sustained training. The new
Jurkat architecture has 33,908,867 trainable parameters with 6506 genes.

First server two-GPU single-update preflight at global batch128, per-rank
microbatch64/accumulation1 failed during backward with CUDA OOM; immutable
receipt: `/data/yilangliu/GraD-Pert/development/v2-glm53-integration-1672bb6/receipt.json`.
Both GPUs were idle before launch. The next isolated config revision uses
microbatch32/accumulation2, retaining global batch128, and needs a new receipt.
That second isolated attempt also failed before the first optimizer update;
receipt: `/data/yilangliu/GraD-Pert/development/v2-glm53-integration-54911b8/receipt.json`.
The next bounded repair checkpoints each sparse query chunk and verifies its
gradient equality against the uncheckpointed reference before a fresh CUDA
preflight. Failed run IDs remain immutable.

Third isolated two-GPU preflight at source `7f4140c90dd123c0bc2444e28da52a437c7e42cf`
passed one global-batch128 optimizer step and checkpoint reload:
`/data/yilangliu/GraD-Pert/development/v2-glm53-integration-7f4140c/receipt.json`,
SHA256 `337560ee9a42db53cf074eaebaa45e897f27fd9fc391673f3949b374e7aa2707`.
Peak allocated was 20,394,624,512 / 20,274,631,168 bytes on GPUs 0/1; the
single update took 30.90 seconds. This does not establish sustained training
or inference capacity. Top100 has config/unit validation only.

Previous completed data-only atlas and report remain in
`docs/experiments/PERTURBATION_DATASET_ANALYSIS_ZH.md`; this new stage does not
change their evidence or data.
