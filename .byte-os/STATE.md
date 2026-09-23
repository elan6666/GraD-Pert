# Current bounded stage: GraD-Pert v2 dual-GPU performance and capacity

Overall outcome: deliver a runnable, clean, published Top500/GenePT-PCA256/
SwiGLU v2 model on two RTX 5090 cards, measure useful batch capacity, and make
this the parent for future Jurkat v2 ablations without modifying v1 or historical
B0/B1 runs. The user-selected scientific default remains global batch 128 and
five training epochs; capacity candidates are engineering evidence only.

Core stage acceptance: the complete model and default config are published;
the bounded two-GPU performance comparison, 128-update capacity at the highest
validated batch 144, full v2 tests, ablation parent/CLI smoke, and design docs
are complete. The capacity-run source is `0a9a67f272978476e104952f3a715fa41969105d`
(the immutable server checkout is
`/data/yilangliu/GraD-Pert/development/source-v2-capacity-clean-0a9a67f`). Work in
`/Users/elan/code/grad-pert-v2-build`; preserve the separate dirty root checkout.

Measured at global batch 128 on both cards: 5-step bounded probe baseline
`cb517ea` 4.561 cells/s, 21.51 GB/card; current fused KoLeo backward `2eb4c4e`
6.511 cells/s, 25.91 GB/card (+42.7% throughput). An attention-only optimization
had negligible full-step benefit. CPU/distributed checks for fused gradients:
53 passed. Detail, limitations, and immutable receipt stems:
`docs/experiments/GRADPERT_V2_GLM53_DUAL_GPU_PERFORMANCE.md`.

Batch boundary: 144 passed a clean 128-update plus 300-control validation
inference probe, receipt
`/data/yilangliu/GraD-Pert/development/v2-m36-capacity-clean-0a9a67f/receipt.json`
SHA256 `aa78cce7b4312d7150744fdae25c25265b37a633cd828a954721327a26e2f1a4`.
It is the highest sustained-tested batch, not an exact physical maximum. 152
passed only 5 updates and reached 32.42 GB/card; 160 failed by update 3; 192
failed at its first update. Disabling outer checkpointing OOMed at global 128.

Completed server process: exact-default global-batch128 128-update plus
validation-inference capacity probe, output
`/data/yilangliu/GraD-Pert/development/v2-m32-capacity-clean-0a9a67f`, log
same stem plus `.log`, clean read-only source checkout
`source-v2-capacity-clean-0a9a67f`, GPUs 0,1. Its receipt passed 128/128
updates, checkpoint continuation, and 300×5000 validation inference; receipt
SHA256 is `51323c14610b8094dbe25e73c9c6ce4663c3325498e9873db97b2f1e9a59b269`.
Both GPUs are idle after completion. Measured throughput was 6.622 cells/s,
with 27.80/27.80 GB peak allocated memory. Source HEAD remains the pinned
`0a9a67f` commit and the server checkout is clean.
The earlier distinct 128-step attempt `v2-m36-capacity-0a9a67f` was stopped
after 32 updates because a concurrent build introduced ignored egg-info into
its active source tree; `audit-stop.json` preserves both hashes. It is not
capacity evidence. The replacement checkout's tree hash matched its pinned
publication receipt before launch and remained clean after the successful 144
probe. Never build in that checkout. Do not overwrite run IDs or change the
active server checkout. The next stage is exact published-source preflight
for a new five-epoch Jurkat baseline. The
bounded Goal is complete. The active thread heartbeat is
`grad-pert-v2-batch128`, every 30 minutes during testing; switch to every two
hours only when formal ablation training begins.
Full v2 test suite: 212 passed; lint,
format and isolated wheel/sdist build passed. Mypy retains five unrelated
existing errors in `training/step.py` and `training/v2/reductions.py`.
A formal five-epoch scientific run is a later background stage and is not
claimed complete by this capacity probe.
