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
active server checkout. The completed capacity receipt is engineering evidence.
The exact published source `536458333e437252178ffe493c5c50c9064e7615`
then passed the same-config two-GPU one-update integration preflight: receipt
`/data/yilangliu/GraD-Pert/development/v2-5364583-b128-integration/receipt.json`,
SHA256 `c8882f28d279ec5cde045fc1b3f9cdfd9be2d8371930994ecb6ad85c2498cc2c`.
Its publication receipt is `gradpert-5364583-publication.json`, SHA256
`37b58e67236317c969b5350e5f25dd1ce0ae3bc16f94eb7e20ad5c19ce8018cf`.

The formal five-epoch Jurkat v2 baseline is now active on GPUs 0,1 at global
batch 128: run ID
`nadig_jurkat-seed1-20260923T194747Z-1b7eda2abd6441f592d0834e1e275e88`,
run root `/data/yilangliu/GraD-Pert/runs-v2-glm53-current/` plus that ID,
log `/data/yilangliu/GraD-Pert/development/v2-jurkat-baseline-5364583.log`,
parent PID file with the same stem and `.pid`. The source checkout is
`/data/yilangliu/GraD-Pert/development/source-v2-formal-5364583`; config
SHA256 `8471f5ea68f4801406497985291a0088116ff5a8435b82f85e55c611548b06c6`.
At handoff the process was alive, epoch state was 0/5, and both GPUs were
occupied by this run. The existing heartbeat `grad-pert-v2-batch128` has been
retargeted to the formal baseline and verified active at every two hours, with
user-requested reports on each check. Verify five committed epochs and actual
best/last tests before treating this run as complete; keep training SHA fixed
at `5364583` even if documentation-only GitHub main advances.

2026-09-24 11:53 +08 milestone: epoch 1/5 committed after 1,081 optimizer
updates. Journal SHA256
`02fe76ec4cd46489a58116a095a3059ac9f6382c1e97f95fe3ce419f7ac76dcb`;
provisional best/last `epoch-0001.pt` SHA256
`c3e1134e441db33556765458f009c88f5bd321389539c1bda1a8118a40647d85`.
Validation prediction loss `0.005269254464738538`; validation Pearson values
TxPert `0.14296990652273225`, TriShift `0.1875737367080305`, Systema
`0.06702235003118254`. The parent and both rank workers remained alive;
`COMPLETE.json` and best/last **test** receipts were absent. Continue training
the existing run through five epochs; no new run ID or ablation group.
Full v2 test suite: 212 passed; lint,
format and isolated wheel/sdist build passed. Mypy retains five unrelated
existing errors in `training/step.py` and `training/v2/reductions.py`.
The formal five-epoch scientific run described above was later stopped by the
user. Capacity and one-step preflight receipts are not scientific best/last
results.

2026-09-24 18:05 +08 user stop: the formal Jurkat v2 run was interrupted
after one committed epoch while the second epoch had no committed receipt.
Verified parent PID 2770034 and torchrun PID 2770045 received SIGTERM;
both rank workers 2770069/2770070 exited. GPUs 0/1 returned to 2 MiB each.
`COMPLETE.json` and best/last test receipts were absent. Preserve the run ID,
epoch-1 checkpoint, source/config hashes, and all existing receipts; do not
mark the five-epoch baseline complete or restart it. The two-hour heartbeat
`grad-pert-v2-batch128` was deleted. Current scope is discussion of speed
changes (Cell/Response Encoder depth 2:1 and four heads 16384→8192
prototypes); no new code or experiment has been launched.
