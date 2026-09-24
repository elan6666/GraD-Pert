# Current bounded stage: Hamiltonian expander / full MLA method implementation

The user requested replacing the old random-permutation expander with
TxPert-style fixed bidirectional Hamiltonian cycles, allowing graph layers
to read updated neighbors, and dropping DSA for full MLA. Preserve the old
stopped B0 and all historical config/source identities. The new explicit
Jurkat config is `configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml`.
It keeps 5 epochs and a **candidate** global batch192; capacity must be
remeasured before formal GPU work. Detailed method and open scientific risks:
`docs/design/GRADPERT_V2_HAMILTONIAN_MLA_METHOD.md`. This stage is local
implementation, tests, documentation and Git publication only, not a
training restart. The historical capacity records below refer to older
architectures and are not evidence for this method.
Local validation: 230 v2 tests, Ruff lint/format, and wheel/sdist build passed.

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

2026-09-24 18:37 +08 new authorized outcome: implement the compact v2 parent
and retest maximum useful dual-GPU batch without restarting the stopped formal
run. User clarified that **each** of Cell and Response has two KDA layers and
one terminal DSA/MLA layer, not a 2:1 depth ratio between encoders. Four
Student projectors and their four Teacher copies use 8192 prototypes; center
buffers follow. New default Jurkat config is explicit; unmodified historical
v2 configs resolve to the old three-KDA plus one-terminal-layer architecture.
Expected Student count: 23,123,847. Isolated server development snapshot
`/data/yilangliu/GraD-Pert/development/v2-compact-preflight-20260924-39473bd`
passed 82 targeted and 227 total v2 tests, Ruff lint/format and isolated
wheel/sdist build. Mypy retained exactly the five previously documented errors
in unchanged `training/step.py` and `training/v2/reductions.py`. This is
prepublication CPU evidence, not a GPU capacity receipt. Next: scoped commit
and GitHub push; clean immutable server checkout and exact publication receipt;
two-GPU short failure bracket then 128-update/validation confirmation at the
highest practical batch. No monitoring is active yet.

2026-09-24 19:07 +08 publication and capacity milestone: scoped source/config
change `974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94` is on GitHub `main` and
in clean immutable server checkout
`/data/yilangliu/GraD-Pert/development/source-v2-compact-974c5ea`.
Publication receipt `gradpert-v2-compact-publication-974c5ea.json` has SHA256
`626bb795287187ea1accec20eacb1bdb906c6bf1ed95fc371ead270424c13760`;
formal server source identity passed. Default global batch128 dual-GPU one-step
integration plus checkpoint reload passed; its receipt SHA256 is
`b7b75b95c22e4ff6cb43aa9f40d39f353a4cc230381d6b63d5fa775bc5698554`.
Five-step short probes passed batch192, 208, and 224, with batch224 throughput
9.5172 cells/s and 32,338,694,656 bytes peak allocated. Batch240 OOMed at
step3. The 128-step batch224 probe
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m56-128`
failed at step6 with CUDA OOM after five completed steps; batch224 is not a
sustained-capacity result. The failed receipt and log are preserved.

The next 128-step probe is running at global batch208, microbatch52 per GPU,
accumulation2, GPUs 0/1, with the same published source and allocator contract.
Run root `/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m52-128`,
PID file same stem plus `.pid`, log same stem plus `.log`, exit-code file same
stem plus `.exit`; config SHA256
`89832263f13fe41a3eda0e0793fea8567df28616d07f92416ae31594b2b84172`.
Check 128 completed steps, checkpoint continuation and 300-control validation
inference before accepting it. If it OOMs, preserve its run and probe 192 with
a distinct ID after the GPUs are idle. The 30-minute heartbeat
`grad-pert-v2-compact-batch-capacity` is active and reports at every check.
This is engineering capacity evidence only; no new five-epoch training or
ablation has been launched.

2026-09-24 19:41 +08 capacity follow-up: the batch208, micro52, accumulation2
128-step run failed on both ranks from CUDA OOM in backward at update24, with
23 completed updates. GPU 0 had about 120 MiB free and the attempted tensor
was 138 MiB. Exit code 1; failed receipt SHA256
`b52aad0d0d1874f4ec04d556d991504123defe0b1d95fe62156221535cd5811a`.
The failure is not a sustained-capacity result. Both GPUs returned to idle.
The next independent full probe started at global batch192, micro48 per GPU,
accumulation2, using the same immutable source and publication receipt:
`/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128`.
Its config SHA256 is
`674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`;
PID/log/exit file names use that run stem. The existing 30-minute monitor was
retargeted to this run. Require all 128 updates, checkpoint continuation and
300-control validation before accepting batch192. If it fails, retain evidence
and narrow to a smaller unused candidate with a new run ID.

2026-09-24 20:50 +08 capacity completion: the new compact v2 source
`974c5eadf35a95fbc3ba46cffe6d9ab34cdd4e94` completed the dual-GPU
global-batch192 (micro48/rank × accumulation2 × two ranks) 128-step probe.
Exit code 0; receipt status `passed`, 128/128 updates, checkpoint continuation
SHA256 `1610f227056f5565697bde00c2ba95345b4a1df3f80b70bec11ff98a3ee1ffbb`,
300×5000 validation inference in 11.986 s, and both GPUs free afterward.
Receipt `/data/yilangliu/GraD-Pert/development/v2-compact-974c5ea-capacity-m48-128/receipt.json`
SHA256 `71d952f90466052a52855a819090d9c5e03ba3e96081eadc03ac2cd3c896529d`.
Post-warmup throughput 8.486 cells/s; end-to-end throughput 7.992 cells/s;
training wall time 3,074.805 s. Peak allocated memory: GPU0 31,011,834,880
bytes, GPU1 31,045,261,312 bytes. The single-condition validation prediction
loss 0.008950 is only an inference finiteness check, not a scientific result.
The source checkout remains clean at its pinned commit. Batch208 failed at
24th update and batch224 at sixth, so 192 is the highest sustained-validated
batch among tested candidates; the exact physical threshold between 192 and
208 remains unknown. Scientific default batch128 and five-epoch plan remain
unchanged. No new formal train or ablation was launched. The capacity monitor
can now be removed.

2026-09-24 21:44 +08 new formal stage: the user selected the sustained-passed
global batch192 profile and authorized B0 formal training. The user then
clarified that B0 here means only the complete prediction + SSL1 + SSL2
baseline, **not** the `prediction_only` contrast in the older group generator.
The tentative two-row local generated group was removed before publication;
no prediction-only training launched. The existing self-contained compact-m48
configuration has micro48/rank, accumulation2, world2, batch192, lambda1=1,
lambda2=0.1, seed1 and five epochs. Source, GitHub main and a fresh immutable
server checkout matched published SHA
`fc90a0d992373e619976504c82bd6f94c930d7b2`; clean source identity and
publication receipt SHA256
`a77a9e63771f1ffc516d123f7280e8cf04401c408f34c820d150dfd487c5c770`
passed. Exact-source, exact-config dual-GPU one-step integration and checkpoint
reload passed, receipt SHA256
`7cdeb101149bbfe5264215b474e2e38d449ca461c6312f11aafc0aa57c0b73e4`.

Formal B0 run ID
`nadig_jurkat-seed1-20260924T134139Z-e5df6111138747e08ba5a582e37c1ed2`,
root `/data/yilangliu/GraD-Pert/runs-v2-b0-compact192-fc90a0d/` plus ID,
started from sealed plan
`/data/yilangliu/GraD-Pert/development/v2-b0-fc90a0d-b192.launch-plan.json`,
SHA256 `8f4dac881eb98bf5f3ed1bf7562e51e897d8ece58cde146727008fc61b2df03a`.
Config SHA256 `674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`;
runtime SHA256 `54691399183a502fcf7f939e0f4e3bad587b00677fc84545cf6d20a582a547c6`.
Parent PID file `/data/yilangliu/GraD-Pert/development/v2-b0-fc90a0d-b192.pid`
contains 2953737; log and exit-code files use the same stem. At first check,
parent was alive, epoch journal 0/5 with 749 updates/epoch, and GPUs 0/1
were both active. Require five committed epochs, best/last tests and
`COMPLETE.json` before reporting a scientific result. The heartbeat
`grad-pert-v2-b0-compact192-formal` checks every two hours and reports every
check. Do not modify the active source, resume the old run, or launch any other
ablation row.

2026-09-25 02:28 +08 user stop: the compact complete B0 run above was
terminated before its first epoch committed. The targeted torchrun PID 2953749
and launcher parent PID 2953737 received SIGTERM after their command lines
were matched to the sealed B0 launch plan. Wrapper PID 2953738 and rank PIDs
2953761/2953762 also exited; GPUs 0/1 returned to 2 MiB each. The fit journal
remains 0/5 with 749 planned optimizer updates per epoch, `history.json` is
empty, and only the epoch-0000 initial checkpoint exists. There is no finite
validation selection, `COMPLETE.json`, or best/last test receipt. Preserve the
run ID, published training SHA `fc90a0d992373e619976504c82bd6f94c930d7b2`,
config SHA256 `674ea9ab4160e10e85de7f6da78137257efee510c97e2f9852a474a55e81acf5`,
and existing run files; do not resume or relabel this interrupted run. The
two-hour `grad-pert-v2-b0-compact192-formal` heartbeat was deleted. No other
ablation or `prediction_only` job was started. Current stage is discussion of
graph propagation and expander design; no server training is active from this
B0 run.
