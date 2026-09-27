# Handoff: B0 three-epoch run failed on CUDA OOM

- Date: 2026-09-28T04:34:09+08:00
- From: codex supervisor `codex:01a0df0b-4142-7df1-86c0-d959471d80a1`
- To: main/build `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`
- Run: `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`, attempt 1
- Source: `e821173b4d11268b621ac8299e45b8d6d272b7e7` (clean); config SHA256 `3f1d3d9dde6bf151eeb3916809863eda0020fa89d8c82c74005d19c4d424dedd`
- Terminal receipt: `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-terminal-failure-20260928.json`
- Exact server failure receipt: `/data/yilangliu/GraD-Pert/runs-v2-b0-gate-m74-3ep-e821173/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143/FAILURE.json` (SHA256 `f3f36227e26adab22144e51269c0dd984a26ab77b5941a521754d9fd3887ddf2`), copied locally after recording its 19,977-byte size.

## Verified terminal state

Exit code is 1. Epoch journal remains 0/3 (434 updates per epoch); no epoch-0001 checkpoint or history rows. `COMPLETE.json`, best/last tests, and the test root are absent. Recursive scan found zero PKLs. No target process remains and both GPUs returned to 0% utilization / 2 MiB.

Rank 0 failed first: CUDA OOM on GPU 0 while `torch.nn.functional.dropout` tried to allocate 128 MiB with only 9.62 MiB free (process memory 31.34 GiB reported). Rank 1 then timed out on NCCL ALLREDUCE sequence 5573 after 300,027 ms and aborted. This is a failed incomplete run, not a result. Source/config identity matched the handoff.

No retry, evaluation, source/config edit, or other experiment was performed. Main should inspect the preserved receipt/log excerpt and decide any future new run/configuration; do not overwrite this run ID.
