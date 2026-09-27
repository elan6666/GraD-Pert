# Handoff: supervise GraD-Pert v2 full B0 for three epochs

- Date: 2026-09-28T03:08:30+08:00
- From: codex (build), `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`
- To: codex (supervise), `codex:01a0df0b-4142-7df1-86c0-d959471d80a1` on `local`
- Code snapshot: detached `e821173b4d11268b621ac8299e45b8d6d272b7e7`, server clean checkout `source-v2-b0-3epoch-e821173`
- Task receipt: `.byte-os/coordination/receipts/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143-launch.json`
- Monitor lease: existing `grad-pert-v2`, exact run `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`, attempt 1, target supervisor, intended 20-minute ACTIVE check, pause/delete at handback; activation must be verified after transfer.
- Event wake: no tested event-to-idle-chat adapter; `.exit` file is evidence, so periodic checks have up to 20-minute detection latency.
- Return wake: `grad-pert-v2-b0-return` is PAUSED; supervisor may activate for terminal handback, but idle main-chat wake has not been proven. Main continuation must be observed, not inferred from message delivery.
- Model provenance: exact supervisor runtime model unverified.

## What was done

- Chose microbatch 74, accumulation 2, two GPUs, global batch 296, graph rows64, sequence chunk16 after 128-step capacity and repeated throughput checks; see `docs/experiments/GRADPERT_V2_GATE_BATCH_CHUNK_20260928.md`.
- Published `e821173`, verified server source tree and publication hash, passed 86 server tests and one-step integration under three-epoch config SHA `3f1d3d9dde6bf151eeb3916809863eda0020fa89d8c82c74005d19c4d424dedd`.
- Launched exact run `nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143` through native `gradpert train`: run manifest and plan exist, wrapper PID 3829919, torchrun plus two ranks alive; GPU 0/1 allocated. No failure or exit file at launch check.

## What changed

The committed code adds an explicit `v2_fixed_3` policy and self-contained B0 config; no v1 code or frozen upstream tree changed. The server's active checkout must remain immutable.

## Result vs acceptance

Launch gate met; scientific completion pending. Never infer completion from process exit alone. Need epoch journal 3/3, COMPLETE, both best/last test receipts, hashes and zero PKL.

## For the next owner

Only monitor this exact run root `/data/yilangliu/GraD-Pert/runs-v2-b0-gate-m74-3ep-e821173/nadig_jurkat-seed1-20260927T190511Z-c619dc8aabbc4b5a9f3a578afd0e8143`. Check `/data/yilangliu/GraD-Pert/development/v2-b0-gate-m74-3ep-e821173-formal-r1.pid/.log/.exit`, fit/epoch_state.json, COMPLETE.json, best/last test JSON, rank/failure receipts, and GPUs. Healthy unchanged checks stay quiet. Preserve any failure evidence and return complex repair or final analysis to main without relaunching or changing source/config/run ID.
