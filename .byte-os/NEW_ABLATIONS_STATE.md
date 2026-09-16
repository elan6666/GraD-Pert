# New R50 ablations

Pre-change published baseline: 7005aa7bd48e283dc5376a95d4b614b654bc4b71.
Scope/protocol: docs/experiments/R50_NEW_ABLATIONS.md.
Main user checkout has unrelated Trackio edits; work is isolated in
/private/tmp/gradpert-r50-txpert-7efd.0jgc4x. Preserve all historical runs.

Seven implemented rows have passed local configuration/gate tests and CPU
step/resume tests (72 passed, one CUDA-only skip in the latter combined run).
They still require publication, server 1-step acceptance then automatic
full50+best/last. MLG/BMP blocked on exact auditable definitions.
Planned fresh server root: /data/yilangliu/GraD-Pert/runs/r50-new-ablation-20260916-v1.
Plan/publication: /data/yilangliu/GraD-Pert/contracts/r50-new-ablation-20260916-v1/.
Runtime start/exit records and plan.json in that root are authoritative for
commit, row, GPU and phase. Never infer success from the design or this state file.
DSH profile disabled, last heartbeat 2026-09-15; do not reactivate.
Existing Codex heartbeat grad-pert-r50 paused until queue starts. Update it,
never duplicate it. Two physical GPUs, max one experiment per GPU; initial
training hour checks every10min, subsequently hourly. While queued rows can
start automatically, lightweight start/exit-record checks stay at 10min to
detect a new attempt; deep training checks switch to hourly after its first
hour. Once no further starts are pending, the heartbeat itself can be hourly.
