# Bounded engineering stage: metric scalar packing

Pre-change baseline: local clean detached HEAD and GitHub main verified at eada917fbbc23934b568f15d972ac32d9d037186. Preserve all scientific R8 artifacts and its immutable server source. User authorizes code simplification/performance during research; no baseline retraining or scientific conclusion is implied.

Candidate: explicitly opt-in benchmark-only batching of the six microbatch term scalar D2H reads. Preserve original default behavior, every safety/finite check, AU/CR timing, autograd, Teacher/center, optimizer, RNG and accumulation order. Preserve FP64 values by scalar promotion before stack. Non-scalar/complex/mixed-device values retain the original fallback/error behavior. No automatic adoption.

Disjoint implementation ownership:
- performance agent: src/gradpert/training/v2/engine.py; tests/v2/test_metric_scalar_batch.py only.
- main: scripts/v2/capacity_probe.py; isolated benchmark/parity orchestration scripts; this plan; ignored reports/receipts.
- read-only reviewer: inspect affected code/tests and report; no edits.

Acceptance: focused CPU scalar/full-update checks; unchanged default; opt-in guarded benchmark flag and truthful receipts; independent review; scoped clean commit pushed to main before any GPU use. Use a new clean server checkout matching the published SHA. No change to active/historical checkouts. Mandatory expandable_segments allocator and original resource gates.

CUDA verification: compare full native same-input, same-initial-state optimizer updates under original BF16/DDP/global320 recipe. Inspect Student/Teacher/centers/optimizer/RNG/metrics, not just forward. No numerical tolerance relaxation. Only after parity, run unprofiled ABBA complete-update timing and compare slow-rank throughput against A/A drift; no claimed training speedup from CPU timing. Candidate remains disabled if no reliable gain. GPU runtime and costs bound each arm; use the existing supervisor/20-minute monitor for a job that outlives this turn. No scientific run or test-set access.

Deliverable: engineering evidence and actual end-to-end gain or rejection. Overall representation acceptance remains pending. Next research candidate is a separate staged SSL-to-joint design, not bundled into this source change.
