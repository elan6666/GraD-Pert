# Loss diagnostic OOM repair - 2026-10-08

Owner: original main codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b. Long-run supervisor only codex:01a0df0b-4142-7df1-86c0-d959471d80a1. No Goal was requested for this repair.

Parent release0d8b88a1b2a48bd8a3a7276084654699645895de and failed queue v2-loss-six-0d8b88a-20261007T120935Z-5911a378 attempt1 remain immutable. All six128-update/checkpoint/inference gates passed. Formal L0 failed at initial diagnostic before epoch1 with allocated30.11GiB and only41.62MiB device free. No scientific results; no remaining formal arms started.

The diagnostic switched to eval, disabling configured activation checkpointing even though autograd was enabled. It also retained graph-only SSL1 activations although dSSL1/d(shared Cell/expression parameters)=0. Repair enables configured checkpointing whenever autograd is enabled, computes graph-only diagnostic values without a gradient graph, and releases all objective graphs before the residual-only read. Train-mode and no-grad inference routing are unchanged. Dropout remains disabled in diagnostic eval.

Acceptance: full-joint synthetic diagnostic values/shared-gradient norms/cosine match; diagnostics preserve existing gradients, RNG, auxiliary counters, model/optimizer/Teacher/center updates exactly across two nonzero-LR steps for baseline/M1/M2. Broader v2 regression plus real dual5090 baseline and largest M2 initial/update diagnostic, checkpoint reload, finite gradients and measured peak memory. Fresh exact-source128-update six-arm preflights now also require initial and updated fixed-training diagnostics and their memory receipts.

Execution: complete local/server tests, scoped publish of a new source, new immutable server checkout; perform short L0/M2 dual-card diagnostic integration with separate IDs; only then start a new six-arm queue with fresh run IDs and original protocol. All six exact-source128-update gates run before formal training. Keep old failure evidence and old passing gates as historical evidence; never relabel them with the new SHA.

Unchanged: orderL0/L1/L2/M2/L3/M1, Jurkat cap40 seed1 fresh6, global272=m68xaccum2xworld2, no validation, finalepoch6 last only, M1 gene1/CLS0, M2 gene1/CLS1, mask20percent, complete four active SSL terms, graph/order/views/Teacher-center semantics and original split/control manifests. No extra scientific group or batch fallback.

The new long queue is handed only after actual diagnostic and training-step startup plus exact supervisor ACK and ACTIVE20min monitor. Failure/terminal returns to this original main for already-authorized repair or results/report; return fallback is inactive during repair/training and retargeted to the actual new queue before handoff.
