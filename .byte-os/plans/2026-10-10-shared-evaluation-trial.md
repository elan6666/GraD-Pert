# U4 evaluation beside active U3 fit

Direct user request: try overlapping training and evaluation; if unsuccessful,
evaluate U4 before continuing U3. U3 was already active when the request arrived.
Base/source of active training: 1929cba76bee4c7f868083616a105d2507c7fa54.

1. Preserve both runs and the active source. U4 epoch6 TRAIN_COMPLETE is sealed;
   U3 is active with the same fixed6/global272/m68*a2*world2 protocol.
2. Published evaluator-only addition: optional allocator fraction, CPU teacher
   and optimizer state, explicit bounded diagnostic subset. No arithmetic,
   batching, controls, truth, masks, parameter or checkpoint changes. An explicit
   diagnostic subset cannot enter execute_evaluation_plan or scientific COMPLETE.
3. Try 32 frozen U4 conditions on GPU0 beside U3, eval batch128 unchanged,
   CUDA allocator cap0.19 (~6.05GiB). Historical U3 10-step peak reserved18.48GiB;
   cap plus current training allocation leaves several GiB headroom. Verify only
   expected project PIDs occupy cards. Keep OMP/MKL2 and immutable source.
4. Compare before/during/after committed U3 update timestamps and wall time, eval
   condition rate, memory peak and failures. Trial is bounded engineering evidence,
   not long-run equivalence or a mathematical/model change. No repeated blind
   tuning after failure. Require continued U3 updates and no failures; only adopt
   overlap if estimated total remaining makespan improves over sequential work.
5. Success: seal full same-checkpoint U4 evaluation on disjoint workers, cap per
   evaluator, record the new evaluation SHA separately. Old controller cannot
   launch duplicate postfit: stop/retire only its exact parent PID while preserving
   child and inherited leases, then adopt the active child in a new saved schedule.
6. Failure: preserve probe evidence; retire old parent without signalling child;
   stop U3 at a validated saved epoch checkpoint, record interrupted attempt and
   checkpoint/RNG/optimizer/Teacher/center hashes. Drain U4 before exact same-source
   resume-v2 of U3. Do not reinterpret as new LR stage or reset any training state.
   Interrupted epoch work is replayed from its saved boundary, explicitly recorded.
7. Use Luna only for short read-only wait supervision. Main verifies outcomes.
   Update canonical Byte state/report; then transfer long jobs to the designated
   supervisor with the updated20min monitor, exact saved schedule and ACK.
