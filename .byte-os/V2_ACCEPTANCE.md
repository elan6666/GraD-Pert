# GraD-Pert v2 acceptance ledger

Scope: method and independent implementation, synchronous two-5090 capacity,
measured grouped experiment configs/scripts, exact launch preflight and a verified
ZCode supervision handoff. Formal50-epoch training results are downstream of this
active implementation goal; the goal does not wait for their completion.
Current implementation reviewed at4e30832. Original evidence SHAs remain attached
to the executions that produced them, not relabeled as the latest publication.

| Requirement | Verified evidence | Remaining acceptance work |
| --- | --- | --- |
| Config-selected v1/v2, historical compatibility | Explicit v2 schema/dispatch; unchanged legacy config serialization; combined v2 and historical step/resume regression230passed,1localCUDA-onlyskip atffd94b9; reporting3passed separately | Final exact-source server entry preflight |
| Complete v2 method | Native graph, gene+expression sum, distinct basal/response CLS, KDA/MLA/mHC, independent SSL heads/centers, loss reductions and committed EMA; operator/gradient/masking tests; prediction-only and full-joint conditional overfit passed | Scientific efficacy remains a formal experiment, not an implementation claim |
| Reuse v1 infrastructure | Canonical data/splits/evaluation manifests, shared epoch iterator, validation metrics, selection, RNG serialization and curves; version-specific teacher/rank checkpoint adapter documented in docs/design/GRADPERT_V2_REUSE.md | Exact final-config canonical-data checks in server preflight |
| Small artifacts and lifecycle | Best by validation prediction loss, true last50, committed-epoch recovery and best/last receipt tests; metrics_only/zeroPKL guards; complete11-condition Norman engineering validation at35c6c1a | Final startup checks; formal50epoch curves/best/last outcomes remain explicitly unexecuted |
| Dual-GPU capacity | Single synchronous world2 job, accumulation1. Micro74/global148 passed128updates+reload+300control5000gene inference twice; adjacent75OOM after20updates. Micro32/global64 and micro64/global128 passed the same protocol with eval128 at7c17d3a | Reference64/rank and H3global64/128/148 frozen;128steps is not proof of50epoch stability |
| Inference execution profile | Fixed-input fp32 batch2/64/128 parity passed for full5000 axis in1000 blocks and single5000 context; receipts in evidence/v2-inference; candidate eval128 | Exact generated-config preflight; single-condition timing is not full Jurkat validation timing |
| Grouped experiments | All15training-group generator paths,16group design includingD1; validation-only H1→H2→H3 selection and rejection of prospective parents tested; all5dataset scripts; actual47-row Jurkat matrix passed schema/hash/declared-factor checks | Publish independent configs; exact B0/H1 preflight and queue dry-run |
| G1/D1 | Holdout training-exclusion tests; Norman all11-condition context and response engineering checks; sealed Jurkat1000-column holdout and validation/context/D1 protocols, ID-only selection | Actual Jurkat holdout-row CUDA integration check; formal trained-checkpoint diagnostics after training |
| Five datasets and baselines | All5native datasets passed one-update/reload at35c6c1a. Existing official GEARS/TxPert/Scouter dispatch preserves model-specific settings; canonical contract equality dry-runs passed; GEARS frozen resource cache verified | Final-source native v1 and official baseline one-step preflights, result provenance audit and launch recipes |
| Verification and publication | Scoped Ruff check,64-file format check,19-file native-v2 Mypy and wheel/sdist build passed; separate clean published worktree protects unrelated main-tree edits | Publish generated final configs/source, deploy new clean server checkout and verify identity before final preflights |
| Relay | Codex remains build owner; scripts/handoff protocol are repository-mediated; exact candidate ZCode task recorded in session digest | Finish implementation/preflight package and resolve destination before supervision transfer |

The successful32 receipt and parameter census are in evidence/v2-dual-gpu.
Jurkat protocol hashes/IDs are in evidence/v2-jurkat-protocols; these are sealed
protocols, not executed scientific evaluations. Larger-context evaluation uses
an unrestricted checkpoint; heldout-column evaluation is a separate fixed-budget
comparison. Test metrics never select hyperparameters or parent configurations.

Per-dataset single-update checks prove integration only, not transferable batch
capacity. Other datasets retain their own canonical split/graph/control hashes
and require their own capacity measurements before a formal distributed batch is
chosen. Historical single-GPU32/64 results are superseded for the dual-GPU default.

The v1 R50 work in STATUS.md is separate. Current process/session handles and next
actions remain in V2_IMPLEMENTATION_STATE.md in the shared project worktree.
No formal50epoch baseline/ablation has been launched or delegated by this stage.
