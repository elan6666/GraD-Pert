# GraD-Pert v2 acceptance ledger

Scope remains the full implementation, two-5090 capacity testing, grouped runnable
experiments and verified supervision handoff. This ledger does not reduce it.
Source inspected: `f92ff5564afac35c48976469dd1d0b4a980bd982`.

| Requirement | Current evidence | Acceptance gap |
| --- | --- | --- |
| Config-selected v1/v2, historical v1 behavior | `config/v2.py`, `execution/train_entry.py`; combined v2 and shared step/resume regression221 passed,1 CUDA-only skipped at86488ff; reporting3 passed separately | Final combined regression on delivery source and real server entry |
| V2 method | `modeling/v2`, `training/v2/objective.py`, native operator tests and joint-loss capacity probes | Final design-to-code audit; no scientific efficacy claim |
| Reuse v1 infrastructure | Canonical data/evaluation, metrics, entry, source checks, selection, RNG restoration, curves and `training/epoch.py` are shared | Review outer lifecycle/postfit differences; preserve necessary teacher/rank-state adapters |
| Small evaluation artifacts | Schema metrics_only, best/last retention, compact validation population, no-PKL completion check | Norman full11-condition engineering validation passed at35c6c1a; actual complete lifecycle with curves and best/last receipts on delivery source remains |
| Two-GPU engineering evidence | Two-rank micro72/global144 completed128 updates, checkpoint reload and300-control/5000-gene inference at ac15422; training19.50884 cells/s; receipt SHA256 `00f3e1dd041fd2b8d545510adf3c8ce24246281b5b0d39985f42d23668b40cac` | Dual-rank micro76 and80 failed OOM after13 updates; adjacent boundary, repeatability, inference batch parity and final default remain. Single-rank defaults are superseded |
| Hyperparameter-first groups | `experiments/v2/groups.json`; standalone generation, validation-only selection, follow-up dependency gates | H3 levels/configs depend on capacity; final complete generated matrix not frozen |
| Execution/recovery/collection | `scripts/v2/run_group.py`, `resume.py`, `collect_results.py`; tests and source gates | Actual server queue dry-run, per-row integration checks and source parity |
| Expression holdout/context generalization | Holdout manifest, training exclusion/poison tests, fixed-axis context-query API | Norman fixed1000-axis context1000/2000/5045 engineering check passed all11 validation conditions; Jurkat ID-only holdout manifest prepared; concrete selected-parent config and real holdout training check remain |
| CLS diagnostics | Response CLS→gene intervention and tests | Norman engineering diagnostics passed (receipt5395b54084c2754e007c815e624d76d06e7a9f8af5d274799d311561a337720a); formal trained-checkpoint diagnostics remain downstream experiments |
| Five datasets and baselines | Existing canonical/shared infrastructure preserved | Five dataset single-update/reload probes passed at35c6c1a; GEARS/TxPert preflights and Scouter official import passed; exact delivery-source startup checks and actual receipt parity remain |
| ZCode supervision | Relay boundary remains Codex build/test first | Exact destination resolution and complete verified handoff package |

Do not use a single-condition capacity inference as evidence for full validation
metrics or50-epoch lifecycle completion. Do not use a one-step integration pass
as sustained memory evidence. Best/last test scores must never choose a parent.
Actual training/evaluation SHAs remain attached to each original receipt; later
code publication must not relabel historical evidence.

The v1 R50 task described in `STATUS.md` is separate historical/parallel work;
this ledger and the v2 plan track this implementation without rewriting its run
state. Mutable active probe details remain in the task's implementation state log.

Engineering validation, G1 and D1 receipts use one-update Norman weights and are not scientific efficacy evidence. The dual-rank micro72 capacity receipt evaluates only one Jurkat validation condition; it does not establish full-validation duration or50-epoch stability. Formal configs remain gated on the measured dual-GPU batch and exact final-source preflight.
