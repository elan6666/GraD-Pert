# R6-EMA99: short-horizon Teacher tracking

Status: authorized representation research; implementation/preflight pending. This is the existing two-epoch research program, not a new R50 protocol or a change to historical experiments. Overall scientific acceptance remains pending.

## Evidence and hypothesis

R5 moved A=.05 and CR=.00004 to the existing SSL2 CLS pre-L2 bottleneck. Compared with reused RI, matched validation MSE worsened 2.682%, versus R4's 14.596%; Student geometry improved but Teacher targets remained nearly uniform. Native and matched clean-Global frozen diagnostics used the same R5 checkpoint, fixed 128-row crossed panel, zero updates and unchanged model/center/RNG/checkpoint states. At the same .04 temperature and frozen center, clean SS/ST/TS/TT JS averaged .016541536/.034367229/.000810475/.001539877. Thus the Teacher encoder path remains weak when augmentation visibility is matched. Head swapping is a functional stress test, not a unique causal decomposition: coordinate compatibility and nonlinear parameter averaging remain limitations.

Native EMA over 300 successful updates rises .99→1. Initial Teacher coefficient at the endpoint is .2218701963; the most recent 30 and100 Student update coefficients sum to .0023472169 and .0822794456. These are recurrence coefficients, not fractions of biological information. Constant .99 has corresponding arithmetic coefficients .0490408941/.2602996266/.6339676587; this alternative has not yet been trained.

Hypothesis: in this short protocol, retaining Teacher tracking through the end may improve input-specific targets without destroying cross-view consistency or prediction utility. This is not an assertion that large EMA is generally harmful, nor a replication claim for DINO/BYOL. Primary-source literature and limits are recorded in the existing R6_EVIDENCE_DESIGN.md (official DINO/BYOL/SimDINO/VICReg references).

## Single intervention

Parent R5-PROJ, no baseline rerun. Only model parameter teacher_end changes 1.0→.99; teacher_start remains .99. Existing lifecycle computes the same cosine formula with equal endpoints, hence constant .99 per successful optimizer update. No new model or loss algorithm. Same initialization/seed, GenePT rule, residuals, activations, bias, loss weights, auxiliary space and warmup30, temperature/center/prototype rules, global320=80×2×accum2, two epochs, sample/view/order/data and validation18/medium/crossed manifests. No test tuning. Native fully resolved standalone config and source identity are mandatory.

No concurrent sharpening, prototype normalization/freeze, optimizer/LR/regularizer, training mask or feature-architecture changes. Clean-input intervention remains diagnostic-only; training uses native views. End-of-fit diagnostics reuse the same panels/aggregation definitions and expose actual sample ceilings. Source and checkpoint versions are separate.

## Decision criteria

Compare R6 against R5 and reused RI/R4, not rank alone. Report native split loss, CE=H+KL, Teacher H/JS, raw CLS and pre-L2 geometry for both roles, same-input GG/GL/LL and matched same/different condition/control separation; matched 18 validation metrics with NA counts and MSE. Alert if MSE exceeds RI by5% or positive GG falls below .90, not automatic proof below those bounds. Seek Teacher input-specific distribution and correct-versus-mismatched gaps with preserved prediction utility; entropy sharpening or lower joint auxiliary loss alone cannot satisfy acceptance. Unseen-expression nonDEG weakness remains explicit. One seed/two epochs gives a bounded hypothesis test, not general resolution.

## Implementation ownership and verification

Agent r6_evidence_design owns new generator scripts/v2/generate_representation_round6.py, its standalone configs/v2/representation_round6_20261024/ and tests/test_representation_round6.py only. Main owns dispatch integration in existing scripts and shared tests; no overlapping edits. Published pre-change source493e295dbd8654876ac4da36710cdea3c400fc57 verified clean and on GitHub main before edits. Preserve src tree exactly; verify R6 differs from parent scientifically only in teacher_end. Native lifecycle tests verify constant .99 and successful-update semantics. Appropriate family/panel/postfit/capacity contract tests, Ruff and independent review precede scoped commit/push. Fresh server checkout must match published main; no active source is modified.

All native 10-step/restore and dual-card eval128/reference128 gates at existing2e-6 tolerance must pass before exact main acceptance. Then existing controller launches one fresh two-epoch R6 run, followed by validation18 and medium/crossed diagnostics. Preserve raw data/checkpoints on server, transfer only reviewed small aggregate/receipts. No resource relaxation. Long formal work goes to existing supervisor with fresh exact run ID, 20-minute native monitor and verified main-return route. At terminal main independently verifies and immediately analyzes/redesigns; no scientific completion claim is implied by phase success.
