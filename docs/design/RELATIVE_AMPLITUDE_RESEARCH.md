# Relative amplitude floor: two-arm representation study

R12 increased the centered covariance effective rank while ResponseCLS variation
became extremely small. Normalizing covariance eigenvalues by their sum constrains
spectral shape, but cannot constrain centered energy. Historical R4 also showed
that large variance and high rank alone do not establish prediction utility.

R13 retains the original RI architecture, data/split/control identities, view
recipe, MSE/SSL weights, optimizer and two-epoch/300-update schedule. Both arms
start from scratch, use effective batch320=80x2devicesx2accumulation, and retain
only the final checkpoint (best remains null). Teacher EMA remains .99→1.

| Arm | Rank target/weight | Extra positive cosine | Relative amplitude target/weight |
|---|---|---|---|
| R13-SRAMP | .20/.02 | .10 | .0225/0.012439571837341717 |
| R13-SRAMPNC | .20/.02 | 0 | .0225/0.012439571837341717 |

For each complete, deduplicated Student Global post-final-RMSNorm ResponseCLS
population, let V=mean(||z−mean(z)||²), E=mean(||z||²), q=V/E. The amplitude loss is
mean_Global ReLU(sqrt(.0225)−sqrt(q+1e−8)). The denominator retains its gradient.
Covariance/statistics use FP64; no row normalization or diagonal jitter enters the
training loss. Existing rank softness .25 and all three auxiliary weights share
min(1, completed successful updates/30), with zero weight at the first update.

The .0225 floor is a project reference-scale choice, not an official VICReg
threshold. It is close to the historical RI relative variance on the fixed
training-side panel. Uniform gamma inflation cannot change q, but common-mean
reduction, anisotropic gamma or nuisance variation can still increase it. Report
raw V/E/mean-energy/q and separate condition differences from control/view
variation. The ordinary common medium/panel currently does not collect final
gamma, affine-free or fixed-row-RMS loophole diagnostics: those postfit fields
remain NA and require a separate frozen zero-update measurement before scientific
acceptance. Calibration-side diagnostics do not stand in for trained endpoints. An exactly constant
population is rejected by the existing spectral near-zero-trace guard; near
collapse and inactive hinges must not be confused with proof of learning.

The common coefficient was selected from eight fixed training populations at the
sealed R12-BAL-last endpoint by the preregistered median Response shared-parameter
amplitude/MSE gradient budget of .10. Init and original RI-last were diagnostics,
with inactive hinges; all 24 observations made zero optimizer/Teacher/center
updates. All144 endpoint/group/arm direction comparisons had no positive MSE
descent direction flipped negative. This is engineering evidence, not validation
or test-based hyperparameter selection or evidence of scientific improvement.

Calibration source: `933dfb5cf546adbf5561a4041fbd4737643f2020`.
Original native result SHA256:
`6a98d66d761f69e65d2f41d35665d3965702602d6ba11ec300b2bb94d721d191`.
Formal configuration/harness source is recorded separately at launch. It must
match clean local/GitHub/server publication. The R13 family manifest pins complete
core/model/config/optimizer/checkpoint/reduction and calibration-module file
hashes to the actual calibration source. Capacity instrumentation may add the
new gate; the exact source segments of its calibration-used state_fingerprint
and validation_state functions must remain unchanged. Compatibility does not
replace the original training, calibration or evaluation SHA.

Before fitting, each arm requires ten actual native successful updates, full
checkpoint/state restoration and both physical GPU eval128/reference128 parity
gates at2e−6. The independent R13 amplitude gate records actual raw and
warmup-weighted rank/cos/amplitude, q/V/E/mean and first-update zero. The first ten
steps may have an inactive amplitude hinge; zero is then a valid measured value.
They do not prove sustained capacity or future task improvement.

The sealed queue is first executed explicitly with --preflight-only and exits
at validated PREFLIGHT_COMPLETE. A MAIN_ACCEPTANCE file cannot make that process
fit. A later, separately invoked --resume-fit promotion must verify successful
preflight state, the exact unchanged queue/source/gates/data/checkpoints and main
acceptance, reacquire resources, and reuse the same fresh run IDs without rerunning
preflights. This engineering stage does not authorize or execute that promotion.

After training, use the same common medium and separate fixed training panel,
and final validation18 metrics plus MSE. Small partial-population probes retain
NA for rank/cos/amplitude, never fabricated zeros. Report saved actual warmup
contributions rather than reconstructing them from epoch-average warmup. Reuse
historical R3-RI as primary reference and RS-B0/RS-RI only as context; no baseline
rerun. Test remains deferred until the research design is locked. Geometric and
Teacher discrimination evidence must accompany validation utility; single-seed,
cross-version comparisons do not establish seed-level significance.

The variance-floor principle is related to [VICReg](https://arxiv.org/abs/2105.04906),
but R13 uses a global relative RMS floor with existing spectral regularization,
not a reproduction of VICReg. [LiDAR](https://arxiv.org/abs/2312.04000) motivates
separating reproducible task-related variation from nuisance rank inflation.
