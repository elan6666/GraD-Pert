# GraD-Pert v2 B0: single-stream residual and six Pearson metrics

This is a new three-epoch Jurkat B0 coordinate. The parent is the published
source-key-gate m74 configuration; it retains the full prediction + SSL1 + SSL2
objective, two-GPU accumulation, query/view rules, test-target expression
exclusion, and all attention operators. Only the residual stream count changes
from four to one. In the native `ManifoldResidual` and `CrossManifoldResidual`,
one stream uses the ordinary residual `x + sublayer(norm(x))` and does not build
or apply mHC routing/Sinkhorn maps. KDA and MLA are still the attention modules.
The student and EMA teacher have the same new architecture.

The fit lifecycle selects `best.pt` by minimum fixed-view validation **joint
loss** and keeps epoch-three `last.pt`. The config monitor is `val/joint_loss`.
Neither validation nor checkpoint selection uses test truth. The six Pearson
outputs are descriptive, not selection criteria.

Evaluation state `evaluation-state-v2` has its own paths, leaving v1 manifests
and historical results untouched. Per condition, one truth-ranked Top20
non-dropout DEG set serves TxPert, TriShift and Systema variants; perturbation
target genes remain eligible. Each family reports `_all` and `_deg`. The
reference vectors retain their established definitions: TxPert subtracts the
mean of the 300 input control rows, TriShift subtracts the matching-context
control-pool mean, and Systema subtracts the equal-weight mean of non-control
condition centroids. Systema validation uses train conditions; test uses
train + validation conditions. The DEG ranking uses evaluator truth and is
never available to model fitting or checkpoint selection.

The new configuration is
`configs/v2/no_mhc_joint_eval_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml`.
An otherwise identical m64 variant is also provided under
`three_epoch_m64_a2` for a lower-memory launch if capacity testing rejects
m74. The parent m74 capacity proof applied to four streams and that formal run
later OOMed during epoch one; neither batch setting is assumed safe for the
new one-stream model. Complete-update semantics and sustained memory require
fresh two-GPU preflight before this B0 launch. The formal run must use a new
ID and published, clean, immutable source.
