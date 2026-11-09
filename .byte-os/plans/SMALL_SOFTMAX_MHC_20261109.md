# Small softmax mHC single-arm implementation

Human scope: one small_softmax_mhc_no_distill arm; frozen historical two-epoch
baseline reused; no DEG, extra arm, baseline retraining, or test tuning.
Design: docs/design/GRADPERT_SMALL_SOFTMAX_MHC_DESIGN_20261109.md revision v2.

Ownership: main owns modeling/config/objective/runtime/controller/docs/tests.
Luna alignment_supervision is read-only on frozen references and those paths.
Existing selected native supervisor owns read-only observation after exact run
handoff and its own acknowledgement. It does not tune/restart experiments.

Implementation: explicit architecture_variant in the existing v2 lifecycle;
four graph layers and three Cell/Response layers; single-pass four-stream mHC;
frozen 2048-wide GenePT; graph-neighborhood relation MLA and shared-KV attention;
two-group rank64 output; FP32 RMS, route/mixing and clipped SwiGLU (limit10).
Teacher, EMA, centers, SSL heads/views absent. Historical payload keys preserved.

Completed local evidence: 128 core/regression tests, 2 graph/recipe checks,
2 equivariance/admission checks; changed-file lint; sdist and wheel. New module
passes mypy; full dependency traversal retains historical type/stub errors.
Server baseline launch hash independently matches the original local plan:
4a6c85d44edb9bb4d675db6c423379ffdfbe6f630c43c256443d4312a77df5ec.

Remaining: clean published source; immutable server checkout; 10 complete
optimizer updates, exact restore and 300-control inference; capacity acceptance;
2 complete epochs with identical data/split/selection/global320/micro80/seed1;
frozen 445-condition expression evaluation; independently verify terminal hashes,
whole-run zero PKL, all18 Pearson entries/counts, Chinese comparison and figures.

Only aggregate receipts/reports/figures may be pulled after dry-run review.
No new scheduler is created. The user has activated the existing supervisor.
