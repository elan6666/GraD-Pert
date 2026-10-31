# R3-RI two-epoch precision ablations

Authorized scope: PREC-SSL32, PREC-PRED32 and PREC-FFN8 only. Reuse the retained
two-epoch R3-RI results; do not retrain a baseline. Each new candidate starts from
scratch with the same seed1, global320/micro80/two ranks/accumulation2, 300 successful
updates and epoch2-last selection. Existing loss weights, architecture, data and
view manifests, 225-column expression exclusion, schedules and optimizer routes
are unchanged. Current R18 and its postfit/MFU dependencies remain separate.

The generator copies the complete checksum-bound historical R3-RI file into each
self-contained config. Its manifest records the original training source and
checkpoint identity. New source must not be attributed to the historical result;
cross-version comparisons require parity evidence or descriptive labeling.

| Arm | Sole precision intervention |
| --- | --- |
| PREC-SSL32 | Complete Student/Teacher SSL projection MLP, L2 and prototype computation in FP32; SSL1 node stays inactive. |
| PREC-PRED32 | Native prediction MLP and Softplus computation in FP32; SSL paths unchanged. |
| PREC-FFN8 | Native graph/Cell/Response GELU FFN up/down GEMMs in real FP8; other operations keep original precision. |

FP32 head execution disables local autocast and rejects CUDA TF32 rather than
silently altering global matmul settings. An input conversion cannot recover
rounding already introduced by the backbone; these are numerical hypotheses,
not evidence that BF16 caused the representation problem.

FFN8 uses stateless per-tensor current scaling, E4M3 forward operands and E5M2
output-gradient operands. Native FP32 parameters and gradient storage, optimizer
states/routes, KDA state, Teacher EMA and center timing remain unchanged. The
adapter must execute real forward/dgrad/wgrad FP8 GEMMs and fail on unsupported
device, layout or backend. Quantize/dequantize followed by BF16 linear is not
accepted as FP8 training. Checkpoint/recompute retains dropout/RNG semantics;
no moving amax state or additional trainable tensors are introduced.

This borrows the sensitive-operation/high-throughput split from
[DeepSeek-V3 section 3.3](https://arxiv.org/html/2412.19437v2), but is **not** its
complete recipe: DeepSeek-V3 uses E4M3 throughout with 1x128 activation and
128x128 weight scaling, plus other storage/communication choices. Its BF16 AdamW
moments are not imported into our Muon/AdamW experiment. GLM-5.3 checkpoint
precision alone does not specify a complete training precision policy.

Before formal fitting, require clean matching published source, unchanged
resource/allocator gates, dtype/kernel observations, ten native updates,
fullrestore and 300-control inference. Local synthetic/fake-GEMM tests cannot
stand in for native CUDA evidence. No test tuning or old-run/source mutation.

The shared runner prepares this exact family with `--preflight-only`. Its
acceptance checks both ranks' complete precision observations and all six
independent arm/GPU inference leaves. The queue, run and test roots must contain
no PKL files. Preflight completion cannot start fitting: the main owner must
independently verify and accept the sealed gates first.

After acceptance, `--resume-fit --fit-only-row PREC-SSL32` explicitly promotes
one fresh arm; use the corresponding name for each later arm. Each successful
fit must have two epochs, 150 updates per epoch and 300 successful updates, then
returns `FIT_COMPLETE.<arm>.json` with scientific completion false. The controller
does not dispatch historical postfit or test paths for this family. Main owns
the separate `common_ri_native` validation and unified diagnostics below. A
later promotion rechecks all original gates and completed fit evidence and
rejects retraining an already completed arm.

Primary prediction evaluation uses `common_ri_native`: the historical RI
evaluator does not enable autocast. Reset only the precision policy in a separate
strictly weight-loaded evaluation Student, leaving training checkpoint identity
intact. Do not infer evaluation dtype from BF16 training. As-trained precision
sensitivity is a separately labeled endpoint. Use the same medium/crossed
populations and unified analysis template:
graph gene tokens, condition e_p, Response gene tokens/CLS; raw/normalized rank,
relative variance, matched same-input/same-condition/different-condition cosine,
active head entropy/JS/CE/KL and gradient/logit scale. Report all 18 validation
metrics and MSE with existing paired uncertainty. Compare full-update latency,
throughput and memory directly; differing precision peak denominators must not
masquerade as speed gains. Neither a sharper Teacher nor higher rank alone is
scientific completion.
