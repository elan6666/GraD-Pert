# Official recovery then five FFN experiments

Owner: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b. Published clean base: 141989eeb18719aad2d790012eb6fc7366c31e9b. No BR or Scouter rerun.

Outcome: finish official scDFM and GEARS (four epochs, existing expression holdout and frozen 300-control/18-metric last evaluation), then run five native four-epoch last-only arms FB/FG/FN/DP/PO using the completed BR400 anchor. Preserve failed receipts and immutable releases. No native arm dispatch before both official baseline terminal acceptances.

Current repair: scDFM batch48 FP32 OOM; CPU saved-activation offload GPU multistep parity failed. Diagnose ordinary repeatability and layout changes under original tolerance before adopting a repair. No approximation of MMD or model rewrite.

Native specifications: FB disables Linear biases in all FFN/role MLPs, retains normalization beta and mechanism biases; FG uses GELU in all role FFNs, transformer hidden1536 matches original SwiGLU hidden1024 weights; FN vector-role pre-LayerNorm eps1e-6 with FP32 statistics, scalar expression lifted first, existing FFN norm replaced not doubled; DP four prototype layers weight_norm(g initialized1), first epoch LR0 (optimizer moments continue), original widths/MLP/init conventions explicitly recorded; PO score head Softplus directly predicts expression, delta=prediction-control, all response inference paths consistent. No silent default changes.

Diagnostics: fixed training-only probes at initialization/epochs; prototype entropy/JS/KL and pre/post projection variance/rank; gene/control/response token variance, delta/negative prediction/tail metrics; fixed-seed order and control/perturbation interventions; shared graph and basal loss gradient norms/cosine at bounded cadence. No excluded expression enters training probes, no optimizer/EMA/center/RNG mutation. Aggregate files only, budget and elapsed time recorded.

Disjoint delegation: native implementation agent owns config/v2.py, modeling/v2/*, optimizer/runtime update hooks and its new tests; diagnostics agent owns NEW process-diagnostic module and NEW tests only, does not wire runtime/lifecycle. Main owns official adapters, config generators/queues, runtime diagnostic integration, lifecycle, documentation/publication/launch. Agents do not commit or launch jobs.

Acceptance: scoped local regression; baseline parity with disabled options; nonzero-LR multistep/checkpoint/EMA/center/RNG tests; official real GPU parity and ten updates/restore/300-control gate; clean local/GitHub/server matching release; actual run launch; unique supervisor ACK and active monitor/return wake. Long tasks handed to designated supervisor 01a0df0b-4142-7df1-86c0-d959471d80a1, bounded probes to Luna without duplicate polling. No Goal requested.
