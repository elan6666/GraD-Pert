# Cap40 full posthoc diagnostics

## Superseding scheduling request: two GPUs (2026-10-15)

Correction after real coexistence probe: startup-only peer admission was
insufficient. Old FG's per-condition guard rejected the GPU1 FB probe and
stopped itself at435conditions/1731rows (no COMPLETE). No worker signal was
sent, but the migration caused this interruption; preserve that evidence and
never relabel it success. The36da82e gate is historical, no formal dual job
was launched with it. Fresh source/run uses one exact live dispatcher as the
ownership root at BOTH startup and every condition. All two-slot workers are
its actual descendants, so the original shared-GPU guard counts both as owned
without changing any policy. Validate parent UID/PID/start/cmd every check.
Old per-worker observers cannot be live-adopted into this mode; retain all
completed DP/FN/PO and rerun only incomplete FG analysis with FB under fresh
IDs. Frozen weights/method/manifest remain unchanged; this is not retraining.
Run both new gates, then require actual simultaneous multi-condition progress
before handoff. Unknown siblings remain foreign, parent death fails closed.

User authorizes independent medium analyses on both existing RTX5090 GPUs,
one worker per device, not training/DDP or any new ablation. Pre-change baseline
47c0ba129b160ec07de5473554936ce06b009f48 verified published on GitHub main.
The frozen single-worker admission classified another same-project worker as
foreign. Recognize only an exact same-UID peer on the other physical GPU using
PID/start-ticks/NUL-command SHA and the common manifest; retain the original
foreign UID/program whitelist, 2GiB physical reserve and 90% own-device budget.
No native method, numerical path, seed, mask, row grouping or checkpoint changes.

Retire only the identity-checked old dispatcher (never its process group),
preserve/adopt its current scientific worker and all completed reports. New
immutable queue/source records per-arm analysis source, including original47
adopted/completed outputs. Pending arms receive fresh output IDs. Reuse five
frozen mathematical gates; test pending workers on the other GPU at indices
0/1/15 with strict native-path2e-6 and state immutability before main acceptance.
New dispatcher fills free slots only, reports each arm once, then runs the same
five-arm comparison. Unknown/recycled peer, missing COMPLETE, output collision,
resource or worker failure stops dispatch without signaling remaining workers.
The old controller state is historical, not completion evidence.

Acceptance: CPU identity/resource/adoption/slot tests; clean matching publication;
GPU gates; actual simultaneous progress on two GPU identities; immutable source,
common manifest and unchanged budgets. One short Luna observer owns test waits;
then the unique20min supervisor is retargeted to the real combined run and ACKed.
Report observed overlap/end-to-end makespan, not a claimed 2x speedup from GPU
utilization. Final Chinese reports/figures/tables remain the overall endpoint.

## Superseding user scope: medium analysis (2026-10-15)

User rejected the20h full run. Main verified exact identities, stopped only the
new priority controller3321763 then read-only DP3317381 with SIGTERM; no other
processes/checkpoints touched. STOP receipt db93990425805e2febaa865b57300174b1109d127f95a3216ee71428c6739a44
preserves DP198conditions/6679rows and full old queue/source/output. Old results
are partial, not completed scientific analysis.

New shared medium manifest: all1335conditions,<=4context-balanced cells each;
Context order must be seeded percondition, not lexical: parent has up to40
contexts/condition and1312conditions exceed4. Initial1539506 seal/gates passed
model parity but its lexical context prefix was rejected before formal launch.
Preserve that seal/gates as superseded; publish/reseal fresh identity. Query,
graph-view/mask recipes depend on seed/query/anchors and cellcount (frozen views.py),
not row/control IDs; if all shapes/queries/seeds match, retain the already hashed
recipes, with every actual native assembly rechecked at gates/full worker.
128deep conditions stratified by context-set and original condition-size bins,
32gradient conditions nested in them; budgets<=5340/512/128 cells. Reuse original
row/control pairing, query panels/order/seeds; reseal actual masks for new cell
counts onCPU. No test-result selection. Condition-equal basic summaries cover
allconditions; deep diagnostics describe only the fixed stratified subset.
Frozen checkpoint/method and actual formulas unchanged. Explicit medium scope
in receipts/plots/reports; do not reuse full-scope gates or claim full coverage.
CLS between-condition decomposition now fixes querypanel+context, control label
kept distinct from paired perturbation. Do not treat4cell ranks as global ranks.

Fresh published source/run/config: verify basic-only and deep/gradient paths,
strict native/cache parity and no state updates for allfive, then automatic
PO→DP→FN→FG→FB and reports/comparison. Gate waits/sealing use one Luna helper;
then main independently accepts, launches full medium queue and hands exact
run to the existing unique20min supervisor. Old monitor stays paused until
new real job/ACK. Estimate duration from actual pilot, never from old four-cell
probe timings. Full Chinese common template/reports/all9arm figures and4master
figures/all aggregate table indices remain required; large raw files stay server.

Outcome: one sealed diagnostic manifest, read-only last-checkpoint analysis for
DP/FN/FG/FB/PO, complete grouped Chinese reports, all figures and underlying
aggregate tables. No training, selection, EMA/center/optimizer updates, or edits
to original runs. Training/method source remains immutable 420d3a1; analysis has
its own published source identity. Baseline 0ee26e71a45bbbceff8007d9e0ce92571178b012
is committed and verified equal to GitHub main before edits.

Scope: all 47,836 cap40 training rows / 1,335 conditions, preserving numeric
expression exclusions and the 1,000-query budget. Fixed per-row control pairing,
query panels, view recipes and orders shared across arms. Full basic output/CLS
statistics; deep diagnostic subset up to eight cells per condition, stratified
by context; 64-condition x four-cell gradient subset. Existing frozen test
metrics remain separate; test-side feature probes use evaluation visibility.
Raw un-downsampled populations are out of scope.

Implementation ownership: main only, scripts/v2/cap40_posthoc*.py,
tests/test_cap40_posthoc*.py, docs/experiments/CAP40_POSTHOC*.md and common
analysis template. No native model or training changes. A bounded Luna helper
may supervise test waits read-only, without a second main poller.

Acceptance: exact unique-row coverage; aligned row/gene/view identities; separate
cell_line/cell_type/batch/condition and control treatment labels; stable streaming
moments/entropy/within-between decomposition and uncertainty; per-view independent
cell counts; no prefix sampling; checkpoint/state/center/RNG immutability;
original-method prediction parity; same manifest hashes across all arms; CPU
tests and guarded server integration before launch. Partial outputs never count
as completion. Original scientific artifacts are read-only; fresh output ID.

Execution: verify ownership and PO terminal evidence; implement/test and publish;
seal manifest on server; measure diagnostic chunk capacity/runtime, then execute
full jobs with approved GPU policy. Longer than one hour uses exact supervisor
handoff/20min monitor and verified return route. Report throughput, completed
rows/conditions and ETA without extrapolating old four-cell timings.

Deliver: common schema v2, each arm's report/receipt and full aggregate CSV index,
comparison report embedding every produced figure; missing/unidentifiable items
explicitly labeled. Per-cell matrices/checkpoints remain server-only. Persist
current run and next action in Byte status; no duplicate monitor.

## Rejected optional transformation (2026-10-15)

Shared manifest sealed: cc208dc56b2eaa45eb0ce4b005729c049f98349860fb6d148576f3fc99fbc6e1,
47836 rows, 1335 conditions, 10532 deep rows, 256 gradient rows, 11793 controls.
First ca65644 DP preflight preserved as FAILED: singleton-cell vs full-group
BF16 forward max tensor difference 0.04460430145263672 exceeded 2e-6.
Do not relax tolerance or adopt cell splitting. Full-group native vs cached
path had passed the preceding strict comparison. Repair only makes optional
chunk rejection explicit; every used path must still pass the original 2e-6
reference/cache comparison. Keep the exact sealed condition groups for all arms.
Record per-tensor errors, no claims about the cause until independently tested.
Fresh preflight/analysis IDs; old source/results untouched.

## Analysis observer repair after first full attempt

Failed a3f8b66 full DP at schedule index22 (AFG3L2+ctrl): first Global had
0 masked tokens, second850. Native iBOT correctly skipped first; analysis
incorrectly used head-call counter0 as view0. Fix only observer metadata:
map call index to actual active view indices for both node heads. Preserve
empty-mask eligibility explicitly; do not create zero-loss rows. Full failed
root22conditions/848rows is preserved, not resumed or relabeled.
Expanded fresh gates use indices0/22/58/295, actual mask patterns both active,
first empty, second empty, both empty (including a singleton condition).
All five arms must pass state/recipe/strict usedpath parity plus these cases.
Formal order/population/queries/masks/native losses remain unchanged.

## User priority update: PO first delivery, preserve active DP

The user requested PO evaluation priority. Its original last test is already
complete; prioritize its new full posthoc analysis, without rerunning that test.
First preserve/wait for existing DP onGPU0, then run PO onGPU1. Retire only
the exact old dispatch parent (not its process group/DP child), receipt both
identities before and after; a fresh scheduling controller adopts the live DP
worker and never relaunches it. Frozen worker/report/comparison source remains
460d5216f33e247e018882a15da18aba5f40e1bb, method420d3a1 and sealed recipes unchanged.
New controller source and schedule identity are recorded separately.

Await/verify adopted DP, launch PO, generate DP's CPU report during PO, then
PO report and FN/FG/FB and the
five-arm comparison. Verify full coverage, checkpoint/manifest/state and output
hashes before adopting outputs. Stop dispatch on missing worker without COMPLETE,
PID reuse, invalid hashes or priority failure; preserve any still-running workers.
No signal to scientific workers, no active source/config/queue edits, no retries
or reused output IDs. PO retains its existing strict native/cache first-condition
gate and final state-immutability checks onGPU1; five frozen model gates already
passed, and unchanged worker/statistical code does not require repeating them.
Parallel DP/PO was rejected before launch: the frozen per-worker admission
classifies an independent peer as foreign. Do not change active DP or weaken
the policy to make concurrency pass. The initial4557e10 concurrent scheduler
was published but never used in an experiment; retain it as superseded design.
The original failed/retired controller state is historical, not a completion
receipt. Unique read-only short Luna startup checks, then updated20min supervisor
with exact combined schedule/DP adopted/PO identities and return conditions.
