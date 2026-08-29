# Subagent Strategy

Subagent mode: on

## Plans 031/032 disjoint implementation — active

| Task | Scope | Write access | Status |
|---|---|---|---|
| `official_graph_models` | Generic matrix-aware census worker, protocols, promotion and aggregation | `scripts/performance/**`, `tests/performance/**` | active |
| `native_architecture_gap` | Diagnostic-only step stage observer and focused training tests | `src/gradpert/training/step.py`, `tests/training/**` | active |
| `genept_shiftblock` | Seed-GO-ProteinPathway superset selection, E configs/matrix/tests and GenePT docs | explicitly assigned GenePT/config files only | active |

The primary agent owns Plans 031/032, Byte OS integration, implementation merge,
server verification and all launch authorization. Implementation ownership is
assigned only after these handoffs and must use disjoint files.

## Current read-only exploration

| Task | Scope | Write access | Status |
|---|---|---|---|
| `design_audit` | All GraD-Pert design and study Markdown under `TxPert/` | none | completed |
| `txpert_audit` | Frozen TxPert official code and local paper | none | completed |
| `gears_data_audit` | GEARS adapters and five-dataset integration/readiness | none | completed |
| `official_graph_models` | Frozen public graph encoders/configs/provenance | none | completed |
| `native_architecture_gap` | Native graph/view/config/execution extension points | none | completed |
| `genept_shiftblock` | TriShift emb_b schema/coverage and shift-block behavior | none | completed |
| `official_graph_models` | Exphormer/Fanout hot-path call-count and synchronization audit | none | completed |
| `native_architecture_gap` | Exact-effect, timing-receipt and matched-A/B gate audit | none | completed |
| `native_architecture_gap` | Ratio config and generalized consistency loss | scoped implementation | completed |
| `official_graph_models` | Generic H-count graph-axis materialization | scoped implementation | completed |
| `genept_shiftblock` | Successor H/L config generator and matrix diff gates | scoped implementation | completed |
| `native_architecture_gap` | Pre-model ratio contract, architecture identity and realized-view receipts | scoped repair | completed |
| `genept_shiftblock` | Runtime schema-v2 semantic matrix enforcement | scoped repair | completed |
| `official_graph_models` | Training-only profiler safety and production cross-H lineage audit | scoped repair | completed |

Completed handoffs:

- `.byte-os/subagents/exploration-design-audit.md`
- `.byte-os/subagents/exploration-txpert-audit.md`
- `.byte-os/subagents/exploration-gears-data-audit.md`

## Boundaries

- Exploration agents do not edit project or Byte OS files.
- The primary agent independently reads the selected design and reference files, resolves conflicts and owns all product decisions.
- Implementation subagents may be assigned only after plan files define disjoint directories, acceptance criteria and verification commands.
- Review subagents run only after implementation evidence exists.

## Plan 029 implementation ownership

- `native_architecture_gap`: only `src/gradpert/config/native.py`,
  `src/gradpert/modeling/losses.py`, and their focused tests.
- `official_graph_models`: only `src/gradpert/pilots/vnext_graph_axis.py` and
  `tests/pilots/test_vnext_graph_axis.py`.
- `genept_shiftblock`: only the Nadig Jurkat ablation generator/config/matrix,
  `tests/config/test_vnext_ablation_matrix.py`, and generator-focused tests.
- Primary agent: graph views, training/execution wiring, receipts, integration
  tests, documentation merge and all final verification.

## Required handoff

Every handoff must include Scope, Allowed files or directories, Files inspected, Files changed, Verification run, Result, Risks and Handoff.
