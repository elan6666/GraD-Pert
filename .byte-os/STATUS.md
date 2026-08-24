---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: blocked
current_workflow: byte-review
next_workflow: byte-build
review_verdict: block
iteration_count: 3
harness_status: ready
hard_blocked: true
updated_at: 2026-08-24T18:03:31+08:00
---

# Status

- Goal mode: on; Codex goal created for the end-to-end GraD-Pert delivery.
- Project state: root Git repository and standalone package shell exist; plans
  002 and 003 are complete, and native model plan 004 is active.
- Discussion source: Codex task `01a00ab3-9864-7032-98d9-45f6d0016838`, `TxPert/grad-pert`, and `.byte-os/DISCUSSION.md`.
- Confirmed scope: all five datasets through `canonical_ready`, training, evaluation, and manifests.
- Training route: one B2 configuration; no B3 and no ablation matrix.
- Benchmarks: GEARS, TxPert, matched-control mean, global train delta, and general train delta with Norman additive seen singles.
- Fairness: one canonical condition split/evaluation manifest; paired run seeds and shared 300-control evaluation draws.
- Metric contract: frozen union with three distinct Pearson headlines: TxPert macro delta, TriShift delta, Systema Pearson.
- Artifact contract: versioned condition-level PKL plus Parquet/JSON/H5AD; notebooks consume artifacts only.
- Execution policy: all 15 learned model/dataset integrations must pass exactly 1 epoch; only GraD-Pert continues to max 200 with validation-only patience 10. GEARS/TxPert are official-package `smoke_only` runners.
- Server topology: formal compute only on `/data/yilangliu`; local/GitHub/server must share one clean commit; large artifacts remain server-only.
- Hardware decision: two RTX 5090 with 32607 MiB each; the sustained global
  gate selected `K_head=16384`. Candidate 65,536 failed on K562 at step 34,
  32,768 failed on Jurkat at step 20, and 16,384 completed 128 real steps on
  each dataset. Its worst peak reserved memory was 24,226,299,904 bytes on
  Jurkat, below the 28,168,037,990-byte threshold.
- Harness: ready; Claude context ready; Codex context ready; `AGENTS.md` ready.
- Config contract: exactly 30 self-contained model/dataset YAML files verified with 30 unique byte hashes; no global config/default/merge/inheritance path.
- Evidence iteration 1 migrated all five splits to the frozen
  official GEARS default-graph representability intersection without
  reshuffling retained conditions. Data/hash verification, graph rebuild and
  evaluator-state rebuild passed. The sustained `K_head` refit is complete and
  the five native configs now freeze 16,384.
- Publication gate: the first public push is paused because the safety review requires explicit approval to publish internal design/server-topology material; no workaround is used.
- Data state: all five official sources remain sealed and all five datasets are
  `canonical_ready` under `datasets-v2`. The canonical H5AD hashes did not
  change; split/control hashes did. Server verification passed for all five,
  and prior split/evaluation/graph receipts are recoverable under
  `/data/yilangliu/GraD-Pert/superseded/20260824-gears-default-intersection`.
- Local prepared/result hygiene: prior datasets-v1 and seed-0 small artifacts
  are preserved under explicit `superseded/` roots. Current v2 receipt chains
  are marked pending sync rather than being inferred from terminal output.
- Local verification: 137 tests pass and 9 optional Torch/PyG/anndata or
  pending-v2-receipt tests skip; all 30 self-contained configs, Ruff
  lint/format, the 20-module available-dependency strict mypy surface, and the
  offline wheel/sdist build pass. Full local strict mypy cannot complete because
  the intentionally lightweight local environment has no Torch/PyG; server
  policy/config/type gates passed before the latest resume/launcher additions,
  so a fresh complete server regression remains required after synchronization.
- Evidence iteration 2 closed the server-operations implementation gap. The
  dry-run-first matrix now freezes 15 learned smoke, 15 paired nonlearned and
  20 native full tasks. Full runs are machine-gated on exact one-epoch training
  receipts, checkpoints, no test Truth during fit, shared commit/config
  identities, and per-dataset equality of canonical-data/split/300-control
  hashes across all three learned models.
- Small-file synchronization now has a sealed staging layer: only allowlisted,
  size-bounded, non-symlink files from named `small_results` or one explicit
  receipt root can enter a new staging tree, and the transferred tree must
  reverify with no missing, extra or changed files.
- Evidence iteration 3 strengthened the leakage and source boundaries. GEARS
  and TxPert now finish fit, checkpoint hashing and training receipt sealing
  before the canonical test reader exists; development Git worktrees reject a
  declared commit that differs from HEAD. The current pre-server review verdict
  is `block`, with exact unresolved formal-execution evidence listed under
  `.byte-os/reviews/2026-08-24-pre-server-review.md`.
- The final notebook surface is now executable rather than implicit. An
  explicit-source, dry-run-first builder seals a catalog only for the exact 45
  formal coordinates (all six models at paired seed 1 plus native seeds 2--4),
  one source commit, shared per-dataset fairness hashes, one native config per
  dataset, and exact three-metric schemas/denominators. The benchmark notebook
  was re-executed and uses the strict final loader; no formal catalog is claimed
  before the actual server results exist.
- TriShift architecture audit: source-level call chain recorded in
  `docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md`; the explicit hash-pinned
  ResultCatalog and evaluator-only condition bundle are implemented and tested,
  without newest-result discovery or runner-side Truth.
- Notebook handoff: the read-only benchmark notebook is generated and executed;
  it reports no result until an explicit hash-pinned catalog is synchronized.
- Hard execution blocker: new remote sessions and sync calls are unavailable
  until the account/tool usage window resets on 2026-08-27 14:11 CST. The
  completed capacity session was collected normally; no alternate SSH path is
  used. Publication/formal execution also still needs approval for the first
  public push of the internal design and server-topology material.
- Next action after access is restored: synchronize the selected configs and
  v2 small receipts, run the fresh server regression, then execute the 15
  learned one-epoch gates and regenerated seed-1 nonlearned matrix.
