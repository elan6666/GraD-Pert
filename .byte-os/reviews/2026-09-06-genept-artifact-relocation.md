# GenePT artifact path relocation

The original `/data/yilangliu/GenePT-Seed` directory is absent. A read-only
server search located the same artifact under
`/data/yilangliu/DinoGenePT/data/embeddings/seed-go-protein-pathway-master-aligned.npz`.
Its SHA-256 is
`34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`
and its size is 68,243,774 bytes, both matching the sealed prior input.

Only the current generator and four E-row artifact paths change; the matrix
records their new config hashes. The scientific prior content, graph, seed,
training settings and all other rows remain unchanged. The path assertion is
updated while retaining the exact content-hash assertion. Historical configs,
results and availability receipts remain immutable.

The blocked P0 under `capacity-m-9301ca1-v1` is preserved. This source change
does not authorize reuse of that receipt: publication, server checks and a
fresh availability receipt bound to the actual path must precede a new P0.
M2/M4 capacity must also cover the isolated-anchor failure after 28 steps.
Neither path discovery nor local tests constitute capacity or formal completion.

Targeted config, matrix and preflight checks: 37 passed. Local full pytest:
513 passed, 10 environment/reference skips. Ruff, format and diff checks pass.
Local mypy is blocked by missing PyTorch (37 diagnostics); it is not recorded
as a pass. Exact-commit server gates with the complete runtime and new server
receipt validation remain required before any CUDA launch.
