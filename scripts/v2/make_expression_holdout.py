"""Seal a response-independent G1 expression partition on the scientific server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradpert.hashing import sha256_file
from gradpert.training.v2.holdout import make_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--excluded-genes", type=Path)
    parser.add_argument("--heldout-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to("/data/yilangliu") for p in (args.genes, args.output)):
        parser.error("scientific expression partitions stay on the server")
    genes = tuple(args.genes.read_text().splitlines())
    excluded = ()
    if args.excluded_genes is not None:
        if not args.excluded_genes.resolve().is_relative_to("/data/yilangliu"):
            parser.error("excluded gene IDs stay on the server")
        excluded = tuple(args.excluded_genes.read_text().splitlines())
    partition = make_partition(
        genes, heldout_count=args.heldout_count, seed=args.seed, excluded_gene_ids=excluded
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(partition, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "training_genes": len(genes) - args.heldout_count,
                "heldout_genes": args.heldout_count,
            }
        )
    )


if __name__ == "__main__":
    main()
