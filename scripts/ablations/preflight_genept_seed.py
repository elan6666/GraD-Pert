#!/usr/bin/env python3
"""Seal the Seed-GO-ProteinPathway prior against one unchanged vNext graph."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradpert.pilots import preflight_genept_seed_vnext  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--runtime-graph-root", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = preflight_genept_seed_vnext(
        parent_root=args.parent_root,
        genept_artifact_path=args.artifact,
        expected_genept_sha256=args.artifact_sha256,
        runtime_graph_root=args.runtime_graph_root,
        availability_receipt_path=args.receipt,
    )
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
