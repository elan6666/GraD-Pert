#!/usr/bin/env python3
"""Select the native batch size from two hash-pinned capacity receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradpert.training.performance import compare_batch_capacity_receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch64", type=Path, required=True)
    parser.add_argument("--batch256", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_batch_capacity_receipts(
        batch64_path=args.batch64,
        batch256_path=args.batch256,
        source_commit=args.source_commit,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
