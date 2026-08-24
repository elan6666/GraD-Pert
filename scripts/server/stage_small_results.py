#!/usr/bin/env python3
"""Dry-run, stage, or verify the allowlisted small server-result snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradpert.execution.small_sync import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    small_sync_plan,
    stage_small_results,
    verify_staged_small_results,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("run-small-results", "explicit-root"),
        default="run-small-results",
    )
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        payload = verify_staged_small_results(args.destination_root)
    else:
        if args.source_root is None:
            parser.error("planning/staging requires --source-root")
        operation = stage_small_results if args.execute else small_sync_plan
        payload = operation(
            args.source_root,
            args.destination_root,
            selection_scope=args.scope,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
