#!/usr/bin/env python3
"""Plan or seal the exact fair 45-run notebook-facing ResultCatalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradpert.artifacts import (
    plan_final_result_catalog,
    seal_final_result_catalog_from_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_final_result_catalog(args.source_spec, trusted_root=args.trusted_root)
    payload = plan.payload()
    payload["execute"] = args.execute
    payload["output"] = str(args.output.resolve())
    if args.execute:
        file_sha256, audit = seal_final_result_catalog_from_spec(
            args.output,
            source_spec_path=args.source_spec,
            trusted_root=args.trusted_root,
        )
        payload["catalog_sha256"] = file_sha256
        payload["audit"] = audit.payload()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
