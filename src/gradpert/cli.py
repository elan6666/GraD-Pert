"""Command-line entry point.

The first build wave exposes only non-mutating shell commands. Product commands
are registered by their owning plans rather than implemented as placeholders
that claim work succeeded.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence

from gradpert._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gradpert", description="GraD-Pert research CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Report the local runtime without changing it")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _doctor(as_json: bool) -> int:
    payload = {
        "gradpert_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "formal_compute_allowed": False,
        "note": "Local doctor is read-only; formal compute is server-only.",
    }
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)
    parser.print_help(sys.stdout)
    return 0
