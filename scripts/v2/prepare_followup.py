"""Generate a later ablation group from a reverified validation-selected parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_group import GROUPS, generate
from select_parent import verify_selection

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file


def validate_dependency(manifest: Path, parent: Path) -> dict | None:
    payload = json.loads(manifest.read_text())
    dependency = payload.get("validation_parent")
    if dependency is None:
        if payload["group"] not in ("B0", "H1"):
            raise ValueError("later groups require a validation-selected parent receipt")
        return None
    selection = Path(dependency["selection"])
    upstream = Path(dependency["manifest"])
    if (
        sha256_file(selection) != dependency["selection_sha256"]
        or sha256_file(upstream) != dependency["manifest_sha256"]
    ):
        raise ValueError("upstream selection or group manifest changed")
    verified = verify_selection(selection, upstream, Path(dependency["parent"]))
    upstream_group = json.loads(upstream.read_text())["group"]
    required = "H1" if payload["group"] == "H2" else "H2" if payload["group"] == "H3" else "H3"
    if upstream_group != required:
        raise ValueError("selected parent comes from the wrong preceding hyperparameter group")
    if sha256_file(parent) != verified["winner"]["sha256"]:
        raise ValueError("downstream parent differs from the validation-selected configuration")
    return verified


def prepare(
    selection: Path,
    manifest: Path,
    parent: Path,
    group: str,
    output: Path,
    *,
    batch_probes: list[dict] | None = None,
) -> dict:
    required = "H1" if group == "H2" else "H2" if group == "H3" else "H3"
    if group in ("B0", "H1") or json.loads(manifest.read_text())["group"] != required:
        raise ValueError("follow-up group must follow the hyperparameter dependency order")
    verified = verify_selection(selection, manifest, parent)
    winner = Path(verified["winner"]["config"])
    result = generate(winner, group, output, batch_probes=batch_probes)
    result["validation_parent"] = {
        "selection": str(selection.resolve()),
        "selection_sha256": verified["selection_sha256"],
        "manifest": str(manifest.resolve()),
        "manifest_sha256": verified["manifest_sha256"],
        "parent": str(parent.resolve()),
    }
    atomic_json(output / "manifest.json", result)
    validate_dependency(output / "manifest.json", winner)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-probes", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.selection,
        args.manifest,
        args.parent,
        args.group,
        args.output,
        batch_probes=json.loads(args.batch_probes.read_text()) if args.batch_probes else None,
    )
    print(json.dumps({"group": result["group"], "parent_sha256": result["parent_sha256"]}))


if __name__ == "__main__":
    main()
