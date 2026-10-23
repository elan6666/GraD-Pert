"""Generate and verify self-contained research configs without runtime inheritance."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file


def verify(source: Path, path: Path, metadata: dict, configs: dict, changes: dict) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata:
        raise ValueError("frozen research protocol changed")
    rows = []
    for name, config in configs.items():
        target = path.parent / name / "gradpert_v2/nadig_jurkat.yaml"
        if target.is_symlink() or not target.resolve().is_relative_to(source):
            raise ValueError("configuration must remain source contained")
        if yaml.safe_load(target.read_text()) != config:
            raise ValueError("frozen research configuration changed")
        load_experiment_config(target)
        rows.append(
            dict(
                name=name,
                config=str(target.relative_to(source)),
                sha256=sha256_file(target),
                changes=changes,
            )
        )
    if value["rows"] != rows:
        raise ValueError("research scope, order or configuration identity changed")
    return value


def generate(source: Path, output: Path, metadata: dict, configs: dict, changes: dict) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    for name, config in configs.items():
        target = output / name / "gradpert_v2/nadig_jurkat.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        rows.append(
            dict(
                name=name,
                config=str(target.relative_to(source)),
                sha256=sha256_file(target),
                changes=changes,
            )
        )
    path = output / "manifest.json"
    path.write_text(json.dumps({**metadata, "rows": rows}, indent=2) + "\n")
    return verify(source, path, metadata, configs, changes)
