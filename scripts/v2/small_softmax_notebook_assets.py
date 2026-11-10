"""Shared binding for notebook outputs in the experiment platform."""

from __future__ import annotations

import hashlib
import json
import re

BOOKS = {
    "01_training_process": "训练过程",
    "02_unseen_perturbations": "未见扰动",
    "03_downstream_tasks": "下游诊断",
}
INPUTS = (
    "analysis/summary.json",
    "MAIN_EXPRESSION_VAL_ACCEPTANCE.json",
    "condition-aggregates.json",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_assets(artifacts, run_id):
    root = artifacts / "notebooks"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != "gradpert-notebook-analysis-v1"
        or manifest["run_id"] != run_id
        or manifest["input_hashes"] != {p: digest(artifacts / p) for p in INPUTS}
    ):
        raise ValueError("notebook manifest differs from archived run inputs")
    assets = {
        "notebook-manifest": (manifest_path, digest(manifest_path), "Notebook manifest", "report")
    }
    per_book = {stem: 0 for stem in BOOKS}
    for relative, checksum in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != checksum
        ):
            raise ValueError(f"notebook output checksum differs: {relative}")
        if relative in {f"{stem}.ipynb" for stem in BOOKS}:
            stem = path.stem
            file = path.with_suffix(".json")
            if not file.is_file() or digest(file) != checksum:
                file.write_bytes(path.read_bytes())
            assets[f"notebook-{stem[:2]}"] = (
                file,
                checksum,
                f"{BOOKS[stem]} · {path.name}",
                "report",
            )
        elif re.fullmatch(r"[a-z0-9_]+/[a-z0-9-]+\.png", relative) and path.parent.name in BOOKS:
            stem = path.parent.name
            per_book[stem] += 1
            assets[f"notebook-{stem[:2]}-{path.stem}"] = (
                path,
                checksum,
                f"{BOOKS[stem]} · {path.stem}",
                "image",
            )
        else:
            raise ValueError(f"unexpected notebook output: {relative}")
    if any(not count or f"notebook-{stem[:2]}" not in assets for stem, count in per_book.items()):
        raise ValueError("missing executed notebook or figures")
    return assets
