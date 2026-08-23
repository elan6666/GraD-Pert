from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "src" / "gradpert" / "modeling"
FORBIDDEN_IMPORT_ROOTS = {"txpert", "gspp", "gears", "dino", "dinov2"}
FORBIDDEN_NATIVE_NAME_PARTS = {"txpert", "dino", "gears"}


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py")) if root.exists() else []


def test_native_modeling_has_no_upstream_imports_or_classes() -> None:
    failures: list[str] = []
    for path in _python_files(MODELING):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0].lower() for alias in node.names}
                if roots & FORBIDDEN_IMPORT_ROOTS:
                    failures.append(f"{path}: forbidden import {sorted(roots)}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0].lower()
                if root in FORBIDDEN_IMPORT_ROOTS:
                    failures.append(f"{path}: forbidden from-import {node.module}")
            elif isinstance(node, ast.ClassDef):
                lowered = node.name.lower()
                if any(part in lowered for part in FORBIDDEN_NATIVE_NAME_PARTS):
                    failures.append(f"{path}: forbidden native class {node.name}")
    assert failures == []


def test_upstream_checkouts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "TxPert/" in ignore
