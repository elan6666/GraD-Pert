#!/usr/bin/env python3
"""Build the isolated Blackwell-compatible TxPert runtime from its frozen lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.txpert.runtime import load_runtime_contract
from gradpert.hashing import sha256_file


def _distribution_name(requirement: str) -> str:
    name = re.split(r"[<>=!~\[ ;@]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def filtered_requirements(exported: str, *, replaced: set[str]) -> tuple[str, ...]:
    """Remove only the CUDA packages explicitly replaced by the runtime contract."""

    retained: list[str] = []
    for line in exported.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _distribution_name(stripped) in replaced:
            continue
        retained.append(stripped)
    return tuple(retained)


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()[-4000:]
        raise RuntimeError(
            f"command failed ({error.returncode}): {command[0]}\n{details}"
        ) from error
    return result.stdout


def build_environment(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.resolve(strict=True)
    checkout_root = args.official_checkout.resolve(strict=True)
    source_python = args.source_python.resolve(strict=True)
    uv = args.uv.resolve(strict=True)
    destination = args.destination.resolve()
    receipt_root = args.receipt_root.resolve()
    contract_path, contract = load_runtime_contract(
        repository_root=repository_root,
        checkout_root=checkout_root,
    )
    official = contract["official_checkout"]
    observed_commit = _run(["git", "-C", str(checkout_root), "rev-parse", "HEAD"]).strip()
    checkout_status = _run(["git", "-C", str(checkout_root), "status", "--porcelain"]).strip()
    if observed_commit != official["commit"] or checkout_status:
        raise ValueError("TxPert environment build requires the exact clean official checkout")
    override = contract["hardware_compatibility_override"]
    plan = {
        "schema_version": "txpert-environment-build-plan-v1",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "official_checkout": str(checkout_root),
        "official_commit": observed_commit,
        "source_python": str(source_python),
        "destination": str(destination),
        "receipt_root": str(receipt_root),
        "execute": bool(args.execute),
    }
    if not args.execute:
        return plan
    if destination.exists() or receipt_root.exists():
        raise FileExistsError("destination and receipt root must both be new")

    export_command = [
        str(uv),
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--no-annotate",
        "--directory",
        str(checkout_root),
    ]
    exported = _run(export_command)
    replaced = {str(value) for value in override["replaced_distributions"]}
    requirements = [
        *filtered_requirements(exported, replaced=replaced),
        *(str(value) for value in override["install_requirements"]),
    ]
    with tempfile.TemporaryDirectory(prefix="gradpert-txpert-env-") as temporary:
        requirements_path = Path(temporary) / "requirements.txt"
        requirements_text = "\n".join(requirements) + "\n"
        requirements_path.write_text(requirements_text, encoding="utf-8")
        _run([str(uv), "venv", str(destination), "--python", str(source_python)])
        _run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(destination / "bin" / "python"),
                "--requirements",
                str(requirements_path),
                "--default-index",
                "https://pypi.org/simple",
                "--index",
                str(override["pytorch_index"]),
                "--find-links",
                str(override["pyg_find_links"]),
                "--index-strategy",
                "unsafe-best-match",
            ]
        )

    python = destination / "bin" / "python"
    freeze = _run([str(uv), "pip", "freeze", "--python", str(python)])
    _run([str(uv), "pip", "check", "--python", str(python)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root / "src"), str(repository_root))
    )
    runtime_json = _run(
        [
            str(python),
            "-m",
            "benchmarks.txpert.runtime",
            "--repository-root",
            str(repository_root),
            "--official-checkout",
            str(checkout_root),
            "--device",
            args.device,
        ],
        environment=environment,
    )
    runtime_receipt = json.loads(runtime_json)
    receipt_root.mkdir(parents=True)
    resolved_path = receipt_root / "requirements.resolved.txt"
    resolved_path.write_text(freeze, encoding="utf-8")
    receipt = {
        **plan,
        "schema_version": "txpert-environment-build-receipt-v1",
        "requirements_count": len([line for line in freeze.splitlines() if line.strip()]),
        "requirements_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "requirements_path": str(resolved_path),
        "runtime": runtime_receipt,
    }
    (receipt_root / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--source-python", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_environment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
