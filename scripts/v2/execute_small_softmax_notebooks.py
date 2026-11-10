"""Execute the three analysis notebooks from verified, small aggregate inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2] / "notebooks/small_softmax"
BOOKS = ("01_training_process", "02_unseen_perturbations", "03_downstream_tasks")
INPUTS = (
    "analysis/summary.json",
    "MAIN_EXPRESSION_VAL_ACCEPTANCE.json",
    "condition-aggregates.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(inputs: Path, output: Path):
    inputs, output = inputs.resolve(strict=True), output.resolve()
    conditions = json.loads((inputs / INPUTS[2]).read_text())
    validation = json.loads((inputs / INPUTS[1]).read_text())
    if (
        conditions["schema"] != "frozen-condition-aggregates-v1"
        or conditions["validation_sha256"] != validation["evaluation_sha256"]
        or conditions["run_id"] != validation["run_id"]
        or len(conditions["rows"]) != validation["validation_conditions"]
    ):
        raise ValueError("notebook inputs differ from the accepted frozen validation")
    hashes = {p: sha(inputs / p) for p in INPUTS}
    source_hashes = {f"{stem}.ipynb": sha(ROOT / f"{stem}.ipynb") for stem in BOOKS} | {
        "analysis_charts.py": sha(ROOT / "analysis_charts.py")
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if (
            previous["input_hashes"] == hashes
            and previous["source_hashes"] == source_hashes
            and all(
                (output / p).is_file() and sha(output / p) == h
                for p, h in previous["files"].items()
            )
        ):
            return previous
    files = {}
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as temp:
        stage = Path(temp)
        kernel_path = Path(install(prefix=str(stage), kernel_name="gradpert-verified"))
        env = {
            **os.environ,
            "JUPYTER_PATH": str(stage / "share/jupyter"),
            "MPLBACKEND": "Agg",
            "GRADPERT_ANALYSIS_INPUT": str(inputs),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }
        for stem in BOOKS:
            src = ROOT / f"{stem}.ipynb"
            rendered = nbformat.read(src, as_version=4)
            nbformat.validate(rendered)
            target = stage / stem
            target.mkdir()
            NotebookClient(
                rendered,
                timeout=180,
                record_timing=False,
                km=KernelManager(
                    kernel_name="gradpert-verified",
                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_path.parent)]),
                ),
                resources={"metadata": {"path": str(ROOT)}},
            ).execute(
                env={**env, "GRADPERT_ANALYSIS_OUTPUT": str(target)},
                cleanup_kc=True,
            )
            rendered.metadata.kernelspec = nbformat.read(src, as_version=4).metadata.kernelspec
            result = stage / f"{stem}.ipynb"
            nbformat.write(rendered, result)
            figures = sorted(target.glob("*.png"))
            inline_count = sum(
                "image/png" in item.get("data", {})
                for cell in rendered.cells
                for item in cell.get("outputs", [])
            )
            if not figures or len(figures) != inline_count:
                raise ValueError(f"{stem}: saved figures differ from notebook output")
            for file in [result, *figures]:
                relative = file.relative_to(stage)
                dest = output / relative
                dest.parent.mkdir(exist_ok=True)
                file.replace(dest)
                files[relative.as_posix()] = sha(dest)
        manifest = {
            "schema": "gradpert-notebook-analysis-v1",
            "run_id": conditions["run_id"],
            "response_diagnostics": conditions["response_diagnostics"] is not None,
            "input_hashes": hashes,
            "source_hashes": source_hashes,
            "files": files,
        }
        receipt = stage / "manifest.json"
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        receipt.replace(output / receipt.name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.inputs, args.output)
