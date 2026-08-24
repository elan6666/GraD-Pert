"""Build and execute the read-only GraD-Pert benchmark result notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "benchmark_results.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# GraD-Pert sealed benchmark results

## tl;dr

This notebook is a read-only consumer of a hash-pinned `ResultCatalog`. It does
not train models, create splits, resample controls, or recompute metrics. Until
a verified small-result catalog is synchronized from the server, the executed
output reports the result set as unavailable and makes no performance claim."""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

The catalog pins one `run_manifest.json`, server-artifact pointer, and metrics
CSV per evaluated run. The loader rejects changed files, path escape, identity
drift, and runs without exactly one test evaluation.

### Key Assumptions

- Formal training and large PKL/H5AD/checkpoint artifacts stay on the server.
- Only checksum-verified `.json`, `.csv`, `.txt`, and `.md` result receipts are
  present below the trusted local result root.
- Model comparisons are shown only when canonical data, split, and 300-control
  manifest hashes agree within a dataset."""
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Resolve the explicit catalog"),
        nbformat.v4.new_code_cell(
            """from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
from IPython.display import display

from gradpert.artifacts import load_result_catalog

repo_root = Path.cwd().resolve()
catalog_path = Path(
    os.environ.get("GRADPERT_RESULT_CATALOG", repo_root / "results" / "result_catalog.json")
).resolve()
catalog_sha256 = os.environ.get("GRADPERT_RESULT_CATALOG_SHA256", "").strip()
trusted_result_root = Path(
    os.environ.get("GRADPERT_RESULT_ROOT", catalog_path.parent)
).resolve()

catalog = None
catalog_state = "available"
catalog_reason = None
if not catalog_path.exists():
    catalog_state = "unavailable"
    catalog_reason = f"missing catalog: {catalog_path}"
elif not catalog_sha256:
    sidecar = catalog_path.with_suffix(catalog_path.suffix + ".sha256")
    if sidecar.exists():
        catalog_sha256 = sidecar.read_text(encoding="utf-8").strip().split()[0]
    else:
        catalog_state = "unavailable"
        catalog_reason = "catalog exists but no trusted SHA-256 was supplied"

if catalog_state == "available":
    catalog = load_result_catalog(
        catalog_path,
        expected_file_sha256=catalog_sha256,
        trusted_root=trusted_result_root,
    )

display(pd.DataFrame([{
    "catalog_state": catalog_state,
    "catalog_path": str(catalog_path),
    "catalog_sha256": catalog_sha256 or None,
    "reason": catalog_reason,
}]))"""
        ),
        nbformat.v4.new_markdown_cell("### 2. Load bounded small-result tables"),
        nbformat.v4.new_code_cell(
            """summary_frames = []
if catalog is not None:
    for entry in catalog.entries:
        frame = pd.read_csv(entry.metrics_path)
        frame.insert(0, "run_seed", entry.run_manifest.run_seed)
        frame.insert(0, "run_id", entry.run_manifest.run_id)
        frame.insert(0, "dataset_id", entry.run_manifest.dataset_id)
        frame.insert(0, "model_id", entry.run_manifest.model_id)
        summary_frames.append(frame)

metrics_summary = (
    pd.concat(summary_frames, ignore_index=True)
    if summary_frames
    else pd.DataFrame(columns=[
        "model_id", "dataset_id", "run_id", "run_seed", "metric_id",
        "available", "macro_mean", "reason", "finite_condition_count",
        "total_condition_count",
    ])
)
display(metrics_summary.head(30))"""
        ),
        nbformat.v4.new_markdown_cell("## Results\n\n### 3. Verify paired comparison hashes"),
        nbformat.v4.new_code_cell(
            """fairness_rows = []
if catalog is not None:
    by_dataset = {}
    for entry in catalog.entries:
        by_dataset.setdefault(entry.run_manifest.dataset_id, []).append(entry.run_manifest)
    for dataset_id, manifests in sorted(by_dataset.items()):
        data_hashes = {item.canonical_data_sha256 for item in manifests}
        split_hashes = {item.split_content_sha256 for item in manifests}
        control_hashes = {item.control_manifest_sha256 for item in manifests}
        fairness_rows.append({
            "dataset_id": dataset_id,
            "n_runs": len(manifests),
            "canonical_data_hashes": len(data_hashes),
            "split_hashes": len(split_hashes),
            "control_manifest_hashes": len(control_hashes),
            "paired_inputs_match": (
                len(data_hashes) == len(split_hashes) == len(control_hashes) == 1
            ),
        })

fairness_check = pd.DataFrame(
    fairness_rows,
    columns=[
        "dataset_id", "n_runs", "canonical_data_hashes", "split_hashes",
        "control_manifest_hashes", "paired_inputs_match",
    ],
)
display(fairness_check)"""
        ),
        nbformat.v4.new_markdown_cell("### 4. Show headline metrics only for fair groups"),
        nbformat.v4.new_code_cell(
            """fair_datasets = set(
    fairness_check.loc[fairness_check["paired_inputs_match"], "dataset_id"]
) if not fairness_check.empty else set()

fair_metrics = metrics_summary[
    metrics_summary["dataset_id"].isin(fair_datasets)
].copy()
display(fair_metrics.head(50))

if fair_metrics.empty:
    print("No hash-matched evaluated runs are available yet.")
else:
    metric_view = fair_metrics.pivot_table(
        index=["dataset_id", "model_id", "run_seed"],
        columns="metric_id",
        values="macro_mean",
        aggfunc="first",
    ).reset_index()
    display(metric_view.head(50))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Takeaways

- Treat the availability table above as authoritative; missing catalogs or
  unavailable metrics are not converted to zeros.
- Compare model values only for dataset groups where `paired_inputs_match` is
  `True`.
- Follow the server pointer in the catalog for full condition arrays and
  downstream experiments; this notebook intentionally uses only small synced
  summaries."""
        ),
    ]
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(notebook, timeout=120, kernel_name="python3")
    client.execute(cwd=str(REPO_ROOT))
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
