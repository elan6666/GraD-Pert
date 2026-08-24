from __future__ import annotations

from pathlib import Path

import nbformat


def test_benchmark_notebook_is_executed_read_only_catalog_consumer() -> None:
    path = Path("notebooks/benchmark_results.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert [
        heading in markdown
        for heading in (
            "## tl;dr",
            "## Context & Methods",
            "## Data",
            "## Results",
            "## Takeaways",
        )
    ] == [True] * 5
    assert "load_final_result_catalog" in code
    assert "paired_inputs_match" in code
    assert "trainer.fit" not in code
    assert "seal_" not in code
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    assert all(
        cell.get("execution_count") is not None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
