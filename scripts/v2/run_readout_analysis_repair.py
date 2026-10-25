"""Recover only R11 diagnostics from the original completed frozen readout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
from pathlib import Path

from cap40_posthoc import read, sha, write
from generate_representation_round11 import SCHEMA

SERVER_ROOT = Path("/data/yilangliu")
TRAINING_SHA = "a0f75472345d82386bbdf88056b63a4fd3bdbbb4"
HELPER_SHA = "e579bd96db4de773a5bc4493ad30bca11a3e4982903e55f8432b0921b5e0c763"
SCOPE_SOURCE_SHA = "04c2ed8ff7114810bf343e6e70a89c62540fa51d841ffd1bfb554aa7f2bab9f2"
ALLOWED_CHANGES = {
    "scripts/v2/cap40_posthoc.py": "M",
    "scripts/v2/representation_panel.py": "M",
    "scripts/v2/diagnostic_view_options.py": "A",
    "scripts/v2/run_fgpo_medium.py": "M",
    "scripts/v2/run_representation_postfit.py": "M",
    "scripts/v2/run_readout_analysis_repair.py": "A",
    "tests/v2/test_frozen_readout_medium.py": "A",
    "tests/v2/test_readout_analysis_repair.py": "A",
    "tests/v2/test_au_posthoc_adapter.py": "M",
}


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True, timeout=30).strip()


def validate_output(queue, output):
    output = Path(output).resolve()
    protected = [
        Path(queue["output"]).resolve(),
        Path(queue["source"]).resolve(),
        Path(__file__).resolve().parents[2],
    ] + [Path(row["plan"]["run_root"]).resolve() for row in queue["rows"]]
    if (
        not output.is_relative_to(SERVER_ROOT)
        or output == SERVER_ROOT
        or any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected)
    ):
        raise ValueError("repair output must be an independent server root")
    return output


def normalized_sampling(source, *, current, panel=False):
    """Allow only the reviewed assembly switch and frozen-status metadata."""
    tree = ast.parse(source)
    target = "assemble_control_only" if panel else "assemble"
    helper = ast.parse("diagnostic_view_options(runtime.options, deep=deep)", mode="eval").body
    scope_call = ast.parse("frozen_readout_scope(obj)", mode="eval").body

    class Normalize(ast.NodeTransformer):
        def visit_Import(self, node):
            if not panel and ast.dump(node) == ast.dump(ast.parse("import dataclasses").body[0]):
                return None
            return node

        def visit_ImportFrom(self, node):
            if node.module == "diagnostic_view_options":
                expected = ast.parse(
                    "from diagnostic_view_options import diagnostic_view_options"
                ).body[0]
                if not current or ast.dump(node) != ast.dump(expected):
                    raise ValueError("unexpected diagnostic helper import")
                return None
            return node

        def visit_FunctionDef(self, node):
            if node.name == "frozen_readout_scope":
                if (
                    not current
                    or hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                    != SCOPE_SOURCE_SHA
                ):
                    raise ValueError("frozen metadata function differs from reviewed repair")
                return None
            if node.name == target:
                assignments = [
                    n
                    for n in node.body
                    if isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "options"
                ]
                if len(assignments) != (0 if current else 1):
                    raise ValueError("diagnostic options assignment differs")
                node.body = [n for n in node.body if n not in assignments]
                calls = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "assemble_batch"
                ]
                if len(calls) != 1 or len(calls[0].args) != 4:
                    raise ValueError("native diagnostic assembly changed")
                expected = helper if current else ast.Name(id="options", ctx=ast.Load())
                if ast.dump(calls[0].args[2]) != ast.dump(expected):
                    raise ValueError("diagnostic sampling switch changed")
                calls[0].args[2] = ast.Constant(value="reviewed_sampling_options")
            return self.generic_visit(node)

        def visit_Dict(self, node):
            pairs = []
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None and ast.dump(value) == ast.dump(scope_call):
                    if not current:
                        raise ValueError("unexpected original frozen metadata")
                    continue
                pairs.append((key, value))
            node.keys = [key for key, _ in pairs]
            node.values = [value for _, value in pairs]
            return self.generic_visit(node)

    return ast.dump(Normalize().visit(tree))


def validate_repair_source(queue, source):
    if (
        queue.get("family") != SCHEMA
        or queue.get("source_commit") != TRAINING_SHA
        or [row.get("name") for row in queue.get("rows", [])] != ["R11-READOUT"]
    ):
        raise ValueError("analysis repair requires the exact original R11 training family/source")
    source = Path(source).resolve()
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("analysis repair source must be clean")
    commit = git(source, "rev-parse", "HEAD")
    if commit == TRAINING_SHA:
        raise ValueError("analysis source must be separately published")
    entries = git(source, "diff", "--name-status", "--no-renames", TRAINING_SHA, "HEAD")
    changed = dict(
        (path, status) for status, path in (line.split("\t") for line in entries.splitlines())
    )
    if changed != ALLOWED_CHANGES:
        raise ValueError("R11 repair changed undeclared source/config/numerical files")
    if sha(source / "scripts/v2/diagnostic_view_options.py") != HELPER_SHA:
        raise ValueError("sampling helper differs from the reviewed validated clone")
    for path in ("scripts/v2/cap40_posthoc.py", "scripts/v2/representation_panel.py"):
        original = git(source, "show", TRAINING_SHA + ":" + path)
        current = (source / path).read_text()
        panel = path.endswith("representation_panel.py")
        if normalized_sampling(original, current=False, panel=panel) != normalized_sampling(
            current, current=True, panel=panel
        ):
            raise ValueError("original diagnostic numerical operations changed")
    return {"commit": commit, "dirty": False, "diff": changed}


def validate_repair_publication(queue, source, publication, publication_sha256):
    from gradpert.execution.identity import inspect_source_identity

    if publication is None or not publication_sha256:
        raise ValueError("R11 repair requires a separate published analysis receipt")
    return inspect_source_identity(
        source,
        formal=True,
        expected_repository=queue["rows"][0]["plan"]["resolved_config"]["source_code"][
            "repository"
        ],
        publication_receipt=publication,
        expected_publication_receipt_sha256=publication_sha256,
    )


def repair_panel_command(queue, row, source, output):
    from run_representation_postfit import panel_command

    command = panel_command(queue, row)
    command[1] = str(Path(source) / "scripts/v2/run_representation_panel.py")
    command[command.index("--output") + 1] = str(output)
    return command


def preserved_files(queue):
    """Hash the old queue and run trees; no old artifact is written by repair."""
    roots = [Path(queue["output"]), *(Path(r["plan"]["run_root"]) for r in queue["rows"])]
    return {
        str(p): {"sha256": sha(p), "bytes": p.stat().st_size}
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def validate_controller(queue):
    from shared_gpu import admit, snapshot

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("repair controller must be CPU-only")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("repair allocator/determinism contract missing")
    observed = snapshot(os.getpid())
    if any(p["identity"].get("uid") == os.getuid() for p in observed["processes"]):
        raise ValueError("another own GPU process is active; preserve it and stop repair")
    admit(queue["shared_gpu_policy"], observed)
    return observed


def execute(queue_path, output, *, analysis_sha, publication, publication_sha256):
    from run_fgpo_medium import execute as medium_execute
    from run_fgpo_medium import validate_medium_complete
    from run_group import lock
    from run_representation_postfit import context, validate_panel, validate_validation

    import gradpert

    source = Path(__file__).resolve().parents[2]
    queue_path = Path(queue_path).resolve()
    queue = read(queue_path)
    if queue_path != (Path(queue["output"]) / "queue.json").resolve():
        raise ValueError("repair must consume the exact original queue path")
    output = validate_output(queue, output)
    source_proof = validate_repair_source(queue, source)
    if not Path(gradpert.__file__).resolve().is_relative_to(Path(queue["source"]) / "src"):
        raise ValueError("native imports must use the unchanged original training checkout")
    identity = validate_repair_publication(queue, source, publication, publication_sha256)
    if (
        identity.commit != analysis_sha
        or identity.commit != source_proof["commit"]
        or identity.dirty
    ):
        raise ValueError("analysis SHA/publication differs from the clean repair checkout")
    row = queue["rows"][0]
    training = context(queue, row)
    validation = validate_validation(queue, row)
    original = preserved_files(queue)
    checkpoint = training["last"]
    admission = validate_controller(queue)
    if output.exists():
        raise ValueError("fresh analysis repair output required")
    with lock(SERVER_ROOT / "GraD-Pert/runtime/v2-r11-analysis-repair.lock"):
        output.mkdir(parents=True)
        inputs = {
            "queue": str(queue_path),
            "queue_sha256": sha(queue_path),
            "training_source_commit": queue["source_commit"],
            "analysis_source": identity.payload(),
            "source_diff": source_proof["diff"],
            "analysis_publication": str(Path(publication).resolve()),
            "analysis_publication_sha256": publication_sha256,
            "training": training,
            "validation": validation,
            "checkpoint": checkpoint,
            "frozen_proof_sha256": sha(
                Path(row["plan"]["run_root"]) / "fit/frozen_readout_proof.json"
            ),
            "original_files": original,
            "admission": admission,
        }
        write(output / "INPUTS.json", inputs)
        try:
            medium_execute(
                queue_path,
                "R11-READOUT",
                repair_output=output / "medium",
                repair_publication=publication,
                repair_publication_sha256=publication_sha256,
            )
            medium = validate_medium_complete(queue, row, root=output / "medium")
            validate_controller(queue)
            command = repair_panel_command(queue, row, source, output / "panel")
            env = dict(
                os.environ,
                PYTHONPATH=str(Path(queue["source"]) / "src"),
                CUDA_VISIBLE_DEVICES="0",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            with (output / "panel.log").open("x") as log:
                subprocess.run(
                    command,
                    cwd=queue["source"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            panel = validate_panel(queue, row, root=output / "panel")
            expected_analysis = {"commit": analysis_sha, "dirty": False}
            if (
                read(output / "medium/MEDIUM_COMPLETE.json").get("report_controller_commit")
                != analysis_sha
                or read(output / "medium/analysis/summary.json").get("analysis")
                != expected_analysis
                or read(output / "panel/COMPLETE.json").get("analysis_source") != expected_analysis
            ):
                raise ValueError(
                    "repaired artifacts are not bound to the published analysis source"
                )
            if preserved_files(queue) != original or validate_validation(queue, row) != validation:
                raise ValueError("original queue/training/validation artifacts changed")
            if validate_repair_source(queue, source) != source_proof:
                raise ValueError("published analysis source changed during repair")
            if (
                validate_repair_publication(
                    queue, source, publication, publication_sha256
                ).payload()
                != identity.payload()
            ):
                raise ValueError("analysis publication changed during repair")
            pkl_roots = [output, Path(queue["output"]), Path(row["plan"]["run_root"])]
            if any(list(root.rglob("*.pkl")) for root in pkl_roots):
                raise ValueError("successful repair and original run roots must contain zero PKL")
            write(
                output / "ANALYSIS_REPAIR_COMPLETE.json",
                {
                    "status": "completed",
                    "scope": "R11-analysis-only-repair-1",
                    "updates": 0,
                    "retrained": False,
                    "validation_rerun": False,
                    "zero_pkl": True,
                    "zero_pkl_roots": [str(root) for root in pkl_roots],
                    "original_artifacts_unchanged": True,
                    "inputs": inputs,
                    "medium": medium,
                    "panel": panel,
                    "outputs": {
                        str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in sorted(output.rglob("*"))
                        if p.is_file()
                    },
                },
            )
        except BaseException as error:
            write(
                output / "FAILURE.json",
                {
                    "status": "failed",
                    "error": repr(error),
                    "queue_sha256": inputs["queue_sha256"],
                    "analysis_sha": analysis_sha,
                    "updates": 0,
                },
            )
            raise
    return output / "ANALYSIS_REPAIR_COMPLETE.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "output", "publication"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("analysis-sha", "publication-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    execute(
        args.queue,
        args.output,
        analysis_sha=args.analysis_sha,
        publication=args.publication,
        publication_sha256=args.publication_sha256,
    )


if __name__ == "__main__":
    main()
