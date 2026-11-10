"""Run training through verified platform archival from the Mac controlling this workspace."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from small_softmax_notebook_assets import notebook_assets

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "small_softmax_workflow_server.py"
BUILDER = HERE / "build_small_softmax_platform_package.py"
NOTEBOOK_RUNNER = HERE / "execute_small_softmax_notebooks.py"
PLATFORM = "http://127.0.0.1:8787"
HOST = "yilangliu@10.24.1.91"
SSH = [
    "ssh",
    "-i",
    "/Users/elan/.ssh/elanquant_yilangliu_ed25519",
    "-S",
    "/tmp/elanquant-agent.sock",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
]
SCOPES = {"all_expression": "all", "seen_expression": "seen", "unseen_expression": "unseen"}
METRICS = {
    "txpert_macro_pearson_delta_all",
    "trishift_pearson_delta_all",
    "systema_pearson_all",
}
SHA = re.compile(r"[0-9a-f]{64}")
ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
GPU = re.compile(r"GPU-[0-9a-f-]{36}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def server(mode, spec, *args):
    command = shlex.join(["python3", "-", mode, json.dumps(spec), *args])
    output = subprocess.check_output([*SSH, HOST, command], input=DRIVER.read_bytes())
    return json.loads(output)


def spec_from_file(path, expected_hash):
    require(digest(path) == expected_hash, "workflow spec hash changed")
    spec = read(path)
    fields = {
        "experiment_id",
        "run_id",
        "training_plan",
        "plan_sha256",
        "expected_updates",
        "training_gpu_uuids",
        "evaluation_gpu",
        "evaluation_launcher_sha256",
        "verifier_sha256",
    }
    require(
        set(spec) in (fields, fields | {"response_diagnostics"})
        and type(spec.get("response_diagnostics", False)) is bool,
        "unexpected workflow spec fields",
    )
    for key in ("plan_sha256", "evaluation_launcher_sha256", "verifier_sha256"):
        require(SHA.fullmatch(spec[key]), f"invalid {key}")
    require(ID.fullmatch(spec["experiment_id"]), "invalid experiment_id")
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", spec["run_id"]), "invalid run_id")
    require(
        type(spec["expected_updates"]) is int
        and spec["expected_updates"] > 0
        and spec["expected_updates"] % 2 == 0,
        "expected updates must cover two epochs",
    )
    uuids = spec["training_gpu_uuids"]
    require(
        type(uuids) is list
        and 1 <= len(uuids) <= 2
        and len(set(uuids)) == len(uuids)
        and all(GPU.fullmatch(uuid) for uuid in uuids),
        "invalid training GPU identities",
    )
    require(spec["evaluation_gpu"] in ("0", "1"), "invalid evaluation GPU")
    require(PurePosixPath(spec["training_plan"]).name == "PLAN.json", "expected a sealed PLAN.json")
    return spec


def fetch_ready(spec, ready, artifacts):
    require(
        ready["run_id"] == spec["run_id"] and ready["status"] == "completed",
        "server readiness receipt is for a different run",
    )
    root = PurePosixPath(spec["training_plan"]).parent
    require(root.is_relative_to("/data/yilangliu/GraD-Pert"), "invalid stage root")
    for path, checksum in ready["files"].items():
        src = PurePosixPath(path)
        require(
            src.is_relative_to(root)
            and ".." not in src.parts
            and re.fullmatch(r"[a-zA-Z0-9_./-]+", path)
            and SHA.fullmatch(checksum),
            "invalid artifact transfer path or hash",
        )
        target = artifacts / src.name
        if src.parent == root / "analysis-20c1bdf":
            target = artifacts / "analysis" / src.name
        else:
            require(
                path
                in {
                    str(root / "PLAN.json"),
                    str(root / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json"),
                    str(root / "runs" / spec["run_id"] / "TRAIN_COMPLETE.json"),
                },
                "only verified training and validation receipts may be transferred",
            )
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".incoming")
            argv = ["rsync", "-ti", "-e", shlex.join(SSH), f"{HOST}:{path}", str(temporary)]
            subprocess.run([argv[0], "--dry-run", *argv[1:]], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
            require(digest(temporary) == checksum, f"received artifact hash differs: {src.name}")
            temporary.replace(target)
        require(digest(target) == checksum, f"cached artifact hash differs: {src.name}")

    aggregate = ready["condition_aggregates"]
    validation = read(artifacts / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json")
    require(
        aggregate["schema"] == "frozen-condition-aggregates-v1"
        and aggregate["run_id"] == spec["run_id"]
        and aggregate["validation_sha256"] == validation["evaluation_sha256"]
        and (aggregate["response_diagnostics"] is not None)
        == spec.get("response_diagnostics", False)
        and len(aggregate["rows"]) == 445,
        "validation aggregate inputs differ from the accepted receipt",
    )
    payload = json.dumps(aggregate, allow_nan=False, sort_keys=True).encode()
    require(len(payload) <= 3 * 1024**2, "condition aggregates are unexpectedly large")
    (artifacts / "condition-aggregates.json").write_bytes(payload)


def read_experiment(spec, base):
    route = f"/api/export?project=grad-pert&id={spec['experiment_id']}"
    with urlopen(base + route, timeout=20) as response:
        experiments = json.load(response)["experiments"]
    require(
        len(experiments) == 1 and experiments[0]["id"] == spec["experiment_id"],
        "platform experiment does not resolve to the sealed ID",
    )
    return experiments[0]


def completed_run(experiment, spec):
    runs = experiment["runs"]
    require(
        not runs or (len(runs) == 1 and runs[0]["id"] == spec["run_id"]),
        "platform experiment contains a different run",
    )
    if runs and runs[0]["statuses"]["analysis"] == "completed":
        return runs[0]
    return None


def verify_platform(run, spec, artifacts, base):
    val_path = artifacts / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json"
    val = read(val_path)
    analysis = artifacts / "analysis/receipt.json"
    expected = {
        (row["metric"], SCOPES[row["scope"]]): (
            row["candidate"]["macro_mean"],
            row["candidate"]["valid"],
        )
        for row in val["comparison"]
    }
    require(
        set(expected) == {(metric, scope) for metric in METRICS for scope in SCOPES.values()},
        "validation receipt lacks nine Pearson metrics",
    )
    actual = {(m["id"], m["gene_group"]): (m["value"], m["valid_count"]) for m in run["metrics"]}
    require(len(run["metrics"]) == 9 and actual == expected, "platform Pearson readback differs")
    require(
        all(
            m["split"] == "val"
            and m["checkpoint_epoch"] == 2
            and m["total_count"] == 445
            and m["source_sha256"] == digest(val_path)
            for m in run["metrics"]
        ),
        "platform metric protocol differs",
    )
    require(
        run["statuses"]
        == {"training": "completed", "evaluation": "completed", "analysis": "completed"}
        and run["source_commit"] == val["training_commit"]
        and run["config_sha256"] == val["config_sha256"]
        and run["checkpoint_sha256"] == val["checkpoint_sha256"]
        and [row["step"] for row in run["history"]]
        == [spec["expected_updates"] // 2, spec["expected_updates"]],
        "platform run identity differs",
    )
    required = {
        "training-terminal": val["training_sha256"],
        "validation-terminal": digest(val_path),
        "cpu-terminal": digest(analysis),
    }
    observed = {a["id"]: a["sha256"] for a in run["artifacts"] if a["id"] in required}
    require(observed == required, "platform evidence hashes differ")
    for artifact in run["artifacts"]:
        if artifact["id"] in required:
            require(
                artifact["file"] == required[artifact["id"]] + ".json",
                "platform receipt attachment name differs",
            )
            with urlopen(base + "/api/asset/" + artifact["file"], timeout=20) as response:
                require(
                    hashlib.sha256(response.read(10 * 1024**2 + 1)).hexdigest()
                    == artifact["sha256"],
                    "platform receipt attachment content differs",
                )
    require(
        len(run["analyses"]) == 1
        and run["analyses"][0]["status"] == "completed"
        and run["analyses"][0]["checkpoint_sha256"] == val["checkpoint_sha256"]
        and run["analyses"][0]["source_commit"] == read(analysis)["analysis_identity"]["commit"],
        "platform training analysis differs",
    )


def verify_notebooks(run, expected, base):
    observed = {
        a["id"]: a for a in run["analyses"][0]["artifacts"] if a["id"].startswith("notebook-")
    }
    require(observed.keys() == expected.keys(), "platform notebook asset list differs")
    for id_, (_, checksum, _, _) in expected.items():
        asset = observed[id_]
        require(
            asset["sha256"] == checksum and asset["file"].startswith(checksum + "."),
            "platform notebook asset identity differs",
        )
        with urlopen(base + "/api/asset/" + asset["file"], timeout=20) as response:
            require(
                hashlib.sha256(response.read(10 * 1024**2 + 1)).hexdigest() == checksum,
                "platform notebook asset content differs",
            )


def write_report(artifacts, spec):
    val = read(artifacts / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json")
    lines = [
        "# Jurkat 冻结验证与训练分析",
        "",
        f"运行 {spec['run_id']} 完成两轮。总步数为 {spec['expected_updates']}。",
        "使用最后的 checkpoint 在 445 个冻结条件上进行表达验证。",
        "",
        "| 范围 | 指标 | 得分 | 有效条件 |",
        "|---|---|---:|---:|",
    ]
    for row in val["comparison"]:
        item = row["candidate"]
        mean = item["macro_mean"]
        score = f"{mean:.5f}" if mean is not None else "NA"
        lines.append(f"| {row['scope']} | {row['metric']} | {score} | {item['valid']} |")
    lines.extend(["", f"验证 MSE 为 {val['candidate_mse']:.8f}。"])
    (artifacts / "REPORT_CN.md").write_text("\n".join(lines) + "\n")


def archive(spec, platform_root, output, artifacts, base):
    locks = platform_root / ".local/platform-workflow-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{spec['experiment_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = completed_run(read_experiment(spec, base), spec)
        expected = notebook_assets(artifacts, spec["run_id"])
        current = {}
        if run is not None:
            verify_platform(run, spec, artifacts, base)
            current = {
                a["id"]: a["sha256"]
                for a in run["analyses"][0]["artifacts"]
                if a["id"].startswith("notebook-")
            }
        if run is None or current != {k: row[1] for k, row in expected.items()}:
            write_report(artifacts, spec)
            package_root = output / "package"
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    spec["experiment_id"],
                    str(artifacts),
                    str(package_root),
                    str(platform_root),
                    str(spec["expected_updates"]),
                    base,
                ],
                check=True,
            )
            importer = [
                "node",
                str(platform_root / "scripts/import-package.mjs"),
                str(package_root / "package.json"),
                f"--base={base}",
            ]
            subprocess.run([*importer, "--check-only"], check=True)
            subprocess.run([*importer, "--replace"], check=True)
            run = completed_run(read_experiment(spec, base), spec)
        require(run is not None, "platform did not return a completed run")
        verify_platform(run, spec, artifacts, base)
        verify_notebooks(run, expected, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--platform-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-url", default=PLATFORM)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--collect-only", action="store_true", help="only accept existing server results"
    )
    group.add_argument(
        "--retry-server", action="store_true", help="reattach or continue a failed server workflow"
    )
    parser.add_argument("--notebook-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    require(not sys.flags.optimize, "verification requires Python assertions enabled")
    require(
        re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", args.platform_url),
        "platform endpoint must be loopback HTTP",
    )
    spec = spec_from_file(args.spec, args.spec_sha256)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = output / "SPEC_SHA256"
        if binding.exists():
            require(
                binding.read_text().strip() == args.spec_sha256, "output belongs to another run"
            )
        else:
            binding.write_text(args.spec_sha256 + "\n")
        stage = "notebook_runtime"
        try:
            notebook_python = args.notebook_python.expanduser().absolute()
            subprocess.run(
                [str(notebook_python), "-c", "import ipykernel, matplotlib, nbclient, nbformat"],
                check=True,
            )
            stage = "server_workflow"
            if args.collect_only:
                ready = server("verify", spec)
            else:
                driver = base64.b64encode(DRIVER.read_bytes()).decode("ascii")
                owner = server("start", spec, driver, "retry" if args.retry_server else "attach")
                print(json.dumps({"status": "server_dispatched", "owner": owner}), flush=True)
                ready = server("wait", spec, json.dumps(owner))
            stage = "artifact_transfer"
            artifacts = output / "artifacts"
            fetch_ready(spec, ready, artifacts)
            stage = "notebook_analysis"
            subprocess.run(
                [
                    str(notebook_python),
                    str(NOTEBOOK_RUNNER),
                    "--inputs",
                    str(artifacts),
                    "--output",
                    str(artifacts / "notebooks"),
                ],
                check=True,
            )
            stage = "platform_archive"
            archive(
                spec, args.platform_root.resolve(strict=True), output, artifacts, args.platform_url
            )
            receipt = {
                "status": "completed",
                "run_id": spec["run_id"],
                "experiment_id": spec["experiment_id"],
                "spec_sha256": args.spec_sha256,
                "validation_sha256": digest(artifacts / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json"),
                "analysis_sha256": digest(artifacts / "analysis/receipt.json"),
                "notebook_manifest_sha256": digest(artifacts / "notebooks/manifest.json"),
                "response_diagnostics_available": spec.get("response_diagnostics", False),
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
            filename = "READY.json"
        except Exception as error:
            receipt = {
                "status": "failed",
                "run_id": spec["run_id"],
                "stage": stage,
                "error": str(error),
            }
            filename = "FAILED.json"
        for name in (filename, "STATUS.json"):
            pending = output / (name + ".tmp")
            pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            os.replace(pending, output / name)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return 0 if filename == "READY.json" else 1


if __name__ == "__main__":
    raise SystemExit(main())
