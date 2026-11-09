"""Guard two sealed expression-only evaluations on the selected physical GPU."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
SERVER_ROOT = Path("/data/yilangliu")
ARMS = ("A_norm_mla", "B_norm_standard")
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "scripts/v2")]

from gradpert.data._io import atomic_json, read_json  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402
from scripts.v2.run_dataset_transfer_300 import native_run  # noqa: E402
from scripts.v2.run_group import lock  # noqa: E402
from scripts.v2.shared_gpu import admit_selected_device, process_info, snapshot  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sealed(path, digest):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(SERVER_ROOT), "sealed evaluation input must stay on server")
    require(sha256_file(path) == digest, f"evaluation input changed: {path.name}")
    return path


def verify(spec_path, expected):
    spec_path = sealed(spec_path, expected)
    root = spec_path.parent
    spec = read_json(spec_path)
    require(
        tuple(spec["order"]) == ARMS
        and spec["split"] == "val"
        and spec["metric_scope"] == "expression_only"
        and all(spec[key] is True for key in ("no_DEG", "no_test", "no_training")),
        "only the two sealed expression-only validation arms are supported",
    )
    require(
        not any(
            (root / name).exists()
            for name in ("OWNER.json", "CONTROLLER_COMPLETE.json", "CONTROLLER_FAILURE.json")
        ),
        "evaluation controller output must be fresh",
    )
    pub = sealed(spec["publication"], spec["publication_sha256"])
    runtime = read_json(sealed(spec["runtime"], spec["runtime_sha256"]))
    require(
        runtime["publication_receipt"] == str(pub)
        and runtime["publication_sha256"] == spec["publication_sha256"],
        "evaluation runtime publication differs",
    )
    source = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=pub,
        expected_publication_receipt_sha256=spec["publication_sha256"],
    ).payload()
    require(
        source["commit"] == spec["source_commit"] and not source["dirty"],
        "evaluation source differs",
    )
    acceptance = read_json(
        sealed(root.parent / "MAIN_TRAINING_ACCEPTANCE_A2.json", spec["training_acceptance_sha256"])
    )
    require(
        acceptance["status"] == "both_arms_main_independently_training_accepted",
        "both training arms need main acceptance",
    )
    policy = read_json(sealed(spec["policy_path"], spec["policy_file_sha256"]))["shared_gpu_policy"]
    for arm in ARMS:
        entry = spec["plans"][arm]
        plan = read_json(sealed(entry["path"], entry["sha256"]))
        training = acceptance["arms"][arm]
        require(
            plan["split"] == "val"
            and plan["metric_scope"] == "expression_only"
            and plan["checkpoint_epoch"] == 2
            and plan["checkpoint_role"] == "last"
            and plan["gpu"] == ["0"]
            and plan["cuda_memory_fraction"] == 0.9
            and plan["cpu_training_state"] is True
            and plan.get("evaluation_precision", "native") == "native"
            and plan["evaluation_source"] == source
            and plan["data_root"] == runtime["data_root"]
            and plan["runtime_sha256"] == spec["runtime_sha256"],
            "evaluation plan differs from the frozen runtime or scope",
        )
        require(
            plan["training_identity"]["run_id"] == training["run_id"]
            and plan["training_identity"]["source"]["commit"] == training["source_commit"]
            and plan["training_run_root"] == training["run_root"]
            and plan["checkpoint_sha256"]
            == entry["checkpoint_sha256"]
            == training["last_checkpoint"]["sha256"]
            and plan["config_sha256"] == training["config_sha256"],
            "evaluation plan differs from the accepted training checkpoint",
        )
        train_root = Path(training["run_root"]).resolve(strict=True)
        require(
            train_root.is_relative_to(root.parent / "runs")
            and Path(plan["checkpoint"]).resolve(strict=True)
            == (train_root / "fit" / training["last_checkpoint"]["file"]).resolve(strict=True)
            and read_json(train_root / "run_manifest.json") == plan["training_identity"],
            "training manifest or final checkpoint path differs",
        )
        output = Path(plan["output_root"]).resolve()
        require(
            str(output) == entry["output"] and output.is_relative_to(root) and not output.exists(),
            "evaluation result root must be fresh and local to the controller",
        )
        sealed(plan["config"], plan["config_sha256"])
        sealed(plan["checkpoint"], plan["checkpoint_sha256"])
    return spec, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    spec, policy = verify(args.spec, args.spec_sha256)
    root = args.spec.resolve().parent
    admit_selected_device(policy, snapshot(os.getpid()), 0)
    if args.check_only:
        print(
            json.dumps({"inputs_verified": True, "admission_passed": True, "CUDA_started": False})
        )
        return
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "evaluation launch environment differs from the sealed GPU0 contract",
    )
    approval = read_json(root / "MAIN_LAUNCH_APPROVAL.json")
    require(
        approval["approved"] is True
        and approval["spec_sha256"] == args.spec_sha256
        and approval["controller_sha256"] == sha256_file(Path(__file__)),
        "main evaluation launch approval differs",
    )
    owner = process_info(os.getpid())
    require(owner is not None, "controller identity unavailable")
    atomic_json(root / "OWNER.json", {"identity": owner, "gpu": 0, "spec_sha256": args.spec_sha256})
    phase = "arming"
    try:
        with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):

            def guard():
                sealed(args.spec, args.spec_sha256)
                sealed(spec["policy_path"], spec["policy_file_sha256"])
                for entry in spec["plans"].values():
                    sealed(entry["path"], entry["sha256"])
                observed = snapshot(os.getpid())
                admit_selected_device(policy, observed, 0)
                atomic_json(
                    root / "RESOURCE_CURRENT.json",
                    {
                        "checked_unix": time.time(),
                        "owner": owner,
                        "gpu": 0,
                        "gpu_state": observed["gpus"][0],
                        "owned_processes": [p for p in observed["processes"] if p["owned"]],
                    },
                )

            receipts = {}
            for arm in ARMS:
                guard()
                entry = spec["plans"][arm]
                phase = "expression_only_validation_" + arm
                plan = read_json(Path(entry["path"]))
                sealed(plan["checkpoint"], plan["checkpoint_sha256"])
                command = [
                    sys.executable,
                    "-c",
                    "import sys,json;from pathlib import Path;"
                    "from gradpert.execution.v2_checkpoint_eval import execute_evaluation_plan;"
                    "execute_evaluation_plan(json.loads(Path(sys.argv[1]).read_text()))",
                    entry["path"],
                ]
                atomic_json(
                    root / "CURRENT_PHASE.json",
                    {
                        "arm": arm,
                        "phase": phase,
                        "output": entry["output"],
                        "command": command,
                        "plan_sha256": entry["sha256"],
                    },
                )
                started = time.monotonic()
                native_run(command, root / (arm + "-eval.log"), policy, admission_check=guard)
                guard()
                output = Path(entry["output"])
                terminal = output / "COMPLETE.json"
                done = read_json(terminal)
                result = done["result"]
                order = [row["condition_id"] for row in result["conditions"]]
                require(
                    done["plan"] == plan
                    and done["zero_pkl"]
                    and result["split"] == "val"
                    and result["metric_scope"] == "expression_only"
                    and len(order) == 445
                    and len(result["metrics"]) == 3
                    and all(
                        not metric["metric_id"].endswith("_deg") for metric in result["metrics"]
                    )
                    and sha256_json(order) == spec["frozen_condition_order_sha256"]
                    and result["control_manifest_sha256"] == spec["frozen_control_manifest_sha256"]
                    and result["reference_sha256"] == spec["frozen_reference_sha256"],
                    "evaluation terminal does not match the frozen expression protocol",
                )
                receipts[arm] = {
                    "COMPLETE_sha256": sha256_file(terminal),
                    "output": str(output),
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "wall_seconds": time.monotonic() - started,
                    "conditions": len(order),
                    "query_recipe": result["query_recipe"],
                    "expression_exposure": result["expression_exposure"],
                }
                atomic_json(root / (arm.upper() + "_SMALL_RECEIPT.json"), receipts[arm])
            require(
                receipts[ARMS[0]]["query_recipe"] == receipts[ARMS[1]]["query_recipe"],
                "evaluation query recipes differ",
            )
            require(
                receipts[ARMS[0]]["expression_exposure"]
                == receipts[ARMS[1]]["expression_exposure"],
                "evaluation expression exposure differs",
            )
            require(
                not any(p.is_file() and p.suffix.lower() == ".pkl" for p in root.parent.rglob("*")),
                "unexpected PKL",
            )
            atomic_json(
                root / "CONTROLLER_COMPLETE.json",
                {
                    "status": "execution_complete_main_verification_pending",
                    "spec_sha256": args.spec_sha256,
                    "arms": receipts,
                    "whole_family_zero_PKL": True,
                    "scientific_complete": False,
                },
            )
    except BaseException as error:
        atomic_json(
            root / "CONTROLLER_FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "owner": owner,
                "phase": phase,
                "preserve_outputs": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
