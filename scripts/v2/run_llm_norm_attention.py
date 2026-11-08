"""Hash-bound orchestration of published native entry points; no model implementation."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[2]
F = Path("/data/yilangliu/GraD-Pert/development/llm-norm-standard-attention-20261109-a1")
# Direct-file launches also need the native scripts namespace.
sys.path[:0] = [str(S / "src"), str(S), str(S / "scripts/v2")]

from gradpert.data._io import atomic_json  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.execution.v2_training_stage import validate_training_stage  # noqa: E402
from gradpert.hashing import sha256_file  # noqa: E402
from scripts.v2.run_dataset_transfer_300 import native_run  # noqa: E402
from scripts.v2.run_group import lock, validate_preflight  # noqa: E402
from scripts.v2.shared_gpu import admit_selected_device, process_info, snapshot  # noqa: E402


def verify(protocol_sha):
    assert sha256_file(F / "PROTOCOL.json") == protocol_sha
    protocol = json.loads((F / "PROTOCOL.json").read_text())
    source = inspect_source_identity(
        S,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=F / "PUBLICATION.json",
        expected_publication_receipt_sha256=protocol["publication_sha256"],
    )
    assert not source.dirty and source.commit == protocol["source_commit"]
    assert sha256_file(F / "RUNTIME.json") == protocol["runtime_sha256"]
    policy_path = Path(protocol["policy_path"])
    assert sha256_file(policy_path) == protocol["policy_sha256"]
    policy = json.loads(policy_path.read_text())["shared_gpu_policy"]
    assert protocol["order"] == ["A_norm_mla", "B_norm_standard"]
    assert protocol["epochs_per_arm"] == 2 and protocol["updates_per_arm"] == 300
    plans = {}
    for arm in protocol["order"]:
        row = protocol["plans"][arm]
        path = Path(row["path"])
        assert sha256_file(path) == row["sha256"]
        p = json.loads(path.read_text())
        assert p["source_commit"] == source.commit and p["gpu"] == "0"
        assert p["postfit_policy"] == "deferred" and p["cuda_memory_fraction"] == 0.9
        assert sha256_file(Path(p["config"])) == p["config_sha256"] == row["config_sha256"]
        assert not Path(p["run_root"]).exists(), "fresh run root required"
        assert p["resolved_config"]["training"]["max_epochs"]["value"] == 2
        params = p["resolved_config"]["model"]["parameters"]
        assert params["normalization_layout"]["value"] == "llm_pre"
        assert params["terminal_attention"]["value"] == (
            "latent" if arm == "A_norm_mla" else "standard"
        )
        assert params["attention"]["value"] == "relay_full"
        assert params["graph_input_norm"]["value"] == "historical"
        assert params.get("mla_query_rank", {"value": 0})["value"] == 0
        for flag in ["relay_validate_once", "cache_kda_constants"]:
            assert p["resolved_config"]["model"]["parameters"][flag]["value"] is True
        assert not (F / ("integration-" + arm + "-a1")).exists()
        plans[arm] = p
    return protocol, plans, policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    protocol, plans, policy = verify(args.protocol_sha256)
    observed = snapshot(os.getpid())
    admit_selected_device(policy, observed, 0)
    if args.check_only:
        print(
            json.dumps(
                {
                    "inputs_verified": True,
                    "admission_passed": True,
                    "gpu": 0,
                    "CUDA_started": False,
                    "model_forward": 0,
                }
            )
        )
        return
    for key, value in [
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("CUDA_VISIBLE_DEVICES", "0"),
    ]:
        assert os.environ.get(key) == value
    approval = json.loads((F / "MAIN_CONTROLLER_ARM_APPROVAL.json").read_text())
    assert approval["approved_to_arm_controller"] is True
    assert approval["protocol_sha256"] == args.protocol_sha256
    assert approval["controller_sha256"] == sha256_file(Path(__file__))
    owner = process_info(os.getpid())
    assert owner is not None
    assert not (F / "OWNER.json").exists()
    atomic_json(
        F / "OWNER.json",
        {
            "identity": owner,
            "gpu": 0,
            "protocol_sha256": args.protocol_sha256,
            "controller_sha256": sha256_file(Path(__file__)),
        },
    )
    phase = "arming"
    try:
        with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):

            def guard():
                snap = snapshot(os.getpid())
                admit_selected_device(policy, snap, 0)
                atomic_json(
                    F / "RESOURCE_CURRENT.json",
                    {
                        "checked_unix": time.time(),
                        "owner": owner,
                        "gpu": 0,
                        "admission_scope": "selected_device",
                        "gpu_state": snap["gpus"][0],
                        "owned_processes": [p for p in snap["processes"] if p["owned"]],
                    },
                )

            def wait_gate(name):
                started = time.monotonic()
                while not (F / name).exists():
                    guard()
                    if time.monotonic() - started > 1200:
                        raise TimeoutError(
                            "main supervision/start gate not armed within 20 minutes"
                        )
                    time.sleep(5)
                gate = json.loads((F / name).read_text())
                assert gate["approved"] is True and gate["owner"] == owner
                assert gate["protocol_sha256"] == args.protocol_sha256
                assert gate["live_observer_verified"] is True
                guard()

            phase = "waiting_capacity_supervision_gate"
            atomic_json(F / "CURRENT_PHASE.json", {"phase": phase, "controller": owner})
            wait_gate("CAPACITY_START_MAIN.json")
            probes = {}
            for arm in protocol["order"]:
                p = plans[arm]
                out = F / ("integration-" + arm + "-a1")
                command = [
                    sys.executable,
                    str(S / "scripts/v2/capacity_probe.py"),
                    "--config",
                    p["config"],
                    "--data-root",
                    p["data_root"],
                    "--output",
                    str(out),
                    "--gpu",
                    "0",
                    "--publication",
                    str(F / "PUBLICATION.json"),
                    "--publication-sha256",
                    protocol["publication_sha256"],
                    "--integration-only",
                ]
                phase = "integration_" + arm
                atomic_json(
                    F / "CURRENT_PHASE.json",
                    {"phase": phase, "arm": arm, "output": str(out), "command": command},
                )
                native_run(
                    command, F / ("integration-" + arm + "-a1.log"), policy, admission_check=guard
                )
                entry = {
                    "receipt": str(out / "receipt.json"),
                    "sha256": sha256_file(out / "receipt.json"),
                }
                validate_preflight(p, entry)
                probes[arm] = entry
            phase = "waiting_training_supervision_gate"
            atomic_json(
                F / "CAPACITY_STAGE_COMPLETE.json",
                {
                    "status": "native_integration_passed_main_acceptance_pending",
                    "controller": owner,
                    "protocol_sha256": args.protocol_sha256,
                    "probes": probes,
                    "training_started": False,
                },
            )
            atomic_json(F / "CURRENT_PHASE.json", {"phase": phase, "controller": owner})
            wait_gate("TRAINING_START_MAIN.json")
            completed = {}
            for arm in protocol["order"]:
                p = plans[arm]
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "1",
                    str(S / "scripts/v2/distributed_train.py"),
                    "--plan",
                    protocol["plans"][arm]["path"],
                ]
                phase = "full_two_epoch_" + arm
                atomic_json(
                    F / "CURRENT_PHASE.json",
                    {
                        "phase": phase,
                        "arm": arm,
                        "run_id": p["run_id"],
                        "output": p["run_root"],
                        "command": command,
                    },
                )
                started = time.monotonic()
                native_run(command, F / (arm + "-fit-a1.log"), policy, admission_check=guard)
                guard()
                validate_training_stage(p)
                root = Path(p["run_root"])
                journal = json.loads((root / "fit/epoch_state.json").read_text())
                history = json.loads((root / "fit/history.json").read_text())
                assert journal["budget"] == [2, 150] and journal["epoch"] == 2
                assert history[-1]["optimizer_steps"] == 300 and not list(root.rglob("*.pkl"))
                completed[arm] = {
                    "run_id": p["run_id"],
                    "run_root": p["run_root"],
                    "TRAIN_COMPLETE_sha256": sha256_file(root / "TRAIN_COMPLETE.json"),
                    "epochs": 2,
                    "optimizer_steps": 300,
                    "fit_subprocess_wall_seconds": time.monotonic() - started,
                    "epoch_training_seconds": [h["training_seconds"] for h in history],
                    "scientific_complete": False,
                }
                atomic_json(F / (arm.upper() + "_EXECUTION_RECEIPT.json"), completed[arm])
            assert not list(F.rglob("*.pkl"))
            atomic_json(
                F / "CONTROLLER_COMPLETE.json",
                {
                    "status": "training_complete_main_acceptance_pending",
                    "arms": completed,
                    "protocol_sha256": args.protocol_sha256,
                    "scientific_complete": False,
                },
            )
    except BaseException as error:
        atomic_json(
            F / "CONTROLLER_FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "phase": phase,
                "owner": owner,
                "preserve_results": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
