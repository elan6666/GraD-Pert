"""One sealed anchor preflight or fit, reusing native execution/resource gates."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.generate_response_anchor import configuration
from scripts.v2.run_group import lock, validate_preflight
from scripts.v2.shared_gpu import (
    admit,
    descendant,
    parent_map,
    process_info,
    snapshot,
    terminate_tree,
)


def checked(binding: dict) -> Path:
    path = Path(binding["path"])
    if (
        not path.resolve(strict=True).is_relative_to("/data/yilangliu")
        or sha256_file(path) != binding["sha256"]
    ):
        raise ValueError("sealed server input changed")
    return path


def recipe(root: Path, arm: str) -> tuple[Path, Path, str]:
    """Explicit recipe selection; historical MSE keeps its original sealed paths."""
    families = {
        "R14-FRGA": ("response_anchor_20261028", "mse"),
        "R15-DIR": ("response_anchor_direction_20261029", "cosine_direction"),
        "R16-GPM": ("response_population_prediction_20261029", "population_prediction"),
        "R18-CPD": ("response_population_contrast_20261030", "population_prediction_contrast"),
        "R17-GPM-HG0": ("response_population_head_protection_20261030", "population_prediction"),
    }
    if arm not in families:
        raise ValueError("unknown fixed response anchor arm")
    family, mode = families[arm]
    base = root / "configs/v2" / family
    return (
        base / "calibration/gradpert_v2/nadig_jurkat.yaml",
        base / "CALIBRATION_DECISION.json",
        mode,
    )


def validate_direction_diagnostics(diagnostic: dict) -> None:
    """Reject undefined frozen targets and unobserved/nonfinite readout summaries."""
    if (
        diagnostic.get("response_anchor_loss") != "cosine_direction"
        or diagnostic.get("eps") != 1e-6
        or diagnostic.get("target_count") != 1322
        or diagnostic.get("target_clamped_count") != 0
        or type(diagnostic.get("head_observation_count")) is not int
        or diagnostic["head_observation_count"] < 1
        or type(diagnostic.get("head_clamped_count")) is not int
        or not 0 <= diagnostic["head_clamped_count"] <= diagnostic["head_observation_count"]
        or any(
            not isinstance(diagnostic.get(key), (int, float))
            or isinstance(diagnostic.get(key), bool)
            or not math.isfinite(diagnostic[key])
            for key in ("target_norm_min", "target_norm_max", "head_norm_min", "head_norm_max")
        )
        or not 1e-6 < diagnostic["target_norm_min"] <= diagnostic["target_norm_max"]
        or not 0 <= diagnostic["head_norm_min"] <= diagnostic["head_norm_max"]
    ):
        raise ValueError("invalid direction objective norm audit")


def _digest(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def validate_pair_diagnostics(result):
    """Fail closed on first8 signal and exact two-match metadata; scalar reads only."""
    from gradpert.training.v2.response_anchor import pair_schedule_identity

    expected = pair_schedule_identity(1322, 150, 20261028)
    if (
        result.get("pair_schedule") != expected
        or result.get("identity", {}).get("response_anchor", {}).get("pair_schedule") != expected
        or result.get("response_anchor_prediction_head_grad") is not True
    ):
        raise ValueError("pair schedule/head calibration identity differs")
    ranks = result.get("pair_diagnostics_ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError("pair first8 diagnostics required on both ranks")
    audits = result.get("pair_replay_audit_ranks")
    if not isinstance(audits, list) or len(audits) != 2:
        raise ValueError("pair replay proof required on both ranks")
    for rank, audit in enumerate(audits):
        from gradpert.training.v2.response_anchor import pair_schedule

        first_pair = list(pair_schedule(1322, 150, 0, 20261028, 2, rank)[0][0])
        comparison = audit.get("gradient_comparison", {})
        deterministic = audit.get("deterministic_audit", {})
        if (
            audit.get("pair") != first_pair
            or audit.get("epoch") != 0
            or audit.get("query_ids_sha256") != result["batches"][0]["query_ids_sha256"]
            or audit.get("schema") != "population-pair-replay-audit-v1"
            or audit.get("complete") is not True
            or audit.get("state_restored") is not True
            or audit.get("replays") != 2
            or audit.get("extra_group_forwards") != 4
            or audit.get("included_in_calibration_norms") is not False
            or any(
                audit.get(k) != 0
                for k in ("optimizer_updates", "teacher_updates", "center_updates")
            )
            or comparison.get("atol") != 2e-6
            or comparison.get("rtol") != 0
            or comparison.get("tensor_count", 0) <= 0
            or any(
                comparison.get(k) != 0
                for k in ("missing_mismatch", "shape_mismatch", "nonfinite_tensors")
            )
            or not math.isfinite(comparison.get("max_abs", math.inf))
            or not 0 <= comparison["max_abs"] <= 2e-6
            or deterministic.get("cuda") is not True
            or deterministic.get("inside") != {"enabled": True, "warn_only": False}
            or deterministic.get("restored") != deterministic.get("before")
            or deterministic.get("settings_restored") is not True
            or deterministic.get("cublas_workspace_config") not in (":4096:8", ":16:8")
            or "failure" in deterministic
        ):
            raise ValueError("pair strict deterministic replay differs")
    from gradpert.training.v2.response_anchor import pair_schedule

    for step in range(8):
        units = pair_schedule(1322, 150, 0, 20261028, 2, 0)[step]
        row = result["batches"][step]
        if row.get("local_group_count") != len(units) or row.get(
            "local_group_indices_sha256"
        ) != sha256_json(list(units)):
            raise ValueError("pair first8 batch schedule binding differs")
    for rank, rows in enumerate(ranks):
        expected_pairs = [
            (step, list(pair))
            for step in range(8)
            for pair in pair_schedule(1322, 150, 0, 20261028, 2, rank)[step]
        ]
        if (
            not isinstance(rows, list)
            or [(x.get("step"), x.get("pair")) for x in rows] != expected_pairs
        ):
            raise ValueError("pair diagnostic first8 native step coverage differs")
        for row in rows:
            if (
                not _digest(row.get("query_ids_sha256"))
                or row["query_ids_sha256"] != result["batches"][row["step"]]["query_ids_sha256"]
                or len(row.get("group_rows", [])) != 2
                or any(type(n) is not int or not 4 <= n <= 40 for n in row["group_rows"])
                or any(
                    not _positive_finite(row.get(k))
                    for k in (
                        "target_contrast_rms",
                        "prediction_contrast_rms",
                        "contrast_error_rms",
                    )
                )
            ):
                raise ValueError("pair scalar signal/query/row audit differs")
            for scope in ("response", "prediction"):
                data = row.get(scope, {})
                cosine = data.get("absolute_pair_gradient_cosine", math.inf)
                if (
                    not _positive_finite(data.get("pair_gradient_norm"))
                    or not _positive_finite(data.get("absolute_gradient_norm"))
                    or not math.isfinite(cosine)
                    or not -1.000001 <= cosine <= 1.000001
                ):
                    raise ValueError("pair gradient diagnostic invalid")


def validate_population_calibration(
    config_path: Path,
    parameters: dict,
    calibration: dict,
    decision: dict,
    *,
    arm: str = "R16-GPM",
) -> dict[str, str | bool]:
    """Bind the arm's initialization, selected rule and server-only target identities.

    This reads scalar receipts only. The native loader separately validates the
    sidecar manifest/arrays and read boundaries; this gate never opens an NPZ.
    """
    head_grad = parameters.get("response_anchor_prediction_head_grad", {"value": True}).get("value")
    mode = parameters.get("response_anchor_loss", {"value": "population_prediction"})["value"]
    pair = mode == "population_prediction_contrast"
    expected_arm = "R18-CPD" if pair else ("R16-GPM" if head_grad else "R17-GPM-HG0")
    if pair and head_grad is not True:
        raise ValueError("pair contrast requires normal prediction head gradients")
    if type(head_grad) is not bool or arm != expected_arm or decision.get("arm") != arm:
        raise ValueError("population arm differs from its prediction head gradient route")
    population = {
        key: parameters.get("response_anchor_" + key, {}).get("value")
        for key in ("population_path", "population_sha256", "population_arrays_sha256")
    }
    path = population["population_path"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or ".." in Path(path).parts
        or not Path(path).is_relative_to("/data/yilangliu")
        or not _digest(population["population_sha256"])
        or not _digest(population["population_arrays_sha256"])
    ):
        raise ValueError("population fit requires all three sealed sidecar configuration fields")
    binding = calibration.get("identity", {}).get("response_anchor", {})
    files = calibration.get("input_files_sha256", {})
    source = calibration.get("source", {})
    weight = calibration.get("selected_weight")
    if (
        calibration.get("response_anchor_loss") != mode
        or decision.get("arm") != arm
        or decision.get("response_anchor_loss") != mode
        or source.get("formal_eligible") is not True
        or source.get("published_commit") != source.get("commit")
        or not _digest(source.get("commit"), 40)
        or calibration.get("calibration_complete") is not True
        or calibration.get("state_restored") is not True
        or type(calibration.get("epoch")) is not int
        or calibration.get("epoch") != 0
        or type(calibration.get("rank")) is not int
        or calibration.get("rank") != 0
        or type(calibration.get("identity", {}).get("run_seed")) is not int
        or calibration.get("identity", {}).get("run_seed") != 1
        or any(
            type(calibration.get(key)) is not int or calibration[key] != 0
            for key in ("optimizer_updates", "teacher_updates", "center_updates")
        )
        or calibration.get("parameter_scope")
        != "student.response.parameters; excludes response_anchor_readout"
        or calibration.get("gradient_bank_dtype") != "float32"
        or calibration.get("norm_accumulation_dtype") != "float64"
        or calibration.get("rule") != "min(1,0.1*median(Gpred)/median(Ganchor))"
        or decision.get("rule") != calibration.get("rule")
        or decision.get("calibration_forward_precision") != "bfloat16_autocast"
        or not _digest(decision.get("calibration_complete_sha256"))
        or decision.get("fit_config_sha256") != sha256_file(config_path)
        or decision.get("training_epochs") != 2
        or decision.get("optimizer_updates") != 300
        or decision.get("global_batch") != 320
        or not _positive_finite(weight)
        or weight > 1
        or decision.get("selected_weight") != weight
        or binding.get("loss") != mode
        or binding.get("head") != "existing_native_prediction_softplus"
        or binding.get("teacher_readout") != "absent_no_additional_forward"
        or binding.get("weight") != 1.0
        or binding.get("seed") != 20261028
        or not isinstance(files, dict)
        or files.get(path) != population["population_sha256"]
        or not any(
            isinstance(filename, str)
            and ".." not in Path(filename).parts
            and Path(filename).is_relative_to(Path(path).parent)
            and Path(filename).suffix == ".npz"
            and digest == population["population_arrays_sha256"]
            for filename, digest in files.items()
        )
    ):
        raise ValueError("population fit requires its own complete initialization calibration")
    for key, expected in (
        ("manifest_sha256", parameters["response_anchor_sha256"]["value"]),
        ("arrays_sha256", parameters["response_anchor_arrays_sha256"]["value"]),
        ("population_manifest_sha256", population["population_sha256"]),
        ("population_arrays_sha256", population["population_arrays_sha256"]),
    ):
        top = key if key.startswith("population_") else "bank_" + key
        if (
            not _digest(expected)
            or binding.get(key) != expected
            or calibration.get(top) != expected
            or decision.get(top) != expected
        ):
            raise ValueError(
                "population calibration/decision sidecar and sampling bank hashes differ"
            )
    batches = calibration.get("batches")
    if (
        not isinstance(batches, list)
        or len(batches) != 8
        or any(not isinstance(row, dict) for row in batches)
        or any(type(row.get("step")) is not int for row in batches)
        or [row.get("step") for row in batches] != list(range(8))
        or len({row.get("ordered_identity_sha256") for row in batches}) != 8
        or any(
            any(
                not _digest(row.get(key))
                for key in (
                    "ordered_identity_sha256",
                    "query_ids_sha256",
                    "local_group_indices_sha256",
                )
            )
            or row.get("global_rows") != 320
            or type(row.get("local_group_count")) is not int
            or row["local_group_count"] < 1
            or not _positive_finite(row.get("prediction_gradient_norm"))
            or not _positive_finite(row.get("unit_anchor_gradient_norm"))
            for row in batches
        )
    ):
        raise ValueError("population calibration requires the exact ordered first eight batches")
    metadata = [
        {
            key: value
            for key, value in row.items()
            if key not in {"prediction_gradient_norm", "unit_anchor_gradient_norm"}
        }
        for row in batches
    ]
    gp = statistics.median(row["prediction_gradient_norm"] for row in batches)
    ga = statistics.median(row["unit_anchor_gradient_norm"] for row in batches)
    if (
        calibration.get("ordered_batch_metadata_sha256") != sha256_json(metadata)
        or calibration.get("median_prediction_gradient_norm") != gp
        or calibration.get("median_unit_anchor_gradient_norm") != ga
        or decision.get("median_prediction_gradient_norm") != gp
        or decision.get("median_unit_anchor_gradient_norm") != ga
        or weight != min(1.0, 0.1 * gp / ga)
        or calibration.get("anchor_coefficient") != 2 * 150 / (661 if pair else 1322)
    ):
        raise ValueError("population calibration first-eight gradient rule/identity differs")
    if pair:
        validate_pair_diagnostics(calibration)
    if not head_grad:
        # Deferred import avoids the resource launcher's existing checked() import
        # cycle; the shared scalar receipt guard remains the single route gate.
        from scripts.v2.run_population_calibration import require_head_route_audit

        if (
            binding.get("prediction_head_grad") is not False
            or binding.get("gradient_routing")
            != "gpm_head_parameters_detached_preserve_input_jacobian"
        ):
            raise ValueError("protected population calibration native identity route differs")
        require_head_route_audit(
            calibration, argparse.Namespace(response_anchor_prediction_head_grad=False)
        )
        population["prediction_head_grad"] = False
    elif (
        binding.get("prediction_head_grad", True) is not True
        or "gradient_routing" in binding
        or calibration.get("response_anchor_prediction_head_grad", True) is not True
        or "gradient_routing" in calibration
        or "population_head_route_audit" in calibration
        or "population_head_route_audit_ranks" in calibration
    ):
        raise ValueError("default population arm cannot reuse a protected head calibration")
    return population


def validate_calibration(config_path: Path, calibration: dict, arm: str = "R14-FRGA") -> None:
    root = Path(__file__).resolve().parents[2]
    probe_path, decision_path, mode = recipe(root, arm)
    decision = json.loads(decision_path.read_text())
    raw = yaml.safe_load(config_path.read_text())
    p = raw["model"]["parameters"]
    actual_mode = p.get("response_anchor_loss", {"value": "mse"})["value"]
    if actual_mode != mode:
        raise ValueError("execution arm differs from the configured objective")
    population = {}
    if mode in ("population_prediction", "population_prediction_contrast"):
        population = validate_population_calibration(config_path, p, calibration, decision, arm=arm)
        if yaml.safe_load(probe_path.read_text()) != configuration(
            root,
            bank_path=p["response_anchor_path"]["value"],
            manifest_sha256=p["response_anchor_sha256"]["value"],
            arrays_sha256=p["response_anchor_arrays_sha256"]["value"],
            weight=1.0,
            loss_mode=mode,
            **population,
        ):
            raise ValueError("population calibration must use the unit-weight native probe recipe")
    if mode == "cosine_direction":
        diagnostic = calibration.get("direction_diagnostics", {})
        validate_direction_diagnostics(diagnostic)
        if (
            decision.get("arm") != arm
            or decision.get("response_anchor_loss") != mode
            or calibration.get("response_anchor_loss") != mode
        ):
            raise ValueError("direction fit requires its own complete initialization calibration")
    if (
        not calibration.get("calibration_complete")
        or not calibration.get("state_restored")
        or calibration.get("forward_precision") != "bfloat16_autocast"
        or calibration.get("world_size") != 2
        or calibration.get("optimizer_updates") != 0
        or calibration.get("teacher_updates") != 0
        or calibration.get("center_updates") != 0
        or len(calibration.get("batches", [])) != 8
        or calibration.get("source", {}).get("commit") != decision["calibration_source"]
        or calibration.get("source", {}).get("dirty") is not False
        or calibration.get("config_sha256") != sha256_file(probe_path)
        or calibration.get("resolved_config_sha256")
        != sha256_json(load_experiment_config(probe_path).model_dump(mode="json"))
        or calibration.get("identity", {}).get("run_seed") != 1
        or calibration.get("bank_manifest_sha256") != p["response_anchor_sha256"]["value"]
        or calibration.get("bank_arrays_sha256") != p["response_anchor_arrays_sha256"]["value"]
        or calibration.get("selected_weight") != p["response_anchor_weight"]["value"]
        or raw
        != configuration(
            Path(__file__).resolve().parents[2],
            bank_path=p["response_anchor_path"]["value"],
            manifest_sha256=p["response_anchor_sha256"]["value"],
            arrays_sha256=p["response_anchor_arrays_sha256"]["value"],
            weight=calibration["selected_weight"],
            loss_mode=mode,
            **population,
        )
    ):
        raise ValueError("fit config must equal the single calibrated RI+anchor recipe")


def execute(spec_path: Path, digest: str) -> None:
    spec_path = checked({"path": str(spec_path), "sha256": digest})
    spec = json.loads(spec_path.read_text())
    if spec["schema"] != "response-anchor-execution-v1" or spec["stage"] not in {
        "preflight",
        "fit",
    }:
        raise ValueError("unknown single-stage execution spec")
    root = Path(__file__).resolve().parents[2]
    config_path, pub, policy_path, calibration_path = (
        checked(spec[name]) for name in ("config", "publication", "policy", "calibration")
    )
    if not config_path.resolve().is_relative_to(root):
        raise ValueError("anchor configuration must belong to the executing clean source")
    config = load_experiment_config(config_path)
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=pub,
        expected_publication_receipt_sha256=spec["publication"]["sha256"],
    )
    if source.commit != spec["source_commit"] or spec["gpu"] != "0,1":
        raise ValueError("anchor source/world differs")
    calibration = json.loads(calibration_path.read_text())
    arm = spec.get("arm", "R14-FRGA")
    validate_calibration(config_path, calibration, arm)
    _, decision_path, mode = recipe(root, arm)
    decision = json.loads(decision_path.read_text())
    if spec["calibration"]["sha256"] != decision["calibration_complete_sha256"]:
        raise ValueError("execution requires the exact frozen calibration endpoint")
    if calibration["resource_policy_sha256"] != spec["policy"]["sha256"]:
        raise ValueError("resource policy differs from calibrated admission")
    policy = json.loads(policy_path.read_text())
    output = Path(spec["output"]).resolve()
    if (
        not output.is_relative_to("/data/yilangliu")
        or output.is_relative_to(root)
        or output.exists()
    ):
        raise ValueError("execution output must be a new independent server directory")
    output.mkdir(parents=True)
    child = None
    known = {}

    def guard() -> None:
        checked({"path": str(spec_path), "sha256": digest})
        names = ("config", "publication", "policy", "calibration") + (
            ("plan", "preflight") if spec["stage"] == "fit" else ()
        )
        for name in names:
            checked(spec[name])
        admit(policy, snapshot(os.getpid()))

    stack = contextlib.ExitStack()
    try:
        for gpu in ("0", "1"):
            stack.enter_context(
                lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock")
            )
        guard()
        if spec["stage"] == "preflight":
            resource_queue = output / "RESOURCE_QUEUE.json"
            atomic_json(
                resource_queue,
                {
                    "schema": "unified-ddp-queue-1",
                    "source_commit": source.commit,
                    "shared_gpu_policy": policy,
                    "rows": [
                        {
                            "name": arm,
                            "plan": {
                                "config": str(config_path),
                                "config_sha256": spec["config"]["sha256"],
                                "data_root": spec["data_root"],
                                "publication": str(pub),
                                "publication_sha256": spec["publication"]["sha256"],
                            },
                        }
                    ],
                },
            )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                str(root / "scripts/v2/capacity_probe.py"),
                "--config",
                str(config_path),
                "--data-root",
                spec["data_root"],
                "--output",
                str(output / f"{arm}-preflight"),
                "--gpu",
                "0,1",
                "--publication",
                str(pub),
                "--publication-sha256",
                spec["publication"]["sha256"],
                "--steps",
                "10",
                "--shared-queue",
                str(resource_queue),
                "--resource-owner-pid",
                str(os.getpid()),
            ]
        else:
            plan_path = checked(spec["plan"])
            plan = json.loads(plan_path.read_text())
            if (
                plan["source_commit"] != source.commit
                or plan["config_sha256"] != spec["config"]["sha256"]
                or plan["postfit_policy"] != "deferred"
                or plan["gpu"] != "0,1"
                or plan["config"] != str(config_path)
                or plan["data_root"] != spec["data_root"]
                or plan["publication"] != str(pub)
                or plan["publication_sha256"] != spec["publication"]["sha256"]
                or plan.get("cuda_memory_fraction") != policy["max_own_memory_fraction"]
            ):
                raise ValueError("native fit plan differs from sealed calibrated preflight")
            validate_preflight(
                plan,
                {
                    "receipt": str(checked(spec["preflight"])),
                    "sha256": spec["preflight"]["sha256"],
                },
            )
            probe = json.loads(Path(spec["preflight"]["path"]).read_text())
            if not probe.get("response_anchor_gate", {}).get("passed"):
                raise ValueError("fit requires the native complete40/full-state anchor preflight")
            if mode == "cosine_direction":
                validate_direction_diagnostics(
                    probe["response_anchor_gate"].get("direction_diagnostics", {})
                )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                str(root / "scripts/v2/distributed_train.py"),
                "--plan",
                str(plan_path),
            ]
        env = {
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root / 'scripts/v2'}:{root}",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (output / "child.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = process_info(child.pid)
            if identity is None:
                raise RuntimeError("native child exited before identity capture")
            known[child.pid] = identity
            atomic_json(
                output / "RUNNING.json",
                {
                    "pid": child.pid,
                    "command": command,
                    "spec_sha256": digest,
                    "source_commit": source.commit,
                    "scientific_complete": False,
                },
            )
            while child.poll() is None:
                parents = parent_map()
                for pid in parents:
                    if (
                        descendant(pid, child.pid, parents)
                        and (info := process_info(pid)) is not None
                    ):
                        known.setdefault(pid, info)
                guard()
                time.sleep(5)
            code = child.wait()
        if code:
            raise RuntimeError(f"native {spec['stage']} failed with exit {code}")
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("owned native ranks remain after launcher exit")
            time.sleep(1)
        guard()
        current_source = inspect_source_identity(
            root,
            formal=True,
            expected_repository=config.source_code.repository,
            publication_receipt=pub,
            expected_publication_receipt_sha256=spec["publication"]["sha256"],
        )
        native = (
            (output / f"{arm}-preflight/receipt.json")
            if spec["stage"] == "preflight"
            else (Path(plan["run_root"]) / "TRAIN_COMPLETE.json")
        )
        if current_source != source or not native.is_file() or list(output.rglob("*.pkl")):
            raise ValueError("native source/receipt/zero-PKL closure failed")
        if spec["stage"] == "fit" and list(Path(plan["run_root"]).rglob("*.pkl")):
            raise ValueError("fit must preserve whole-run zero-PKL")
        atomic_json(
            output / "COMPLETE.json",
            {
                "stage": spec["stage"],
                "exit_code": code,
                "spec_sha256": digest,
                "source_commit": source.commit,
                "native_receipt": str(native),
                "native_receipt_sha256": sha256_file(native),
                "scientific_complete": False,
            },
        )
    except BaseException as error:
        cleanup_errors = []
        for pid, identity in reversed(list(known.items())):
            try:
                terminate_tree(pid, identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=30)
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        atomic_json(
            output / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "cleanup_errors": cleanup_errors,
                "source_commit": source.commit,
                "spec_sha256": digest,
                "scientific_complete": False,
            },
        )
        raise
    finally:
        stack.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    execute(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
