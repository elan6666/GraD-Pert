"""V3 metadata-only sealer: fixed R3 GPU0 / R14 GPU1, unchanged scientific specs.

--check-only reads only exact allowlisted metadata; --seal-dir writes a fresh
metadata directory. Neither mode opens checkpoints, matrices, H5AD or NPZ.
Native validate_spec and checkpoint byte verification remain unchanged at launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from itertools import pairwise
from pathlib import Path

BASE = Path("/data/yilangliu/GraD-Pert")
DEV = BASE / "development"
SHA = "7ae8824c7e8c65bb34c89df302f57999cbf38cef"
ANALYSIS = DEV / "source-response-anchor-analysis-7ae8824-20261028-a1"
ANALYSIS_PUB = DEV / "response-anchor-analysis-7ae8824-20261028-a1/PUBLICATION.json"
ANALYSIS_PUB_SHA = "e7d20a22770cbc9c287c7acd13ae21b0032d7793a845d60204ef602cff09f78e"
DATA = BASE / "data-vnext-a942114"
DATASET = DATA / "nadig_jurkat/within_cell_unseen_single"
JOB = DEV / "response-anchor-postfit-7ae8824-20261028-a1"
POLICY = DEV / "response-anchor-preflight-dc5233e-20261028-a2/RESOURCE_POLICY.json"
PYTHON = BASE / "source/.venv/bin/python"
R3_SOURCE = DEV / "source-representation-round3-au-f0cb20e"
R3_RUN = (
    BASE
    / "runs-representation-round3-au-f0cb20e-20261023-a2"
    / "nadig_jurkat-seed1-20261022T172341Z-0d539c670fc0452a8d7d5be881becdbf"
)
R14_SOURCE = DEV / "source-response-anchor-dc5233e-20261028-a2"
R14_RUN = (
    DEV
    / "response-anchor-fit-dc5233e-20261028-a1/runs"
    / "nadig_jurkat-seed1-20261028T104437Z-1584dd32edaa44a495c830e531868123"
)
QUEUE = DEV / "representation-round3-au-f0cb20e-20261023-a2/queue/queue.json"
PARENT = DEV / "cap40-medium-47c0ba1-20261015/manifest/manifest.json"
PERFORMANCE = DEV / "medium-observer-c9efc24-20261015/COMPLETE.json"
PANEL = QUEUE.parent / "R3-RI-panel/panel.json"
VIEWS = QUEUE.parent / "R3-RI-panel/matched_expression_views.json"

ENDPOINTS = [
    dict(
        name="R3-RI",
        root=R3_RUN,
        source=R3_SOURCE,
        commit="f0cb20ebb2efab749970f2bf5d205ed2f358592c",
        publication=DEV / "representation-round3-au-f0cb20e-20261023-a2/R3_PUBLICATION.json",
        publication_sha256="6737976d5a80e6db5aa2e8b6b520d9ce7accd5c6a37935be6e20a7aeaad25441",
        config=R3_SOURCE
        / "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml",
        config_sha256="65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7",
        manifest_sha256="ef626f70df0f7697d21d429955782309d438d408644b6e1a705efe5aca275c83",
        journal_sha256="fe6a0321ff0cc47dd037f270352b1da0cfb62b1272ec8caaa85a47d0e1c3ff31",
        resolved_sha256="734c7be128d721d36c6e00da1cbb79691b2dc0b5c383ee121f70ea3ffc411ef9",
        checkpoint_sha256="96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4",
    ),
    dict(
        name="FixedResponseGroupAnchor",
        root=R14_RUN,
        source=R14_SOURCE,
        commit="dc5233ef1a243b2144107e5441a5a8db4f800fc6",
        publication=DEV / "response-anchor-preflight-dc5233e-20261028-a2/PUBLICATION.json",
        publication_sha256="a1fc0878971b1cea58feb48f40293f5ec16a7f3e84842678aff9c9d416608505",
        config=R14_SOURCE
        / "configs/v2/response_anchor_20261028/R14-FRGA/gradpert_v2/nadig_jurkat.yaml",
        config_sha256="194b10ad499e2dc20b21f75f152dbc87344b482f00fab62410bd5c57bc9eda32",
        manifest_sha256="c13a037d00729696161cc16cb7ddd3c005429784653b2e81b519e54e7d4433bb",
        journal_sha256="af03dcfeb172bd6717a4b401adbe0e34bf4010b49eb01b791730907039d178f2",
        resolved_sha256="163d064bd390496b8af5133a8da55c0c6b56fb71a23af46c68baca49b88f150e",
        checkpoint_sha256="7ad95c52d04aa3accec6a6a82006f5d9c8111c2d602c058898b25706e99cb467",
    ),
]
# Exact allowlist: no checkpoint, state_arrays, target bank, expression matrix,
# test control manifest or directory traversal is exposed through the reader.
EXPECTED = {
    ANALYSIS_PUB: ANALYSIS_PUB_SHA,
    POLICY: "2a2adb50c25df9285ae7cba4b48856b671920175d4dad54014eec094b3ebaea2",
    DATASET / "manifests/canonical.json": (
        "cb23e89cb7e7b0697e70a16230e6cd7c3dd53ec37d0152d75638008f10e1d95b"
    ),
    DATASET / "manifests/split.json": (
        "d8807af3f03f4ba6c57ddb4f77acdb915b3b180592bca4ba1619f3e49cbe6890"
    ),
    DATASET / "manifests/evaluation_controls.val.json": (
        "f76df958701858c2c488bb701f0594ce224779b15e28d0b9567a4d3a824cc528"
    ),
    DATASET / "validation_evaluation_v2/state_manifest.json": (
        "a24f138f2851719d19fc2821bc2c7a00aa4d845a1f197eea98e1472e8f8cab49"
    ),
    DATASET / "canonical/expression_gene_ids.txt": (
        "b713f1fe7e3cf05371668326973fc323a1a2d3c1382163a7fa842218e983c09b"
    ),
    QUEUE: "1d15fa1c24d77b9ed77d88c67da6caac1090f289626550138d96605bf3ec3d44",
    PARENT: "19c814aac4642ae74b6d9d3d8830d0ca24a2725c8ecd112c7165f6d8fb87b1d8",
    PERFORMANCE: "aeccac2f41c3d112d5a5b9917c8d399bcc688d1c5ffed4ed98fb0854d47b2460",
    PANEL: "d4455c25a767218513f6967d744eb1867ad7f4161762473f3ea806ba5fc76c31",
    VIEWS: "a76b57ca79fb9855a41bb1f3c9dabb989631bd24cbcf97fafb106b84d2dccb6f",
}
for e in ENDPOINTS:
    EXPECTED.update(
        {
            e["publication"]: e["publication_sha256"],
            e["config"]: e["config_sha256"],
            e["root"] / "run_manifest.json": e["manifest_sha256"],
            e["root"] / "fit/epoch_state.json": e["journal_sha256"],
            e["root"] / "resolved_config.json": e["resolved_sha256"],
        }
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def canonical(path, exists=True):
    path = Path(path)
    actual = path.resolve(strict=exists)
    if not actual.is_relative_to("/data/yilangliu") or actual != path.absolute():
        raise ValueError("canonical server path required")
    return actual


def binding(path):
    return dict(path=str(path), sha256=EXPECTED[Path(path)])


def read_metadata(path):
    path = canonical(path)
    if path not in EXPECTED or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("outside bounded exact metadata allowlist")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED[path]:
        raise ValueError("frozen metadata hash mismatch: " + str(path))
    return raw


def read_json(path):
    return json.loads(read_metadata(path))


def source(root, commit, publication):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, env=env
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True, env=env
    ).strip()
    pub = read_json(publication)
    if actual != commit or dirty or pub["published_commit"] != commit:
        raise ValueError("source clean published identity differs")
    return dict(root=str(root), commit=commit, publication=binding(publication))


POSTFIT_SPEC = JOB / "metadata-a1/SPEC.json"
POSTFIT_SPEC_SHA = "8e69e4ff75a677b7aa0838499a1475b3fa8ebbd826ea0e32ee5887d34c7d07b0"
# Reduce the inherited exact registry to the metadata needed by this operation.
KEEP = {ANALYSIS_PUB, POLICY, QUEUE, PARENT, PERFORMANCE, PANEL, VIEWS}
for endpoint in ENDPOINTS:
    KEEP.update(
        {
            endpoint["publication"],
            endpoint["config"],
            endpoint["root"] / "run_manifest.json",
            endpoint["root"] / "fit/epoch_state.json",
            endpoint["root"] / "resolved_config.json",
        }
    )
EXPECTED = {p: h for p, h in EXPECTED.items() if p in KEEP}
EXPECTED[POSTFIT_SPEC] = POSTFIT_SPEC_SHA
COUNTS = dict(
    conditions=1335,
    rows=5317,
    deep_conditions=128,
    deep_rows=506,
    gradient_conditions=32,
    gradient_rows=125,
)
SCHEMA = "response-anchor-direct-medium-spec-v1"
POPULATION_SHA = "afa9c95fa26dc19167e9919bc09c86a8d99f9446466de95a1efd613b1f93bd86"
ALLOWED_AXIS_SHA = "50a508fcd24f37084cb51e71acf1b1db1756d3dbbe5301bd40a2c1b98c1b7b0f"
PARALLEL_JOB = DEV / "response-anchor-medium-parallel-7ae8824-20261029-a2"
OUTPUT_ROOT = PARALLEL_JOB / "execution"
GPU_MAP = {"R3-RI": "0", "R14-FRGA": "1"}
APPROVED_CHANGES = {
    "R3-RI": [],
    "R14-FRGA": [
        dict(
            path="/response_anchor",
            before=dict(present=False),
            after=dict(
                present=True,
                value=dict(
                    arrays_sha256="28bc66a2b02f5435ee7fdec46be808461e1a33cd13fd9be35dd3d6a7bd21ff6d",
                    head="Linear256_256_default_random_bias_true_private_rng",
                    manifest_sha256="86901e2829d7964f41d39bc958d81f1b150584810861d98fa6fb1b4c5eaf012a",
                    schedule="hash_permutation_step_i_mod_T_rank_floor_i_div_T_mod_world",
                    seed=20261028,
                    teacher_readout="unused_ema_mirror_no_forward_or_target_values",
                    weight=0.03148168297045377,
                ),
            ),
        )
    ],
}
PROTECTED = (DATA, ANALYSIS, R3_SOURCE, R14_SOURCE, R3_RUN, R14_RUN)


def identity_diff(before, after, prefix=""):
    changes = []
    for key in sorted(before.keys() | after.keys()):
        path = prefix + "/" + key
        a, b = key in before, key in after
        if a and b and isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.extend(identity_diff(before[key], after[key], path))
        elif a != b or before.get(key) != after.get(key):
            changes.append(
                dict(
                    path=path,
                    before=dict(present=a, **({"value": before[key]} if a else {})),
                    after=dict(present=b, **({"value": after[key]} if b else {})),
                )
            )
    return changes


def fresh(path, *, extra_protected=()):
    path = canonical(path, exists=False)
    if path.exists() or any(
        path.is_relative_to(p) or p.is_relative_to(path) for p in (*PROTECTED, *extra_protected)
    ):
        raise ValueError("fresh isolated output/metadata directory required: " + str(path))
    return path


def source_direct(root, commit, publication):
    checked_source = source(root, commit, publication)
    return dict(
        root=checked_source["root"],
        commit=checked_source["commit"],
        publication=str(publication),
        publication_sha256=EXPECTED[publication],
    )


def validate_population(parent, panel, views, medium):
    counts = dict(
        conditions=len(parent["schedule"]),
        rows=sum(len(g["rows"]) for g in parent["schedule"]),
        deep_conditions=sum(bool(g["deep_rows"]) for g in parent["schedule"]),
        deep_rows=sum(len(g["deep_rows"]) for g in parent["schedule"]),
        gradient_conditions=sum(bool(g["gradient_rows"]) for g in parent["schedule"]),
        gradient_rows=sum(len(g["gradient_rows"]) for g in parent["schedule"]),
    )
    fields = ("index", "condition_id", "rows", "deep_rows", "gradient_rows", "queries", "view_seed")
    population_sha = digest([{k: g[k] for k in fields} for g in parent["schedule"]])
    if (
        counts != COUNTS
        or any(parent.get(k) != v for k, v in COUNTS.items())
        or parent.get("analysis_scope") != "medium_cap40_stratified"
        or population_sha != POPULATION_SHA
        or medium["population_sha256"] != POPULATION_SHA
        or medium["counts"] != COUNTS
    ):
        raise ValueError("frozen medium population/counts changed")
    if (
        panel.get("conditions"),
        panel.get("rows"),
        panel.get("controls_per_context"),
        panel.get("condition_context_groups"),
        len(views),
    ) != (32, 128, 4, 32, 32) or panel["panel_sha256"] != digest(panel["schedule"]):
        raise ValueError("fixed crossed32x4 panel changed")
    contexts = {}
    train = {g["condition_id"] for g in parent["schedule"]}
    for group, view in zip(panel["schedule"], views, strict=True):
        context = tuple(group["context"])
        if (
            len(context) != 4
            or group["condition_id"] not in train
            or len(group["rows"]) != 4
            or group["queries"] != parent["schedule"][0]["queries"]
            or view["condition_id"] != group["condition_id"]
            or tuple(view["context"]) != context
        ):
            raise ValueError("frozen condition/context/view/query order changed")
        contexts.setdefault(context, []).append(group)
    if len(contexts) != 16:
        raise ValueError("exact16 crossed contexts required")
    for groups in contexts.values():
        if len(groups) != 2 or len({g["condition_id"] for g in groups}) != 2:
            raise ValueError("two distinct conditions per context required")
        controls = [[r["control_row_id"] for r in g["rows"]] for g in groups]
        if controls[0] != controls[1] or len(set(controls[0])) != 4:
            raise ValueError("crossed shared four ordered controls changed")


def validate_performance(performance):
    parity = performance.get("full_parity", {})
    fast, reference = (
        performance.get(k, float("nan")) for k in ("full_fast_seconds", "full_reference_seconds")
    )
    if (
        performance.get("status") != "completed"
        or performance.get("same_sample_scope") is not True
        or performance.get("deterministic_algorithms") is not True
        or performance.get("model_updates") != 0
        or parity.get("passed") is not True
        or parity.get("atol") != 1e-8
        or not 0 <= parity.get("max_abs", float("inf")) <= 1e-8
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (fast, reference))
        or not 0 < fast < reference
    ):
        raise ValueError("frozen full-medium performance/correctness receipt differs")


def owner_binding(path, expected_sha, output_root):
    if path is None and expected_sha is None:
        return None
    if path is None or expected_sha is None or len(expected_sha) != 64:
        raise ValueError("both exact owner path and SHA256 required")
    path = canonical(path)
    if not path.is_relative_to(output_root) or path.name != "DISPATCH_OWNER.json":
        raise ValueError("owner must be the fresh execution root DISPATCH_OWNER.json")
    if path.stat().st_size > 16384:
        raise ValueError("owner metadata exceeds bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("owner metadata SHA256 differs")
    owner = json.loads(raw)
    process = owner.get("process", {})
    if (
        set(owner) != {"schema", "process"}
        or owner.get("schema") != "response-anchor-analysis-owner-v1"
        or set(process) != {"pid", "uid", "start_ticks", "command_sha256", "script_basename"}
        or type(process["pid"]) is not int
        or process["pid"] <= 0
        or type(process["uid"]) is not int
        or process["uid"] != os.getuid()
        or not isinstance(process["start_ticks"], str)
        or not process["start_ticks"].isdigit()
        or int(process["start_ticks"]) <= 0
        or len(process["command_sha256"]) != 64
        or not process["script_basename"]
    ):
        raise ValueError("exact owner receipt schema differs")
    # Metadata only. Published native resource_snapshots verifies current PID,
    # start ticks, command, UID and worker ancestry at actual execution.
    return dict(path=str(path), sha256=expected_sha)


def build(output_root, owner_path, owner_sha):
    if not BASE.is_dir() or not Path.cwd().resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("run only on the authorized server under /data/yilangliu")
    output_root = canonical(output_root, exists=False)
    if any(output_root.is_relative_to(p) or p.is_relative_to(output_root) for p in PROTECTED):
        raise ValueError("execution root overlaps protected data/source/run")
    postfit = read_json(POSTFIT_SPEC)
    medium = postfit["medium"]
    if (
        postfit["schema"] != "response-anchor-postfit-input-1"
        or postfit["data_root"] != str(DATA)
        or medium["original_queue"] != binding(QUEUE)
        or medium["parent"] != binding(PARENT)
        or medium["performance"] != binding(PERFORMANCE)
        or medium["panel"]["population"] != binding(PANEL)
        or medium["panel"]["matched_views"] != binding(VIEWS)
        or postfit["shared_policy"] != binding(POLICY)
        or postfit["analysis_source"]["root"] != str(ANALYSIS)
        or postfit["analysis_source"]["commit"] != SHA
        or postfit["analysis_source"]["publication"] != binding(ANALYSIS_PUB)
    ):
        raise ValueError("existing postfit medium pointers/source/policy differ")
    queue = read_json(QUEUE)
    if queue["medium_analysis"] != dict(
        parent_manifest=str(PARENT),
        parent_manifest_sha256=EXPECTED[PARENT],
        performance_receipt=str(PERFORMANCE),
        performance_receipt_sha256=EXPECTED[PERFORMANCE],
    ):
        raise ValueError("original queue medium pointers differ")
    parent, panel, views = read_json(PARENT), read_json(PANEL), read_json(VIEWS)
    validate_population(parent, panel, views, medium)
    validate_performance(read_json(PERFORMANCE))
    read_json(POLICY)
    allowed = postfit["validation"]["allowed_gene_indices"]
    if (
        len(allowed) != 4775
        or any(type(i) is not int or i < 0 or i >= 5000 for i in allowed)
        or any(a >= b for a, b in pairwise(allowed))
        or digest(allowed) != ALLOWED_AXIS_SHA
        or parent["allowed_expression_ids"] != allowed
    ):
        raise ValueError("fixed4775 ordered allowed axis differs")
    analysis_source = source_direct(ANALYSIS, SHA, ANALYSIS_PUB)
    owner = owner_binding(owner_path, owner_sha, output_root)
    specs, endpoint_summaries = {}, {}
    for number, e in enumerate(ENDPOINTS):
        name = "R3-RI" if number == 0 else "R14-FRGA"
        frozen = postfit["endpoints"][number]
        for key in ("run_manifest", "journal", "config"):
            path = (
                e["root"]
                / {"run_manifest": "run_manifest.json", "journal": "fit/epoch_state.json"}.get(
                    key, ""
                )
                if key != "config"
                else e["config"]
            )
            if frozen[key] != binding(path):
                raise ValueError("postfit endpoint metadata binding differs")
        if (
            frozen["name"] != e["name"]
            or frozen["run_root"] != str(e["root"])
            or frozen["source"]["root"] != str(e["source"])
            or frozen["source"]["commit"] != e["commit"]
            or frozen["source"]["publication"] != binding(e["publication"])
            or frozen["checkpoint"]["sha256"] != e["checkpoint_sha256"]
        ):
            raise ValueError("postfit endpoint source/checkpoint differs")
        identity = read_json(e["root"] / "run_manifest.json")
        journal = read_json(e["root"] / "fit/epoch_state.json")
        resolved = read_json(e["root"] / "resolved_config.json")
        read_metadata(e["config"])
        if (
            journal["identity"] != identity
            or journal["last"] != dict(epoch=2, file="epoch-0002.pt", sha256=e["checkpoint_sha256"])
            or identity["source"]["commit"] != e["commit"]
            or identity["source"]["dirty"] is not False
            or digest(resolved) != identity["resolved_config_sha256"]
            or identity.get("config_sha256", e["config_sha256"]) != e["config_sha256"]
        ):
            raise ValueError("exact original run/config/journal identity differs")
        changes = identity_diff(parent["data"], identity["data"])
        if changes != APPROVED_CHANGES[name]:
            raise ValueError("unapproved parent data/protocol difference")
        policy = identity["data"]["training_expression_policy"]
        if (
            policy["exclude_test_target_expression"] is not True
            or len(policy["excluded_expression_gene_ids"]) != 225
            or identity["data"]["effective_training_expression_ids_sha256"]
            != postfit["validation"]["allowed_gene_ids_sha256"]
        ):
            raise ValueError("original225 exclusion policy differs")
        checkpoint = canonical(e["root"] / "fit/epoch-0002.pt")
        if not checkpoint.is_file() or frozen["checkpoint"]["path"] != str(checkpoint):
            raise ValueError("retained checkpoint path missing/different")
        output = fresh(output_root / name)
        spec = dict(
            schema=SCHEMA,
            name=name,
            analysis_source=analysis_source,
            native_source=source_direct(e["source"], e["commit"], e["publication"]),
            run_root=str(e["root"]),
            run_manifest=binding(e["root"] / "run_manifest.json"),
            resolved_config=binding(e["root"] / "resolved_config.json"),
            config=binding(e["config"]),
            journal=binding(e["root"] / "fit/epoch_state.json"),
            checkpoint=dict(role="last", epoch=2, sha256=e["checkpoint_sha256"]),
            data_root=str(DATA),
            medium=dict(
                parent=binding(PARENT),
                performance=binding(PERFORMANCE),
                population_sha256=POPULATION_SHA,
                identity_changes=changes,
            ),
            panel=dict(population=binding(PANEL), matched_views=binding(VIEWS)),
            allowed_axis=dict(count=4775, sha256=ALLOWED_AXIS_SHA),
            resource_policy=binding(POLICY),
            resource_owner=owner,
            output=str(output),
            gpu=GPU_MAP[name],
        )
        specs[name] = spec
        endpoint_summaries[name] = dict(
            training_run_id=identity["run_id"],
            native_source_commit=e["commit"],
            config_sha256=e["config_sha256"],
            checkpoint_epoch=2,
            checkpoint_role="last",
            checkpoint_sha256=e["checkpoint_sha256"],
            checkpoint_size=checkpoint.stat().st_size,
            parent_data_sha256=digest(parent["data"]),
            training_data_sha256=digest(identity["data"]),
            approved_identity_changes=changes,
            sampling_changed=False,
        )
    summary = dict(
        status="metadata_checked_only",
        schema=SCHEMA,
        analysis_source_commit=SHA,
        metadata_files=len(EXPECTED),
        endpoint_gpu_map=GPU_MAP,
        metadata_hashes_verified=True,
        owner_status="exact_metadata_bound_live_ancestry_pending"
        if owner
        else "required_before_seal",
        owner_live_process_checked=False,
        counts=COUNTS,
        population_sha256=POPULATION_SHA,
        panel_conditions=32,
        panel_rows=128,
        panel_contexts=16,
        controls_per_context=4,
        allowed_axis=dict(count=4775, sha256=ALLOWED_AXIS_SHA),
        excluded_count=225,
        endpoints=endpoint_summaries,
        native_validate_spec_executed=False,
        checkpoint_bytes_opened=False,
        expression_or_bank_arrays_opened=False,
        gpu_used=False,
        scientific_complete=False,
        output_root=str(output_root),
        internal_probe_gate=dict(
            indices=[0, 1, 15],
            conditions=3,
            backends=["reference", "fast"],
            standalone_pilot_cli=False,
            continues_full_medium_if_passed=True,
        ),
    )
    return specs, summary


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--seal-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--owner-path", type=Path)
    parser.add_argument("--owner-sha256")
    args = parser.parse_args()
    if args.seal_dir is not None and (args.owner_path is None or args.owner_sha256 is None):
        parser.error(
            "--seal-dir requires exact --owner-path and --owner-sha256 from live dispatcher"
        )
    specs, summary = build(args.output_root, args.owner_path, args.owner_sha256)
    if args.check_only:
        print(json.dumps(summary, sort_keys=True))
        return
    seal = fresh(args.seal_dir, extra_protected=(canonical(args.output_root, exists=False),))
    # All checks above precede every write; a fresh directory is never overwritten.
    seal.mkdir(parents=True, exist_ok=False)
    commands = []
    seals = {}
    for name, spec in specs.items():
        spec_path = seal / (name + ".SPEC.json")
        write_new(spec_path, spec)
        spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        seals[name] = dict(path=str(spec_path), sha256=spec_sha, digest=digest(spec))
        environment = dict(
            CUDA_VISIBLE_DEVICES=spec["gpu"],
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTHONPATH=spec["native_source"]["root"] + "/src:" + spec["native_source"]["root"],
        )
        commands.append(
            dict(
                endpoint=name,
                env=environment,
                modes=[
                    dict(
                        mode=mode,
                        command=[
                            str(PYTHON),
                            str(ANALYSIS / "scripts/v2/response_anchor_medium.py"),
                            mode,
                            "--spec",
                            str(spec_path),
                            "--spec-sha256",
                            spec_sha,
                        ],
                    )
                    for mode in ("seal", "medium", "panel", "validate")
                ],
            )
        )
    write_new(
        seal / "COMMANDS.json",
        dict(
            execute="one_owner_parallel_endpoints_sequential_phases",
            endpoint_gpu_map=GPU_MAP,
            endpoint_order=list(specs),
            commands=commands,
            internal_probe_gate=summary["internal_probe_gate"],
        ),
    )
    write_new(
        seal / "METADATA_SEAL.json",
        dict(
            **summary,
            specs=seals,
            metadata_bindings=[binding(p) for p in EXPECTED],
            checkpoint_sha_basis="exact terminal journal; native launch rehashes retained CP bytes",
            native_seal_is_not_metadata_only=True,
        ),
    )
    print(
        json.dumps(
            dict(**summary, sealed_specs=seals, commands=str(seal / "COMMANDS.json")),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
