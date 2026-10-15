"""Recognize one exact same-project analysis peer without changing GPU budgets."""

from __future__ import annotations

import copy
import os

from shared_gpu import admit, process_info


def dispatcher_owner(receipt, *, lookup=process_info, parent=None, uid=None):
    """Both slots share their exact live parent for every resource snapshot."""
    uid = os.getuid() if uid is None else uid
    parent = os.getppid() if parent is None else parent
    expected = receipt["process"]
    actual = lookup(expected["pid"])
    if (
        receipt.get("schema") != "cap40-analysis-owner-1"
        or receipt.get("uid") != uid
        or expected["pid"] != parent
        or actual is None
        or actual.get("uid") != uid
        or actual.get("script_basename") != "cap40_posthoc_dual.py"
        or any(actual[k] != expected[k] for k in ("pid", "start_ticks", "command_sha256"))
    ):
        raise ValueError("analysis dispatcher parent identity changed")
    return parent


def admit_analysis(policy, observed, own_gpu, peer=None, *, lookup=process_info, uid=None):
    uid = os.getuid() if uid is None else uid
    if own_gpu not in (0, 1):
        raise ValueError("analysis needs one physical GPU")
    view = copy.deepcopy(observed)
    if peer is not None:
        if (
            peer.get("schema") != "cap40-exact-analysis-peer-1"
            or peer.get("gpu") not in (0, 1)
            or peer["gpu"] == own_gpu
            or peer.get("arm") not in ("DP", "FN", "FG", "FB", "PO")
            or peer.get("uid") != uid
        ):
            raise ValueError("peer must be our exact analysis on the other GPU")
        expected = peer["process"]
        actual = lookup(expected["pid"])
        if actual is not None and (
            actual["uid"] != uid
            or actual.get("script_basename") != "cap40_posthoc.py"
            or any(actual[k] != expected[k] for k in ("pid", "start_ticks", "command_sha256"))
        ):
            raise ValueError("analysis peer PID identity changed")
        devices = {d["gpu_index"]: d["gpu_uuid"] for d in policy["devices"]}
        for entry in view["processes"]:
            if entry["pid"] == expected["pid"]:
                info = entry.get("identity")
                if (
                    actual is None
                    or entry["gpu_uuid"] != devices[peer["gpu"]]
                    or info is None
                    or info["uid"] != uid
                    or any(info[k] != expected[k] for k in expected)
                ):
                    raise ValueError("peer GPU telemetry identity differs")
                entry["owned"] = True
    # Includes the peer's memory in the original per-device own budget. Unknown
    # jobs retain foreign classification; no whitelist or headroom is widened.
    admit(policy, view)
    return view
