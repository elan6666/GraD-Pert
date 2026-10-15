"""Lossless, non-pickle checkpoints for read-only analysis sufficient statistics.

Store only accumulators, scalar report rows and RNG bytes, never model weights,
per-cell activations or logits. Recovery always targets a fresh analysis root.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from cap40_posthoc_stats import Distribution, Moments


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_recovery(folder, *, binding, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index = state["next_index"]
    target = folder / f"{index:04d}.npz"
    receipt = target.with_suffix(".json")
    if target.exists() or receipt.exists():
        raise FileExistsError("analysis recovery generations are immutable")
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            name = f"array{len(arrays)}"
            arrays[name] = value
            return {"kind": "array", "name": name}
        if isinstance(value, (Moments, Distribution)):
            return {"kind": type(value).__name__, "fields": pack(vars(value))}
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[pack(k), pack(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": "list", "items": [pack(v) for v in value]}
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported recovery type {type(value).__name__}")

    payload = {"binding": binding, "state": pack(state)}
    arrays["metadata"] = np.frombuffer(
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), dtype=np.uint8
    )
    temporary = target.with_suffix(".tmp")
    with temporary.open("xb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(target)
    meta = {
        "schema": "medium-analysis-recovery-1",
        "binding": binding,
        "next_index": index,
        "file": str(target),
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "contains_model_or_cell_matrices": False,
    }
    receipt.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return receipt


def load_recovery(path, *, binding, schedule_size):
    path = Path(path)
    meta = json.loads(path.read_text())
    target = Path(meta["file"])
    if (
        meta.get("schema") != "medium-analysis-recovery-1"
        or meta["binding"] != binding
        or target.parent.resolve() != path.parent.resolve()
        or target.stat().st_size != meta["bytes"]
        or digest(target) != meta["sha256"]
        or not 0 < meta["next_index"] <= schedule_size
    ):
        raise ValueError("analysis recovery identity, bytes or coverage mismatch")
    with np.load(target, allow_pickle=False) as arrays:
        payload = json.loads(arrays["metadata"].tobytes().decode())
        if payload["binding"] != binding:
            raise ValueError("internal recovery binding differs")

        def unpack(value):
            if not isinstance(value, dict):
                return value
            kind = value.get("kind")
            if kind == "array":
                return arrays[value["name"]].copy()
            if kind == "list":
                return [unpack(v) for v in value["items"]]
            if kind == "dict":
                return {unpack(k): unpack(v) for k, v in value["items"]}
            if kind in ("Moments", "Distribution"):
                fields = unpack(value["fields"])
                obj = (
                    Moments(len(fields["mean"]))
                    if kind == "Moments"
                    else Distribution(len(fields["sum"]))
                )
                obj.__dict__.update(fields)
                return obj
            raise ValueError("unknown recovery node")

        state = unpack(payload["state"])
    if state["next_index"] != meta["next_index"]:
        raise ValueError("internal recovery coverage differs")
    return state, meta
