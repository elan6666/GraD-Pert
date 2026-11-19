"""Local GPU lease and admission primitives shared by experiment runners."""

import contextlib
import fcntl
from pathlib import Path


@contextlib.contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns lease {path}") from error
        yield handle.fileno()


def admit_selected_idle(observed: dict, gpu: int, uuid: str) -> None:
    admit_selected_devices(observed, {gpu: uuid})


def admit_selected_devices(observed: dict, selected: dict[int, str]) -> None:
    for gpu, uuid in selected.items():
        device = observed["gpus"].get(gpu)
        if device is None or device["uuid"] != uuid:
            raise RuntimeError("selected GPU identity changed")
        if device["free_mib"] < 2048:
            raise RuntimeError("selected GPU free-memory reserve exhausted")
    uuids = set(selected.values())
    for process in observed["processes"]:
        if process["gpu_uuid"] in uuids and not process["owned"]:
            raise RuntimeError("selected GPU has foreign work; preserve it")
        if process["owned"] and process["gpu_uuid"] not in uuids:
            raise RuntimeError("owned process escaped selected GPU")
