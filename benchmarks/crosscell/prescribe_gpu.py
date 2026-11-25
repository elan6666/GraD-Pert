"""Capped, single-device admission for the PRESCRIBE C2 experiment."""

from __future__ import annotations

import os
import time

import torch

from scripts.v2.shared_gpu import snapshot


class SharedGpu:
    def __init__(self, gpu, uuid, config, *, allow_same_account_resident=False):
        self.gpu, self.uuid = gpu, uuid
        self.allocator_mib = config["allocator_mib"]
        self.process_mib = config["process_mib"]
        self.reserve_mib = config["reserve_mib"]
        self.foreign_limit_mib = config["foreign_limit_mib"]
        self.pause_ratio = config["pause_ratio"]
        self.allow_same_account_resident = allow_same_account_resident
        self.max_residents = 2 if allow_same_account_resident else 1
        self.peak_process_mib = 0
        self.min_free_mib = None

    def _read(self):
        for attempt in range(3):
            state = snapshot(os.getpid())
            missing = any(
                process["gpu_uuid"] == self.uuid
                and not process["owned"]
                and process["identity"] is None
                for process in state["processes"]
            )
            if not missing or attempt == 2:
                break
            # A peer can exit between the NVIDIA and process identity reads.
            time.sleep(0.2)
        device = state["gpus"].get(self.gpu)
        if device is None or device["uuid"] != self.uuid:
            raise RuntimeError("selected GPU identity changed")
        foreign, own_mib = {}, 0
        for process in state["processes"]:
            if process["owned"] and process["gpu_uuid"] != self.uuid:
                raise RuntimeError("owned process escaped selected GPU")
            if process["gpu_uuid"] != self.uuid:
                continue
            if process["owned"]:
                own_mib += process["memory_mib"]
                continue
            identity = process["identity"]
            if identity is None or (
                identity["uid"] == os.getuid() and not self.allow_same_account_resident
            ):
                raise RuntimeError("selected GPU coexistence unavailable: unverified resident")
            key = (
                identity["pid"],
                identity["uid"],
                identity["start_ticks"],
                identity["command_sha256"],
            )
            foreign[key] = process["memory_mib"]
        return device, foreign, own_mib

    def admit(self):
        device, foreign, own_mib = self._read()
        if (
            len(foreign) > self.max_residents
            or sum(foreign.values()) > self.foreign_limit_mib
            or device["free_mib"] < self.process_mib + self.reserve_mib
            or own_mib
        ):
            raise RuntimeError(
                "selected GPU coexistence unavailable: insufficient reserved headroom"
            )
        self.residents = set(foreign)
        self.jax_allocator = False
        self.min_free_mib = device["free_mib"]
        self.total_mib = device["total_mib"]

    def configure_allocator(self):
        torch.cuda.set_per_process_memory_fraction(self.allocator_mib / self.total_mib, 0)

    def configure_jax_allocator(self):
        # Configure the BFC pool before JAX initializes the selected device.
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.allocator_mib / self.total_mib)
        self.jax_allocator = True

    def check(self):
        device, foreign, own_mib = self._read()
        self.peak_process_mib = max(self.peak_process_mib, own_mib)
        self.min_free_mib = min(self.min_free_mib, device["free_mib"])
        if (
            (not self.allow_same_account_resident and not foreign.keys() <= self.residents)
            or len(foreign) > self.max_residents
            or sum(foreign.values()) > self.foreign_limit_mib
            or device["free_mib"] < self.reserve_mib
            or own_mib > self.process_mib
        ):
            raise RuntimeError("PRESCRIBE shared-GPU reserve or resident changed")
        return bool(foreign)

    def finish_batch(self, started):
        if not self.jax_allocator:
            torch.cuda.synchronize()
        if self.check():
            time.sleep(self.pause_ratio * (time.monotonic() - started))

    def evidence(self):
        self.check()
        mib = 1024**2
        return {
            "allocator_cap_mib": self.allocator_mib,
            "process_limit_mib": self.process_mib,
            "free_reserve_mib": self.reserve_mib,
            "minimum_observed_free_mib": self.min_free_mib,
            "peak_observed_process_mib": self.peak_process_mib,
            **(
                {}
                if self.jax_allocator
                else {
                    "peak_pytorch_reserved_mib": torch.cuda.max_memory_reserved() / mib,
                    "peak_pytorch_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                }
            ),
            "resident_count_at_admission": len(self.residents),
        }


def scbig_shared_gpu(gpu, uuid):
    """Bound the frozen scBIG JAX pool and keep headroom for verified peers."""
    return SharedGpu(
        gpu,
        uuid,
        {
            "allocator_mib": 8192,
            "process_mib": 9216,
            "reserve_mib": 4096,
            "foreign_limit_mib": 18432,
            "pause_ratio": 1.0,
        },
        allow_same_account_resident=True,
    )
