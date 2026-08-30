"""Small, dependency-free host resource probes used by bounded CUDA tooling."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _linux_mem_available_bytes(meminfo_path: Path) -> int | None:
    """Return Linux reclaimable available memory from ``/proc/meminfo``."""

    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    matches = [line for line in lines if line.startswith("MemAvailable:")]
    if len(matches) != 1:
        return None
    fields = matches[0].split()
    if len(fields) != 3 or fields[0] != "MemAvailable:" or fields[2] != "kB":
        return None
    try:
        kilobytes = int(fields[1])
    except ValueError:
        return None
    if kilobytes < 0:
        return None
    return kilobytes * 1024


def host_available_memory_bytes(*, meminfo_path: Path = Path("/proc/meminfo")) -> int | None:
    """Return memory available without treating reclaimable page cache as used.

    Linux ``SC_AVPHYS_PAGES`` reports currently free pages, which can fall near
    zero after hashing large immutable inputs even when the page cache is safely
    reclaimable.  ``MemAvailable`` is the kernel estimate intended for capacity
    decisions.  The POSIX page-count probe remains a fallback for non-Linux
    hosts and unusual Linux environments without a readable procfs.
    """

    if sys.platform.startswith("linux"):
        available = _linux_mem_available_bytes(meminfo_path)
        if available is not None:
            return available
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None
    if pages < 0 or page_size <= 0:
        return None
    return pages * page_size
