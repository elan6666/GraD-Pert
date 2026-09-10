"""Event-driven legacy checkpoint preservation, without editing active source.

Only atomically published last.pt in the exact supplied checkpoint directory
is linked. The archive is outside the immutable run. Final-epoch identity must
still be checked by the evaluator; a missed final event is never fabricated.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import struct
from pathlib import Path


def preserve(source: Path, archive: Path) -> bool:
    if not source.exists():
        return False
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint source must be a regular file")
    temporary = archive.with_name(".last-link.tmp")
    if temporary.exists():
        raise FileExistsError("stale archival link; preserve and inspect")
    try:
        os.link(source, temporary)
    except FileNotFoundError:
        return False
    os.replace(temporary, archive)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.training_root.resolve(strict=True)
    archive = args.archive_root.resolve()
    if archive.is_relative_to(root) or root.is_relative_to(archive):
        raise ValueError("archive must be outside training evidence")
    archive.mkdir(parents=True, exist_ok=False)
    source = root / "checkpoints/last.pt"
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    wd = libc.inotify_add_watch(fd, os.fsencode(source.parent), 0x80 | 0x200)
    if wd < 0:
        raise OSError(ctypes.get_errno(), "inotify_add_watch")
    count = 0
    try:
        while True:
            if preserve(source, archive / "last.pt"):
                count += 1
            if (root / "small_results/selection_receipt.json").exists():
                print(
                    json.dumps(
                        {
                            "status": "training_terminal_observed",
                            "captures": count,
                            "final_epoch_verified": False,
                        }
                    ),
                    flush=True,
                )
                return
            ready, _, _ = select.select([fd], [], [], 60)
            if ready:
                events = os.read(fd, 65536)
                offset = 0
                while offset + 16 <= len(events):
                    _, mask, _, length = struct.unpack_from("iIII", events, offset)
                    if mask & 0x4000:
                        raise RuntimeError("inotify overflow; archive cannot certify final")
                    offset += 16 + length
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
