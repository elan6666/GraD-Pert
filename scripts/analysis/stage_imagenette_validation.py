"""Stage exactly the official validation set from verified complete prefix ranges.

The unused training tail need not finish. This does not claim whole-archive
CRC or whole-archive SHA verification: each selected JPEG and the official CSV
are independently checked and hashed instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import tarfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from dinov2_reference import sha256, write_json
from PIL import Image


class JoinedFiles:
    def __init__(self, paths):
        with ExitStack() as stack:
            self.files = [stack.enter_context(Path(p).open("rb")) for p in paths]
            self.resources = stack.pop_all()
        self.index = 0

    def read(self, size):
        chunks = []
        while size and self.index < len(self.files):
            data = self.files[self.index].read(size)
            chunks.append(data)
            size -= len(data)
            if not data:
                self.index += 1
        return b"".join(chunks)

    def close(self):
        self.resources.close()


def stage(parts, output, expected_count=3925, expected_classes=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = [{"path": str(p), "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in parts]
    expected = None
    inventory = {}
    csv_hash = None
    stream = JoinedFiles(parts)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe archive member")
                if not member.isfile():
                    continue
                if path.name == "noisy_imagenette.csv":
                    data = tar.extractfile(member).read()
                    csv_hash = hashlib.sha256(data).hexdigest()
                    records = list(csv.DictReader(io.StringIO(data.decode())))
                    expected = {r["path"] for r in records if r["is_valid"].lower() == "true"}
                    if len(expected) != expected_count:
                        raise ValueError("unexpected official validation count")
                    if any(
                        len(Path(p).parts) != 3
                        or Path(p).parts[0] != "val"
                        or ".." in Path(p).parts
                        for p in expected
                    ):
                        raise ValueError("invalid validation identity")
                    if len({Path(p).parts[1] for p in expected}) != expected_classes:
                        raise ValueError("unexpected class count")
                elif len(path.parts) == 4 and path.parts[1] == "val":
                    key = Path(*path.parts[1:]).as_posix()
                    if expected is None or key not in expected or key in inventory:
                        raise ValueError("unlisted, early or repeated validation image")
                    data = tar.extractfile(member).read()
                    Image.open(io.BytesIO(data)).verify()
                    target = output / key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    inventory[key] = {
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    if set(inventory) == expected:
                        break
    finally:
        stream.close()
    if expected is None or set(inventory) != expected:
        raise ValueError("incomplete validation set; preserve partial staging")
    write_json(output / "server_only_image_inventory.json", inventory)
    receipt = {
        "schema": "imagenette-validation-prefix-v1",
        "source_ranges": source,
        "source_url": "https://fast-ai-imageclas.s3.amazonaws.com/imagenette2-320.tgz",
        "source_etag": '"894c3361e6e919faf86e611f55c90813-41"',
        "official_csv_sha256": csv_hash,
        "validation_images": len(inventory),
        "class_counts": dict(Counter(Path(p).parts[1] for p in inventory)),
        "all_images_verified": True,
        "validation_complete_against_official_csv": True,
        "training_images_extracted": 0,
        "whole_archive_crc_verified": False,
        "whole_archive_sha256": None,
        "inventory_file_sha256": sha256(output / "server_only_image_inventory.json"),
    }
    write_json(output / "COMPLETE.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_sha = [
        "f2822f7f2ab85f214267740d5f7ecf2f70a607f8a53f79f424b00f9ed7453b09",
        "a092ce0a76978b24401e4fa77effb891aaf78d95b325fe804cd264e13861da42",
    ]
    if [sha256(p) for p in args.ranges] != expected_sha:
        raise RuntimeError("prefix differs from independently verified ranges")
    stage(args.ranges, args.output)
