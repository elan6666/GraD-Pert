"""Bounded four-request download of the official Imagenette320 reference data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
SIZE = 341663724
ETAG = '"894c3361e6e919faf86e611f55c90813-41"'


def spans(size, parts=4):
    return [(size * i // parts, size * (i + 1) // parts - 1) for i in range(parts)]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def fetch(pair):
        start, end = pair
        path = output / f"range-{start}-{end}"
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("ETag") != ETAG:
                raise RuntimeError("range/ETag changed; preserve files and stop")
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                raise RuntimeError("server returned wrong interval")
            with path.open("xb") as f:
                shutil.copyfileobj(response, f, length=262144)
        if path.stat().st_size != end - start + 1:
            raise RuntimeError("incomplete range")
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, spans(SIZE)))
    archive = output / "imagenette2-320.tgz"
    h = hashlib.sha256()
    with archive.open("xb") as f:
        for path in paths:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1048576), b""):
                    h.update(block)
                    f.write(block)
    assert archive.stat().st_size == SIZE
    selected = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            # Do not materialize the training split; reject unsafe names/links.
            parts = Path(member.name).parts
            if ".." in parts or member.name.startswith("/"):
                raise RuntimeError("unsafe archive name")
            if len(parts) == 4 and parts[:2] == ("imagenette2-320", "val") and member.isfile():
                if Path(member.name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                target = output / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                selected.append(str(target.relative_to(output)))
    if len(selected) < 1335:
        raise RuntimeError("incomplete validation set")
    receipt = {
        "url": URL,
        "etag": ETAG,
        "bytes": SIZE,
        "sha256": h.hexdigest(),
        "range_requests": 4,
        "validation_images": len(selected),
        "training_split_extracted": False,
        "elapsed_seconds": time.monotonic() - started,
        "archive": str(archive),
        "dataset": str(output / "imagenette2-320"),
    }
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
