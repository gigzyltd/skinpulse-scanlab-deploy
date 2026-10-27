#!/usr/bin/env python3
"""Download the pinned MediaPipe Face Landmarker task file.

The file is written only after the sha256 matches the pin in
scanlab.cv.model_pin. The destination is the gitignored cache
services/scanlab/.cache/models/. This script is not used by tests or by
the analysis runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SERVICES = Path(__file__).resolve().parents[2]
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))

from scanlab.cv.model_pin import (  # noqa: E402
    FACE_LANDMARKER_BYTES,
    FACE_LANDMARKER_SHA256,
    FACE_LANDMARKER_URL,
    FACE_LANDMARKER_VERSION,
    cache_path,
    pin_status,
    verify_model_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the pinned Face Landmarker task file.")
    parser.add_argument("--check", action="store_true", help="Verify the cache and do not download.")
    parser.add_argument("--print-pin", action="store_true", help="Print the pinned URL and sha256. No network.")
    args = parser.parse_args(argv)
    if args.print_pin:
        print(f"{FACE_LANDMARKER_VERSION} {FACE_LANDMARKER_SHA256} {FACE_LANDMARKER_BYTES}")
        print(FACE_LANDMARKER_URL)
        return 0
    destination = cache_path()
    status = pin_status(destination)
    if args.check:
        if status != "ok":
            print(f"model pin check failed: {status}", file=sys.stderr)
            return 1
        print(destination)
        return 0
    if status == "ok":
        print(destination)
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="face_landmarker-", suffix=".partial", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        request = urllib.request.Request(FACE_LANDMARKER_URL, headers={"User-Agent": "skinpulse-scanlab-model-pin"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        verify_model_file(temporary)
        temporary.replace(destination)
    except Exception as exc:  # noqa: BLE001 — download or checksum failure
        temporary.unlink(missing_ok=True)
        print(f"model download failed: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
