"""Pinned MediaPipe Face Landmarker task file.

The bytes are not vendored and this module never downloads them. The local
script `services/scanlab/scripts/download_face_landmarker.py` fetches the
pinned URL and checks this sha256 before writing the gitignored cache.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from scanlab.cv.support import ScanLabError

FACE_LANDMARKER_VERSION = "face_landmarker/float16/1"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
FACE_LANDMARKER_BYTES = 3_758_596
FACE_LANDMARKER_FILENAME = "face_landmarker.task"


def cache_dir() -> Path:
    override = os.environ.get("SCANLAB_MODEL_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / ".cache" / "models"


def cache_path() -> Path:
    return cache_dir() / FACE_LANDMARKER_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_status(path: Path) -> str:
    """Return ok, missing, size_mismatch, or checksum_mismatch."""

    if not path.is_file():
        return "missing"
    if path.stat().st_size != FACE_LANDMARKER_BYTES:
        return "size_mismatch"
    digest = sha256_file(path)
    if not hmac.compare_digest(digest, FACE_LANDMARKER_SHA256):
        return "checksum_mismatch"
    return "ok"


def verify_model_file(path: Path) -> None:
    status = pin_status(path)
    if status == "ok":
        return
    raise ScanLabError(
        "model_checksum_mismatch",
        "The Face Landmarker file does not match the pinned float16/1 sha256.",
        503,
    )


def resolved_default_model() -> str | None:
    """Use the cache only when the pinned checksum matches. Never download."""

    path = cache_path()
    if pin_status(path) == "ok":
        return str(path)
    return None
