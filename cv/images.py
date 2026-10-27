"""In-memory image decoding. Bytes are not logged or written."""

from __future__ import annotations

import hashlib

import numpy as np

from scanlab.cv.support import ScanLabError, ensure_deterministic_cv
from scanlab.cv.versions import MIN_EDGE_PX


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, *, max_edge_px: int, max_bytes: int) -> np.ndarray:
    """Decode PNG or JPEG bytes to contiguous RGB uint8."""

    if not data:
        raise ScanLabError("empty_image", "The image payload is empty.")
    if len(data) > max_bytes:
        raise ScanLabError("payload_too_large", "The image exceeds the size limit.", 413)
    ensure_deterministic_cv()
    import cv2

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ScanLabError("undecodable_image", "The payload is not a readable image.")
    rgb = _to_rgb(array)
    height, width = rgb.shape[:2]
    if min(height, width) < MIN_EDGE_PX:
        raise ScanLabError("image_too_small", "The image is below the minimum edge length.")
    if max(height, width) > max_edge_px:
        raise ScanLabError("image_too_large", "The image exceeds the maximum edge length.")
    return np.ascontiguousarray(rgb)


def encode_png(rgb: np.ndarray) -> bytes:
    ensure_deterministic_cv()
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ScanLabError("encode_failed", "The image could not be encoded.")
    return buffer.tobytes()


def _to_rgb(array: np.ndarray) -> np.ndarray:
    import cv2

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim != 3:
        raise ScanLabError("unsupported_channels", "The image channel layout is unsupported.")
    channels = array.shape[2]
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if channels == 4:
        bgr = array[:, :, :3].astype(np.float64)
        alpha = array[:, :, 3:4].astype(np.float64) / 255.0
        composited = bgr * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    raise ScanLabError("unsupported_channels", "The image channel layout is unsupported.")
