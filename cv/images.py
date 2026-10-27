"""In-memory image decoding. Bytes are not logged or written.

Capture preparation applies EXIF orientation, downscales the longer edge to
the working maximum, and returns pixels only. The numpy array has no EXIF
and no GPS. This module does not contact a network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from scanlab.cv.support import ScanLabError, ensure_deterministic_cv
from scanlab.cv.versions import ABSOLUTE_MAX_SOURCE_EDGE_PX, MIN_EDGE_PX

_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_SWAP_ORIENTATION = {5, 6, 7, 8}


@dataclass(frozen=True)
class PreparedCapture:
    rgb: np.ndarray
    orientation: int
    source_width: int
    source_height: int
    downscaled: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_capture(data: bytes, *, max_edge_px: int, max_bytes: int) -> PreparedCapture:
    """Decode, orient, and downscale. The result carries no metadata."""

    if not data:
        raise ScanLabError("empty_image", "The image payload is empty.")
    if max_edge_px < 1 or max_bytes < 1:
        raise ScanLabError("invalid_limits", "The image limits are not usable.")
    if len(data) > max_bytes:
        raise ScanLabError("payload_too_large", "The image exceeds the size limit.", 413)
    orientation, _gps = read_orientation_and_gps(data)
    header = _header_size(data)
    if header is not None:
        oriented_width, oriented_height = _oriented_size(header[0], header[1], orientation)
        if max(oriented_width, oriented_height) > ABSOLUTE_MAX_SOURCE_EDGE_PX:
            raise ScanLabError("image_too_large", "The image exceeds the maximum source edge.")
    ensure_deterministic_cv()
    import cv2

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ScanLabError("undecodable_image", "The payload is not a readable image.")
    rgb = apply_orientation(_to_rgb(array), orientation)
    height, width = rgb.shape[:2]
    if max(height, width) > ABSOLUTE_MAX_SOURCE_EDGE_PX:
        raise ScanLabError("image_too_large", "The image exceeds the maximum source edge.")
    source_width, source_height = width, height
    downscaled = max(height, width) > max_edge_px
    if downscaled:
        rgb = downscale_longer_edge(rgb, max_edge_px)
    out_height, out_width = rgb.shape[:2]
    if min(out_height, out_width) < MIN_EDGE_PX:
        raise ScanLabError("image_too_small", "The image is below the minimum edge length.")
    if max(out_height, out_width) > max_edge_px:
        raise ScanLabError("image_too_large", "The image exceeds the maximum edge length.")
    return PreparedCapture(
        rgb=rgb,
        orientation=orientation,
        source_width=source_width,
        source_height=source_height,
        downscaled=downscaled,
    )


def decode_image(data: bytes, *, max_edge_px: int, max_bytes: int) -> np.ndarray:
    """Decode PNG or JPEG bytes to contiguous RGB uint8."""

    return prepare_capture(data, max_edge_px=max_edge_px, max_bytes=max_bytes).rgb


def downscale_longer_edge(rgb: np.ndarray, max_edge_px: int) -> np.ndarray:
    """Shrink so the longer edge equals max_edge_px. Never upscales."""

    height, width = rgb.shape[:2]
    longer = max(height, width)
    if longer <= max_edge_px:
        return np.ascontiguousarray(rgb)
    if width >= height:
        new_width = max_edge_px
        new_height = max(1, int(round(height * max_edge_px / width)))
    else:
        new_height = max_edge_px
        new_width = max(1, int(round(width * max_edge_px / height)))
    ensure_deterministic_cv()
    import cv2

    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def apply_orientation(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation tag. Unknown values leave the pixels unchanged."""

    if orientation == 2:
        return np.ascontiguousarray(rgb[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(rgb[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(rgb[::-1, :])
    if orientation == 5:
        return np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    if orientation == 6:
        return np.ascontiguousarray(np.rot90(rgb, k=-1))
    if orientation == 7:
        transposed = np.transpose(rgb, (1, 0, 2))
        return np.ascontiguousarray(transposed[::-1, ::-1])
    if orientation == 8:
        return np.ascontiguousarray(np.rot90(rgb, k=1))
    return np.ascontiguousarray(rgb)


def read_orientation_and_gps(data: bytes) -> tuple[int, bool]:
    """Return the EXIF orientation (1 when absent) and whether a GPS IFD is present."""

    tiff = _exif_tiff(data)
    if tiff is None:
        return 1, False
    return _tiff_orientation_and_gps(tiff)


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


def _oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    if orientation in _SWAP_ORIENTATION:
        return height, width
    return width, height


def _header_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24 and data[12:16] == b"IHDR":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width > 0 and height > 0:
            return width, height
        return None
    if not data.startswith(b"\xff\xd8"):
        return None
    for marker, payload in _jpeg_segments(data):
        if marker in _JPEG_SOF and len(payload) >= 5:
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            if width > 0 and height > 0:
                return width, height
    return None


def _exif_tiff(data: bytes) -> bytes | None:
    if data.startswith(b"\xff\xd8"):
        for marker, payload in _jpeg_segments(data):
            if marker == 0xE1 and payload.startswith(b"Exif\x00\x00") and len(payload) > 6:
                return payload[6:]
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png_exif(data)
    return None


def _png_exif(data: bytes) -> bytes | None:
    offset = 8
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if length < 0 or end + 4 > len(data):
            return None
        if chunk_type == b"eXIf":
            return data[start:end]
        if chunk_type == b"IEND":
            return None
        offset = end + 4
    return None


def _jpeg_segments(data: bytes):
    offset = 2
    size = len(data)
    while offset + 1 < size:
        if data[offset] != 0xFF:
            return
        offset += 1
        while offset < size and data[offset] == 0xFF:
            offset += 1
        if offset >= size:
            return
        marker = data[offset]
        offset += 1
        if marker in {0xD9, 0xDA}:
            return
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > size:
            return
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > size:
            return
        yield marker, data[offset + 2 : offset + length]
        offset += length


def _tiff_orientation_and_gps(tiff: bytes) -> tuple[int, bool]:
    if len(tiff) < 8:
        return 1, False
    order: Literal["little", "big"]
    if tiff[:2] == b"II":
        order = "little"
    elif tiff[:2] == b"MM":
        order = "big"
    else:
        return 1, False

    def u16(pos: int) -> int:
        return int.from_bytes(tiff[pos : pos + 2], order)

    def u32(pos: int) -> int:
        return int.from_bytes(tiff[pos : pos + 4], order)

    if u16(2) != 42:
        return 1, False
    ifd = u32(4)
    orientation = 1
    gps = False
    seen = 0
    while 0 < ifd < len(tiff) - 1 and seen < 8:
        seen += 1
        if ifd + 2 > len(tiff):
            break
        count = u16(ifd)
        entry = ifd + 2
        for _ in range(count):
            if entry + 12 > len(tiff):
                break
            tag = u16(entry)
            typ = u16(entry + 2)
            count_values = u32(entry + 4)
            if tag == 0x0112 and typ == 3 and count_values >= 1:
                orientation = u16(entry + 8)
            if tag == 0x8825:
                gps = True
            entry += 12
        if entry + 4 > len(tiff):
            break
        ifd = u32(entry)
    if orientation not in range(1, 9):
        orientation = 1
    return orientation, gps
