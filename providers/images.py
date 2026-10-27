"""Local image gates. These reject bytes before any provider request is built.

Dimension checks read JPEG SOF and PNG IHDR only. They do not decode pixels
and they do not decide that a photo is a usable face scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from scanlab.providers.types import ImageRejected

YOUCAM_MAX_BYTES = 10 * 1024 * 1024
YOUCAM_HD_MIN_SHORT_SIDE = 1080
AILAB_MAX_BYTES = 2 * 1024 * 1024
AILAB_MIN_SIDE = 200
AILAB_MAX_SIDE = 4096
GEMINI_MAX_BYTES = 20 * 1024 * 1024

_JPEG_SOF = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


@dataclass(frozen=True)
class ImageFacts:
    mime_type: str
    width: int | None
    height: int | None
    size: int

    @property
    def short_side(self) -> int | None:
        if self.width is None or self.height is None:
            return None
        return min(self.width, self.height)


def inspect_image(data: bytes) -> ImageFacts:
    if not data:
        raise ImageRejected("empty image")
    if data.startswith(b"\xff\xd8"):
        width, height = _jpeg_size(data)
        return ImageFacts("image/jpeg", width, height, len(data))
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = _png_size(data)
        return ImageFacts("image/png", width, height, len(data))
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ImageFacts("image/webp", None, None, len(data))
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"mif1"}:
            return ImageFacts("image/heic", None, None, len(data))
        if brand == b"heif":
            return ImageFacts("image/heif", None, None, len(data))
    raise ImageRejected("unsupported image format")


def require_youcam_hd(data: bytes) -> ImageFacts:
    """HD Skin Analysis limits published by Perfect Corp."""
    facts = inspect_image(data)
    if facts.mime_type not in {"image/jpeg", "image/png"}:
        raise ImageRejected("YouCam HD accepts jpg, jpeg, or png")
    if facts.size >= YOUCAM_MAX_BYTES:
        raise ImageRejected("YouCam image must be under 10 MB")
    if facts.short_side is None or facts.short_side < YOUCAM_HD_MIN_SHORT_SIDE:
        raise ImageRejected("YouCam HD short side must be at least 1080 pixels")
    return facts


def require_ailab(data: bytes) -> ImageFacts:
    """Skin Analyze image limits published by AILabTools."""
    facts = inspect_image(data)
    if facts.mime_type != "image/jpeg":
        raise ImageRejected("AILabTools Skin Analyze accepts JPG or JPEG")
    if facts.size > AILAB_MAX_BYTES:
        raise ImageRejected("AILabTools image must be 2 MB or smaller")
    if facts.width is None or facts.height is None:
        raise ImageRejected("AILabTools image dimensions could not be read")
    for side in (facts.width, facts.height):
        if side <= AILAB_MIN_SIDE or side >= AILAB_MAX_SIDE:
            raise ImageRejected("AILabTools image must be larger than 200 px and smaller than 4096 px")
    return facts


def require_gemini(data: bytes, declared: str | None = None) -> ImageFacts:
    """Inline image limits for the Gemini generateContent vision call."""
    facts = inspect_image(data)
    if facts.mime_type not in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
        raise ImageRejected("Gemini vision accepts png, jpeg, webp, heic, or heif")
    if facts.size >= GEMINI_MAX_BYTES:
        raise ImageRejected("Gemini inline image must be under 20 MB")
    if declared and declared != facts.mime_type and not (
        declared == "image/jpg" and facts.mime_type == "image/jpeg"
    ):
        raise ImageRejected("declared mime type does not match the image bytes")
    return facts


def _jpeg_size(data: bytes) -> tuple[int, int]:
    index = 2
    size = len(data)
    while index + 3 < size:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < size and data[index] == 0xFF:
            index += 1
        if index >= size:
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            break
        if marker == 0xDA:
            break
        if index + 1 >= size:
            break
        segment = int.from_bytes(data[index:index + 2], "big")
        if segment < 2 or index + segment > size:
            raise ImageRejected("truncated jpeg")
        if marker in _JPEG_SOF:
            if segment < 7:
                raise ImageRejected("truncated jpeg")
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            if width <= 0 or height <= 0:
                raise ImageRejected("jpeg dimensions are not usable")
            return width, height
        index += segment
    raise ImageRejected("jpeg dimensions are missing")


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ImageRejected("png IHDR is missing")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        raise ImageRejected("png dimensions are not usable")
    return width, height
