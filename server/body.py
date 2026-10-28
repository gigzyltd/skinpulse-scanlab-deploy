"""In-memory multipart reads. Nothing is spooled to disk."""

from __future__ import annotations

import logging

from python_multipart.multipart import MultipartParser, MultipartParseError, parse_options_header
from starlette.requests import Request

from scanlab.cv.versions import ABSOLUTE_MAX_SOURCE_EDGE_PX
from scanlab.server.errors import ApiError

logger = logging.getLogger("scanlab.server")

_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"


class MemoryImage:
    def __init__(self, data: bytes, media_type: str) -> None:
        self.data = data
        self.media_type = media_type


async def read_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            size = int(declared)
        except ValueError as exc:
            raise ApiError(400, "invalid_request", "Content-Length is not an integer.") from exc
        if size < 0:
            raise ApiError(400, "invalid_request", "Content-Length is invalid.")
        if size > limit:
            raise ApiError(413, "payload_too_large", "The upload exceeds the size limit.")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            chunks.clear()
            raise ApiError(413, "payload_too_large", "The upload exceeds the size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_image(body: bytes, content_type: str | None, *, max_bytes: int, normalize: bool = True) -> MemoryImage:
    """Read the single 'file' part. With normalize=False the caller must run
    normalize_upload() itself (the server does it off the event loop)."""
    if not content_type:
        raise ApiError(400, "unsupported_media_type", "Send the photo as multipart/form-data.")
    kind, options = parse_options_header(content_type)
    if kind != b"multipart/form-data":
        raise ApiError(400, "unsupported_media_type", "Send the photo as multipart field 'file'.")
    boundary = options.get(b"boundary")
    if not boundary or not body:
        raise ApiError(400, "invalid_multipart", "The multipart body is incomplete.")
    file_headers: list[tuple[str, str]] = []
    file_data = bytearray()
    saw_file = False
    header_field = bytearray()
    header_value = bytearray()
    current_headers: list[tuple[str, str]] = []
    current = bytearray()
    is_file = False

    def on_part_begin() -> None:
        nonlocal is_file
        current_headers.clear()
        current.clear()
        header_field.clear()
        header_value.clear()
        is_file = False

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_field.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        current_headers.append((bytes(header_field).decode("latin-1"), bytes(header_value).decode("latin-1")))
        header_field.clear()
        header_value.clear()

    def on_headers_finished() -> None:
        nonlocal is_file, saw_file
        name = _field_name(current_headers)
        if name != "file":
            raise ApiError(400, "unexpected_field", "The only accepted field is 'file'.")
        if saw_file:
            raise ApiError(400, "unexpected_field", "Send one photo.")
        is_file = True
        saw_file = True
        file_headers.extend(current_headers)

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if not is_file:
            return
        piece = data[start:end]
        if len(current) + len(piece) > max_bytes:
            current.clear()
            raise ApiError(413, "payload_too_large", "The upload exceeds the size limit.")
        current.extend(piece)

    def on_part_end() -> None:
        if is_file:
            file_data.extend(current)
            current.clear()

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
        max_size=max(len(body), 1),
        max_header_count=8,
        max_header_size=1024,
    )
    parser.logger.setLevel(logging.CRITICAL)
    try:
        parser.write(body)
        parser.finalize()
    except ApiError:
        file_data.clear()
        current.clear()
        raise
    except MultipartParseError as exc:
        file_data.clear()
        raise ApiError(400, "invalid_multipart", "The multipart body could not be read.") from exc
    finally:
        # Drop the caller's copy as soon as the part bytes are decided.
        del body
    if not saw_file:
        raise ApiError(400, "missing_file", "The multipart body needs a 'file' field.")
    data = bytes(file_data)
    file_data.clear()
    media_type = _check_media(data, _part_media_type(file_headers))
    image = MemoryImage(data, media_type)
    return normalize_upload(image) if normalize else image


def normalize_upload(image: MemoryImage) -> MemoryImage:
    """CPU-bound: call from a worker thread, never directly on the event loop."""
    image.data, image.media_type = normalize_image(image.data, image.media_type)
    return image


def _field_name(headers: list[tuple[str, str]]) -> str:
    for name, value in headers:
        if name.lower() != "content-disposition":
            continue
        _kind, options = parse_options_header(value)
        raw = options.get(b"name", b"").decode("latin-1")
        return raw
    return ""


def _part_media_type(headers: list[tuple[str, str]]) -> str | None:
    for name, value in headers:
        if name.lower() == "content-type":
            kind, _options = parse_options_header(value)
            text = kind.decode("latin-1").strip().lower()
            return text or None
    return None


def _check_media(data: bytes, declared: str | None) -> str:
    """Decide the format from the bytes. The declared part Content-Type is advisory only:
    phones often send ``image/jpg``, ``application/octet-stream`` or a stale HEIC label."""
    sniffed = _sniff(data)
    if sniffed is None:
        if declared:
            logger.info("rejected upload declared=%s (bytes not a supported image)", declared[:40])
        raise ApiError(400, "unsupported_media_type", "Upload a JPEG, PNG, WebP or HEIC photo.")
    return sniffed


# Decoded-size budget, checked from the header BEFORE any pixel decode. A byte limit
# does not bound decoded allocation (a small HEIC/JPEG can declare a huge canvas).
# 50 MP covers 48 MP phone sensors; the edge cap matches the CV pipeline's.
MAX_DECODE_PIXELS = 50_000_000
MAX_DECODE_EDGE_PX = ABSOLUTE_MAX_SOURCE_EDGE_PX


def _check_decode_budget(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ApiError(400, "undecodable_image", "The photo could not be decoded.")
    if max(width, height) > MAX_DECODE_EDGE_PX or width * height > MAX_DECODE_PIXELS:
        raise ApiError(400, "image_too_large", "The photo exceeds the maximum pixel size.")


_HEIF_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1"}


def _sniff(data: bytes) -> str | None:
    if data.startswith(_JPEG):
        return "image/jpeg"
    if data.startswith(_PNG):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brands = {data[8:12]}
        box = int.from_bytes(data[:4], "big")
        if 16 <= box <= min(len(data), 256):
            brands.update(data[i:i + 4] for i in range(16, box - 3, 4))
        if brands & _HEIF_BRANDS:
            return "image/heif"
    return None


def normalize_image(data: bytes, media_type: str) -> tuple[bytes, str]:
    """Decode HEIC/HEIF to JPEG and bake EXIF orientation into pixels.

    OpenCV ignores HEIF and (with IMREAD_UNCHANGED) EXIF orientation, so we hand it an
    upright JPEG/PNG/WebP. Undecodable bytes are returned unchanged and rejected downstream.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover
        return data, media_type
    import io

    if media_type == "image/heif":
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError as exc:  # pragma: no cover
            raise ApiError(400, "unsupported_media_type", "HEIC decoding is unavailable.") from exc
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Image.open() reads only the header; enforce the budget before load().
            _check_decode_budget(*img.size)
            orientation = img.getexif().get(0x0112, 1)
            if media_type != "image/heif" and orientation in (None, 1):
                return data, media_type
            img.load()
            upright = ImageOps.exif_transpose(img)
            if upright.mode not in ("RGB", "L"):
                upright = upright.convert("RGB")
            out = io.BytesIO()
            upright.save(out, format="JPEG", quality=92)
            return out.getvalue(), "image/jpeg"
    except ApiError:
        raise
    except Exception:
        if media_type == "image/heif":
            raise ApiError(400, "undecodable_image", "The HEIC photo could not be decoded.") from None
        return data, media_type
