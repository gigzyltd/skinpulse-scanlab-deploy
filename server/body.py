"""In-memory multipart reads. Nothing is spooled to disk."""

from __future__ import annotations

import logging

from python_multipart.multipart import MultipartParser, MultipartParseError, parse_options_header
from starlette.requests import Request

from scanlab.server.errors import ApiError

logger = logging.getLogger("scanlab.server")

_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_ALLOWED = {"image/jpeg", "image/png", "image/webp"}
_ALIASES = {"image/jpg": "image/jpeg"}


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


def parse_image(body: bytes, content_type: str | None, *, max_bytes: int) -> MemoryImage:
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
    return MemoryImage(data, media_type)


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
    sniffed = _sniff(data)
    if declared in _ALIASES:
        declared = _ALIASES[declared]
    if declared and declared not in _ALLOWED:
        raise ApiError(400, "unsupported_media_type", "Upload a JPEG, PNG, or WebP photo.")
    if sniffed is None:
        raise ApiError(400, "unsupported_media_type", "Upload a JPEG, PNG, or WebP photo.")
    if declared and declared != sniffed:
        raise ApiError(400, "media_type_mismatch", "The file type does not match its contents.")
    return sniffed


def _sniff(data: bytes) -> str | None:
    if data.startswith(_JPEG):
        return "image/jpeg"
    if data.startswith(_PNG):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
