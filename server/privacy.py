"""Logging helpers that refuse to record photo bytes."""

from __future__ import annotations

import logging
import re

_DIGIT = re.compile(r"\d")
_KNOWN_PATHS = frozenset({
    "/healthz",
    "/v1/quality",
    "/v1/analyze",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
})


class RedactBytesFilter(logging.Filter):
    """Drop byte payloads before a log record is formatted."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, (bytes, bytearray)):
            record.msg = "[redacted-bytes]"
            record.args = ()
            return True
        record.args = _redact_args(record.args)
        if isinstance(record.msg, str) and _looks_like_image(record.msg):
            record.msg = "[redacted-image]"
            record.args = ()
        return True


def install_redaction() -> None:
    root = logging.getLogger()
    if not any(isinstance(item, RedactBytesFilter) for item in root.filters):
        root.addFilter(RedactBytesFilter())
    for name in ("python_multipart", "multipart", "scanlab.server"):
        logging.getLogger(name).addFilter(RedactBytesFilter())
    logging.getLogger("python_multipart").setLevel(logging.ERROR)
    logging.getLogger("multipart").setLevel(logging.ERROR)


def public_path(path: str) -> str:
    if path in _KNOWN_PATHS:
        return path
    return "[unlisted]"


def public_explanation(text: str | None) -> str | None:
    """Keep short prose. Withhold anything that contains a digit."""

    if not text:
        return None
    kept = []
    for char in text:
        code = ord(char)
        if char in "\n\t" or code >= 32:
            kept.append(char)
    cleaned = "".join(kept).strip()
    if len(cleaned) > 1000:
        cleaned = cleaned[:1000].rstrip()
    if not cleaned or _DIGIT.search(cleaned):
        return None
    return cleaned


def _redact_args(args: object) -> object:
    if isinstance(args, dict):
        return {key: _redact_value(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_redact_value(value) for value in args)
    return args


def _redact_value(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return "[redacted-bytes]"
    if isinstance(value, str) and _looks_like_image(value):
        return "[redacted-image]"
    return value


def _looks_like_image(value: str) -> bool:
    return "\x89PNG" in value or "\xff\xd8\xff" in value or "iVBORw0KGgo" in value
