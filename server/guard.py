"""API key, rate limit, and in-memory idempotency. Photos are not stored."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

from starlette.requests import Request

from scanlab.server.config import ServerSettings
from scanlab.server.errors import ApiError

_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class CachedResponse:
    status_code: int
    payload: dict


class IdempotencyConflict(Exception):
    pass


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, *, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        moment = time.monotonic() if now is None else now
        with self._lock:
            if len(self._events) > self.max_keys:
                self._events.clear()
            bucket = self._events.setdefault(key, deque())
            while bucket and moment - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry = int(self.window_seconds - (moment - bucket[0])) + 1
                return False, max(retry, 1)
            bucket.append(moment)
            return True, 0


class IdempotencyStore:
    """Remember status and JSON only. The image hash is the conflict check."""

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[tuple[str, str, str], tuple[float, str, CachedResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def lookup(self, scope: str, key: str, path: str, digest: str, now: float | None = None) -> CachedResponse | None:
        moment = time.monotonic() if now is None else now
        token = (scope, path, key)
        with self._lock:
            self._prune(moment)
            found = self._items.get(token)
            if found is None:
                return None
            _stored_at, previous, cached = found
            if not hmac.compare_digest(previous, digest):
                raise IdempotencyConflict()
            self._items.move_to_end(token)
            return cached

    def save(self, scope: str, key: str, path: str, digest: str, cached: CachedResponse, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        token = (scope, path, key)
        with self._lock:
            self._prune(moment)
            self._items[token] = (moment, digest, cached)
            self._items.move_to_end(token)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def _prune(self, now: float) -> None:
        expired = [token for token, (stored_at, _digest, _cached) in self._items.items() if now - stored_at >= self.ttl_seconds]
        for token in expired:
            del self._items[token]


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if value in _LOOPBACK:
        return True
    if value.startswith("::ffff:"):
        return value.removeprefix("::ffff:") == "127.0.0.1"
    return False


def peer_host(request: Request) -> str:
    client = request.client
    if client is None or not client.host:
        return ""
    return client.host


def authorize(request: Request, settings: ServerSettings) -> str:
    """Return a cache scope. Loopback may omit the key; other peers may not.

    X-Forwarded-For is ignored here. A header must not turn a remote peer
    into loopback.
    """

    host = peer_host(request)
    local = is_loopback(host) and not settings.require_api_key
    provided = _presented_key(request)
    if local and provided is None:
        return "loopback"
    expected = settings.api_key or ""
    if not expected or provided is None or not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise ApiError(401, "unauthorized", "A valid API key is required.")
    digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]
    return "key:" + digest


def enforce_rate_limit(request: Request, settings: ServerSettings, limiter: SlidingWindowLimiter) -> None:
    allowed, retry = limiter.allow(_rate_key(request, settings))
    if not allowed:
        raise ApiError(
            429,
            "rate_limited",
            "Too many requests. Try again later.",
            headers={"Retry-After": str(retry)},
        )


def idempotency_key(request: Request) -> str | None:
    if "idempotency-key" not in request.headers:
        return None
    raw = request.headers.get("idempotency-key", "").strip()
    if not _KEY.fullmatch(raw):
        raise ApiError(400, "invalid_idempotency_key", "Idempotency-Key must be 1-128 letters, digits, or . _ : -.")
    return raw


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization")
    bearer: str | None = None
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise ApiError(401, "unauthorized", "A valid API key is required.")
        bearer = token.strip()
    if header is not None:
        header = header.strip()
        if not header:
            raise ApiError(401, "unauthorized", "A valid API key is required.")
    if header is not None and bearer is not None and not hmac.compare_digest(header.encode("utf-8"), bearer.encode("utf-8")):
        raise ApiError(401, "unauthorized", "A valid API key is required.")
    if header is not None:
        return header
    return bearer


def _rate_key(request: Request, settings: ServerSettings) -> str:
    host = peer_host(request) or "unknown"
    if not settings.trust_proxy:
        return host
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if not first or len(first) > 128 or any(ord(char) < 32 or ord(char) == 127 for char in first):
        return host
    return first
