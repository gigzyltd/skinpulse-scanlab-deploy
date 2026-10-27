"""HTTP transports.

The default transport refuses to open a socket. Tests pass a scripted
transport. A live call is possible only when allow_network is set on purpose.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from scanlab.providers.types import NetworkDisabled


@dataclass
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None = None


@dataclass
class HttpResponse:
    status: int
    body: bytes
    latency_ms: float
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        parsed = json.loads(self.body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("response JSON must be an object")
        return parsed


class DisabledTransport:
    """Used unless a caller injects a script or opts into the network."""

    def send(self, request: HttpRequest) -> HttpResponse:
        raise NetworkDisabled(
            "Live provider calls are disabled. Pass a scripted transport. "
            "allow_network spends provider units or credits and is not used by tests."
        )


class ScriptedTransport:
    """Replay status, body, and latency. This never opens a socket."""

    def __init__(self, steps: list[tuple[int, dict | bytes | str, float]]):
        self._steps = list(steps)
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if not self._steps:
            raise NetworkDisabled("scripted transport has no response left")
        status, payload, latency_ms = self._steps.pop(0)
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = payload
        return HttpResponse(status=status, body=body, latency_ms=float(latency_ms))


class UrllibTransport:
    """Opt-in HTTP. Tests must inject urlopen; they must not use the network."""

    def __init__(self, *, allow_network: bool = False, urlopen=None, timeout: float = 60):
        if urlopen is None and not allow_network:
            raise NetworkDisabled(
                "UrllibTransport refuses to call the network unless allow_network is true."
            )
        self._urlopen = urlopen if urlopen is not None else urllib.request.urlopen
        self._timeout = timeout

    def send(self, request: HttpRequest) -> HttpResponse:
        started = time.perf_counter()
        outgoing = urllib.request.Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        try:
            with self._urlopen(outgoing, timeout=self._timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
        elapsed = (time.perf_counter() - started) * 1000
        return HttpResponse(status=status, body=raw, latency_ms=elapsed)
