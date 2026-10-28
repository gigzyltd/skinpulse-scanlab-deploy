"""Mock eligibility contract (SP-016), isolated to tests.

The tester path is ``scanlab.server.declaration`` (eligibility contract v2,
source ``self_declared``). This module does not estimate age, detect a face,
or test liveness. There is no provider, no model, and no network call.
Do not present a mock ``adult_verified`` token as real verification.

Capability stays ``verification_required`` / ``NOT_READY``. Unknown and
unavailable are not a pass. A client age, checkbox, boolean, or forged token
is not proof.

Entry points today
------------------
Protected (fail closed before any score or stored bytes):

- ``POST /v1/analyze`` on the deployed Scan Lab API (``scanlab.server.app``).
  Cosmetic optical proxies and Skin Score run only after a server-issued v2
  adult declaration token. In engineering mode only, a mock ``adult_verified``
  token can also reach the score. The default mode never issues a mock token.
- ``POST /v1/contributions`` (and only that write). Photo and metadata are
  stored only after that same kind of decision, and only with separate photo
  consent. A self-declared adult decision is not research consent. A refusal
  leaves the bucket unchanged. ``POST /v1/contributions/delete`` is deletion,
  not intake.

Not a score and not storage:

- ``POST /v1/quality`` — blur, exposure, size, pose, lighting, ROI only.
  Those checks do not establish age or liveness. The photo is not stored.
- ``POST /v1/eligibility`` — ephemeral mock assessment. Bytes are wiped.
  No score and no contribution write.
- ``POST /v1/repeatability`` and ``scanlab.cv.api`` — local CV lab process.
  ``serve_options`` does not publish that app. It has no contribution store.
  It is not this gate. Do not expose it as a public analyze route.

Modes
-----
``enforce`` (default, and the only mode for a shared or public endpoint):
a mock token is ignored. Scoring requires a v2 declaration token from
``AgeDeclarationIssuer``.

``engineering`` (``SCANLAB_ELIGIBILITY_MODE=engineering`` only): an isolated
process may prime a scripted verdict in memory. The mock can then sign a
short-lived token bound to that capture. Nothing in the HTTP request can
choose the verdict. Responses stay labelled NOT verified. This is not a
protection claim and not a remote bypass. A v2 declaration token is still
the primary decision when it is present.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from typing import Callable, Literal

CAPABILITY: Literal["verification_required"] = "verification_required"
READINESS: Literal["NOT_READY"] = "NOT_READY"
PROVIDER: Literal["none"] = "none"

NOT_VERIFIED_LABEL = "NOT verified — mock test contract only, not an age declaration"
NOT_READY_LABEL = "NOT verified — mock test contract only, not an age declaration"

Verdict = Literal[
    "adult_verified",
    "age_uncertain",
    "underage",
    "no_face",
    "multiple_faces",
    "spoof_suspected",
    "verification_unavailable",
]

VERDICTS: frozenset[str] = frozenset({
    "adult_verified",
    "age_uncertain",
    "underage",
    "no_face",
    "multiple_faces",
    "spoof_suspected",
    "verification_unavailable",
})

# Scripted labels for tests. Not measurements from pixels.
_SIGNALS: dict[str, dict[str, str]] = {
    "adult_verified": {
        "face": "single",
        "adult_eligibility": "adult",
        "live_capture": "live",
        "manipulation": "none_indicated",
    },
    "age_uncertain": {
        "face": "single",
        "adult_eligibility": "uncertain",
        "live_capture": "live",
        "manipulation": "none_indicated",
    },
    "underage": {
        "face": "single",
        "adult_eligibility": "underage",
        "live_capture": "live",
        "manipulation": "none_indicated",
    },
    "no_face": {
        "face": "none",
        "adult_eligibility": "unknown",
        "live_capture": "unknown",
        "manipulation": "unknown",
    },
    "multiple_faces": {
        "face": "multiple",
        "adult_eligibility": "unknown",
        "live_capture": "unknown",
        "manipulation": "unknown",
    },
    "spoof_suspected": {
        "face": "single",
        "adult_eligibility": "unknown",
        "live_capture": "not_live",
        "manipulation": "suspected",
    },
    "verification_unavailable": {
        "face": "unknown",
        "adult_eligibility": "unknown",
        "live_capture": "unknown",
        "manipulation": "unknown",
    },
}

_PASS_SIGNALS = _SIGNALS["adult_verified"]

_MESSAGES = {
    "adult_verified": "Mock adult check only. This is not a verified age or live-person result.",
    "age_uncertain": "We could not confirm this photo clearly enough to continue. Nothing was saved.",
    "underage": "This photo cannot be used for a scan. Nothing was saved. You can try again.",
    "no_face": "We could not see one face clearly enough to continue. Nothing was saved.",
    "multiple_faces": "We could see more than one face. Nothing was saved. Try again with one person.",
    "spoof_suspected": "This photo does not look like a live capture. Nothing was saved. Try a new photo.",
    "verification_unavailable": (
        "An age declaration recorded by the server is required before a result or a stored photo. Nothing was saved."
    ),
    "eligibility_invalid": "The test eligibility token could not be confirmed. Nothing was saved.",
    "eligibility_expired": "The test eligibility token expired. Nothing was saved. You can try again.",
    "eligibility_replayed": "This test eligibility token was already used. Nothing was saved.",
    "eligibility_capture_mismatch": "The test eligibility token does not match this photo. Nothing was saved.",
}

TTL_SECONDS = 120
_MAX_TOKEN_CHARS = 4096
_USED_KEEP_SECONDS = TTL_SECONDS + 60

Now = Callable[[], dt.datetime]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(text: str) -> bytes | None:
    try:
        pad = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + pad)
    except Exception:  # noqa: BLE001 — a bad token is invalid, not a crash
        return None


def signals_for(verdict: str) -> dict[str, str]:
    return dict(_SIGNALS[verdict])


def capability_status(mode: str) -> dict:
    """Public capability. Always NOT READY. Never a protection claim."""

    return {
        "capability": CAPABILITY,
        "readiness": READINESS,
        "provider": PROVIDER,
        "mode": mode,
        "mock": True,
        "verified": False,
        "age_model": False,
        "liveness_model": False,
        "label": NOT_VERIFIED_LABEL if mode == "engineering" else NOT_READY_LABEL,
    }


def public_block(verdict: str, signals: dict[str, str], *, request_id: str | None = None) -> dict:
    block = {
        "capability": CAPABILITY,
        "readiness": READINESS,
        "verdict": verdict,
        "signals": signals,
        "mock": True,
        "verified": False,
        "provider": PROVIDER,
        "age_model": False,
        "liveness_model": False,
        "label": NOT_VERIFIED_LABEL,
    }
    if request_id:
        block["request_id"] = request_id
    return block


@dataclass(frozen=True)
class Decision:
    allowed: bool
    error: str
    verdict: str
    signals: dict[str, str]
    message: str

    def public(self, request_id: str | None = None) -> dict:
        return public_block(self.verdict, self.signals, request_id=request_id)


def _refuse(code: str, verdict: str | None = None) -> Decision:
    name = verdict or (code if code in VERDICTS else "verification_unavailable")
    signals = signals_for(name) if name in _SIGNALS else signals_for("verification_unavailable")
    return Decision(False, code if code in _MESSAGES else "verification_unavailable", name, signals, _MESSAGES.get(code, _MESSAGES["verification_unavailable"]))


class MockEligibilityVerifier:
    """In-process mock. ``prime`` is not reachable from HTTP."""

    def __init__(self, mode: str, *, now: Now | None = None, secret: bytes | None = None, ttl_seconds: int = TTL_SECONDS) -> None:
        if mode not in {"enforce", "engineering"}:
            raise ValueError("eligibility mode must be enforce or engineering")
        self.mode = mode
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        self.secret = secret or secrets.token_bytes(32)
        self.ttl_seconds = ttl_seconds
        self._primed: list[str] = []
        self._used: dict[str, float] = {}
        self._lock = threading.Lock()

    def prime(self, verdict: str) -> None:
        """Queue one scripted verdict. Ignored unless mode is engineering. HTTP cannot call this."""

        if verdict not in VERDICTS:
            raise ValueError("unknown eligibility verdict")
        with self._lock:
            self._primed.append(verdict)

    def assess(self) -> tuple[str, dict[str, str]]:
        """Next scripted verdict in engineering mode, otherwise unavailable.

        The image is not an argument. Pixels are not read.
        """

        verdict = "verification_unavailable"
        if self.mode == "engineering":
            with self._lock:
                if self._primed:
                    verdict = self._primed.pop(0)
        return verdict, signals_for(verdict)

    def mint(self, capture_sha256: str, verdict: str, signals: dict[str, str] | None = None) -> str | None:
        """Sign a token for this capture. Engineering mode only. Returns None in enforce mode."""

        if self.mode != "engineering" or verdict not in VERDICTS:
            return None
        signed_signals = dict(signals) if signals is not None else signals_for(verdict)
        now = self.now()
        payload = {
            "v": 1,
            "issuer": "mock",
            "mock": True,
            "verdict": verdict,
            "signals": signed_signals,
            "capture": capture_sha256,
            "nonce": secrets.token_hex(16),
            "jti": secrets.token_hex(16),
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + self.ttl_seconds,
        }
        return self._sign(payload)

    def issue(self, capture_sha256: str, verdict: str) -> str | None:
        return self.mint(capture_sha256, verdict, signals_for(verdict))

    def decide(self, capture_sha256: str, token: str | None) -> Decision:
        """Fail closed. Enforce mode accepts nothing. Engineering accepts one bound mock pass."""

        if self.mode != "engineering":
            return _refuse("verification_unavailable")
        if not token or not isinstance(token, str):
            return _refuse("verification_unavailable")
        if len(token) > _MAX_TOKEN_CHARS or token.count(".") != 1:
            return _refuse("eligibility_invalid")
        payload = self._open(token)
        if payload is None:
            return _refuse("eligibility_invalid")
        now = self.now()
        try:
            exp = int(payload.get("exp"))
            jti = str(payload.get("jti") or "")
            bound = str(payload.get("capture") or "")
            verdict = str(payload.get("verdict") or "")
            issuer = payload.get("issuer")
            raw_signals = payload.get("signals")
        except (TypeError, ValueError):
            return _refuse("eligibility_invalid")
        if issuer != "mock" or payload.get("mock") is not True or payload.get("v") != 1:
            return _refuse("eligibility_invalid")
        if not jti or len(jti) > 64 or verdict not in VERDICTS or not isinstance(raw_signals, dict):
            return _refuse("eligibility_invalid")
        if int(now.timestamp()) >= exp:
            return _refuse("eligibility_expired")
        if len(bound) != len(capture_sha256) or not hmac.compare_digest(bound, capture_sha256):
            return _refuse("eligibility_capture_mismatch")
        with self._lock:
            self._sweep(now.timestamp())
            if jti in self._used:
                return _refuse("eligibility_replayed")
            self._used[jti] = now.timestamp()
        signals = {key: str(raw_signals.get(key, "")) for key in ("face", "adult_eligibility", "live_capture", "manipulation")}
        if verdict != "adult_verified":
            return _refuse(verdict, verdict)
        if signals != _PASS_SIGNALS:
            return _refuse("eligibility_invalid", "verification_unavailable")
        return Decision(True, "adult_verified", "adult_verified", signals, _MESSAGES["adult_verified"])

    def _sign(self, payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        sig = hmac.new(self.secret, raw, hashlib.sha256).digest()
        return _b64(raw) + "." + _b64(sig)

    def _open(self, token: str) -> dict | None:
        body, sig = token.split(".", 1)
        raw = _b64decode(body)
        mac = _b64decode(sig)
        if raw is None or mac is None:
            return None
        expected = hmac.new(self.secret, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected):
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _sweep(self, now_ts: float) -> None:
        stale = [key for key, seen in self._used.items() if now_ts - seen > _USED_KEEP_SECONDS]
        for key in stale:
            del self._used[key]
