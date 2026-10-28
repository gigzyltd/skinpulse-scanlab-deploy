"""Server-recorded 18+ age declaration (eligibility contract v2).

This is not an age model, a face check, or liveness. The tester chooses
``18_or_older`` or ``under_18``. The server records that choice and, only for
an adult choice, returns an HMAC token. Exact age, gender, and race are not
accepted and not stored.

A client header or field that merely claims adulthood is not a token. The
signing secret comes from ``SCANLAB_AGE_DECLARATION_SECRET``. With no secret
the issuer fails closed and will not mint or accept a declaration token.
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

from scanlab.server.errors import ApiError

CONTRACT = "eligibility-v2"
SOURCE = "self_declared"
POLICY_VERSION = "age-declaration-v1"
LABEL = "Age declaration only. This is not identity or age verification."
TTL_SECONDS = 900
MIN_SECRET_CHARS = 32
_MAX_TOKEN_CHARS = 4096
_MAX_RECORDS = 4096
_CHOICES = frozenset({"18_or_older", "under_18"})
# Never accepted, stored, or echoed. Exact age and demographics are out of this contract.
_FORBIDDEN = frozenset({
    "age",
    "exact_age",
    "years",
    "dob",
    "date_of_birth",
    "birthdate",
    "birth_date",
    "birthday",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "adult_verified",
    "verified",
})

Now = Callable[[], dt.datetime]
OutcomeStatus = Literal["allow", "reject", "mock"]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(text: str) -> bytes | None:
    try:
        pad = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + pad)
    except Exception:  # noqa: BLE001 — a bad token is invalid, not a crash
        return None


def _iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat()


@dataclass(frozen=True)
class DeclarationOutcome:
    status: OutcomeStatus
    error: str = ""
    message: str = ""
    http_status: int = 403
    public_body: dict | None = None


def _public(*, decision: str, expires_at: str | None) -> dict:
    body = {
        "contract": CONTRACT,
        "source": SOURCE,
        "decision": decision,
        "policy_version": POLICY_VERSION,
        "verified": False,
        "age_verified": False,
        "identity_verified": False,
        "age_model": False,
        "liveness_model": False,
        "label": LABEL,
    }
    if expires_at:
        body["expires_at"] = expires_at
    return body


def _reject(error: str, message: str, *, decision: str = "adult", expires_at: str | None = None, http_status: int = 403) -> DeclarationOutcome:
    return DeclarationOutcome("reject", error, message, http_status, _public(decision=decision, expires_at=expires_at))


class AgeDeclarationIssuer:
    """In-memory record of choices plus an HMAC token for an adult declaration."""

    def __init__(self, secret: bytes | None, *, now: Now | None = None, ttl_seconds: int = TTL_SECONDS) -> None:
        self.secret = secret
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        self.ttl_seconds = ttl_seconds
        self._records: list[dict] = []
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.secret)

    def records(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._records]

    def submit(self, body: object) -> dict:
        """Record one choice. An adult choice returns a token. Under 18 does not."""

        if not self.secret:
            raise ApiError(
                503,
                "declaration_unavailable",
                "An age declaration cannot be recorded right now. The camera stays closed.",
            )
        choice = _choice(body)
        now = self.now()
        expires = now + dt.timedelta(seconds=self.ttl_seconds)
        record = {
            "jti": secrets.token_hex(16),
            "choice": choice,
            "policy_version": POLICY_VERSION,
            "issued_at": _iso(now),
            "expires_at": _iso(expires),
        }
        with self._lock:
            self._records.append(dict(record))
            if len(self._records) > _MAX_RECORDS:
                del self._records[: len(self._records) - _MAX_RECORDS]
        if choice == "under_18":
            public = _public(decision="under_18", expires_at=None)
            public["allowed"] = False
            public["message"] = "Skin Pulse is for people who are 18 or older. The camera stays closed and nothing is uploaded."
            return public
        token = self._sign({
            "v": 2,
            "source": SOURCE,
            "decision": "adult",
            "policy": POLICY_VERSION,
            "jti": record["jti"],
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
        })
        public = _public(decision="adult", expires_at=record["expires_at"])
        public["allowed"] = True
        public["token"] = token
        # Remaining life is a server duration. Clients must not compare expires_at to the phone clock.
        public["issued_at"] = record["issued_at"]
        public["server_now"] = record["issued_at"]
        public["expires_in_seconds"] = self.ttl_seconds
        return public

    def classify(self, token: str | None) -> DeclarationOutcome:
        """Validate a bearer token. Anything that is not a declaration token is ``mock`` (not this contract)."""

        if token is None or token == "":
            return DeclarationOutcome("mock")
        if not isinstance(token, str) or len(token) > _MAX_TOKEN_CHARS or token.count(".") != 1:
            return DeclarationOutcome("mock")
        payload = _peek(token)
        if payload is None or not _declaration_attempt(payload):
            return DeclarationOutcome("mock")
        if not self.secret:
            return _reject(
                "declaration_unavailable",
                "An age declaration cannot be confirmed right now. Nothing was saved.",
                http_status=403,
            )
        opened = self._open(token)
        if opened is None:
            return _reject("declaration_invalid", "The age declaration could not be confirmed. Nothing was saved.")
        try:
            exp = int(opened.get("exp"))
            decision = opened.get("decision")
            source = opened.get("source")
            policy = opened.get("policy")
            version = opened.get("v")
            jti = str(opened.get("jti") or "")
        except (TypeError, ValueError):
            return _reject("declaration_invalid", "The age declaration could not be confirmed. Nothing was saved.")
        if version != 2 or source != SOURCE or policy != POLICY_VERSION or not jti or len(jti) > 64:
            return _reject("declaration_invalid", "The age declaration could not be confirmed. Nothing was saved.")
        now = self.now()
        expires_at = dt.datetime.fromtimestamp(exp, dt.timezone.utc).isoformat()
        remaining = exp - int(now.timestamp())
        if decision != "adult":
            return _reject(
                "declaration_invalid",
                "This age declaration does not allow a photo. Nothing was saved.",
                decision="under_18" if decision == "under_18" else "adult",
                expires_at=expires_at,
            )
        if remaining <= 0:
            return _reject(
                "declaration_expired",
                "The age declaration expired. Nothing was saved. You can declare again.",
                expires_at=expires_at,
            )
        # Drop demographic keys if a future signer ever added them. A token that carries them is invalid.
        if any(key in opened for key in _FORBIDDEN):
            return _reject("declaration_invalid", "The age declaration could not be confirmed. Nothing was saved.")
        issued = opened.get("iat")
        try:
            issued_at = _iso(dt.datetime.fromtimestamp(int(issued), dt.timezone.utc))
        except (TypeError, ValueError):
            issued_at = _iso(now)
        public = _public(decision="adult", expires_at=expires_at)
        public["expires_in_seconds"] = remaining
        public["server_now"] = _iso(now)
        public["issued_at"] = issued_at
        return DeclarationOutcome("allow", public_body=public)

    def _sign(self, payload: dict) -> str:
        assert self.secret is not None
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        sig = hmac.new(self.secret, raw, hashlib.sha256).digest()
        return _b64(raw) + "." + _b64(sig)

    def _open(self, token: str) -> dict | None:
        assert self.secret is not None
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


def _choice(body: object) -> str:
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_request", "Send a declaration choice.")
    lowered = {str(key).lower() for key in body}
    if lowered - {"choice"} or lowered & _FORBIDDEN:
        raise ApiError(400, "invalid_request", "Only a declaration choice is accepted.")
    choice = body.get("choice")
    if choice not in _CHOICES:
        raise ApiError(400, "invalid_request", "Choice must be 18_or_older or under_18.")
    return str(choice)


def _peek(token: str) -> dict | None:
    body, _sig = token.split(".", 1)
    raw = _b64decode(body)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _declaration_attempt(payload: dict) -> bool:
    return payload.get("v") == 2 or payload.get("source") == SOURCE
