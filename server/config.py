"""Process configuration. Flags default off. Values are names, never logged secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from scanlab.cv.pipeline import Settings
from scanlab.cv.versions import DEFAULT_MAX_BYTES

_TRUE = {"1", "true", "yes", "on"}
_MAX_CONFIGURED_BYTES = 20_000_000


class ConfigError(ValueError):
    pass


def env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE


def load_age_declaration_secret(env: dict[str, str] | os._Environ[str]) -> bytes | None:
    """HMAC key for eligibility contract v2. Missing in production fails closed at startup.

    Tests pass a secret in directly. There is no built-in production secret.
    """

    raw = str(env.get("SCANLAB_AGE_DECLARATION_SECRET", "")).strip()
    production = str(env.get("SCANLAB_ENV", "")).strip().lower() in {"production", "prod"}
    if not raw:
        if production:
            raise ConfigError("SCANLAB_AGE_DECLARATION_SECRET is required when SCANLAB_ENV is production")
        return None
    if len(raw) < 32:
        raise ConfigError("SCANLAB_AGE_DECLARATION_SECRET must be at least 32 characters")
    return raw.encode("utf-8")


def parse_eligibility_mode(raw: str | None) -> Literal["enforce", "engineering"]:
    """Default is enforce. Only the exact word engineering opens the isolated mock issuer.

    allow, off, observe, and any other value are refused at startup so a typo cannot pass.
    """

    value = (raw or "").strip().lower()
    if value in {"", "enforce"}:
        return "enforce"
    if value == "engineering":
        return "engineering"
    raise ConfigError("SCANLAB_ELIGIBILITY_MODE must be enforce or engineering")


def parse_origins(raw: str | None) -> tuple[str, ...]:
    """Comma-separated origins. A wildcard is rejected."""

    if not raw or not raw.strip():
        return ()
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip()
        if not origin:
            continue
        if origin == "*":
            raise ConfigError("SCANLAB_CORS_ORIGINS must not be *")
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError("SCANLAB_CORS_ORIGINS entries must be http(s) origins")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise ConfigError("SCANLAB_CORS_ORIGINS entries must not include a path, query, or user")
        origins.append(f"{parsed.scheme}://{parsed.netloc}")
    return tuple(origins)


def _positive_int(source: os._Environ[str] | dict[str, str], name: str, default: int, *, ceiling: int | None = None) -> int:
    raw = source.get(name)
    if raw is None or not str(raw).strip():
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc
    if value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    if ceiling is not None and value > ceiling:
        raise ConfigError(f"{name} exceeds {ceiling}")
    return value


@dataclass(frozen=True)
class ServerSettings:
    cv: Settings
    api_key: str | None
    require_api_key: bool
    cors_origins: tuple[str, ...]
    rate_limit: int
    rate_window_seconds: float
    enable_youcam: bool
    enable_ailabtools: bool
    enable_gemini: bool
    gemini_allow_image: bool
    trust_proxy: bool
    idempotency_ttl_seconds: float
    idempotency_max_entries: int
    eligibility_mode: Literal["enforce", "engineering"] = "enforce"
    age_declaration_secret: bytes | None = None

    @staticmethod
    def from_env(source: dict[str, str] | None = None) -> "ServerSettings":
        env = os.environ if source is None else source
        key = env.get("SCANLAB_API_KEY", "").strip()
        window = _positive_int(env, "SCANLAB_RATE_WINDOW_SECONDS", 60, ceiling=3600)
        return ServerSettings(
            cv=Settings(
                landmark_backend=env.get("SCANLAB_LANDMARK_BACKEND", "mediapipe"),
                face_landmarker_model=env.get("SCANLAB_FACE_LANDMARKER_MODEL") or None,
                max_edge_px=_positive_int(env, "SCANLAB_MAX_EDGE_PX", Settings().max_edge_px, ceiling=8000),
                max_bytes=_positive_int(env, "SCANLAB_MAX_BYTES", DEFAULT_MAX_BYTES, ceiling=_MAX_CONFIGURED_BYTES),
            ),
            api_key=key or None,
            require_api_key=env_flag(env.get("SCANLAB_REQUIRE_API_KEY")),
            cors_origins=parse_origins(env.get("SCANLAB_CORS_ORIGINS")),
            rate_limit=_positive_int(env, "SCANLAB_RATE_LIMIT", 30, ceiling=10_000),
            rate_window_seconds=float(window),
            enable_youcam=env_flag(env.get("SCANLAB_ENABLE_YOUCAM")),
            enable_ailabtools=env_flag(env.get("SCANLAB_ENABLE_AILABTOOLS")),
            enable_gemini=env_flag(env.get("SCANLAB_ENABLE_GEMINI")),
            gemini_allow_image=env_flag(env.get("SCANLAB_GEMINI_ALLOW_IMAGE")),
            trust_proxy=env_flag(env.get("SCANLAB_TRUST_PROXY")),
            idempotency_ttl_seconds=float(_positive_int(env, "SCANLAB_IDEMPOTENCY_TTL_SECONDS", 600, ceiling=86_400)),
            idempotency_max_entries=_positive_int(env, "SCANLAB_IDEMPOTENCY_MAX", 256, ceiling=10_000),
            eligibility_mode=parse_eligibility_mode(env.get("SCANLAB_ELIGIBILITY_MODE")),
            age_declaration_secret=load_age_declaration_secret(env),
        )
