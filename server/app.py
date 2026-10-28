"""FastAPI application for the Scan Lab HTTP API."""

from __future__ import annotations

import gc
import json
import logging
import datetime as dt
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from scanlab.cv.images import sha256_bytes
from scanlab.cv.mediapipe_backend import mediapipe_status
from scanlab.cv.pipeline import analyze_bytes
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import NUMERIC_SOURCE, PIPELINE_VERSION
from scanlab.providers.types import ScanImage
from scanlab.score.engine import ACTIVE_VERSION
from scanlab.server import API_VERSION
from scanlab.server.body import MemoryImage, normalize_upload, parse_image, read_limited
from scanlab.server.config import ServerSettings
from scanlab.server.contributions import TESTER_ID, object_keys, parse_meta, store_from_env
from scanlab.server.errors import ApiError
from scanlab.server.guard import (
    CachedResponse,
    IdempotencyConflict,
    IdempotencyStore,
    SlidingWindowLimiter,
    authorize,
    enforce_rate_limit,
    idempotency_key,
)
from scanlab.server.models import ApiAnalysis, HealthBody
from scanlab.server.privacy import install_redaction, public_path
from scanlab.server.providers import build_ports, idle_providers, run_providers
from scanlab.server.scoring import skin_score_from_proxies

logger = logging.getLogger("scanlab.server")
_BODY_MARGIN = 65_536
_FILENAMES = {"image/png": "scan.png", "image/webp": "scan.webp", "image/jpeg": "scan.jpg"}


def create_app(settings: ServerSettings | None = None, *, ports=None, contrib_store="env") -> FastAPI:
    settings = settings or ServerSettings.from_env()
    if ports is None:
        ports = build_ports(settings)
    install_redaction()
    app = FastAPI(
        title="SkinPulse Scan Lab API",
        version=API_VERSION,
        description=(
            "Capture quality, deterministic optical-proxy metrics, and Skin Score 1.0.0. "
            "Photos are processed in memory and are not stored, except via the opt-in /v1/contributions endpoint with explicit consent. External providers are off "
            "unless an environment flag enables one. Numbers are not a clinical result."
        ),
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.contrib_store = store_from_env(os.environ) if contrib_store == "env" else contrib_store
    app.state.ports = ports
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit, settings.rate_window_seconds)
    app.state.idempotency = IdempotencyStore(
        ttl_seconds=settings.idempotency_ttl_seconds,
        max_entries=settings.idempotency_max_entries,
    )
    app.router.redirect_slashes = False

    @app.exception_handler(ApiError)
    async def on_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"error": exc.code, "message": exc.message}},
            headers=exc.headers,
        )
        return response

    @app.exception_handler(Exception)
    async def on_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": {"error": "internal_error", "message": "The request could not be completed."}},
        )

    @app.middleware("http")
    async def privacy_boundary(request: Request, call_next) -> Response:
        if request.url.query:
            response: Response = JSONResponse(
                status_code=400,
                content={"detail": {"error": "invalid_request", "message": "Query parameters are not accepted."}},
            )
            _stamp(response)
            _access(request, response.status_code)
            return response
        response = await call_next(request)
        _stamp(response)
        _access(request, response.status_code)
        return response

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Content-Type", "Idempotency-Key", "X-API-Key", "Authorization"],
            allow_credentials=False,
            max_age=600,
        )

    @app.get("/healthz", response_model=HealthBody)
    def health() -> HealthBody:
        return HealthBody(
            status="ok",
            service="scanlab-api",
            api_version=API_VERSION,
            pipeline_version=PIPELINE_VERSION,
            score_version=ACTIVE_VERSION,
            numeric_source=NUMERIC_SOURCE,
            llm_used=False,
            landmark_backend=settings.cv.landmark_backend,
            mediapipe=mediapipe_status(settings.cv.face_landmarker_model),
            providers={
                "youcam": settings.enable_youcam,
                "ailabtools": settings.enable_ailabtools,
                "gemini": settings.enable_gemini,
                "gemini_allow_image": settings.gemini_allow_image,
            },
        )

    @app.post("/v1/quality", response_model=ApiAnalysis, response_model_exclude_none=True)
    async def quality(request: Request) -> Response:
        return await _handle(request, include_metrics=False)

    @app.post("/v1/analyze", response_model=ApiAnalysis, response_model_exclude_none=True)
    async def analyze(request: Request) -> Response:
        return await _handle(request, include_metrics=True)

    @app.post("/v1/contributions")
    async def contribute(request: Request) -> Response:
        """Store a photo ONLY with explicit consent (X-Contribution-Meta.consent == true)."""
        settings: ServerSettings = request.app.state.settings
        enforce_rate_limit(request, settings, request.app.state.limiter)
        authorize(request, settings)
        meta = parse_meta(request.headers.get("x-contribution-meta"))
        store = request.app.state.contrib_store
        if store is None:
            raise ApiError(503, "contribution_storage_unavailable", "Photo contributions are not enabled on this server.")
        raw = await read_limited(request, settings.cv.max_bytes + _BODY_MARGIN)
        try:
            upload = parse_image(raw, request.headers.get("content-type"), max_bytes=settings.cv.max_bytes, normalize=False)
        finally:
            raw = b""
        # HEIC/EXIF decode is CPU-bound: keep it off the event loop, serialised with analysis.
        upload = await run_in_threadpool(_normalize_serialised, upload)
        now = dt.datetime.now(dt.timezone.utc)
        photo_key, meta_key = object_keys(meta["tester_id"], upload.media_type, now)
        record = {**meta, "stored_at": now.isoformat(), "media_type": upload.media_type,
                  "sha256": sha256_bytes(upload.data), "photo_key": photo_key, "api_version": API_VERSION}
        try:
            await run_in_threadpool(store.put, photo_key, upload.data, upload.media_type)
            await run_in_threadpool(store.put, meta_key, json.dumps(record).encode("utf-8"), "application/json")
        except Exception as exc:  # noqa: BLE001
            logger.error("contribution store failed: %s", type(exc).__name__)
            raise ApiError(502, "contribution_store_failed", "The photo could not be stored.") from exc
        finally:
            _wipe(upload)
        return JSONResponse(status_code=201, content={"stored": True, "id": photo_key.rsplit("/", 1)[-1].split(".")[0]})

    @app.post("/v1/contributions/delete")
    async def delete_contributions(request: Request) -> Response:
        """Delete every stored photo + metadata for X-Tester-Id."""
        settings: ServerSettings = request.app.state.settings
        enforce_rate_limit(request, settings, request.app.state.limiter)
        authorize(request, settings)
        tester = request.headers.get("x-tester-id", "")
        if not TESTER_ID.match(tester):
            raise ApiError(400, "invalid_request", "X-Tester-Id is invalid.")
        store = request.app.state.contrib_store
        if store is None:
            return JSONResponse(status_code=200, content={"deleted": 0})
        try:
            count = await run_in_threadpool(store.delete_prefix, f"contributions/{tester}/")
        except Exception as exc:  # noqa: BLE001
            logger.error("contribution delete failed: %s", type(exc).__name__)
            raise ApiError(502, "contribution_delete_failed", "Stored photos could not be deleted.") from exc
        return JSONResponse(status_code=200, content={"deleted": count})

    return app


_production_app: FastAPI | None = None


def __getattr__(name: str):
    """Build the process-wide app on first use so imports stay side-effect free."""

    global _production_app
    if name == "app":
        if _production_app is None:
            _production_app = create_app()
        return _production_app
    raise AttributeError(name)


_ANALYSIS_LOCK = threading.Lock()


def _normalize_serialised(upload: MemoryImage) -> MemoryImage:
    with _ANALYSIS_LOCK:
        try:
            return normalize_upload(upload)
        except BaseException:
            _wipe(upload)
            raise


def _analyze_serialised(data: bytes, cv_settings, include_metrics: bool):
    """One analysis at a time bounds peak memory on a 512MB instance."""
    with _ANALYSIS_LOCK:
        try:
            return analyze_bytes(data, cv_settings, include_metrics=include_metrics)
        finally:
            gc.collect()


async def _handle(request: Request, *, include_metrics: bool) -> Response:
    settings: ServerSettings = request.app.state.settings
    enforce_rate_limit(request, settings, request.app.state.limiter)
    scope = authorize(request, settings)
    _check_accept(request)
    key = idempotency_key(request)
    raw = await read_limited(request, settings.cv.max_bytes + _BODY_MARGIN)
    try:
        upload = parse_image(raw, request.headers.get("content-type"), max_bytes=settings.cv.max_bytes, normalize=False)
    finally:
        raw = b""
    # HEIC/EXIF decode is CPU-bound: keep it off the event loop, serialised with analysis.
    upload = await run_in_threadpool(_normalize_serialised, upload)
    digest = sha256_bytes(upload.data)
    path = request.url.path
    store: IdempotencyStore = request.app.state.idempotency
    if key:
        try:
            cached = store.lookup(scope, key, path, digest)
        except IdempotencyConflict as exc:
            _wipe(upload)
            raise ApiError(409, "idempotency_conflict", "This idempotency key was already used for a different image.") from exc
        if cached is not None:
            _wipe(upload)
            return JSONResponse(status_code=cached.status_code, content=cached.payload)
    try:
        # CPU-bound: run off the event loop so /healthz stays responsive on 0.1 CPU.
        result = await run_in_threadpool(_analyze_serialised, upload.data, settings.cv, include_metrics)
    except ScanLabError as exc:
        _wipe(upload)
        body = {"detail": {"error": exc.code, "message": exc.message}}
        if key and exc.status_code in {400, 503}:
            store.save(scope, key, path, digest, CachedResponse(exc.status_code, body))
        raise ApiError(exc.status_code, exc.code, exc.message) from exc
    try:
        payload = _payload(request, result, upload, include_metrics=include_metrics)
    finally:
        _wipe(upload)
    if key:
        store.save(scope, key, path, digest, CachedResponse(200, payload))
    return JSONResponse(status_code=200, content=payload)


def _payload(request: Request, result, upload: MemoryImage, *, include_metrics: bool) -> dict:
    settings: ServerSettings = request.app.state.settings
    skin = skin_score_from_proxies(result.metrics) if include_metrics and result.accepted else None
    notes: list[str] = []
    if include_metrics and result.accepted and result.metrics_included:
        providers, notes = run_providers(
            settings,
            request.app.state.ports,
            ScanImage(upload.data, upload.media_type, _FILENAMES.get(upload.media_type, "scan.jpg")),
            _stored(result, skin),
        )
    else:
        providers = idle_providers(settings, request.app.state.ports)
    data = result.model_dump()
    data["warnings"] = [*result.warnings, *notes]
    data["skin_score"] = skin
    data["providers"] = providers.model_dump()
    model = ApiAnalysis.model_validate(data)
    return json.loads(model.model_dump_json(exclude_none=True))


def _stored(result, skin: dict | None) -> dict:
    return {
        "pipeline_version": result.pipeline_version,
        "numeric_source": result.numeric_source,
        "accepted": result.accepted,
        "quality_checks": [
            {
                "name": item.name,
                "passed": item.passed,
                "value": item.value,
                "confidence": item.confidence,
                "version": item.version,
            }
            for item in result.quality_checks
        ],
        "metrics": [
            {
                "name": item.name,
                "value": item.value,
                "confidence": item.confidence,
                "version": item.version,
                "clinical_validity": item.clinical_validity,
            }
            for item in result.metrics
        ],
        "skin_score": skin,
        "limitations": list(result.limitations),
    }


def _check_accept(request: Request) -> None:
    accept = request.headers.get("accept")
    if not accept:
        return
    pieces = [part.split(";", 1)[0].strip().lower() for part in accept.split(",")]
    if not any(part in {"application/json", "*/*"} for part in pieces):
        raise ApiError(406, "not_acceptable", "This API only returns JSON.")


def _wipe(upload: MemoryImage) -> None:
    upload.data = b""


def _stamp(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Scanlab-Api"] = API_VERSION


def _access(request: Request, status: int) -> None:
    logger.info("%s %s %s", request.method, public_path(request.url.path), status)
