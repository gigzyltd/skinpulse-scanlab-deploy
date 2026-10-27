"""Optional provider calls. Flags default off, and tests inject fakes.

A live transport is built only when a flag is on and credentials exist.
Gemini receives structured fields. The photo is attached only when
SCANLAB_GEMINI_ALLOW_IMAGE is also set. YouCam and AILabTools are photo
APIs: turning either flag on sends that photo to the vendor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from scanlab.providers.types import (
    ConfigurationError,
    ImageRejected,
    MetricObservation,
    NetworkDisabled,
    ProviderAnalysis,
    ProviderError,
    ProviderTimeout,
    ScanImage,
)
from scanlab.server.config import ServerSettings
from scanlab.server.models import ProviderBlock, ProviderMetricBody, ProviderSlot
from scanlab.server.privacy import public_explanation

logger = logging.getLogger("scanlab.server")
_CODE = re.compile(r"^[a-z0-9_]{1,80}$")
_PROVIDER_NOTE = "External provider output is separate from the deterministic metrics and is not a clinical result."


@dataclass
class ProviderPorts:
    youcam: Any | None = None
    ailabtools: Any | None = None
    gemini: Any | None = None
    youcam_error: str | None = None
    ailabtools_error: str | None = None
    gemini_error: str | None = None


def build_ports(settings: ServerSettings, env: Mapping[str, str] | None = None) -> ProviderPorts:
    import os

    source = os.environ if env is None else env
    ports = ProviderPorts()
    if settings.enable_youcam:
        key = source.get("YOUCAM_API_KEY", "")
        secret = source.get("YOUCAM_SECRET_KEY", "")
        if not key or not secret:
            ports.youcam_error = "not_configured"
        else:
            from scanlab.providers.transport import UrllibTransport
            from scanlab.providers.youcam import YouCamClient

            ports.youcam = YouCamClient(
                key,
                secret,
                transport=UrllibTransport(allow_network=True),
                allow_network=True,
            )
    if settings.enable_ailabtools:
        key = source.get("AILABTOOLS_API_KEY", "")
        if not key:
            ports.ailabtools_error = "not_configured"
        else:
            from scanlab.providers.ailabtools import AILabToolsClient
            from scanlab.providers.transport import UrllibTransport

            ports.ailabtools = AILabToolsClient(
                key,
                transport=UrllibTransport(allow_network=True),
                allow_network=True,
            )
    if settings.enable_gemini:
        key = source.get("GEMINI_API_KEY", "")
        if not key:
            ports.gemini_error = "not_configured"
        else:
            from scanlab.providers.gemini import GeminiExplainer
            from scanlab.providers.transport import UrllibTransport

            ports.gemini = GeminiExplainer(
                key,
                transport=UrllibTransport(allow_network=True),
                allow_network=True,
            )
    return ports


def idle_providers(settings: ServerSettings, ports: ProviderPorts) -> ProviderBlock:
    return ProviderBlock(
        youcam=_idle(settings.enable_youcam, ports.youcam_error),
        ailabtools=_idle(settings.enable_ailabtools, ports.ailabtools_error),
        gemini=_idle(settings.enable_gemini, ports.gemini_error),
    )


def run_providers(
    settings: ServerSettings,
    ports: ProviderPorts,
    image: ScanImage,
    stored: dict[str, Any],
) -> tuple[ProviderBlock, list[str]]:
    notes: list[str] = []
    youcam = _call(settings.enable_youcam, ports.youcam, ports.youcam_error, image, stored, include_image=True)
    ailab = _call(settings.enable_ailabtools, ports.ailabtools, ports.ailabtools_error, image, stored, include_image=True)
    gemini = _call(
        settings.enable_gemini,
        ports.gemini,
        ports.gemini_error,
        image,
        stored,
        include_image=settings.gemini_allow_image,
    )
    if any(slot.called for slot in (youcam, ailab, gemini)):
        notes.append(_PROVIDER_NOTE)
    return ProviderBlock(youcam=youcam, ailabtools=ailab, gemini=gemini), notes


def _idle(enabled: bool, error_code: str | None) -> ProviderSlot:
    return ProviderSlot(enabled=enabled, called=False, error_code=error_code)


def _call(
    enabled: bool,
    client: Any | None,
    setup_error: str | None,
    image: ScanImage,
    stored: dict[str, Any],
    *,
    include_image: bool,
) -> ProviderSlot:
    if not enabled:
        return ProviderSlot(enabled=False, called=False)
    if client is None:
        return ProviderSlot(enabled=True, called=False, error_code=setup_error or "not_configured")
    try:
        if _accepts_structured(client):
            analysis = client.analyze(image, stored_output=stored, include_image=include_image)
        else:
            analysis = client.analyze(image)
    except (ImageRejected, ConfigurationError, ProviderError, ProviderTimeout, NetworkDisabled) as exc:
        return ProviderSlot(enabled=True, called=True, success=False, error_code=_error_code(exc))
    except Exception as exc:
        logger.error("provider failed: %s", type(exc).__name__)
        return ProviderSlot(enabled=True, called=True, success=False, error_code="provider_failed")
    return _from_analysis(analysis)


def _accepts_structured(client: Any) -> bool:
    return getattr(client, "provider", "") == "gemini" or client.__class__.__name__ == "GeminiExplainer"


def _from_analysis(analysis: ProviderAnalysis) -> ProviderSlot:
    explanation = public_explanation(analysis.explanation)
    error_code = analysis.error_code if analysis.error_code and _CODE.fullmatch(analysis.error_code) else None
    success = bool(analysis.success)
    if analysis.explanation and explanation is None:
        success = False
        error_code = error_code or "explanation_withheld_numeric"
    warnings = []
    for item in analysis.warnings[:12]:
        text = public_explanation(str(item))
        if text:
            warnings.append(text[:200])
    return ProviderSlot(
        enabled=True,
        called=True,
        success=success,
        error_code=error_code,
        emits_scores=bool(analysis.emits_scores),
        metrics=[_metric(item) for item in list(analysis.metrics.values())[:16]],
        warnings=warnings,
        explanation=explanation,
        cost_consumed=bool(analysis.cost_consumed),
        cost_unit=_short(analysis.cost_unit),
        cost_amount=analysis.cost_amount if isinstance(analysis.cost_amount, (int, float)) else None,
        latency_ms=float(analysis.latency_ms) if isinstance(analysis.latency_ms, (int, float)) else None,
    )


def _metric(item: MetricObservation) -> ProviderMetricBody:
    value = item.value
    if isinstance(value, (int, float)):
        value = float(value)
    else:
        value = None
    confidence = item.confidence if isinstance(item.confidence, (int, float)) else None
    return ProviderMetricBody(
        metric=str(item.metric)[:80],
        value=value,
        scale=str(item.scale)[:80],
        confidence=float(confidence) if confidence is not None else None,
    )


def _error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _CODE.fullmatch(code):
        return code
    if isinstance(exc, ProviderTimeout):
        return "timeout"
    if isinstance(exc, NetworkDisabled):
        return "network_disabled"
    if isinstance(exc, ImageRejected):
        return "image_rejected"
    if isinstance(exc, ConfigurationError):
        return "not_configured"
    return "provider_failed"


def _short(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_./-]{1,40}", value):
        return None
    return value
