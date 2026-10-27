"""AILabTools Skin Analyze adapter.

Official reference, read 2026-10-27:

- POST https://www.ailabapi.com/api/portrait/analysis/skin-analysis
- Header ``ailabapi-api-key``
- multipart field ``image`` (JPG/JPEG, ≤ 2 MB, larger than 200 px and smaller than 4096 px)
- https://www.ailabtools.com/docs/ai-portrait/analysis/skin-analysis/api
- Billing: Skin Analyze deducts 15 credits for each successful call
  (https://www.ailabtools.com/doc/ai-portrait/analysis/skin-analysis)

The basic API reports presence and categories. It does not report redness,
texture, or a 0–100 severity. Those metrics are left absent. Presence is not
converted into a fake severity score. The key is read from ``AILABTOOLS_API_KEY``.
"""

from __future__ import annotations

import json
import os
import secrets
from typing import Any, Mapping

from scanlab.providers.images import require_ailab
from scanlab.providers.transport import DisabledTransport, HttpRequest
from scanlab.providers.types import (
    ConfigurationError,
    MetricObservation,
    ProviderAnalysis,
    ProviderError,
    ScanImage,
)

ADAPTER_VERSION = "1.0.0"
API_VERSION = "skin-analysis"
ENDPOINT = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis"
CREDITS_PER_SUCCESS = 15

SKIN_TYPES = {0: "oily", 1: "dry", 2: "neutral", 3: "combination"}
PORE_FIELDS = ("pores_forehead", "pores_left_cheek", "pores_right_cheek", "pores_jaw")


def metrics_from_result(result: Mapping[str, Any]) -> tuple[dict[str, MetricObservation], list[str]]:
    warnings = [
        "redness is not reported by Skin Analyze",
        "texture is not reported by Skin Analyze",
    ]
    metrics: dict[str, MetricObservation] = {}

    pores = _presence_group(result, PORE_FIELDS)
    if pores is not None:
        metrics["pores"] = pores
    acne = _presence(result.get("acne"), "acne", "result.acne")
    if acne is not None:
        metrics["acne"] = acne
    fine = _presence(result.get("eye_finelines"), "fine_lines", "result.eye_finelines")
    if fine is not None:
        metrics["fine_lines"] = fine
    circles = _presence(result.get("dark_circle"), "dark_circles", "result.dark_circle")
    if circles is not None:
        metrics["dark_circles"] = circles
    spots = _presence(result.get("skin_spot"), "pigmentation", "result.skin_spot")
    if spots is not None:
        metrics["pigmentation"] = spots

    skin = result.get("skin_type")
    if isinstance(skin, dict) and skin.get("skin_type") in SKIN_TYPES:
        code = int(skin["skin_type"])
        details = skin.get("details") if isinstance(skin.get("details"), dict) else {}
        selected = details.get(str(code)) if isinstance(details.get(str(code)), dict) else {}
        confidence = selected.get("confidence")
        metrics["dryness_oiliness"] = MetricObservation(
            metric="dryness_oiliness",
            value=None,
            scale="category",
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            source="result.skin_type",
            category=SKIN_TYPES[code],
            detail={"skin_type": code},
        )
    return metrics, warnings


class AILabToolsClient:
    provider = "ailabtools"
    emits_scores = True

    def __init__(self, api_key: str, *, transport=None, allow_network: bool = False):
        if not api_key:
            raise ConfigurationError("AILABTOOLS_API_KEY is required")
        self._api_key = api_key
        self._transport = transport if transport is not None else DisabledTransport()
        self._allow_network = allow_network

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs) -> "AILabToolsClient":
        source = os.environ if env is None else env
        api_key = source.get("AILABTOOLS_API_KEY", "")
        if not api_key:
            raise ConfigurationError("missing AILABTOOLS_API_KEY")
        return cls(api_key, **kwargs)

    def analyze(self, image: ScanImage, *, image_id: str | None = None, transport=None) -> ProviderAnalysis:
        require_ailab(image.data)
        client = transport or self._transport
        boundary, body = _multipart("image", image.filename or "scan.jpg", "image/jpeg", image.data)
        response = client.send(
            HttpRequest(
                "POST",
                ENDPOINT,
                {
                    "ailabapi-api-key": self._api_key,
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                },
                body,
            )
        )
        if response.status >= 400:
            raise ProviderError(f"http_{response.status}", "AILabTools request failed")
        try:
            payload = response.json()
        except (ValueError, UnicodeError) as exc:
            raise ProviderError("invalid_response", "AILabTools response was not JSON") from exc
        error_code = payload.get("error_code")
        if error_code not in (0, "0"):
            return ProviderAnalysis(
                provider=self.provider,
                adapter_version=ADAPTER_VERSION,
                api_version=API_VERSION,
                success=False,
                metrics={},
                raw=_public(payload),
                latency_ms=response.latency_ms,
                cost_amount=0,
                cost_unit="credit",
                cost_consumed=False,
                emits_scores=True,
                error_code=str(error_code),
                error_message=_safe(str(payload.get("error_msg") or "provider error"), self._api_key),
                image_id=image_id,
                calls=1,
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            metrics, warnings = {}, ["result was missing on a successful response"]
        else:
            metrics, warnings = metrics_from_result(result)
        if payload.get("warning"):
            warnings.append("provider warning present")
        return ProviderAnalysis(
            provider=self.provider,
            adapter_version=ADAPTER_VERSION,
            api_version=API_VERSION,
            success=True,
            metrics=metrics,
            raw=_public(payload),
            latency_ms=response.latency_ms,
            cost_amount=float(CREDITS_PER_SUCCESS),
            cost_unit="credit",
            cost_consumed=True,
            emits_scores=True,
            warnings=warnings,
            image_id=image_id,
            calls=1,
        )


def _presence(node: Any, metric: str, source: str) -> MetricObservation | None:
    if not isinstance(node, dict) or node.get("value") not in (0, 1):
        return None
    confidence = node.get("confidence")
    return MetricObservation(
        metric=metric,
        value=float(node["value"]),
        scale="presence_0_absent_1_present",
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        source=source,
        detail={"value": int(node["value"])},
    )


def _presence_group(result: Mapping[str, Any], fields: tuple[str, ...]) -> MetricObservation | None:
    present: list[int] = []
    confidences: list[float] = []
    for name in fields:
        node = result.get(name)
        if not isinstance(node, dict) or node.get("value") not in (0, 1):
            return None
        present.append(int(node["value"]))
        if isinstance(node.get("confidence"), (int, float)):
            confidences.append(float(node["confidence"]))
    return MetricObservation(
        metric="pores",
        value=1.0 if any(present) else 0.0,
        scale="presence_0_absent_1_present",
        confidence=min(confidences) if confidences else None,
        source="result.pores_*",
        detail={name: present[index] for index, name in enumerate(fields)},
    )


def _multipart(field: str, filename: str, content_type: str, data: bytes) -> tuple[str, bytes]:
    boundary = "scanlab" + secrets.token_hex(16)
    while boundary.encode("ascii") in data:
        boundary = "scanlab" + secrets.token_hex(16)
    crlf = b"\r\n"
    body = b"".join(
        [
            b"--" + boundary.encode("ascii") + crlf,
            (
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'
            ).encode("utf-8")
            + crlf,
            f"Content-Type: {content_type}".encode("utf-8") + crlf,
            crlf,
            data,
            crlf,
            b"--" + boundary.encode("ascii") + b"--" + crlf,
        ]
    )
    return boundary, body


def _public(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _safe(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:300]
