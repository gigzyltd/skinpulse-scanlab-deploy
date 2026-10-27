"""Gemini vision explainer.

Official reference, read 2026-10-27:

- POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent
- Header ``x-goog-api-key: $GEMINI_API_KEY``
- Image part ``inline_data`` with ``mime_type`` and base64 ``data``
- https://ai.google.dev/gemini-api/docs/generate-content/image-understanding
- Model id from the catalog: gemini-3.8-flash
  (https://ai.google.dev/gemini-api/docs/models)

This adapter explains stored provider output. It does not produce skin
scores. A reply that contains a digit is withheld, because the prompt forbids
numbers and this lab will not parse a measurement out of prose. Token counts
are kept as usage metadata. No dollar price is assigned: the deployment
project price was not verified, and this lab does not purchase credits.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Mapping

from scanlab.providers.images import require_gemini
from scanlab.providers.transport import DisabledTransport, HttpRequest
from scanlab.providers.types import (
    ConfigurationError,
    ProviderAnalysis,
    ProviderError,
    ScanImage,
)

ADAPTER_VERSION = "1.0.0"
MODEL = "gemini-3.8-flash"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)
_SECRET_KEYS = {"api_key", "authorization", "secret", "token", "ailabapi-api-key"}
_SYSTEM = (
    "You explain stored skin-analysis JSON for a laboratory reviewer. "
    "Describe only what the stored fields say. Do not invent measurements, "
    "scores, ages, diagnoses, or changes. Do not output numbers."
)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                continue
            cleaned[key] = redact_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def explanation_from_response(payload: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Return prose only when it contains no digits. Never return metrics."""
    warnings: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProviderError("invalid_response", "Gemini response had no candidates")
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    finish = str(first.get("finishReason") or "")
    if finish and finish not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        raise ProviderError("generation_stopped", finish)
    content = first.get("content") if isinstance(first.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    texts = [str(part["text"]) for part in parts if isinstance(part, dict) and "text" in part]
    explanation = "\n".join(texts).strip()
    if not explanation:
        raise ProviderError("empty_explanation", "Gemini returned no text")
    if re.search(r"\d", explanation):
        warnings.append("explanation_withheld_numeric")
        return None, warnings
    return explanation, warnings


class GeminiExplainer:
    provider = "gemini"
    emits_scores = False

    def __init__(self, api_key: str, *, transport=None, allow_network: bool = False, model: str = MODEL):
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY is required")
        if model != MODEL:
            raise ConfigurationError("this adapter is pinned to gemini-3.8-flash")
        self._api_key = api_key
        self._transport = transport if transport is not None else DisabledTransport()
        self._allow_network = allow_network

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs) -> "GeminiExplainer":
        source = os.environ if env is None else env
        api_key = source.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ConfigurationError("missing GEMINI_API_KEY")
        return cls(api_key, **kwargs)

    def analyze(
        self,
        image: ScanImage,
        *,
        stored_output: Mapping[str, Any] | None = None,
        image_id: str | None = None,
        transport=None,
        include_image: bool = True,
    ) -> ProviderAnalysis:
        """Explain stored output.

        ``include_image`` defaults to true so existing callers keep the vision
        request. The HTTP API passes false unless image upload is separately
        enabled: the explanation should normally see structured fields only.
        """

        stored = redact_secrets(stored_output or {})
        stored_json = json.dumps(stored, sort_keys=True, separators=(",", ":"))
        parts: list[dict[str, Any]] = []
        if include_image:
            facts = require_gemini(image.data, image.mime_type)
            prompt = (
                "Explain only the stored provider output below. "
                "Use the image only to say whether a face appears to be present, without numbers. "
                "Do not invent measurements.\n"
                + stored_json
            )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": facts.mime_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    }
                }
            )
        else:
            prompt = (
                "Explain only the stored provider output below. "
                "There is no image in this request. "
                "Do not invent measurements.\n"
                + stored_json
            )
        parts.append({"text": prompt})
        request_body = {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0},
        }
        client = transport or self._transport
        response = client.send(
            HttpRequest(
                "POST",
                ENDPOINT,
                {
                    "x-goog-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
            )
        )
        if response.status >= 400:
            raise ProviderError(f"http_{response.status}", "Gemini request failed")
        try:
            payload = response.json()
        except (ValueError, UnicodeError) as exc:
            raise ProviderError("invalid_response", "Gemini response was not JSON") from exc
        if "error" in payload:
            message = str((payload.get("error") or {}).get("message") or "provider error")
            raise ProviderError("gemini_error", _safe(message, self._api_key))
        explanation, warnings = explanation_from_response(payload)
        usage = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else {}
        return ProviderAnalysis(
            provider=self.provider,
            adapter_version=ADAPTER_VERSION,
            api_version=MODEL,
            success=explanation is not None,
            metrics={},
            raw={"usageMetadata": usage, "withheld": explanation is None},
            latency_ms=response.latency_ms,
            cost_amount=None,
            cost_unit="unpriced",
            cost_consumed=False,
            emits_scores=False,
            explanation=explanation,
            error_code="explanation_withheld_numeric" if explanation is None else None,
            error_message="Explanation withheld because it contained a number." if explanation is None else None,
            warnings=warnings,
            image_id=image_id,
            calls=1,
        )


def _safe(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:300]
