"""Perfect Corp YouCam AI Skin Analysis adapter.

Official references, read 2026-10-27:

- https://docs.perfectcorp.com/reference/ai_skin_analysis/v2.1
- https://docs.perfectcorp.com/reference/file
- https://docs.perfectcorp.com/develop/webhook
- https://yce-api-01.makeupar.com

v2 request authentication is ``Authorization: Bearer <YOUCAM_API_KEY>``.
The console also shows a secret once. v2 skin analysis does not send that
secret. When ``YOUCAM_SECRET_KEY`` is a ``whsec_`` value, it is the Standard
Webhooks signing secret from the YouCam webhook console. The v1 RSA
client_secret is a different value and is not placed on these requests.

Scores copied into metrics are ``raw_score`` (1–100, higher means a healthier
appearance in Perfect Corp's scale). ``ui_score`` is a psychological
adjustment and is kept only in the raw payload. Perfect Corp's ``all`` score
and ``skin_age`` are not SkinPulse metrics.

HD unit table for one successful task (failures and in-progress polls consume
nothing, per the provider docs):

- 1–4 concerns: 12 units
- 5–8 concerns: 16 units
- 9–12 concerns: 20 units
- 13–16 concerns: 22 units

This adapter requests 9 HD concerns, so a successful scan is 20 units.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import urllib.parse
from typing import Any, Mapping

from scanlab.providers.images import require_youcam_hd
from scanlab.providers.transport import DisabledTransport, HttpRequest
from scanlab.providers.types import (
    ConfigurationError,
    MetricObservation,
    ProviderAnalysis,
    ProviderError,
    ProviderTimeout,
    ScanImage,
)

ADAPTER_VERSION = "1.0.0"
API_VERSION = "v2.1"
BASE_URL = "https://yce-api-01.makeupar.com"
FILE_PATH = "/s2s/v2.0/file"
TASK_PATH = "/s2s/v2.1/task/skin-analysis"
POLL_INTERVAL_SECONDS = 10

# Nine HD concerns. HD and SD must not be mixed. This is the 9–12 band.
HD_ACTIONS: tuple[str, ...] = (
    "hd_redness",
    "hd_oiliness",
    "hd_age_spot",
    "hd_moisture",
    "hd_dark_circle",
    "hd_texture",
    "hd_acne",
    "hd_pore",
    "hd_wrinkle",
)

_ACTION_METRIC = {
    "hd_redness": "redness",
    "hd_texture": "texture",
    "hd_pore": "pores",
    "hd_age_spot": "pigmentation",
    "hd_acne": "acne",
    "hd_wrinkle": "fine_lines",
    "hd_dark_circle": "dark_circles",
}

_BEARER = re.compile(r"Bearer\s+\S+", re.I)


def hd_units(concern_count: int) -> int:
    """Return the published HD unit cost for a successful analysis."""
    if concern_count < 1 or concern_count > 16:
        raise ValueError("HD concern count must be from 1 to 16")
    if concern_count <= 4:
        return 12
    if concern_count <= 8:
        return 16
    if concern_count <= 12:
        return 20
    return 22


def combine_dryness_oiliness(
    oiliness: float | None, moisture: float | None
) -> tuple[float | None, str]:
    """Map the two YouCam concerns onto the single dryness/oiliness input.

    Both raw scores use the same 1–100 higher-is-healthier direction. The
    mean is an adapter normalization, not a Perfect Corp field.
    """
    if oiliness is None and moisture is None:
        return None, ""
    if oiliness is not None and moisture is not None:
        return (oiliness + moisture) / 2, "mean of hd_oiliness and hd_moisture raw_score"
    if oiliness is not None:
        return oiliness, "hd_oiliness raw_score only; hd_moisture was not returned"
    return moisture, "hd_moisture raw_score only; hd_oiliness was not returned"


def metrics_from_output(output: list[Any]) -> tuple[dict[str, MetricObservation], list[str]]:
    """Read the format=json ``results.output`` array. Ignore ui_score."""
    warnings: list[str] = []
    chosen: dict[str, tuple[float, str, dict[str, Any]]] = {}
    components: dict[str, float] = {}

    for item in output:
        if not isinstance(item, dict):
            warnings.append("skipped a non-object output row")
            continue
        action = str(item.get("type") or "")
        region = item.get("region")
        raw = item.get("raw_score")
        if action in {"hd_oiliness", "hd_moisture"}:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                components[action] = float(raw)
            else:
                warnings.append(f"{action} had no raw_score; ui_score was ignored")
            continue
        metric = _ACTION_METRIC.get(action)
        if metric is None:
            continue
        if region not in (None, "whole"):
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            warnings.append(f"{action} had no raw_score; ui_score was ignored")
            continue
        source = action if region in (None, "") else f"{action}.{region}"
        chosen[metric] = (float(raw), source, {"region": region, "raw_score": float(raw)})

    metrics: dict[str, MetricObservation] = {}
    for metric, (value, source, detail) in chosen.items():
        metrics[metric] = MetricObservation(
            metric=metric,
            value=value,
            scale="higher_better_1_to_100",
            confidence=None,
            source=source,
            detail=detail,
        )

    dryness, note = combine_dryness_oiliness(
        components.get("hd_oiliness"), components.get("hd_moisture")
    )
    if dryness is not None:
        metrics["dryness_oiliness"] = MetricObservation(
            metric="dryness_oiliness",
            value=dryness,
            scale="higher_better_1_to_100",
            confidence=None,
            source=note,
            detail={key: components[key] for key in ("hd_oiliness", "hd_moisture") if key in components},
        )
    return metrics, warnings


def verify_webhook(
    secret: str,
    webhook_id: str,
    timestamp: str,
    body: bytes,
    signature_header: str,
) -> bool:
    """Check a YouCam Standard Webhooks signature.

    Signed content is ``{webhook-id}.{webhook-timestamp}.{raw body}``.
    The secret is ``whsec_`` plus base64 key bytes. Header form: ``v1,<base64>``.
    """
    if not secret.startswith("whsec_"):
        raise ConfigurationError(
            "YOUCAM_SECRET_KEY is not a whsec_ webhook secret. "
            "v2 skin-analysis requests authenticate with YOUCAM_API_KEY. "
            "Webhook checks need the signing secret from the YouCam webhook console."
        )
    key = base64.b64decode(secret[len("whsec_"):], validate=True)
    signed = f"{webhook_id}.{timestamp}.".encode("utf-8") + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    presented = []
    for piece in signature_header.split():
        if piece.startswith("v1,"):
            presented.append(piece[3:])
    return any(hmac.compare_digest(item, expected) for item in presented)


class YouCamClient:
    provider = "youcam"
    emits_scores = True

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        transport=None,
        allow_network: bool = False,
        sleeper=None,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        max_polls: int = 30,
    ):
        if not api_key or not secret_key:
            raise ConfigurationError("YOUCAM_API_KEY and YOUCAM_SECRET_KEY are both required")
        self._api_key = api_key
        self._secret_key = secret_key
        self._allow_network = allow_network
        self._transport = transport if transport is not None else DisabledTransport()
        self._sleeper = sleeper
        self._poll_interval = poll_interval
        self._max_polls = max_polls

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs) -> "YouCamClient":
        source = os.environ if env is None else env
        api_key = source.get("YOUCAM_API_KEY", "")
        secret = source.get("YOUCAM_SECRET_KEY", "")
        missing = [
            name
            for name, value in (("YOUCAM_API_KEY", api_key), ("YOUCAM_SECRET_KEY", secret))
            if not value
        ]
        if missing:
            raise ConfigurationError("missing " + ", ".join(missing))
        return cls(api_key, secret, **kwargs)

    def verify_webhook(self, webhook_id: str, timestamp: str, body: bytes, signature_header: str) -> bool:
        return verify_webhook(self._secret_key, webhook_id, timestamp, body, signature_header)

    def analyze(self, image: ScanImage, *, image_id: str | None = None, transport=None) -> ProviderAnalysis:
        facts = require_youcam_hd(image.data)
        client = transport or self._transport
        content_type = "image/jpg" if facts.mime_type == "image/jpeg" else "image/png"
        latency = 0.0
        calls = 0

        file_body = {
            "files": [
                {
                    "content_type": content_type,
                    "file_name": image.filename,
                    "file_size": facts.size,
                }
            ]
        }
        created = self._send(
            client,
            HttpRequest(
                "POST",
                BASE_URL + FILE_PATH,
                self._json_headers(),
                _dump(file_body),
            ),
        )
        calls += 1
        latency += created.latency_ms
        file_id, upload = _parse_file(created.json(), self._api_key)
        uploaded = self._send(
            client,
            HttpRequest(upload["method"], upload["url"], dict(upload["headers"]), image.data),
        )
        calls += 1
        latency += uploaded.latency_ms
        if uploaded.status not in {200, 201, 204}:
            raise ProviderError("upload_failed", f"file upload returned HTTP {uploaded.status}")

        task_body = {
            "src_file_id": file_id,
            "dst_actions": list(HD_ACTIONS),
            "format": "json",
        }
        task = self._send(
            client,
            HttpRequest(
                "POST",
                BASE_URL + TASK_PATH,
                self._json_headers(),
                _dump(task_body),
            ),
        )
        calls += 1
        latency += task.latency_ms
        task_id = _parse_task_id(task.json(), self._api_key)
        poll_url = BASE_URL + TASK_PATH + "/" + urllib.parse.quote(task_id, safe="")

        for attempt in range(self._max_polls):
            polled = self._send(
                client,
                HttpRequest("GET", poll_url, self._json_headers(), None),
            )
            calls += 1
            latency += polled.latency_ms
            payload = polled.json()
            status = str((payload.get("data") or {}).get("task_status") or "")
            if status == "running":
                if attempt == self._max_polls - 1:
                    raise ProviderTimeout(
                        f"YouCam task stayed running after {self._max_polls} polls"
                    )
                self._wait()
                continue
            if status == "error":
                code = str((payload.get("data") or {}).get("error") or "task_error")
                return _finished(
                    success=False,
                    metrics={},
                    raw=_public_raw(payload),
                    latency_ms=latency,
                    calls=calls,
                    image_id=image_id,
                    error_code=code,
                    error_message=_safe(str((payload.get("data") or {}).get("error_message") or code), self._api_key),
                    warnings=[],
                    cost_consumed=False,
                )
            if status == "success":
                results = (payload.get("data") or {}).get("results") or {}
                output = results.get("output")
                if not isinstance(output, list):
                    raise ProviderError(
                        "json_result_missing",
                        "format=json did not return results.output. Zip URLs are not downloaded.",
                    )
                metrics, warnings = metrics_from_output(output)
                return _finished(
                    success=True,
                    metrics=metrics,
                    raw=_public_raw(payload),
                    latency_ms=latency,
                    calls=calls,
                    image_id=image_id,
                    error_code=None,
                    error_message=None,
                    warnings=warnings,
                    cost_consumed=True,
                )
            raise ProviderError("unexpected_task_status", _safe(status or "missing", self._api_key))
        raise ProviderTimeout("YouCam poll loop ended without a terminal status")

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _send(self, client, request: HttpRequest):
        if client is None:
            raise ConfigurationError("no transport")
        response = client.send(request)
        if response.status >= 400:
            code, message = _http_error(response, self._api_key)
            raise ProviderError(code, message)
        return response

    def _wait(self) -> None:
        if self._sleeper is not None:
            self._sleeper(self._poll_interval)
            return
        if self._allow_network:
            import time

            time.sleep(self._poll_interval)


def _finished(**kwargs: Any) -> ProviderAnalysis:
    consumed = bool(kwargs.pop("cost_consumed"))
    cost = float(hd_units(len(HD_ACTIONS))) if consumed else 0.0
    return ProviderAnalysis(
        provider="youcam",
        adapter_version=ADAPTER_VERSION,
        api_version=API_VERSION,
        cost_amount=cost,
        cost_unit="youcam_unit",
        cost_consumed=consumed,
        emits_scores=True,
        **kwargs,
    )


def _dump(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _parse_file(payload: dict, api_key: str) -> tuple[str, dict[str, Any]]:
    files = payload.get("files")
    if files is None and isinstance(payload.get("data"), dict):
        files = payload["data"].get("files")
    if not isinstance(files, list) or not files:
        raise ProviderError("invalid_file_response", "file response did not include files")
    item = files[0]
    file_id = item.get("file_id")
    requests = item.get("requests") or []
    if not file_id or not requests:
        raise ProviderError("invalid_file_response", "file response did not include file_id and requests")
    upload = requests[0]
    headers = dict(upload.get("headers") or {})
    _reject_secret_echo(headers, api_key)
    return str(file_id), {
        "method": str(upload.get("method") or "PUT"),
        "url": str(upload["url"]),
        "headers": headers,
    }


def _parse_task_id(payload: dict, api_key: str) -> str:
    raw_data = payload.get("data")
    data = raw_data if isinstance(raw_data, dict) else {}
    task_id = data.get("task_id")
    if not task_id:
        raise ProviderError("invalid_task_response", _safe(json.dumps(payload)[:180], api_key))
    return str(task_id)


def _http_error(response, api_key: str) -> tuple[str, str]:
    try:
        payload = response.json()
    except (ValueError, UnicodeError):
        return "http_error", f"HTTP {response.status}"
    code = str(payload.get("error_code") or payload.get("error") or f"http_{response.status}")
    message = str(payload.get("error") or payload.get("error_message") or code)
    return code, _safe(message, api_key)


def _public_raw(payload: dict) -> dict:
    """Keep scores. Drop mask URLs so the lab record does not store face images."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if key == "mask_urls":
                    cleaned["mask_url_count"] = len(item) if isinstance(item, list) else 0
                    continue
                cleaned[key] = scrub(item)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(json.loads(json.dumps(payload)))


def _safe(text: str, api_key: str) -> str:
    cleaned = _BEARER.sub("Bearer [redacted]", text)
    if api_key:
        cleaned = cleaned.replace(api_key, "[redacted]")
    return cleaned[:300]


def _reject_secret_echo(headers: Mapping[str, str], api_key: str) -> None:
    blob = " ".join(f"{key}:{value}" for key, value in headers.items())
    if api_key and api_key in blob and "Authorization" not in blob:
        raise ProviderError("unexpected_credential_echo", "upload headers echoed the API key")
