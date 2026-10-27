"""Shared Scan Lab result types.

Numeric values stay on the scale the provider actually returned, or on the
single documented normalization noted by the adapter. Raw payloads are kept
so a later score version can be replayed without calling the provider again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ScanLabError(Exception):
    """Base error. Messages must not contain credentials or image bytes."""

    code = "scanlab_error"


class ConfigurationError(ScanLabError):
    code = "configuration"


class ImageRejected(ScanLabError):
    code = "image_rejected"


class ProviderError(ScanLabError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class ProviderTimeout(ScanLabError):
    code = "timeout"


class NetworkDisabled(ScanLabError):
    code = "network_disabled"


@dataclass
class ScanImage:
    data: bytes
    mime_type: str
    filename: str = "scan.jpg"


@dataclass
class MetricObservation:
    """One sub-metric, or an explicit statement that it is not a 0–100 score."""

    metric: str
    value: float | None
    scale: str
    confidence: float | None = None
    source: str = ""
    category: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderAnalysis:
    provider: str
    adapter_version: str
    api_version: str
    success: bool
    metrics: dict[str, MetricObservation]
    raw: dict[str, Any]
    latency_ms: float
    cost_amount: float | None
    cost_unit: str
    cost_consumed: bool
    emits_scores: bool
    explanation: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    image_id: str | None = None
    calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = {key: value.to_dict() for key, value in self.metrics.items()}
        return payload
