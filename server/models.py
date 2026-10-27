"""HTTP bodies. The analysis object stays scanlab-analysis-v1, plus a lab score."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scanlab.cv.models import AnalysisResult


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkinScoreBody(_Model):
    version: Literal["1.0.0"]
    overall: float | None
    confidence: float
    noise_threshold: float
    threshold_kind: Literal["proposed_lab_gate"]
    weights_used: dict[str, float]
    used: dict[str, float]
    missing: list[str]
    confidence_method: str
    input_mapping: Literal["optical-proxy-invert/1.0.0"]
    clinical_validity: Literal["not_established"]


class ProviderMetricBody(_Model):
    metric: str
    value: float | None
    scale: str
    confidence: float | None = None


class ProviderSlot(_Model):
    enabled: bool
    called: bool
    success: bool | None = None
    error_code: str | None = None
    emits_scores: bool | None = None
    metrics: list[ProviderMetricBody] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    explanation: str | None = None
    cost_consumed: bool | None = None
    cost_unit: str | None = None
    cost_amount: float | None = None
    latency_ms: float | None = None


class ProviderBlock(_Model):
    youcam: ProviderSlot
    ailabtools: ProviderSlot
    gemini: ProviderSlot


class ApiAnalysis(AnalysisResult):
    skin_score: SkinScoreBody | None = None
    providers: ProviderBlock


class HealthBody(_Model):
    status: Literal["ok"]
    service: Literal["scanlab-api"]
    api_version: str
    pipeline_version: str
    score_version: Literal["1.0.0"]
    numeric_source: Literal["deterministic_cv"]
    llm_used: Literal[False]
    landmark_backend: str
    mediapipe: dict[str, bool]
    providers: dict[str, bool]
