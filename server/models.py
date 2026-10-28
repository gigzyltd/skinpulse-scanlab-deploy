"""HTTP bodies. The analysis object stays scanlab-analysis-v1, plus a lab score."""

from __future__ import annotations

from typing import Annotated, Literal, Union

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


class EligibilitySignals(_Model):
    """Separate signals. A face in a still image is not adult eligibility or liveness."""

    face: Literal["none", "single", "multiple", "unknown"]
    adult_eligibility: Literal["adult", "underage", "uncertain", "unknown"]
    live_capture: Literal["live", "not_live", "unknown"]
    manipulation: Literal["none_indicated", "suspected", "unknown"]


class EligibilityBlock(_Model):
    """Mock contract only. verified is always false. readiness stays NOT_READY."""

    capability: Literal["verification_required"]
    readiness: Literal["NOT_READY"]
    verdict: Literal[
        "adult_verified",
        "age_uncertain",
        "underage",
        "no_face",
        "multiple_faces",
        "spoof_suspected",
        "verification_unavailable",
    ]
    signals: EligibilitySignals
    mock: Literal[True]
    verified: Literal[False]
    provider: Literal["none"]
    age_model: Literal[False]
    liveness_model: Literal[False]
    label: str
    request_id: str | None = None


class DeclaredEligibilityBlock(_Model):
    """Eligibility contract v2. A self-declared adult decision, not identity or age verification."""

    contract: Literal["eligibility-v2"]
    source: Literal["self_declared"]
    decision: Literal["adult"]
    policy_version: Literal["age-declaration-v1"]
    expires_at: str
    # Remaining seconds and the server clock at this response. Not a phone-clock comparison.
    expires_in_seconds: int
    server_now: str
    issued_at: str
    verified: Literal[False]
    age_verified: Literal[False]
    identity_verified: Literal[False]
    age_model: Literal[False]
    liveness_model: Literal[False]
    label: str
    request_id: str | None = None


class AgeDeclarationCapability(_Model):
    contract: Literal["eligibility-v2"]
    source: Literal["self_declared"]
    available: bool
    policy_version: Literal["age-declaration-v1"]
    verified: Literal[False]
    age_verified: Literal[False]
    identity_verified: Literal[False]
    age_model: Literal[False]
    liveness_model: Literal[False]


class EligibilityCapability(_Model):
    capability: Literal["verification_required"]
    readiness: Literal["NOT_READY"]
    provider: Literal["none"]
    mode: Literal["enforce", "engineering"]
    mock: Literal[True]
    verified: Literal[False]
    age_model: Literal[False]
    liveness_model: Literal[False]
    label: str


class ApiAnalysis(AnalysisResult):
    received_at: str | None = None
    skin_score: SkinScoreBody | None = None
    providers: ProviderBlock
    eligibility: Annotated[
        Union[DeclaredEligibilityBlock, EligibilityBlock],
        Field(union_mode="left_to_right"),
    ] | None = None


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
    eligibility: EligibilityCapability
    age_declaration: AgeDeclarationCapability
