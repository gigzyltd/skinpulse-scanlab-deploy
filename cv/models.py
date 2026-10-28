"""JSON contracts for analysis and repeatability. Numeric fields are measurements."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scanlab.cv.versions import (
    CV_PERCENT_MAX,
    ICC_MIN,
    MEDIAN_ABS_DELTA_MAX,
    P90_ABS_DELTA_MAX,
    REPEATABILITY_STATEMENT,
)

MetricName = Literal["redness", "texture", "pores", "pigmentation", "shine_oiliness"]
GateResult = Literal["GO", "NO-GO", "INCONCLUSIVE"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityCheck(_Model):
    name: str
    passed: bool
    value: float
    confidence: float
    version: str
    components: dict[str, float]


class SubMetric(_Model):
    name: MetricName
    value: float
    confidence: float
    version: str
    unit: Literal["score_0_100"] = "score_0_100"
    measurement_kind: Literal["optical_proxy"] = "optical_proxy"
    clinical_validity: Literal["not_established"] = "not_established"
    description: str
    raw: dict[str, float]


class NormalisationInfo(_Model):
    version: str
    colour_mode: str
    illumination: str


class CapturePreparation(_Model):
    version: Literal["capture-resolution-v1"]
    orientation: int
    source_width: int
    source_height: int
    downscaled: bool
    exif_stripped: Literal[True] = True
    gps_stripped: Literal[True] = True


class AnalysisResult(_Model):
    schema_version: Literal["scanlab-analysis-v1"]
    pipeline_version: str
    numeric_source: Literal["deterministic_cv"]
    llm_used: Literal[False] = False
    image_sha256: str
    width: int
    height: int
    view: Literal["frontal", "left_45", "right_45", "profile"] = "frontal"
    scoring_policy: Literal["frontal", "withheld_until_calibrated", "views_calibrated", "never_scored"] = "frontal"
    capture_preparation: CapturePreparation | None = None
    landmark_backend: str
    landmark_backend_version: str
    normalisation: NormalisationInfo | None
    accepted: bool
    metrics_included: bool
    rejection_reasons: list[str]
    quality_checks: list[QualityCheck]
    metrics: list[SubMetric]
    warnings: list[str]
    limitations: list[str]


class GateThresholds(_Model):
    icc_min: float = ICC_MIN
    median_abs_delta_max: float = MEDIAN_ABS_DELTA_MAX
    p90_abs_delta_max: float = P90_ABS_DELTA_MAX
    cv_percent_max: float = CV_PERCENT_MAX
    icc_definition: str = "icc-2.1-absolute-agreement-v1"
    delta_definition: str = "within-target-pairwise-abs-v1"
    cv_definition: str = "cv-pooled-within-target-v1"


class MetricRepeatability(_Model):
    metric: str
    metric_version: str | None
    targets: int
    replicates: int
    icc: float | None
    icc_version: str
    icc_1_1: float | None
    icc_3_1: float | None
    median_abs_delta: float | None
    p90_abs_delta: float | None
    cv_percent: float | None
    cv_version: str
    delta_version: str
    repeatability_gate: GateResult
    gate_reasons: list[str]


class RepeatabilityReport(_Model):
    schema_version: Literal["scanlab-repeatability-v1"]
    decision_scope: Literal["repeatability_thresholds_only"] = "repeatability_thresholds_only"
    product_validity: Literal["not_established"] = "not_established"
    clinical_go: Literal[False] = False
    statement: str = REPEATABILITY_STATEMENT
    dataset_class: str
    independent_captures: bool | None
    numeric_source: Literal["deterministic_cv", "supplied_measurements"]
    llm_used: Literal[False] = False
    thresholds: GateThresholds = Field(default_factory=GateThresholds)
    metrics: list[MetricRepeatability]
    repeatability_gate: GateResult
    gate_reasons: list[str]
    warnings: list[str]
    dropped_targets: list[dict[str, str]]
    limitations: list[str]
