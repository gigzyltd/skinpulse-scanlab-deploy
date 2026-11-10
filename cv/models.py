"""JSON contracts for analysis and repeatability. Numeric fields are measurements."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

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


RegionId = Literal["image_left_cheek", "image_right_cheek"]
RegionStatus = Literal["measured", "unavailable", "uncertain", "saturated"]


class RegionProvenance(_Model):
    """Identity of the measurement. Image sides are picture sides, not a person's left or right."""

    method: Literal["roi-mask-cie-a-star"] = "roi-mask-cie-a-star"
    pipeline_version: str
    metric_version: str
    normalisation_version: str
    illumination: str


class RegionRedness(_Model):
    """One cheek's redness on that ROI mask.

    ``value`` is present only when ``status`` is ``measured``. ``unavailable``
    means the mask is too small, ``saturated`` means a colour channel on the
    source image or the illumination-normalised image is clipped on more than
    the exposure clip fraction of this mask and was not reconstructed, and
    ``uncertain`` means the capture is outside the existing quality gate so no
    number is reported.
    """

    id: RegionId
    metric: Literal["redness"] = "redness"
    status: RegionStatus
    value: float | None = None
    unit: Literal["score_0_100"] = "score_0_100"
    measurement_kind: Literal["optical_proxy"] = "optical_proxy"
    clinical_validity: Literal["not_established"] = "not_established"
    pixels: int = Field(ge=0)
    description: str
    provenance: RegionProvenance

    @model_validator(mode="after")
    def value_follows_status(self) -> "RegionRedness":
        if self.status == "measured" and self.value is None:
            raise ValueError("A measured region includes a value.")
        if self.status != "measured" and self.value is not None:
            raise ValueError("A region without a measurement has no value.")
        return self


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
    regions: list[RegionRedness] | None = Field(
        default=None,
        description=(
            "Optional image-left and image-right cheek redness from the ROI masks. "
            "Omitted when the analyze response has no region evidence. "
            "Image-left and image-right are sides of the picture. Not a clinical result."
        ),
    )
    warnings: list[str]
    limitations: list[str]

    @model_serializer(mode="wrap")
    def _omit_absent_region_evidence(self, handler, _info):
        data = handler(self)
        if not isinstance(data, dict):
            return data
        if data.get("regions") is None:
            data.pop("regions", None)
            return data
        for region in data["regions"]:
            if isinstance(region, dict) and region.get("value") is None:
                region.pop("value", None)
        return data


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
