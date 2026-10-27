"""Versioned constants for the phase-0 Scan Lab CV pipeline.

A formula or threshold change requires a version bump. Reported numbers are
optical proxies, not clinical measurements.
"""

from __future__ import annotations

from typing import Literal

PIPELINE_VERSION = "scanlab-cv-0.2.0"
SCHEMA_ANALYSIS: Literal["scanlab-analysis-v1"] = "scanlab-analysis-v1"
SCHEMA_REPEATABILITY: Literal["scanlab-repeatability-v1"] = "scanlab-repeatability-v1"
SCHEMA_PRIVATE_DATASET: Literal["scanlab-private-dataset-v1"] = "scanlab-private-dataset-v1"

CAPTURE_PREP_VERSION: Literal["capture-resolution-v1"] = "capture-resolution-v1"
NORMALISATION_VERSION = "colour-illum-v1"
COLOUR_NEUTRAL_VERSION = "highlight-neutral-v1"
COLOUR_IDENTITY = "identity-no-neutral-reference"
ILLUMINATION_VERSION = "masked-gaussian-v1"
POSE_VERSION = "pose-proxy-v1"
POSE_OBLIQUE_VERSION = "pose-oblique-window-v1"
CONFIDENCE_VERSION = "confidence-quality-min-v1"

BLUR_VERSION = "blur-laplacian-variance-v1"
EXPOSURE_VERSION = "exposure-luma-v1"
FACE_SIZE_VERSION = "face-area-fraction-v1"
LIGHTING_VERSION = "lighting-grid-cv-v1"
ROI_VERSION = "roi-pixel-count-v1"
ROI_OBLIQUE_VERSION = "roi-oblique-near-cheek-v1"

REDNESS_VERSION = "redness-a-star-v1"
TEXTURE_VERSION = "texture-laplacian-energy-v1"
PORES_VERSION = "pores-blackhat-v1"
PIGMENTATION_VERSION = "pigmentation-local-contrast-v1"
SHINE_VERSION = "shine-specular-fraction-v1"

ICC_VERSION = "icc-2.1-absolute-agreement-v1"
ICC_11_VERSION = "icc-1.1-one-way-v1"
ICC_31_VERSION = "icc-3.1-consistency-v1"
CV_VERSION = "cv-pooled-within-target-v1"
DELTA_VERSION = "within-target-pairwise-abs-v1"

GEOMETRIC_BACKEND_VERSION = "geometric-synthetic-protocol-v1"
MEDIAPIPE_BACKEND_VERSION = "mediapipe-face-landmarker-v1"

METRIC_NAMES = (
    "redness",
    "texture",
    "pores",
    "pigmentation",
    "shine_oiliness",
)

METRIC_VERSIONS = {
    "redness": REDNESS_VERSION,
    "texture": TEXTURE_VERSION,
    "pores": PORES_VERSION,
    "pigmentation": PIGMENTATION_VERSION,
    "shine_oiliness": SHINE_VERSION,
}

METRIC_DESCRIPTIONS = {
    "redness": (
        "Mean CIE a* on the cheeks after colour normalisation, mapped to 0-100. "
        "Optical redness proxy, not a clinical erythema index."
    ),
    "texture": (
        "Laplacian energy of illumination-normalised cheek luma, mapped to 0-100. "
        "Optical texture proxy."
    ),
    "pores": (
        "Fraction of cheek pixels whose black-hat response exceeds a fixed threshold, "
        "mapped to 0-100. Optical pore proxy."
    ),
    "pigmentation": (
        "Mean dark residual against a local luminance baseline on the cheeks and "
        "forehead, mapped to 0-100. Optical pigment-variation proxy."
    ),
    "shine_oiliness": (
        "Fraction of T-zone pixels that are bright and low-chroma after colour "
        "normalisation, mapped to 0-100. Specular highlight proxy, not sebum."
    ),
}

# Capture-quality gate. Comfort bands are where confidence saturates; failing
# a hard threshold rejects the capture and emits no skin metric.
BLUR_LAPLACIAN_MIN = 40.0
BLUR_COMFORT = 120.0

EXPOSURE_MEAN_MIN = 40.0
EXPOSURE_MEAN_MAX = 215.0
EXPOSURE_COMFORT_MIN = 80.0
EXPOSURE_COMFORT_MAX = 190.0
EXPOSURE_CLIP_MAX = 0.10
EXPOSURE_CLIP_LUMA_LOW = 8.0
EXPOSURE_CLIP_LUMA_HIGH = 247.0

FACE_AREA_FRACTION_MIN = 0.08
FACE_AREA_COMFORT = 0.18
FACE_MIN_SIDE_PX = 80.0

POSE_ABS_MAX_DEG = 15.0
POSE_COMFORT_DEG = 5.0
POSE_YAW_SCALE = 45.0
POSE_FRONTAL_PITCH_RATIO = 0.55
# Oblique windows are separate constants. They do not change the frontal gate.
OBLIQUE_YAW_TARGET_DEG = 45.0
OBLIQUE_YAW_TOLERANCE_DEG = 15.0
OBLIQUE_PITCH_ABS_MAX_DEG = 15.0
OBLIQUE_ROLL_ABS_MAX_DEG = 15.0

LIGHTING_CELL_CV_MAX = 0.15
LIGHTING_CELL_CV_COMFORT = 0.05
LIGHTING_CELL_RANGE_MAX = 45.0
LIGHTING_MIN_CELLS = 4
LIGHTING_MIN_CELL_PIXELS = 30

ROI_MIN_CHEEK_PIXELS = 200
ROI_MIN_TZONE_PIXELS = 80
ROI_MIN_SKIN_PIXELS = 500
ROI_CONFIDENCE_PIXELS = 1500
ROI_OBLIQUE_NEAR_CHEEK_MIN = 200
ROI_OBLIQUE_TZONE_MIN = 80
ROI_OBLIQUE_SKIN_MIN = 500

# Repeatability gate. Comparisons use the reported figures.
ICC_MIN = 0.90
MEDIAN_ABS_DELTA_MAX = 2.0
P90_ABS_DELTA_MAX = 3.0
CV_PERCENT_MAX = 5.0
SMALL_SAMPLE_TARGETS = 5
SMALL_SAMPLE_REPLICATES = 3

MIN_EDGE_PX = 64
DEFAULT_MAX_EDGE_PX = 1600
# Reject before decode when the oriented edge exceeds this. Phone captures,
# including the 3088 px intake photos, stay under it and are downscaled.
ABSOLUTE_MAX_SOURCE_EDGE_PX = 12_000
DEFAULT_MAX_BYTES = 8_000_000

REDNESS_A_LOW = 5.0
REDNESS_A_SPAN = 25.0
TEXTURE_LAP_SPAN = 80.0
PORE_FRACTION_SPAN = 0.08
PORE_BLACKHAT_THRESHOLD = 12.0
PIGMENT_RESIDUAL_SPAN = 8.0
SHINE_FRACTION_SPAN = 0.08
SHINE_LUMA_MIN = 220.0
SHINE_SPREAD_MAX = 18.0

LIMITATIONS = [
    "Optical proxies only. Not hydration, sebum, skin age, or a medical diagnosis.",
    "A repeatability-threshold GO is not an accuracy, clinical, or product GO.",
    "Phase 0 does not include a consented multi-device capture study.",
    "EXIF orientation is applied and the longer edge is downscaled before the quality gate. EXIF and GPS are not retained.",
    "Left and right 45 degree views have their own pose and ROI gates. Optical scores use the frontal view unless an explicit views_calibrated flag is set, and that flag is not a completed calibration.",
]

REPEATABILITY_STATEMENT = (
    "Repeatability thresholds only. Not an accuracy, clinical, or product GO decision."
)

NUMERIC_SOURCE = "deterministic_cv"
