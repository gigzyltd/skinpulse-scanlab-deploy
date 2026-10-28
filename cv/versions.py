"""Versioned constants for the phase-0 Scan Lab CV pipeline.

A formula or threshold change requires a version bump. Reported numbers are
optical proxies, not clinical measurements.
"""

from __future__ import annotations

from typing import Literal

PIPELINE_VERSION = "scanlab-cv-0.2.1"
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
POSE_PROFILE_VERSION = "pose-profile-window-v1"
CONFIDENCE_VERSION = "confidence-quality-min-v1"

BLUR_VERSION = "blur-laplacian-variance-v1"
EXPOSURE_VERSION = "exposure-luma-v1"
FACE_SIZE_VERSION = "face-area-fraction-v1"
LIGHTING_VERSION = "lighting-cheek-balance-v3"
ROI_VERSION = "roi-pixel-count-v1"
ROI_OBLIQUE_VERSION = "roi-oblique-near-cheek-v1"
ROI_PROFILE_VERSION = "roi-profile-near-cheek-v1"

REDNESS_VERSION = "redness-a-star-v1"
TEXTURE_VERSION = "texture-laplacian-facescale-v0.3"
PORES_VERSION = "pores-blackhat-facescale-v2"
PIGMENTATION_VERSION = "pigmentation-local-contrast-v1"
SHINE_VERSION = "shine-specular-fraction-uncalibrated-v0.2"

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
        "Laplacian energy of illumination-normalised cheek luma, mapped to 0-100 "
        "with the uncalibrated v0.2 span. Optical texture proxy. A value of 0 or 100 "
        "is clipped and is not a calibrated measurement."
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
        "normalisation, mapped to 0-100 with the uncalibrated v0.2 specular gate. "
        "Specular highlight proxy, not sebum. A value of 0 or 100 is clipped."
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
# |yaw| 25..60, not a symmetric ±15 around 45: a real left view at -28.7° was
# outside the old -60..-30 window. Pitch and roll stay ±15.
OBLIQUE_YAW_TARGET_DEG = 45.0
OBLIQUE_YAW_ABS_MIN_DEG = 25.0
OBLIQUE_YAW_ABS_MAX_DEG = 60.0
OBLIQUE_PITCH_ABS_MAX_DEG = 15.0
OBLIQUE_ROLL_ABS_MAX_DEG = 15.0
# Profile is either side, past the oblique window. Uncalibrated lab window.
# It does not relax the frontal ±15° gate. A profile is never scored.
PROFILE_YAW_ABS_MIN_DEG = 60.0
PROFILE_YAW_ABS_MAX_DEG = 100.0
PROFILE_PITCH_ABS_MAX_DEG = 20.0
PROFILE_ROLL_ABS_MAX_DEG = 20.0

# lighting-cheek-balance-v2 (SP-013 #3): side-light asymmetry between the two cheek
# medians, |L-R| / max(L,R). Overhead/indoor light that darkens jaw or eye sockets is
# ignored (v1's 3x3 face-grid range rejected normal room light with confidence 0).
# Calibration: 9 private frontal/oblique indoor phone photos 0.03-0.12; synthetic
# horizontal ramp 0.2 -> 0.13, 0.4 -> 0.25, 0.55 -> 0.32.
LIGHTING_SIDE_COMFORT = 0.20  # at or below: confidence 1, no warning (SP-026: was 0.15)
LIGHTING_SIDE_FAIL = 0.42     # above: hard reject, one side clearly half-dark (SP-026: was 0.30)
LIGHTING_WARN_MIN_CONFIDENCE = 0.4  # confidence at the fail edge of the warning band
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
# Uncalibrated v0.2. Span 80 was tuned on protocol drawings and pinned every
# real cheek at 100. 2000 puts a laplacian variance of a few hundred near the
# middle of the scale. It is not a fit to a photo set.
TEXTURE_LAP_SPAN = 2000.0
PORE_FRACTION_SPAN = 0.08
PORE_BLACKHAT_THRESHOLD = 12.0
# SP-013 #4 (v0.3 texture / v2 pores): real Android JPEGs (sensor noise + in-camera
# sharpening) pinned the v0.2 texture and v1 pores at 100. Cheeks are now measured on
# a face-normalised scale (face width -> FACE_SCALE_WIDTH_PX, area downsample only),
# lightly denoised (Gaussian sigma TEXTURE_DENOISE_SIGMA) and mapped with a soft
# curve 100*(1-exp(-raw/span)) so nothing hard-clips at 100. Private indoor phone
# photos: lap var ~26-34, pore fraction ~0.02-0.04 (score ~30-40); simulated heavy
# sharpening / dim+noise: lap var 60-170, fraction 0.03-0.16 (score 55-88).
FACE_SCALE_WIDTH_PX = 320.0
TEXTURE_DENOISE_SIGMA = 1.0
TEXTURE_SOFT_SPAN = 80.0
PORE_SOFT_SPAN = 0.08
PORE_KERNEL_PX = 9
PORE_BLACKHAT_THRESHOLD_V2 = 6.0
PIGMENT_RESIDUAL_SPAN = 8.0
# Uncalibrated v0.2. Luma 220 and spread 18 counted no T-zone pixel on real
# skin, so shine was 0 on every photo. 190 / 24 admits a bright, low-chroma
# highlight without treating ordinary cheek chroma as shine. Span 0.15 keeps
# a small highlight fraction off the ceiling. Not a sebum calibration.
SHINE_FRACTION_SPAN = 0.15
SHINE_LUMA_MIN = 190.0
SHINE_SPREAD_MAX = 24.0
# max/min face-mask area fraction within one target. Above this, ingest warns
# face_distance_mismatch. It does not change the repeatability gate.
FACE_DISTANCE_RATIO_MAX = 1.5

LIMITATIONS = [
    "Optical proxies only. Not hydration, sebum, skin age, or a medical diagnosis.",
    "A repeatability-threshold GO is not an accuracy, clinical, or product GO.",
    "Phase 0 does not include a consented multi-device capture study.",
    "EXIF orientation is applied and the longer edge is downscaled before the quality gate. EXIF and GPS are not retained.",
    "Left and right 45 degree views have their own pose and ROI gates. The near cheek is the camera-facing side: nose toward image-right means the image-left cheek. Optical scores use the frontal view unless an explicit views_calibrated flag is set, and that flag is not a completed calibration.",
    "A profile view is quality-gated and never scored.",
    "Texture and shine use uncalibrated v0.2 cutoffs. A 0 or 100 on those proxies emits metric_saturated and is not a calibrated measurement.",
    "Skin Score on a private ingest inverts these proxies with optical-proxy-invert/1.0.0. That adapter is uncalibrated and is not a clinical result.",
]

REPEATABILITY_STATEMENT = (
    "Repeatability thresholds only. Not an accuracy, clinical, or product GO decision."
)

NUMERIC_SOURCE = "deterministic_cv"
