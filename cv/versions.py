"""Versioned constants for the phase-0 Scan Lab CV pipeline.

A formula or threshold change requires a version bump. Reported numbers are
optical proxies, not clinical measurements.
"""

from __future__ import annotations

from typing import Literal

PIPELINE_VERSION = "scanlab-cv-0.2.2"
SCHEMA_ANALYSIS: Literal["scanlab-analysis-v1"] = "scanlab-analysis-v1"
SCHEMA_REPEATABILITY: Literal["scanlab-repeatability-v1"] = "scanlab-repeatability-v1"
SCHEMA_PRIVATE_DATASET: Literal["scanlab-private-dataset-v1"] = "scanlab-private-dataset-v1"

CAPTURE_PREP_VERSION: Literal["capture-resolution-v1"] = "capture-resolution-v1"
NORMALISATION_VERSION = "colour-illum-v2"
COLOUR_NEUTRAL_VERSION = "highlight-neutral-v1"
COLOUR_IDENTITY = "identity-no-neutral-reference"
ILLUMINATION_VERSION = "masked-gaussian-luma-anchor-v2"
# Report-only background gray-world illuminant estimate (#222 calibration capture).
# It never feeds a score; changing any constant below requires a version bump.
ILLUMINANT_VERSION = "grayworld-background-v1"
ILLUMINANT_FACE_MARGIN = 0.15
ILLUMINANT_DARK_LUMA = 20.0
ILLUMINANT_CLIP = 250
ILLUMINANT_MIN_PIXELS = 2000
ILLUMINANT_MIN_FRACTION = 0.05
# Uncalibrated target for the illumination anchor. Inside the exposure comfort
# band and below the shine luma gate. Not a skin-tone calibration.
ILLUMINATION_LUMA_ANCHOR = 140.0
# Same bounds as the per-pixel shading gain. The anchor must not amplify further.
ILLUMINATION_ANCHOR_GAIN_MIN = 0.4
ILLUMINATION_ANCHOR_GAIN_MAX = 2.5
POSE_VERSION = "pose-proxy-v2"
POSE_OBLIQUE_VERSION = "pose-oblique-window-v1"
POSE_PROFILE_VERSION = "pose-profile-window-v1"
CONFIDENCE_VERSION = "confidence-quality-min-v1"

BLUR_VERSION = "blur-laplacian-variance-v1"
EXPOSURE_VERSION = "exposure-luma-v2"
FACE_SIZE_VERSION = "face-area-fraction-v2"
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

# Hard reject is extreme darkness or brightness on the protocol drawing.
# Exposure 0.20 still reads the face (mean luma about 33). Exposure 0.18
# (mean about 29.8) stays a reject. Exposure 1.42 (mean about 228) still
# scores. Exposure 1.5 (mean about 238) stays a reject. Comfort is unchanged.
# 40 and 215 are the plain-English note lines, not the reject lines.
EXPOSURE_MEAN_MIN = 32.0
EXPOSURE_MEAN_MAX = 232.0
EXPOSURE_COMFORT_MIN = 80.0
EXPOSURE_COMFORT_MAX = 190.0
EXPOSURE_NOTE_MIN = 40.0
EXPOSURE_NOTE_MAX = 215.0
EXPOSURE_CLIP_MAX = 0.10
EXPOSURE_CLIP_LUMA_LOW = 8.0
EXPOSURE_CLIP_LUMA_HIGH = 247.0

# 0.05 still requires the 80 px short side. A face around 5% of the frame
# scores. The 1% protocol control stays rejected. 0.08 is the note line.
FACE_AREA_FRACTION_MIN = 0.05
FACE_AREA_COMFORT = 0.18
FACE_AREA_NOTE_FRACTION = 0.08
FACE_MIN_SIDE_PX = 80.0

# Frontal accept meets the oblique yaw floor. A measured 45° turn still fails
# frontal. Comfort stays 5°. 15° is the plain-English note line: an accepted
# photo past it still scores, and the app says the angle is not straight-on.
# Side light on the protocol drawing shifts the yaw proxy to about 15–16°
# while the nose is still centred; that photo is not a turned head.
POSE_ABS_MAX_DEG = 25.0
POSE_COMFORT_DEG = 5.0
POSE_NOTE_DEG = 15.0
POSE_YAW_SCALE = 45.0
POSE_FRONTAL_PITCH_RATIO = 0.55
# Oblique windows are separate named views. They do not silently score a
# profile. |yaw| 25..60, not a symmetric ±15 around 45: a real left view at
# -28.7° was outside the old -60..-30 window. Oblique pitch and roll stay ±15.
OBLIQUE_YAW_TARGET_DEG = 45.0
OBLIQUE_YAW_ABS_MIN_DEG = 25.0
OBLIQUE_YAW_ABS_MAX_DEG = 60.0
OBLIQUE_PITCH_ABS_MAX_DEG = 15.0
OBLIQUE_ROLL_ABS_MAX_DEG = 15.0
# Profile is either side, past the oblique window. Uncalibrated lab window.
# It does not relax the frontal ±25° gate. A profile is never scored.
PROFILE_YAW_ABS_MIN_DEG = 60.0
PROFILE_YAW_ABS_MAX_DEG = 100.0
PROFILE_PITCH_ABS_MAX_DEG = 20.0
PROFILE_ROLL_ABS_MAX_DEG = 20.0

# lighting-cheek-balance-v3: side-light asymmetry between the two cheek
# medians, |L-R| / max(L,R). Overhead light that darkens the jaw is ignored.
# v2 hard-rejected at 0.30. On the protocol drawing that is a mild ramp
# (gradient 0.55 -> about 0.32, gradient 0.70 -> about 0.40), not a hood
# shadow. 0.15-0.45 scores with a lighting_uneven warning. Above 0.45
# (gradient 0.9 -> about 0.48) is the hard reject.
LIGHTING_SIDE_COMFORT = 0.15  # at or below: confidence 1, no warning
LIGHTING_SIDE_FAIL = 0.45     # above: hard reject (extreme side light)
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
# Per-cheek floor for the optional redness evidence. Below this the cheek is
# unavailable and is not filled from the other cheek. This does not change
# ROI_MIN_CHEEK_PIXELS or the lighting gate.
REGION_REDNESS_MIN_PIXELS = 200
REGION_REDNESS_PROVENANCE = "roi-mask-cie-a-star"
# Optional cheek colour evidence. A cheek is saturated when the fraction of its
# mask with any source or illumination-normalised colour channel outside the
# exposure rails exceeds EXPOSURE_CLIP_MAX. The rails stay 8 and 247. Luma is
# not the predicate. Specular highlights are excluded from the CIE a* mean and
# are not reconstructed; they still count in this fraction, and a set at or
# below the fraction does not withhold the cheek. This rule does not change
# the exposure gate, the lighting gate, or the five proxy formulas.
REGION_CHANNEL_CLIP_VERSION = "region-channel-clip-v1"
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
    "Illumination correction anchors median skin luma to an uncalibrated constant after shading removal. Clipped channels are not reconstructed, and skin is not a white-balance reference.",
    "Skin Score on a private ingest inverts these proxies with optical-proxy-invert/1.0.0. That adapter is uncalibrated and is not a clinical result.",
]

REPEATABILITY_STATEMENT = (
    "Repeatability thresholds only. Not an accuracy, clinical, or product GO decision."
)

NUMERIC_SOURCE = "deterministic_cv"
