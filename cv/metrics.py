"""Deterministic optical-proxy sub-metrics.

Every public value is rounded half-up to 4 decimal places. Nothing in this
module calls a model or a network.
"""

from __future__ import annotations

import math
import numpy as np

from scanlab.cv.colorimetry import rec709_luma, rgb_to_lab
from scanlab.cv.geometry import Regions
from scanlab.cv.models import SubMetric
from scanlab.cv.normalise import Normalised, masked_gaussian
from scanlab.cv.support import ensure_deterministic_cv, round_half_up, round_map
from scanlab.cv.versions import (
    FACE_SCALE_WIDTH_PX,
    PORE_BLACKHAT_THRESHOLD_V2,
    PORE_KERNEL_PX,
    PORE_SOFT_SPAN,
    TEXTURE_DENOISE_SIGMA,
    TEXTURE_SOFT_SPAN,
    COLOUR_IDENTITY,
    METRIC_DESCRIPTIONS,
    METRIC_VERSIONS,
    PIGMENT_RESIDUAL_SPAN,
    PORE_BLACKHAT_THRESHOLD,
    PORE_FRACTION_SPAN,
    REDNESS_A_LOW,
    REDNESS_A_SPAN,
    ROI_CONFIDENCE_PIXELS,
    SHINE_FRACTION_SPAN,
    SHINE_LUMA_MIN,
    SHINE_SPREAD_MAX,
    TEXTURE_LAP_SPAN,
)


def _score(raw: float, span: float, low: float = 0.0) -> float:
    if span <= 0:
        return 0.0
    value = (raw - low) / span * 100.0
    return float(min(max(value, 0.0), 100.0))


def _specular(rgb: np.ndarray) -> np.ndarray:
    luma = rec709_luma(rgb)
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return (luma >= SHINE_LUMA_MIN) & (spread <= SHINE_SPREAD_MAX)


def _metric(
    name: str,
    value: float,
    confidence: float,
    raw: dict[str, float],
) -> SubMetric:
    return SubMetric(
        name=name,  # type: ignore[arg-type]
        value=round_half_up(value, 4),
        confidence=round_half_up(min(max(confidence, 0.0), 1.0), 4),
        version=METRIC_VERSIONS[name],
        description=METRIC_DESCRIPTIONS[name],
        raw=round_map(raw),
    )


def _roi_confidence(pixels: int, quality_confidence: float, colour_penalty: float = 1.0) -> float:
    coverage = min(pixels / float(ROI_CONFIDENCE_PIXELS), 1.0)
    return min(quality_confidence, coverage) * colour_penalty


def saturation_warnings(metrics: list[SubMetric]) -> list[str]:
    """Name a texture or shine proxy that is clipped at 0 or 100.

    Those two scales are uncalibrated v0.2. A floor or ceiling means the cutoff
    still does not describe the capture. Other proxies are unchanged.
    """

    warnings = []
    for metric in metrics:
        if metric.name in {"texture", "shine_oiliness"} and metric.value in (0.0, 100.0):
            warnings.append(f"metric_saturated:{metric.name}")
    return warnings


def compute_metrics(normalised: Normalised, regions: Regions, quality_confidence: float) -> list[SubMetric]:
    colour_penalty = 0.9 if normalised.colour_mode == COLOUR_IDENTITY else 1.0
    specular_colour = _specular(normalised.colour)
    specular_illum = _specular(normalised.illuminated)
    return [
        _redness(normalised, regions, quality_confidence, colour_penalty, specular_illum),
        _texture(normalised, regions, quality_confidence, specular_illum),
        _pores(normalised, regions, quality_confidence, specular_illum),
        _pigmentation(normalised, regions, quality_confidence, specular_illum),
        _shine(normalised, regions, quality_confidence, specular_colour),
    ]


def _redness(normalised, regions, quality_confidence, colour_penalty, specular) -> SubMetric:
    mask = regions.cheeks & ~specular
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        a_mean = 0.0
    else:
        lab = rgb_to_lab(np.clip(normalised.illuminated, 0, 255))
        a_mean = float(lab[:, :, 1][mask].mean())
    return _metric(
        "redness",
        _score(a_mean, REDNESS_A_SPAN, REDNESS_A_LOW),
        _roi_confidence(pixels, quality_confidence, colour_penalty),
        {"a_star_mean": a_mean, "pixels": float(pixels)},
    )


def _soft_score(raw: float, span: float) -> float:
    if span <= 0 or raw <= 0:
        return 0.0
    return float(min(100.0 * (1.0 - math.exp(-raw / span)), 99.99))


def _face_scaled_cheeks(normalised, regions, specular):
    """Luma and cheek mask resampled so the face is FACE_SCALE_WIDTH_PX wide, lightly denoised."""
    ensure_deterministic_cv()
    import cv2

    mask = regions.cheeks & ~specular
    luma = np.clip(rec709_luma(normalised.illuminated), 0, 255).astype(np.float32)
    scale = FACE_SCALE_WIDTH_PX / max(float(regions.face_width), 1.0)
    if scale < 1.0:
        luma = cv2.resize(luma, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (luma.shape[1], luma.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    smooth = cv2.GaussianBlur(luma, (0, 0), TEXTURE_DENOISE_SIGMA)
    return smooth, mask


def _texture(normalised, regions, quality_confidence, specular) -> SubMetric:
    import cv2

    smooth, mask = _face_scaled_cheeks(normalised, regions, specular)
    laplacian = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
    values = laplacian[mask]
    variance = float(values.var()) if values.size else 0.0
    return _metric(
        "texture",
        _soft_score(variance, TEXTURE_SOFT_SPAN),
        _roi_confidence(int(np.count_nonzero(regions.cheeks & ~specular)), quality_confidence),
        {"laplacian_variance": variance, "pixels": float(values.size)},
    )


def _pores(normalised, regions, quality_confidence, specular) -> SubMetric:
    import cv2

    smooth, mask = _face_scaled_cheeks(normalised, regions, specular)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PORE_KERNEL_PX, PORE_KERNEL_PX))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    pixels = int(np.count_nonzero(mask))
    fraction = float(np.mean(blackhat[mask] >= PORE_BLACKHAT_THRESHOLD_V2)) if pixels else 0.0
    return _metric(
        "pores",
        _soft_score(fraction, PORE_SOFT_SPAN),
        _roi_confidence(int(np.count_nonzero(regions.cheeks & ~specular)), quality_confidence),
        {"pore_fraction": fraction, "pixels": float(pixels), "kernel": float(PORE_KERNEL_PX)},
    )


def _pigmentation(normalised, regions, quality_confidence, specular) -> SubMetric:
    mask = (regions.cheeks | regions.forehead) & ~specular
    luma = rec709_luma(normalised.illuminated)
    sigma = min(36.0, max(2.0, regions.face_width * 0.12))
    local = masked_gaussian(luma, mask, sigma)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        residual = 0.0
    else:
        dark = np.clip(local - luma, 0.0, None)
        residual = float(dark[mask].mean())
    return _metric(
        "pigmentation",
        _score(residual, PIGMENT_RESIDUAL_SPAN),
        _roi_confidence(pixels, quality_confidence),
        {"mean_dark_residual": residual, "pixels": float(pixels)},
    )


def _shine(normalised, regions, quality_confidence, specular) -> SubMetric:
    mask = regions.tzone
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        fraction = 0.0
    else:
        fraction = float(np.count_nonzero(specular & mask)) / float(pixels)
    return _metric(
        "shine_oiliness",
        _score(fraction, SHINE_FRACTION_SPAN),
        _roi_confidence(pixels, quality_confidence),
        {"specular_fraction": fraction, "pixels": float(pixels)},
    )
