"""Deterministic optical-proxy sub-metrics.

Every public value is rounded half-up to 4 decimal places. Nothing in this
module calls a model or a network.
"""

from __future__ import annotations

import numpy as np

from scanlab.cv.colorimetry import rec709_luma, rgb_to_lab
from scanlab.cv.geometry import Regions
from scanlab.cv.models import SubMetric
from scanlab.cv.normalise import Normalised, masked_gaussian
from scanlab.cv.support import ensure_deterministic_cv, round_half_up, round_map
from scanlab.cv.versions import (
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


def _texture(normalised, regions, quality_confidence, specular) -> SubMetric:
    ensure_deterministic_cv()
    import cv2

    mask = regions.cheeks & ~specular
    luma = rec709_luma(normalised.illuminated)
    gray = np.clip(luma, 0, 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    values = laplacian[mask]
    variance = float(values.var()) if values.size else 0.0
    return _metric(
        "texture",
        _score(variance, TEXTURE_LAP_SPAN),
        _roi_confidence(int(values.size), quality_confidence),
        {"laplacian_variance": variance, "pixels": float(values.size)},
    )


def _pores(normalised, regions, quality_confidence, specular) -> SubMetric:
    ensure_deterministic_cv()
    import cv2

    mask = regions.cheeks & ~specular
    luma = rec709_luma(normalised.illuminated)
    gray = np.clip(luma, 0, 255).astype(np.uint8)
    kernel_size = int(round(regions.face_width / 35.0))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(max(kernel_size, 3), 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        fraction = 0.0
    else:
        fraction = float(np.mean(blackhat[mask] >= PORE_BLACKHAT_THRESHOLD))
    return _metric(
        "pores",
        _score(fraction, PORE_FRACTION_SPAN),
        _roi_confidence(pixels, quality_confidence),
        {"pore_fraction": fraction, "pixels": float(pixels), "kernel": float(kernel_size)},
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
