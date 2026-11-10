"""Colour and illumination normalisation.

Colour gains come only from bright low-chroma pixels (a gray card or a specular
neutral). Skin itself is not the white balance reference, so a red cast that is
the skin is not removed. Illumination is a masked Gaussian of luminance; metrics
that should ignore shading use the corrected image, and shine does not.

The shading removal leaves a global exposure scale. CIE a* and high-pass energy
then track brightness. After the shading field is divided out, one gain puts
the median skin luma on a fixed uncalibrated anchor. The gain is the same on
every channel, so a real chromatic difference is not rebalanced away. The
shading scale and that gain stay inside 0.4 to 2.5, and the combined scale is
clipped to the same range, so a dark capture is not amplified further. Clipped
source channels are not reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scanlab.cv.colorimetry import rec709_luma
from scanlab.cv.geometry import Regions
from scanlab.cv.support import ensure_deterministic_cv
from scanlab.cv.versions import (
    COLOUR_IDENTITY,
    COLOUR_NEUTRAL_VERSION,
    ILLUMINATION_ANCHOR_GAIN_MAX,
    ILLUMINATION_ANCHOR_GAIN_MIN,
    ILLUMINATION_LUMA_ANCHOR,
)


@dataclass
class Normalised:
    colour: np.ndarray
    illuminated: np.ndarray
    colour_mode: str
    warnings: list[str]


def normalise(rgb: np.ndarray, regions: Regions) -> Normalised:
    gains, mode = neutral_gains(rgb)
    colour = np.clip(rgb.astype(np.float64) * gains.reshape(1, 1, 3), 0.0, 255.0)
    warnings = ["colour_reference_absent"] if mode == COLOUR_IDENTITY else []
    illuminated = _illuminate(colour, regions)
    return Normalised(colour=colour, illuminated=illuminated, colour_mode=mode, warnings=warnings)


def neutral_gains(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    luma = rec709_luma(rgb)
    spread = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    neutral = (spread <= 12.0) & (luma >= 40.0)
    if int(np.count_nonzero(neutral)) < 40:
        return np.ones(3, dtype=np.float64), COLOUR_IDENTITY
    cutoff = float(np.median(luma[neutral]))
    chosen = neutral & (luma >= cutoff)
    if int(np.count_nonzero(chosen)) < 40:
        chosen = neutral
    mean = np.array([red[chosen].mean(), green[chosen].mean(), blue[chosen].mean()], dtype=np.float64)
    gray = float(mean.mean())
    gains = np.clip(gray / np.maximum(mean, 1e-6), 0.6, 1.6)
    return gains, COLOUR_NEUTRAL_VERSION


def _illuminate(colour: np.ndarray, regions: Regions) -> np.ndarray:
    luma = rec709_luma(colour)
    sigma = min(41.0, max(2.0, regions.face_width * 0.08))
    illumination = masked_gaussian(luma, regions.skin, sigma)
    valid = regions.skin & (luma > 1.0)
    reflectance = np.ones_like(luma)
    if np.any(valid):
        reflectance[valid] = luma[valid] / np.maximum(illumination[valid], 1e-3)
        median_luma = float(np.median(luma[valid]))
        median_reflectance = float(np.median(reflectance[valid]))
        scale = median_luma / max(median_reflectance, 1e-3)
        target = reflectance * scale
        scale_map = np.ones_like(luma)
        scale_map[valid] = target[valid] / np.maximum(luma[valid], 1e-3)
        scale_map = np.clip(scale_map, ILLUMINATION_ANCHOR_GAIN_MIN, ILLUMINATION_ANCHOR_GAIN_MAX)
        corrected = luma * scale_map
        median_corrected = float(np.median(corrected[valid]))
        if median_corrected > 1e-3:
            gain = ILLUMINATION_LUMA_ANCHOR / median_corrected
            gain = float(min(max(gain, ILLUMINATION_ANCHOR_GAIN_MIN), ILLUMINATION_ANCHOR_GAIN_MAX))
            scale_map = np.clip(scale_map * gain, ILLUMINATION_ANCHOR_GAIN_MIN, ILLUMINATION_ANCHOR_GAIN_MAX)
    else:
        scale_map = np.ones_like(luma)
    return np.clip(colour * scale_map[:, :, None], 0.0, 255.0)


def masked_gaussian(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    ensure_deterministic_cv()
    import cv2

    sigma = max(float(sigma), 0.8)
    kernel = int(round(sigma * 6.0)) | 1
    kernel = max(3, kernel)
    limit = int(min(values.shape[0], values.shape[1], 255))
    if limit % 2 == 0:
        limit -= 1
    kernel = min(kernel, max(limit, 3))
    if kernel % 2 == 0:
        kernel -= 1
    weight = mask.astype(np.float64)
    numerator = cv2.GaussianBlur(values * weight, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.GaussianBlur(weight, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT)
    out = np.zeros_like(values, dtype=np.float64)
    valid = denominator > 1e-6
    out[valid] = numerator[valid] / denominator[valid]
    return out
