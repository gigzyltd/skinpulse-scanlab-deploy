"""Report-only scene illuminant estimate for paired calibration captures (#222).

A gray-world estimate over the background: every pixel outside an expanded face
box that is neither dark nor clipped. It never changes a score, a quality check
or the colour normalisation. Its purpose is that the export can tell, *relative
to the same person's neutral reference in the same session*, whether a second
capture was taken under warmer or cooler light. Users never declare the
condition.

warmth = log2(mean R / mean B)  (positive = warmer, e.g. tungsten lamp)
tint   = log2(mean G / sqrt(mean R * mean B))  (positive = greener)

A gray-world value is scene-dependent (a red wall reads warm), which is why
the absolute number is not classified here. Only within-session differences
are meaningful.
"""

from __future__ import annotations

import math

import numpy as np

from scanlab.cv.colorimetry import rec709_luma
from scanlab.cv.models import IlluminantEstimate
from scanlab.cv.versions import (
    ILLUMINANT_CLIP,
    ILLUMINANT_DARK_LUMA,
    ILLUMINANT_FACE_MARGIN,
    ILLUMINANT_MIN_FRACTION,
    ILLUMINANT_MIN_PIXELS,
    ILLUMINANT_VERSION,
)


def estimate_illuminant(rgb: np.ndarray, face_bbox: tuple[int, int, int, int]) -> IlluminantEstimate:
    height, width = rgb.shape[:2]
    usable = np.ones((height, width), dtype=bool)
    x, y, w, h = face_bbox
    if w > 0 and h > 0:
        mx, my = int(round(w * ILLUMINANT_FACE_MARGIN)), int(round(h * ILLUMINANT_FACE_MARGIN))
        # The face box plus a margin (hair, ears) and everything below it (neck,
        # shoulders, clothing) is excluded: skin is not an illuminant reference.
        usable[max(0, y - my):, max(0, x - mx):min(width, x + w + mx)] = False
    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    clipped = (red >= ILLUMINANT_CLIP) | (green >= ILLUMINANT_CLIP) | (blue >= ILLUMINANT_CLIP)
    usable &= ~clipped & (rec709_luma(rgb) >= ILLUMINANT_DARK_LUMA)
    count = int(np.count_nonzero(usable))
    fraction = round(count / float(height * width), 4) if height * width else 0.0
    if count < ILLUMINANT_MIN_PIXELS or fraction < ILLUMINANT_MIN_FRACTION:
        return IlluminantEstimate(version=ILLUMINANT_VERSION, measured=False, warmth=None, tint=None,
                                  background_fraction=fraction)
    r, g, b = (float(channel[usable].mean()) for channel in (red, green, blue))
    warmth = round(math.log2(max(r, 1e-6) / max(b, 1e-6)), 3)
    tint = round(math.log2(max(g, 1e-6) / math.sqrt(max(r, 1e-6) * max(b, 1e-6))), 3)
    return IlluminantEstimate(version=ILLUMINANT_VERSION, measured=True, warmth=warmth, tint=tint,
                              background_fraction=fraction)
