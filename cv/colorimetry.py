"""Deterministic sRGB to CIE Lab (D65) conversion."""

from __future__ import annotations

import numpy as np

# IEC 61966-2-1 linear-sRGB to XYZ matrix. The white point is this matrix
# applied to (1, 1, 1), so reference white lands on a*=0, b*=0.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_WHITE = _RGB_TO_XYZ.sum(axis=1)
_DELTA = 6.0 / 29.0
_DELTA_CUBE = _DELTA ** 3
_DELTA_SCALE = 3.0 * _DELTA * _DELTA


def srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    c = channel.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab_f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA_CUBE, np.cbrt(t), t / _DELTA_SCALE + 4.0 / 29.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Return an array of L, a, b with L in about 0..100."""

    red = srgb_to_linear(rgb[:, :, 0])
    green = srgb_to_linear(rgb[:, :, 1])
    blue = srgb_to_linear(rgb[:, :, 2])
    x = _RGB_TO_XYZ[0, 0] * red + _RGB_TO_XYZ[0, 1] * green + _RGB_TO_XYZ[0, 2] * blue
    y = _RGB_TO_XYZ[1, 0] * red + _RGB_TO_XYZ[1, 1] * green + _RGB_TO_XYZ[1, 2] * blue
    z = _RGB_TO_XYZ[2, 0] * red + _RGB_TO_XYZ[2, 1] * green + _RGB_TO_XYZ[2, 2] * blue
    fx = _lab_f(x / _WHITE[0])
    fy = _lab_f(y / _WHITE[1])
    fz = _lab_f(z / _WHITE[2])
    lightness = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return np.dstack([lightness, a_star, b_star])


def rec709_luma(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue
