"""Procedural face drawings for tests.

These are not photographs, not people, and not measurement evidence. The
geometric backend is the detector paired with this drawing protocol.
"""

from __future__ import annotations

import math

import numpy as np

from scanlab.cv.geometric_backend import skin_mask
from scanlab.cv.support import ensure_deterministic_cv


def render_synthetic_face(
    *,
    width: int = 400,
    height: int = 520,
    face_rx: float = 120.0,
    face_ry: float = 160.0,
    center: tuple[float, float] | None = None,
    roll_deg: float = 0.0,
    yaw_shift_px: float = 0.0,
    pitch_ratio: float = 0.55,
    exposure: float = 1.0,
    blur_sigma: float = 0.0,
    gradient: float = 0.0,
    redness_add: float = 0.0,
    texture_amp: float = 0.0,
    pore_spacing: float = 0.0,
    pigment_strength: float = 0.0,
    shine_count: int = 0,
    gray_card: bool = False,
    colour_cast: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    if center is None:
        cx, cy = width * 0.5, height * 0.48
    else:
        cx, cy = center
    yy, xx = np.mgrid[0:height, 0:width]
    angle = math.radians(-roll_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = xx - cx
    dy = yy - cy
    xr = dx * cos_a - dy * sin_a
    yr = dx * sin_a + dy * cos_a
    face = (xr / face_rx) ** 2 + (yr / face_ry) ** 2 <= 1.0
    eye_y = -0.18 * face_ry
    eye_x = 0.38 * face_rx
    eye_rx = 0.12 * face_rx
    eye_ry = 0.07 * face_ry
    mouth_y = 0.32 * face_ry
    mouth_rx = 0.22 * face_rx
    mouth_ry = 0.08 * face_ry
    nose_y = eye_y + pitch_ratio * (mouth_y - eye_y)
    nose_x = yaw_shift_px
    left_eye = ((xr + eye_x) / eye_rx) ** 2 + ((yr - eye_y) / eye_ry) ** 2 <= 1.0
    right_eye = ((xr - eye_x) / eye_rx) ** 2 + ((yr - eye_y) / eye_ry) ** 2 <= 1.0
    mouth = (xr / mouth_rx) ** 2 + ((yr - mouth_y) / mouth_ry) ** 2 <= 1.0
    nose = ((xr - nose_x) / (0.07 * face_rx)) ** 2 + ((yr - nose_y) / (0.08 * face_ry)) ** 2 <= 1.0
    cheeks = face & (yr > eye_y + 0.12 * face_ry) & (yr < mouth_y) & (np.abs(xr) > 0.22 * face_rx)
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:] = (70.0, 110.0, 150.0)
    image[face] = (214.0, 158.0, 136.0)
    image[nose & face] = (155.0, 105.0, 90.0)
    if redness_add:
        image[face, 0] += redness_add
    if texture_amp:
        pattern = texture_amp * np.sin(xr / 2.5) * np.sin(yr / 2.5)
        image[cheeks] += pattern[cheeks, None]
    if pore_spacing > 0:
        pores = cheeks & (np.abs(xr - np.round(xr / pore_spacing) * pore_spacing) < 0.9)
        pores &= np.abs(yr - np.round(yr / pore_spacing) * pore_spacing) < 0.9
        image[pores] = (36.0, 28.0, 24.0)
    if pigment_strength:
        patch = ((xr - 0.70 * face_rx) / (0.16 * face_rx)) ** 2 + ((yr - 0.02 * face_ry) / (0.12 * face_ry)) ** 2 <= 1.0
        image[patch & face] -= pigment_strength
    if shine_count > 0:
        for index in range(shine_count):
            sx = -0.45 * face_rx + index * (0.9 * face_rx / max(shine_count, 1))
            sy = eye_y - 0.28 * face_ry
            spot = ((xr - sx) / 5.0) ** 2 + ((yr - sy) / 5.0) ** 2 <= 1.0
            image[spot & face] = (250.0, 250.0, 248.0)
    image[left_eye | right_eye] = (25.0, 20.0, 18.0)
    image[mouth] = (150.0, 55.0, 65.0)
    if gray_card:
        image[8:28, 8:28] = (180.0, 180.0, 180.0)
    cast = np.array(colour_cast, dtype=np.float64)
    image *= cast.reshape(1, 1, 3)
    image *= float(exposure)
    if gradient:
        ramp = 1.0 + float(gradient) * (2.0 * xx / max(width - 1, 1) - 1.0)
        image *= ramp[:, :, None]
    if blur_sigma > 0:
        ensure_deterministic_cv()
        import cv2

        image = np.asarray(cv2.GaussianBlur(image, (0, 0), float(blur_sigma)), dtype=np.float64)
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def render_two_faces() -> np.ndarray:
    """Two separated protocol faces. The detector must reject this as multiple faces."""

    left = render_synthetic_face(width=700, height=520, face_rx=90, face_ry=130, center=(180, 250), gray_card=False)
    right = render_synthetic_face(width=700, height=520, face_rx=90, face_ry=130, center=(520, 250), gray_card=False)
    skin = skin_mask(right)
    merged = left.copy()
    merged[skin] = right[skin]
    return merged
