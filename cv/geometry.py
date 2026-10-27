"""Face anchors, pose proxy, and measurement regions.

Pose is a 2D landmark-ratio proxy (pose-proxy-v1), not a calibrated goniometer.
Yaw is the nose offset along the eye line, scaled by 45 degrees per interocular
distance. Pitch is the nose position between the eyes and the mouth relative to
a frontal ratio of 0.55, scaled the same way. Roll is the eye-line angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scanlab.cv.support import ensure_deterministic_cv
from scanlab.cv.versions import POSE_FRONTAL_PITCH_RATIO, POSE_YAW_SCALE


@dataclass
class Pose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    @property
    def worst_abs_deg(self) -> float:
        return max(abs(self.yaw_deg), abs(self.pitch_deg), abs(self.roll_deg))


@dataclass
class FaceAnchors:
    backend: str
    backend_version: str
    face_mask: np.ndarray
    left_eye_mask: np.ndarray
    right_eye_mask: np.ndarray
    mouth_mask: np.ndarray
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    mouth_center: tuple[float, float]
    forehead: tuple[float, float]
    chin: tuple[float, float]


@dataclass
class Detection:
    anchors: FaceAnchors | None
    failure: str | None


@dataclass
class Regions:
    skin: np.ndarray
    cheeks: np.ndarray
    forehead: np.ndarray
    nose: np.ndarray
    tzone: np.ndarray
    face_bbox: tuple[int, int, int, int]
    face_width: float
    interocular: float
    image_left_cheek: np.ndarray
    image_right_cheek: np.ndarray


def pose_from_points(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    nose: tuple[float, float],
    mouth: tuple[float, float],
) -> Pose | None:
    """Return the pose proxy, or None when the points cannot form a face."""

    lx, ly = left_eye
    rx, ry = right_eye
    if rx < lx:
        lx, rx = rx, lx
        ly, ry = ry, ly
    dx = rx - lx
    dy = ry - ly
    iod = float(np.hypot(dx, dy))
    if iod < 8.0:
        return None
    roll = float(np.degrees(np.arctan2(dy, dx)))
    axis_x = dx / iod
    axis_y = dy / iod
    normal_x = axis_y
    normal_y = -axis_x
    eye_mid_x = (lx + rx) / 2.0
    eye_mid_y = (ly + ry) / 2.0
    mouth_dx = mouth[0] - eye_mid_x
    mouth_dy = mouth[1] - eye_mid_y
    if mouth_dx * normal_x + mouth_dy * normal_y < 0.0:
        normal_x = -normal_x
        normal_y = -normal_y
    mouth_along = mouth_dx * normal_x + mouth_dy * normal_y
    if mouth_along < 8.0:
        return None
    nose_dx = nose[0] - eye_mid_x
    nose_dy = nose[1] - eye_mid_y
    across = nose_dx * axis_x + nose_dy * axis_y
    along = nose_dx * normal_x + nose_dy * normal_y
    yaw = (across / iod) * POSE_YAW_SCALE
    pitch_ratio = along / mouth_along
    pitch = ((pitch_ratio - POSE_FRONTAL_PITCH_RATIO) / POSE_FRONTAL_PITCH_RATIO) * POSE_YAW_SCALE
    return Pose(yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=roll)


def anchors_pose(anchors: FaceAnchors) -> Pose | None:
    return pose_from_points(anchors.left_eye, anchors.right_eye, anchors.nose_tip, anchors.mouth_center)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    return (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    ensure_deterministic_cv()
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return eroded.astype(bool)


def build_regions(anchors: FaceAnchors) -> Regions:
    """Split the face into cheeks, forehead, nose, and T-zone."""

    skin = anchors.face_mask & ~anchors.left_eye_mask & ~anchors.right_eye_mask & ~anchors.mouth_mask
    skin = _erode(skin, 2)
    height, width = skin.shape
    yy, xx = np.ogrid[0:height, 0:width]
    eye_y = (anchors.left_eye[1] + anchors.right_eye[1]) / 2.0
    eye_x = (anchors.left_eye[0] + anchors.right_eye[0]) / 2.0
    iod = float(np.hypot(
        anchors.right_eye[0] - anchors.left_eye[0],
        anchors.right_eye[1] - anchors.left_eye[1],
    ))
    iod = max(iod, 1.0)
    mouth_y = anchors.mouth_center[1]
    nose_x = anchors.nose_tip[0]
    forehead = skin & (yy < eye_y - 0.15 * iod)
    cheek_band = skin & (yy > eye_y + 0.15 * iod) & (yy < mouth_y + 0.25 * iod)
    left_cheek = cheek_band & (xx < nose_x - 0.18 * iod)
    right_cheek = cheek_band & (xx > nose_x + 0.18 * iod)
    cheeks = left_cheek | right_cheek
    nose = skin & (np.abs(xx - nose_x) <= 0.22 * iod) & (yy > eye_y) & (yy < mouth_y)
    # Keep the nose corridor from swallowing a cheek when iod is tiny.
    _ = (eye_x, width)
    tzone = forehead | nose
    bbox = _bbox(anchors.face_mask)
    return Regions(
        skin=skin,
        cheeks=cheeks,
        forehead=forehead,
        nose=nose,
        tzone=tzone,
        face_bbox=bbox,
        face_width=float(bbox[2]),
        interocular=iod,
        image_left_cheek=left_cheek,
        image_right_cheek=right_cheek,
    )


def draw_overlay(rgb: np.ndarray, anchors: FaceAnchors, regions: Regions) -> np.ndarray:
    """Draw ROI contours. The overlay is not an input to any metric."""

    ensure_deterministic_cv()
    import cv2

    out = rgb.copy()

    def paint(mask: np.ndarray, color: tuple[int, int, int]) -> None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)

    paint(anchors.face_mask, (99, 102, 241))
    paint(regions.cheeks, (40, 180, 90))
    paint(regions.tzone, (230, 180, 40))
    return out
