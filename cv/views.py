"""Explicit capture views.

The frontal gate is yaw, pitch, and roll each within ±15 degrees. Left and
right views are optional and named. They are not a silent relaxation of the
frontal gate: a 45 degree yaw still fails `frontal`.

Yaw sign, from pose-proxy-v1: positive yaw means the nose tip lies toward
image-right of the eye midpoint. `right_45` is that direction. `left_45` is
the opposite. The near cheek is the cheek on the same image side as the nose
shift. This is an image-space rule, not a calibrated anatomical measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scanlab.cv.geometry import Pose, Regions
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import (
    OBLIQUE_PITCH_ABS_MAX_DEG,
    OBLIQUE_ROLL_ABS_MAX_DEG,
    OBLIQUE_YAW_TARGET_DEG,
    OBLIQUE_YAW_TOLERANCE_DEG,
    POSE_ABS_MAX_DEG,
)

VIEW_NAMES = ("frontal", "left_45", "right_45")


@dataclass(frozen=True)
class PoseWindow:
    name: str
    yaw_min_deg: float
    yaw_max_deg: float
    pitch_abs_max_deg: float
    roll_abs_max_deg: float
    yaw_target_deg: float

    def excess_deg(self, pose: Pose) -> float:
        return max(
            _outside(pose.yaw_deg, self.yaw_min_deg, self.yaw_max_deg),
            _outside(pose.pitch_deg, -self.pitch_abs_max_deg, self.pitch_abs_max_deg),
            _outside(pose.roll_deg, -self.roll_abs_max_deg, self.roll_abs_max_deg),
        )

    def contains(self, pose: Pose) -> bool:
        return self.excess_deg(pose) <= 0.0


def canonical_view(view: str) -> str:
    if view in VIEW_NAMES:
        return view
    raise ScanLabError("unknown_view", "The capture view is not frontal, left_45, or right_45.")


def pose_window(view: str) -> PoseWindow:
    name = canonical_view(view)
    if name == "frontal":
        return PoseWindow(
            "frontal",
            -POSE_ABS_MAX_DEG,
            POSE_ABS_MAX_DEG,
            POSE_ABS_MAX_DEG,
            POSE_ABS_MAX_DEG,
            0.0,
        )
    target = -OBLIQUE_YAW_TARGET_DEG if name == "left_45" else OBLIQUE_YAW_TARGET_DEG
    return PoseWindow(
        name,
        target - OBLIQUE_YAW_TOLERANCE_DEG,
        target + OBLIQUE_YAW_TOLERANCE_DEG,
        OBLIQUE_PITCH_ABS_MAX_DEG,
        OBLIQUE_ROLL_ABS_MAX_DEG,
        target,
    )


def scoring_allowed(view: str, *, views_calibrated: bool) -> bool:
    """Optical scores use the frontal view unless calibration is explicit."""

    name = canonical_view(view)
    if name == "frontal":
        return True
    return views_calibrated


def scoring_policy(view: str, *, views_calibrated: bool) -> str:
    name = canonical_view(view)
    if name == "frontal":
        return "frontal"
    if views_calibrated:
        return "views_calibrated"
    return "withheld_until_calibrated"


def near_cheek(regions: Regions, view: str) -> np.ndarray:
    name = canonical_view(view)
    if name == "right_45":
        return regions.image_right_cheek
    if name == "left_45":
        return regions.image_left_cheek
    return regions.cheeks


def far_cheek(regions: Regions, view: str) -> np.ndarray:
    name = canonical_view(view)
    if name == "right_45":
        return regions.image_left_cheek
    if name == "left_45":
        return regions.image_right_cheek
    return regions.cheeks


def _outside(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0
