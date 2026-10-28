"""Explicit capture views.

The frontal gate is yaw, pitch, and roll each within ±15 degrees. Left and
right views are optional and named. They are not a silent relaxation of the
frontal gate: a 45 degree yaw still fails `frontal`.

Yaw sign, from pose-proxy-v1: positive yaw means the nose tip lies toward
image-right of the eye midpoint. `right_45` is that direction. `left_45` is
the opposite. The camera-facing (near) cheek is the other image side from the
nose: when the nose points image-right, the near cheek is image-left. The far
cheek is the side the nose points toward and is not a coverage gate. This is
an image-space rule, not a calibrated anatomical measurement.

`profile` is quality-gated and never scored, including when views_calibrated
is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from scanlab.cv.geometry import Pose, Regions
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import (
    OBLIQUE_PITCH_ABS_MAX_DEG,
    OBLIQUE_ROLL_ABS_MAX_DEG,
    OBLIQUE_YAW_ABS_MAX_DEG,
    OBLIQUE_YAW_ABS_MIN_DEG,
    OBLIQUE_YAW_TARGET_DEG,
    POSE_ABS_MAX_DEG,
    PROFILE_PITCH_ABS_MAX_DEG,
    PROFILE_ROLL_ABS_MAX_DEG,
    PROFILE_YAW_ABS_MAX_DEG,
    PROFILE_YAW_ABS_MIN_DEG,
)

VIEW_NAMES = ("frontal", "left_45", "right_45", "profile")
SCORED_VIEW_NAMES = ("frontal", "left_45", "right_45")
ScoringPolicy = Literal["frontal", "withheld_until_calibrated", "views_calibrated", "never_scored"]


@dataclass(frozen=True)
class PoseWindow:
    name: str
    yaw_min_deg: float
    yaw_max_deg: float
    pitch_abs_max_deg: float
    roll_abs_max_deg: float
    yaw_target_deg: float
    yaw_spans: tuple[tuple[float, float], ...] = ()

    def _spans(self) -> tuple[tuple[float, float], ...]:
        if self.yaw_spans:
            return self.yaw_spans
        return ((self.yaw_min_deg, self.yaw_max_deg),)

    def yaw_excess(self, yaw_deg: float) -> float:
        return min(_outside(yaw_deg, low, high) for low, high in self._spans())

    def active_yaw_span(self, yaw_deg: float) -> tuple[float, float]:
        return min(self._spans(), key=lambda span: (_outside(yaw_deg, span[0], span[1]), span[0]))

    def excess_deg(self, pose: Pose) -> float:
        return max(
            self.yaw_excess(pose.yaw_deg),
            _outside(pose.pitch_deg, -self.pitch_abs_max_deg, self.pitch_abs_max_deg),
            _outside(pose.roll_deg, -self.roll_abs_max_deg, self.roll_abs_max_deg),
        )

    def contains(self, pose: Pose) -> bool:
        return self.excess_deg(pose) <= 0.0


def canonical_view(view: str) -> str:
    if view in VIEW_NAMES:
        return view
    raise ScanLabError(
        "unknown_view",
        "The capture view is not frontal, left_45, right_45, or profile.",
    )


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
    if name == "profile":
        return PoseWindow(
            "profile",
            PROFILE_YAW_ABS_MIN_DEG,
            PROFILE_YAW_ABS_MAX_DEG,
            PROFILE_PITCH_ABS_MAX_DEG,
            PROFILE_ROLL_ABS_MAX_DEG,
            90.0,
            (
                (-PROFILE_YAW_ABS_MAX_DEG, -PROFILE_YAW_ABS_MIN_DEG),
                (PROFILE_YAW_ABS_MIN_DEG, PROFILE_YAW_ABS_MAX_DEG),
            ),
        )
    if name == "left_45":
        return PoseWindow(
            name,
            -OBLIQUE_YAW_ABS_MAX_DEG,
            -OBLIQUE_YAW_ABS_MIN_DEG,
            OBLIQUE_PITCH_ABS_MAX_DEG,
            OBLIQUE_ROLL_ABS_MAX_DEG,
            -OBLIQUE_YAW_TARGET_DEG,
        )
    return PoseWindow(
        name,
        OBLIQUE_YAW_ABS_MIN_DEG,
        OBLIQUE_YAW_ABS_MAX_DEG,
        OBLIQUE_PITCH_ABS_MAX_DEG,
        OBLIQUE_ROLL_ABS_MAX_DEG,
        OBLIQUE_YAW_TARGET_DEG,
    )


def scoring_allowed(view: str, *, views_calibrated: bool) -> bool:
    """Optical scores use the frontal view unless calibration is explicit.

    A profile is never scored. views_calibrated does not change that.
    """

    name = canonical_view(view)
    if name == "profile":
        return False
    if name == "frontal":
        return True
    return views_calibrated


def scoring_policy(view: str, *, views_calibrated: bool) -> ScoringPolicy:
    name = canonical_view(view)
    if name == "profile":
        return "never_scored"
    if name == "frontal":
        return "frontal"
    if views_calibrated:
        return "views_calibrated"
    return "withheld_until_calibrated"


def near_cheek(regions: Regions, view: str, yaw_deg: float | None = None) -> np.ndarray:
    """Camera-facing cheek. Nose toward image-right means the near cheek is image-left."""

    name = canonical_view(view)
    if name == "frontal":
        return regions.cheeks
    if name == "profile":
        if yaw_deg is None:
            return np.zeros_like(regions.cheeks)
        if yaw_deg >= 0.0:
            return regions.image_left_cheek
        return regions.image_right_cheek
    if name == "right_45":
        return regions.image_left_cheek
    return regions.image_right_cheek


def far_cheek(regions: Regions, view: str, yaw_deg: float | None = None) -> np.ndarray:
    name = canonical_view(view)
    if name == "frontal":
        return regions.cheeks
    if name == "profile":
        if yaw_deg is None:
            return np.zeros_like(regions.cheeks)
        if yaw_deg >= 0.0:
            return regions.image_right_cheek
        return regions.image_left_cheek
    if name == "right_45":
        return regions.image_right_cheek
    return regions.image_left_cheek


def _outside(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0
