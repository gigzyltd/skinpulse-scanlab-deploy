"""Capture-quality gate: blur, exposure, face size, pose, lighting, ROI coverage."""

from __future__ import annotations

import math

import numpy as np

from scanlab.cv.colorimetry import rec709_luma
from scanlab.cv.geometry import FaceAnchors, Regions, anchors_pose
from scanlab.cv.models import QualityCheck
from scanlab.cv.support import ensure_deterministic_cv, round_half_up, round_map
from scanlab.cv.views import canonical_view, far_cheek, near_cheek, pose_window
from scanlab.cv.versions import (
    BLUR_COMFORT,
    BLUR_LAPLACIAN_MIN,
    BLUR_VERSION,
    EXPOSURE_CLIP_LUMA_HIGH,
    EXPOSURE_CLIP_LUMA_LOW,
    EXPOSURE_CLIP_MAX,
    EXPOSURE_COMFORT_MAX,
    EXPOSURE_COMFORT_MIN,
    EXPOSURE_MEAN_MAX,
    EXPOSURE_MEAN_MIN,
    EXPOSURE_VERSION,
    FACE_AREA_COMFORT,
    FACE_AREA_FRACTION_MIN,
    FACE_MIN_SIDE_PX,
    FACE_SIZE_VERSION,
    LIGHTING_CELL_CV_COMFORT,
    LIGHTING_CELL_CV_MAX,
    LIGHTING_CELL_RANGE_MAX,
    LIGHTING_MIN_CELL_PIXELS,
    LIGHTING_MIN_CELLS,
    LIGHTING_VERSION,
    POSE_ABS_MAX_DEG,
    POSE_COMFORT_DEG,
    POSE_OBLIQUE_VERSION,
    POSE_VERSION,
    ROI_CONFIDENCE_PIXELS,
    ROI_MIN_CHEEK_PIXELS,
    ROI_MIN_SKIN_PIXELS,
    ROI_MIN_TZONE_PIXELS,
    ROI_OBLIQUE_NEAR_CHEEK_MIN,
    ROI_OBLIQUE_SKIN_MIN,
    ROI_OBLIQUE_TZONE_MIN,
    ROI_OBLIQUE_VERSION,
    ROI_VERSION,
)


def _lower_confidence(value: float, fail_at: float, comfort_at: float) -> float:
    if value >= comfort_at:
        return 1.0
    if value <= fail_at:
        return 0.0
    return (value - fail_at) / (comfort_at - fail_at)


def _upper_confidence(value: float, comfort_at: float, fail_at: float) -> float:
    if value <= comfort_at:
        return 1.0
    if value >= fail_at:
        return 0.0
    return (fail_at - value) / (fail_at - comfort_at)


def _band_confidence(value: float, fail_lo: float, comfort_lo: float, comfort_hi: float, fail_hi: float) -> float:
    if comfort_lo <= value <= comfort_hi:
        return 1.0
    if value <= fail_lo or value >= fail_hi:
        return 0.0
    if value < comfort_lo:
        return (value - fail_lo) / (comfort_lo - fail_lo)
    return (fail_hi - value) / (fail_hi - comfort_hi)


def _check(name: str, passed: bool, value: float, confidence: float, version: str, components: dict[str, float]) -> QualityCheck:
    return QualityCheck(
        name=name,
        passed=passed,
        value=round_half_up(value, 4),
        confidence=round_half_up(min(max(confidence, 0.0), 1.0), 4),
        version=version,
        components=round_map(components),
    )


def quality_checks(
    rgb: np.ndarray,
    anchors: FaceAnchors,
    regions: Regions,
    view: str = "frontal",
) -> list[QualityCheck]:
    """Run the capture gate. The default view is the unchanged frontal gate."""

    selected = canonical_view(view)
    luma = rec709_luma(rgb)
    return [
        _blur(rgb, anchors),
        _exposure(luma, regions),
        _face_size(anchors, rgb.shape),
        _pose(anchors, selected),
        _lighting(luma, regions),
        _roi(regions, selected),
    ]


def quality_confidence(checks: list[QualityCheck]) -> float:
    if not checks:
        return 0.0
    return min(check.confidence for check in checks)


def _blur(rgb: np.ndarray, anchors: FaceAnchors) -> QualityCheck:
    ensure_deterministic_cv()
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    values = laplacian[anchors.face_mask]
    variance = float(values.var()) if values.size else 0.0
    passed = bool(values.size) and variance >= BLUR_LAPLACIAN_MIN
    confidence = _lower_confidence(variance, BLUR_LAPLACIAN_MIN, BLUR_COMFORT)
    return _check(
        "blur",
        passed,
        variance,
        confidence,
        BLUR_VERSION,
        {"laplacian_variance": variance, "minimum": BLUR_LAPLACIAN_MIN},
    )


def _exposure(luma: np.ndarray, regions: Regions) -> QualityCheck:
    values = luma[regions.skin]
    if values.size == 0:
        return _check("exposure", False, 0.0, 0.0, EXPOSURE_VERSION, {"pixels": 0})
    mean = float(values.mean())
    clip_low = float(np.mean(values <= EXPOSURE_CLIP_LUMA_LOW))
    clip_high = float(np.mean(values >= EXPOSURE_CLIP_LUMA_HIGH))
    clip = max(clip_low, clip_high)
    passed = EXPOSURE_MEAN_MIN <= mean <= EXPOSURE_MEAN_MAX and clip <= EXPOSURE_CLIP_MAX
    confidence = min(
        _band_confidence(mean, EXPOSURE_MEAN_MIN, EXPOSURE_COMFORT_MIN, EXPOSURE_COMFORT_MAX, EXPOSURE_MEAN_MAX),
        _upper_confidence(clip, EXPOSURE_CLIP_MAX * 0.25, EXPOSURE_CLIP_MAX),
    )
    return _check(
        "exposure",
        passed,
        mean,
        confidence,
        EXPOSURE_VERSION,
        {"mean_luma": mean, "clip_low": clip_low, "clip_high": clip_high},
    )


def _face_size(anchors: FaceAnchors, shape: tuple[int, ...]) -> QualityCheck:
    height, width = shape[:2]
    area_fraction = float(np.count_nonzero(anchors.face_mask)) / float(height * width)
    bbox_width = float(np.max(np.where(anchors.face_mask)[1]) - np.min(np.where(anchors.face_mask)[1]) + 1)
    bbox_height = float(np.max(np.where(anchors.face_mask)[0]) - np.min(np.where(anchors.face_mask)[0]) + 1)
    min_side = min(bbox_width, bbox_height)
    passed = area_fraction >= FACE_AREA_FRACTION_MIN and min_side >= FACE_MIN_SIDE_PX
    confidence = 0.0 if min_side < FACE_MIN_SIDE_PX else _lower_confidence(
        area_fraction, FACE_AREA_FRACTION_MIN, FACE_AREA_COMFORT
    )
    return _check(
        "face_size",
        passed,
        area_fraction,
        confidence,
        FACE_SIZE_VERSION,
        {"area_fraction": area_fraction, "min_side_px": min_side, "minimum_fraction": FACE_AREA_FRACTION_MIN},
    )


def _pose(anchors: FaceAnchors, view: str = "frontal") -> QualityCheck:
    pose = anchors_pose(anchors)
    if pose is None:
        return _check("pose", False, POSE_ABS_MAX_DEG, 0.0, POSE_VERSION, {"valid": 0})
    if view == "frontal":
        # Frontal gate: any of yaw, pitch, roll above 15 degrees fails.
        worst = pose.worst_abs_deg
        passed = worst <= POSE_ABS_MAX_DEG
        confidence = _upper_confidence(worst, POSE_COMFORT_DEG, POSE_ABS_MAX_DEG)
        return _check(
            "pose",
            passed,
            worst,
            confidence,
            POSE_VERSION,
            {"yaw_deg": pose.yaw_deg, "pitch_deg": pose.pitch_deg, "roll_deg": pose.roll_deg, "maximum_deg": POSE_ABS_MAX_DEG},
        )
    window = pose_window(view)
    excess = window.excess_deg(pose)
    passed = window.contains(pose)
    yaw_distance = abs(pose.yaw_deg - window.yaw_target_deg)
    deviation = max(yaw_distance, abs(pose.pitch_deg), abs(pose.roll_deg))
    confidence = _upper_confidence(deviation, POSE_COMFORT_DEG, window.pitch_abs_max_deg)
    return _check(
        "pose",
        passed,
        excess,
        confidence,
        POSE_OBLIQUE_VERSION,
        {
            "yaw_deg": pose.yaw_deg,
            "pitch_deg": pose.pitch_deg,
            "roll_deg": pose.roll_deg,
            "yaw_min_deg": window.yaw_min_deg,
            "yaw_max_deg": window.yaw_max_deg,
            "pitch_abs_max_deg": window.pitch_abs_max_deg,
            "roll_abs_max_deg": window.roll_abs_max_deg,
        },
    )


def _lighting(luma: np.ndarray, regions: Regions) -> QualityCheck:
    x, y, width, height = regions.face_bbox
    if width < 3 or height < 3:
        return _check("lighting_uniformity", False, 1.0, 0.0, LIGHTING_VERSION, {"valid_cells": 0})
    medians: list[float] = []
    for row in range(3):
        for col in range(3):
            x0 = x + col * width // 3
            x1 = x + (col + 1) * width // 3
            y0 = y + row * height // 3
            y1 = y + (row + 1) * height // 3
            cell = np.zeros(luma.shape, dtype=bool)
            cell[y0:y1, x0:x1] = True
            values = luma[cell & regions.skin]
            if values.size >= LIGHTING_MIN_CELL_PIXELS:
                medians.append(float(np.median(values)))
    if len(medians) < LIGHTING_MIN_CELLS:
        return _check(
            "lighting_uniformity",
            False,
            1.0,
            0.0,
            LIGHTING_VERSION,
            {"valid_cells": float(len(medians))},
        )
    center = sum(medians) / len(medians)
    variance = sum((value - center) ** 2 for value in medians) / len(medians)
    cell_cv = math.sqrt(variance) / center if center > 1e-6 else 1.0
    cell_range = max(medians) - min(medians)
    passed = cell_cv <= LIGHTING_CELL_CV_MAX and cell_range <= LIGHTING_CELL_RANGE_MAX
    confidence = min(
        _upper_confidence(cell_cv, LIGHTING_CELL_CV_COMFORT, LIGHTING_CELL_CV_MAX),
        _upper_confidence(cell_range, LIGHTING_CELL_RANGE_MAX * 0.4, LIGHTING_CELL_RANGE_MAX),
    )
    return _check(
        "lighting_uniformity",
        passed,
        cell_cv,
        confidence,
        LIGHTING_VERSION,
        {"cell_cv": cell_cv, "cell_range": cell_range, "valid_cells": float(len(medians))},
    )


def _roi(regions: Regions, view: str = "frontal") -> QualityCheck:
    if view == "frontal":
        cheeks = int(np.count_nonzero(regions.cheeks))
        tzone = int(np.count_nonzero(regions.tzone))
        skin = int(np.count_nonzero(regions.skin))
        passed = cheeks >= ROI_MIN_CHEEK_PIXELS and tzone >= ROI_MIN_TZONE_PIXELS and skin >= ROI_MIN_SKIN_PIXELS
        confidence = min(cheeks / ROI_CONFIDENCE_PIXELS, tzone / ROI_CONFIDENCE_PIXELS, 1.0) if passed else 0.0
        return _check(
            "roi_coverage",
            passed,
            float(cheeks),
            confidence,
            ROI_VERSION,
            {"cheek_pixels": float(cheeks), "tzone_pixels": float(tzone), "skin_pixels": float(skin)},
        )
    near = int(np.count_nonzero(near_cheek(regions, view)))
    far = int(np.count_nonzero(far_cheek(regions, view)))
    tzone = int(np.count_nonzero(regions.tzone))
    skin = int(np.count_nonzero(regions.skin))
    passed = near >= ROI_OBLIQUE_NEAR_CHEEK_MIN and tzone >= ROI_OBLIQUE_TZONE_MIN and skin >= ROI_OBLIQUE_SKIN_MIN
    confidence = min(near / ROI_CONFIDENCE_PIXELS, tzone / ROI_CONFIDENCE_PIXELS, 1.0) if passed else 0.0
    return _check(
        "roi_coverage",
        passed,
        float(near),
        confidence,
        ROI_OBLIQUE_VERSION,
        {
            "near_cheek_pixels": float(near),
            "far_cheek_pixels": float(far),
            "far_cheek_required": 0.0,
            "tzone_pixels": float(tzone),
            "skin_pixels": float(skin),
        },
    )
