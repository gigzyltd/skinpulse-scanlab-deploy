"""Deterministic landmark backend for the synthetic capture protocol.

This is a classical OpenCV detector for procedural faces drawn with a skin-coloured
ellipse, dark eyes, a dark mouth, and a darker nose. It is not a general
photograph face detector. Photographic scans use the MediaPipe backend.
"""

from __future__ import annotations

import numpy as np

from scanlab.cv.colorimetry import rec709_luma
from scanlab.cv.geometry import Detection, FaceAnchors
from scanlab.cv.support import ensure_deterministic_cv
from scanlab.cv.versions import GEOMETRIC_BACKEND_VERSION

_DARK_MEDIAN_FRACTION = 0.62


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    # Margins stay true for the protocol colours after a moderate exposure
    # scale. A gray card and clipped white highlights stay outside the rule.
    return (red > 20.0) & (red > green + 8.0) & (green > blue) & ((red - blue) > 6.0)


class GeometricBackend:
    name = "geometric"
    version = GEOMETRIC_BACKEND_VERSION

    def detect(self, rgb: np.ndarray) -> Detection:
        ensure_deterministic_cv()
        import cv2

        skin = skin_mask(rgb)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)  # type: ignore[call-overload]
        components = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area >= 80:
                components.append((area, index))
        if not components:
            return Detection(None, "face_not_found")
        components.sort(reverse=True)
        if len(components) > 1 and components[1][0] > 0.30 * components[0][0]:
            return Detection(None, "multiple_faces")
        face_area, face_label = components[0]
        component = labels == face_label
        filled = _fill_holes(component)
        luma = rec709_luma(rgb)
        skin_luma = luma[component]
        if skin_luma.size == 0:
            return Detection(None, "face_not_found")
        median = float(np.median(skin_luma))
        dark = filled & ~skin & (luma < median * _DARK_MEDIAN_FRACTION)
        ys, xs = np.where(filled)
        if len(xs) == 0:
            return Detection(None, "face_not_found")
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        height_span = max(y1 - y0, 1)
        min_feature = max(20, int(0.004 * face_area))
        eye_zone = dark & (np.arange(rgb.shape[0])[:, None] < y0 + 0.58 * height_span)
        mouth_zone = dark & (np.arange(rgb.shape[0])[:, None] > y0 + 0.55 * height_span)
        eyes = [blob for blob in _blobs(eye_zone) if blob["area"] >= min_feature]
        eyes.sort(key=lambda blob: blob["area"], reverse=True)
        if len(eyes) < 2 or eyes[1]["area"] < 0.25 * eyes[0]["area"]:
            return Detection(None, "landmarks_incomplete")
        pair = sorted(eyes[:2], key=lambda blob: blob["centroid"][0])
        mouths = [blob for blob in _blobs(mouth_zone) if blob["area"] >= min_feature]
        if not mouths:
            return Detection(None, "landmarks_incomplete")
        mouth = max(mouths, key=lambda blob: blob["area"])
        nose = _nose_centroid(luma, skin, pair[0]["centroid"], pair[1]["centroid"], mouth["centroid"])
        if nose is None:
            # Frontal protocol noses sit between the eyes and are unchanged.
            # A wider window runs only when that search finds nothing, so a
            # synthetic ~45 degree nose can be measured without moving frontal results.
            nose = _nose_centroid(
                luma,
                skin,
                pair[0]["centroid"],
                pair[1]["centroid"],
                mouth["centroid"],
                lateral_margin_iod=1.35,
            )
        if nose is None:
            return Detection(None, "landmarks_incomplete")
        forehead = _extreme(filled, top=True)
        chin = _extreme(filled, top=False)
        if forehead is None or chin is None or not (forehead[1] < pair[0]["centroid"][1] < chin[1]):
            return Detection(None, "landmarks_implausible")
        left_mask = pair[0]["labels"] == pair[0]["label"]
        right_mask = pair[1]["labels"] == pair[1]["label"]
        mouth_mask = mouth["labels"] == mouth["label"]
        anchors = FaceAnchors(
            backend=self.name,
            backend_version=self.version,
            face_mask=filled,
            left_eye_mask=left_mask,
            right_eye_mask=right_mask,
            mouth_mask=mouth_mask,
            left_eye=pair[0]["centroid"],
            right_eye=pair[1]["centroid"],
            nose_tip=nose,
            mouth_center=mouth["centroid"],
            forehead=forehead,
            chin=chin,
        )
        from scanlab.cv.geometry import anchors_pose

        if anchors_pose(anchors) is None:
            return Detection(None, "landmarks_implausible")
        return Detection(anchors, None)


def _blobs(mask: np.ndarray) -> list[dict]:
    ensure_deterministic_cv()
    import cv2

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)  # type: ignore[call-overload]
    blobs = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        blobs.append(
            {
                "area": area,
                "label": index,
                "labels": labels,
                "bbox": (
                    int(stats[index, cv2.CC_STAT_LEFT]),
                    int(stats[index, cv2.CC_STAT_TOP]),
                    int(stats[index, cv2.CC_STAT_WIDTH]),
                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                ),
                "centroid": (float(centroids[index, 0]), float(centroids[index, 1])),
            }
        )
    return blobs


def _nose_centroid(
    luma: np.ndarray,
    skin: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    mouth: tuple[float, float],
    lateral_margin_iod: float = 0.05,
) -> tuple[float, float] | None:
    iod = float(np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
    if iod < 8.0:
        return None
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    y0 = eye_y + 0.05 * (mouth[1] - eye_y)
    y1 = eye_y + 0.95 * (mouth[1] - eye_y)
    x0 = min(left_eye[0], right_eye[0]) - lateral_margin_iod * iod
    x1 = max(left_eye[0], right_eye[0]) + lateral_margin_iod * iod
    height, width = luma.shape
    yy, xx = np.ogrid[0:height, 0:width]
    window = skin & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    values = luma[window]
    if values.size < 10:
        return None
    center = float(np.median(values))
    if float(values.min()) > center - 8.0:
        return None
    chosen = window & (luma <= center - 8.0)
    ys, xs = np.where(chosen)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _fill_holes(component: np.ndarray) -> np.ndarray:
    ensure_deterministic_cv()
    import cv2

    height, width = component.shape
    inv = np.where(component, np.uint8(0), np.uint8(1))
    seed = _border_seed(inv)
    if seed is None:
        return component
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(inv, flood_mask, seed, 2)
    holes = inv == 1
    return component | holes


def _border_seed(inv: np.ndarray) -> tuple[int, int] | None:
    height, width = inv.shape
    for x in range(width):
        if inv[0, x] == 1:
            return (x, 0)
        if inv[height - 1, x] == 1:
            return (x, height - 1)
    for y in range(height):
        if inv[y, 0] == 1:
            return (0, y)
        if inv[y, width - 1] == 1:
            return (width - 1, y)
    return None


def _extreme(mask: np.ndarray, *, top: bool) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y = int(ys.min() if top else ys.max())
    row_xs = xs[ys == y]
    return (float(np.median(row_xs)), float(y))
