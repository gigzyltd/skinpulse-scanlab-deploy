"""MediaPipe Face Landmarker adapter.

The landmarker model is a local file. This module never downloads it and never
calls a hosted vision API. A present file must match the pinned float16/1
sha256. If MediaPipe or the model file is absent, detection raises
landmark_backend_unavailable instead of inventing landmarks.
"""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from scanlab.cv.geometry import Detection, FaceAnchors, anchors_pose
from scanlab.cv.model_pin import pin_status
from scanlab.cv.support import ScanLabError, ensure_deterministic_cv
from scanlab.cv.versions import MEDIAPIPE_BACKEND_VERSION

# Ordered contours from the MediaPipe 468-point face mesh. Subject-left eye in
# MediaPipe's own naming is the 263 contour; subject-right eye is the 33 contour.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
EYE_33 = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_263 = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


def mediapipe_status(model_path: str | None) -> dict[str, bool]:
    try:
        import mediapipe  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    path = Path(model_path) if model_path else None
    present = path is not None and path.is_file()
    checksum_ok = path is not None and pin_status(path) == "ok"
    return {
        "installed": installed,
        "model_present": present,
        "checksum_ok": checksum_ok,
        "ready": installed and checksum_ok,
    }


class MediaPipeBackend:
    name = "mediapipe"
    version = MEDIAPIPE_BACKEND_VERSION

    def __init__(self, model_path: str | None) -> None:
        self.model_path = model_path
        self._lock = threading.Lock()
        self._landmarker = None

    def detect(self, rgb: np.ndarray) -> Detection:
        status = mediapipe_status(self.model_path)
        if not status["model_present"]:
            raise ScanLabError(
                "landmark_backend_unavailable",
                "The local Face Landmarker model file is missing. Refusing to invent landmarks.",
                503,
            )
        if not status["checksum_ok"]:
            raise ScanLabError(
                "model_checksum_mismatch",
                "The Face Landmarker file does not match the pinned float16/1 sha256.",
                503,
            )
        if not status["installed"]:
            raise ScanLabError(
                "landmark_backend_unavailable",
                "MediaPipe is not installed. Refusing to invent landmarks.",
                503,
            )
        ensure_deterministic_cv()
        import mediapipe as mp

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb, dtype=np.uint8),
        )
        with self._lock:
            result = self._get(mp).detect(image)
        faces = result.face_landmarks or []
        if len(faces) >= 2:
            return Detection(None, "multiple_faces")
        if len(faces) == 0:
            return Detection(None, "face_not_found")
        anchors = _anchors_from_landmarks(faces[0], rgb.shape[1], rgb.shape[0])
        if anchors is None or anchors_pose(anchors) is None:
            return Detection(None, "landmarks_incomplete")
        return Detection(anchors, None)

    def _get(self, mp):
        if self._landmarker is not None:
            return self._landmarker
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self._landmarker


def _anchors_from_landmarks(landmarks, width: int, height: int) -> FaceAnchors | None:
    if len(landmarks) < 468:
        return None

    def point(index: int) -> tuple[float, float]:
        landmark = landmarks[index]
        x = min(max(float(landmark.x) * (width - 1), 0.0), width - 1)
        y = min(max(float(landmark.y) * (height - 1), 0.0), height - 1)
        return (x, y)

    def poly(indices: tuple[int, ...]) -> np.ndarray:
        return np.array([point(index) for index in indices], dtype=np.float64)

    shape = (height, width)
    eye_a = _fill(shape, poly(EYE_33))
    eye_b = _fill(shape, poly(EYE_263))
    mouth = _fill(shape, poly(LIPS))
    face = _fill(shape, poly(FACE_OVAL))
    center_a = _mask_center(eye_a)
    center_b = _mask_center(eye_b)
    mouth_center = _mask_center(mouth)
    if center_a is None or center_b is None or mouth_center is None:
        return None
    left, right = (center_a, center_b) if center_a[0] <= center_b[0] else (center_b, center_a)
    left_mask, right_mask = (eye_a, eye_b) if center_a[0] <= center_b[0] else (eye_b, eye_a)
    return FaceAnchors(
        backend="mediapipe",
        backend_version=MEDIAPIPE_BACKEND_VERSION,
        face_mask=face,
        left_eye_mask=left_mask,
        right_eye_mask=right_mask,
        mouth_mask=mouth,
        left_eye=left,
        right_eye=right,
        nose_tip=point(NOSE_TIP),
        mouth_center=mouth_center,
        forehead=point(FOREHEAD),
        chin=point(CHIN),
    )


def _fill(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    ensure_deterministic_cv()
    import cv2

    mask = np.zeros(shape, np.uint8)
    if len(polygon) < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def _mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))
