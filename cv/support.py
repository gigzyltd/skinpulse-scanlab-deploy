"""Shared errors, rounding, and deterministic OpenCV setup."""

from __future__ import annotations

import math
import os
from decimal import Decimal, ROUND_HALF_UP

_CV_READY = False


class ScanLabError(Exception):
    """A caller-facing failure that must not include image pixels."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def ensure_deterministic_cv() -> None:
    """Force single-threaded OpenCV so repeated calls match on this machine."""

    global _CV_READY
    if _CV_READY:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    import cv2

    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    _CV_READY = True


def round_half_up(value: float, places: int) -> float:
    if not math.isfinite(value):
        raise ScanLabError("non_finite_result", "A measurement was not finite.")
    quantum = Decimal("1").scaleb(-places)
    rounded = Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)
    return float(rounded)


def round_map(values: dict[str, float], places: int = 4) -> dict[str, float]:
    return {key: round_half_up(float(value), places) for key, value in values.items()}
