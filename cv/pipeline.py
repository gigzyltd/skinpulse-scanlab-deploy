"""Stateless analysis entry points.

The numeric source is always deterministic computer vision. An LLM is not on
this path. Rejected captures produce quality reasons and no skin metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import numpy as np

from scanlab.cv.geometric_backend import GeometricBackend
from scanlab.cv.geometry import build_regions
from scanlab.cv.images import decode_image, sha256_bytes
from scanlab.cv.mediapipe_backend import MediaPipeBackend
from scanlab.cv.metrics import compute_metrics
from scanlab.cv.models import AnalysisResult, NormalisationInfo
from scanlab.cv.normalise import normalise
from scanlab.cv.quality import quality_checks, quality_confidence
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_EDGE_PX,
    ILLUMINATION_VERSION,
    LIMITATIONS,
    NORMALISATION_VERSION,
    NUMERIC_SOURCE,
    PIPELINE_VERSION,
    SCHEMA_ANALYSIS,
)

logger = logging.getLogger("scanlab.cv")


@dataclass(frozen=True)
class Settings:
    landmark_backend: str = "mediapipe"
    face_landmarker_model: str | None = None
    max_edge_px: int = DEFAULT_MAX_EDGE_PX
    max_bytes: int = DEFAULT_MAX_BYTES

    @staticmethod
    def from_env() -> "Settings":
        model = os.environ.get("SCANLAB_FACE_LANDMARKER_MODEL") or None
        return Settings(
            landmark_backend=os.environ.get("SCANLAB_LANDMARK_BACKEND", "mediapipe"),
            face_landmarker_model=model,
            max_edge_px=int(os.environ.get("SCANLAB_MAX_EDGE_PX", str(DEFAULT_MAX_EDGE_PX))),
            max_bytes=int(os.environ.get("SCANLAB_MAX_BYTES", str(DEFAULT_MAX_BYTES))),
        )

    def overridden(self, backend: str | None = None, model: str | None = None) -> "Settings":
        return Settings(
            landmark_backend=backend or self.landmark_backend,
            face_landmarker_model=self.face_landmarker_model if model is None else model,
            max_edge_px=self.max_edge_px,
            max_bytes=self.max_bytes,
        )


def build_backend(settings: Settings):
    if settings.landmark_backend == "geometric":
        return GeometricBackend()
    if settings.landmark_backend == "mediapipe":
        return MediaPipeBackend(settings.face_landmarker_model)
    raise ScanLabError("unknown_backend", "The landmark backend is not recognised.")


def analyze_bytes(data: bytes, settings: Settings, *, include_metrics: bool = True) -> AnalysisResult:
    digest = sha256_bytes(data)
    rgb = decode_image(data, max_edge_px=settings.max_edge_px, max_bytes=settings.max_bytes)
    return analyze_rgb(rgb, settings, image_sha256=digest, include_metrics=include_metrics)


def analyze_rgb(
    rgb: np.ndarray,
    settings: Settings,
    *,
    image_sha256: str,
    include_metrics: bool = True,
) -> AnalysisResult:
    backend = build_backend(settings)
    detection = backend.detect(rgb)
    height, width = rgb.shape[:2]
    base = dict(
        schema_version=SCHEMA_ANALYSIS,
        pipeline_version=PIPELINE_VERSION,
        numeric_source=NUMERIC_SOURCE,
        llm_used=False,
        image_sha256=image_sha256,
        width=width,
        height=height,
        landmark_backend=backend.name,
        landmark_backend_version=backend.version,
        limitations=list(LIMITATIONS),
    )
    if detection.failure or detection.anchors is None:
        reason = detection.failure or "face_not_found"
        result = AnalysisResult(
            **base,
            normalisation=None,
            accepted=False,
            metrics_included=False,
            rejection_reasons=[reason],
            quality_checks=[],
            metrics=[],
            warnings=[],
        )
        _log(result)
        return result
    regions = build_regions(detection.anchors)
    checks = quality_checks(rgb, detection.anchors, regions)
    failed = [check.name for check in checks if not check.passed]
    if failed or not include_metrics:
        result = AnalysisResult(
            **base,
            normalisation=None,
            accepted=not failed,
            metrics_included=False,
            rejection_reasons=failed,
            quality_checks=checks,
            metrics=[],
            warnings=[],
        )
        _log(result)
        return result
    normalised = normalise(rgb, regions)
    metrics = compute_metrics(normalised, regions, quality_confidence(checks))
    result = AnalysisResult(
        **base,
        normalisation=NormalisationInfo(
            version=NORMALISATION_VERSION,
            colour_mode=normalised.colour_mode,
            illumination=ILLUMINATION_VERSION,
        ),
        accepted=True,
        metrics_included=True,
        rejection_reasons=[],
        quality_checks=checks,
        metrics=metrics,
        warnings=list(normalised.warnings),
    )
    _log(result)
    return result


def _log(result: AnalysisResult) -> None:
    logger.info(
        "analyzed sha=%s accepted=%s reasons=%s",
        result.image_sha256[:12],
        result.accepted,
        ",".join(result.rejection_reasons),
    )
