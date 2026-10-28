"""Stateless analysis entry points.

The numeric source is always deterministic computer vision. An LLM is not on
this path. Rejected captures produce quality reasons and no skin metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
import threading

import numpy as np

from scanlab.cv.geometric_backend import GeometricBackend
from scanlab.cv.geometry import anchors_pose, build_regions
from scanlab.cv.images import prepare_capture, sha256_bytes
from scanlab.cv.mediapipe_backend import MediaPipeBackend
from scanlab.cv.metrics import compute_metrics, saturation_warnings
from scanlab.cv.model_pin import resolved_default_model
from scanlab.cv.models import AnalysisResult, CapturePreparation, NormalisationInfo
from scanlab.cv.normalise import normalise
from scanlab.cv.quality import lighting_warnings, quality_checks, quality_confidence
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import (
    CAPTURE_PREP_VERSION,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_EDGE_PX,
    ILLUMINATION_VERSION,
    LIMITATIONS,
    NORMALISATION_VERSION,
    NUMERIC_SOURCE,
    PIPELINE_VERSION,
    SCHEMA_ANALYSIS,
)
from scanlab.cv.views import canonical_view, near_cheek, scoring_allowed, scoring_policy

logger = logging.getLogger("scanlab.cv")


@dataclass(frozen=True)
class Settings:
    landmark_backend: str = "mediapipe"
    face_landmarker_model: str | None = None
    max_edge_px: int = DEFAULT_MAX_EDGE_PX
    max_bytes: int = DEFAULT_MAX_BYTES

    @staticmethod
    def from_env() -> "Settings":
        model = os.environ.get("SCANLAB_FACE_LANDMARKER_MODEL") or resolved_default_model()
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


_MEDIAPIPE_BACKENDS: dict[str | None, MediaPipeBackend] = {}
_MEDIAPIPE_BACKENDS_LOCK = threading.Lock()


def _mediapipe_singleton(model: str | None) -> MediaPipeBackend:
    """One loaded Face Landmarker per model path for the whole process."""
    with _MEDIAPIPE_BACKENDS_LOCK:
        backend = _MEDIAPIPE_BACKENDS.get(model)
        if backend is None:
            backend = MediaPipeBackend(model)
            _MEDIAPIPE_BACKENDS[model] = backend
        return backend


def build_backend(settings: Settings):
    if settings.landmark_backend == "geometric":
        return GeometricBackend()
    if settings.landmark_backend == "mediapipe":
        return _mediapipe_singleton(settings.face_landmarker_model)
    raise ScanLabError("unknown_backend", "The landmark backend is not recognised.")


def analyze_bytes(
    data: bytes,
    settings: Settings,
    *,
    include_metrics: bool = True,
    view: str = "frontal",
    views_calibrated: bool = False,
) -> AnalysisResult:
    digest = sha256_bytes(data)
    prepared = prepare_capture(data, max_edge_px=settings.max_edge_px, max_bytes=settings.max_bytes)
    preparation = CapturePreparation(
        version=CAPTURE_PREP_VERSION,
        orientation=prepared.orientation,
        source_width=prepared.source_width,
        source_height=prepared.source_height,
        downscaled=prepared.downscaled,
    )
    return analyze_rgb(
        prepared.rgb,
        settings,
        image_sha256=digest,
        include_metrics=include_metrics,
        view=view,
        views_calibrated=views_calibrated,
        capture_preparation=preparation,
    )


def analyze_rgb(
    rgb: np.ndarray,
    settings: Settings,
    *,
    image_sha256: str,
    include_metrics: bool = True,
    view: str = "frontal",
    views_calibrated: bool = False,
    capture_preparation: CapturePreparation | None = None,
) -> AnalysisResult:
    selected = canonical_view(view)
    policy = scoring_policy(selected, views_calibrated=views_calibrated)
    backend = build_backend(settings)
    try:
        return _analyze_detected(
            rgb,
            backend,
            selected=selected,
            policy=policy,
            image_sha256=image_sha256,
            include_metrics=include_metrics,
            views_calibrated=views_calibrated,
            capture_preparation=capture_preparation,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _analyze_detected(
    rgb: np.ndarray,
    backend,
    *,
    selected: str,
    policy: str,
    image_sha256: str,
    include_metrics: bool,
    views_calibrated: bool,
    capture_preparation: CapturePreparation | None,
) -> AnalysisResult:
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
        view=selected,
        scoring_policy=policy,
        capture_preparation=capture_preparation,
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
    checks = quality_checks(rgb, detection.anchors, regions, view=selected)
    failed = [check.name for check in checks if not check.passed]
    score = include_metrics and not failed and scoring_allowed(selected, views_calibrated=views_calibrated)
    if not score:
        warnings = lighting_warnings(checks) if not failed else []
        if include_metrics and not failed and policy == "never_scored":
            warnings.append("profile_excluded_from_scoring")
        elif include_metrics and not failed and selected != "frontal":
            warnings.append("oblique_view_excluded_from_scoring")
        result = AnalysisResult(
            **base,
            normalisation=None,
            accepted=not failed,
            metrics_included=False,
            rejection_reasons=failed,
            quality_checks=checks,
            metrics=[],
            warnings=warnings,
        )
        _log(result)
        return result
    pose = anchors_pose(detection.anchors)
    yaw = None if pose is None else pose.yaw_deg
    metric_regions = regions if selected == "frontal" else replace(regions, cheeks=near_cheek(regions, selected, yaw))
    normalised = normalise(rgb, regions)
    metrics = compute_metrics(normalised, metric_regions, quality_confidence(checks))
    warnings = list(normalised.warnings)
    warnings.extend(lighting_warnings(checks))
    warnings.extend(saturation_warnings(metrics))
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
        warnings=warnings,
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
