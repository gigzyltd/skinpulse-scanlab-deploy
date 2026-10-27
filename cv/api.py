"""Local FastAPI surface for the Scan Lab CV pipeline.

There is no cloud client, no object storage, and no language model. Images are
read into memory and discarded. A repeatability GO is not a clinical GO.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scanlab.cv.mediapipe_backend import mediapipe_status
from scanlab.cv.models import AnalysisResult, RepeatabilityReport
from scanlab.cv.pipeline import Settings, analyze_bytes
from scanlab.cv.repeatability import report_from_image_manifest
from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import NUMERIC_SOURCE, PIPELINE_VERSION
from scanlab.cv.views import VIEW_NAMES


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="SkinPulse Scan Lab CV",
        version=PIPELINE_VERSION,
        description=(
            "Deterministic phase-0 capture quality, optical-proxy metrics, and "
            "repeatability thresholds. Numbers are not produced by an LLM and are "
            "not a clinical or product GO."
        ),
    )
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "pipeline_version": PIPELINE_VERSION,
            "numeric_source": NUMERIC_SOURCE,
            "llm_used": False,
            "landmark_backend": settings.landmark_backend,
            "mediapipe": mediapipe_status(settings.face_landmarker_model),
        }

    @app.post("/v1/quality", response_model=AnalysisResult)
    async def quality(
        file: UploadFile = File(...),
        backend: str | None = Form(default=None),
        view: str = Form(default="frontal"),
    ) -> AnalysisResult:
        data = await _read_limited(file, settings.max_bytes)
        return _run(data, settings, backend, include_metrics=False, view=view)

    @app.post("/v1/analyze", response_model=AnalysisResult)
    async def analyze(
        file: UploadFile = File(...),
        backend: str | None = Form(default=None),
        view: str = Form(default="frontal"),
        views_calibrated: str = Form(default="false"),
    ) -> AnalysisResult:
        data = await _read_limited(file, settings.max_bytes)
        return _run(
            data,
            settings,
            backend,
            include_metrics=True,
            view=view,
            views_calibrated=_flag(views_calibrated),
        )

    @app.post("/v1/repeatability", response_model=RepeatabilityReport)
    async def repeatability(
        manifest: str = Form(...),
        files: list[UploadFile] = File(...),
        backend: str | None = Form(default=None),
    ) -> RepeatabilityReport:
        try:
            payload = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail={"error": "invalid_manifest", "message": "Manifest JSON is invalid."}) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail={"error": "invalid_manifest", "message": "Manifest must be an object."})
        saved: dict[str, bytes] = {}
        for upload in files:
            name = upload.filename or ""
            if not name or "/" in name or "\\" in name or name.startswith("."):
                raise HTTPException(status_code=400, detail={"error": "invalid_filename", "message": "Upload names must be bare filenames."})
            saved[name] = await _read_limited(upload, settings.max_bytes)
        rewritten = _manifest_with_memory_names(payload)
        active = _backend_settings(settings, backend)
        try:
            return _repeat_from_memory(rewritten, saved, active)
        except ScanLabError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    return app


def _backend_settings(settings: Settings, backend: str | None) -> Settings:
    if backend is None:
        return settings
    if backend not in {"geometric", "mediapipe"}:
        raise HTTPException(status_code=400, detail={"error": "unknown_backend", "message": "The landmark backend is not recognised."})
    return settings.overridden(backend=backend)


def _flag(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"", "0", "false", "no", "n"}:
        return False
    raise HTTPException(status_code=400, detail={"error": "invalid_flag", "message": "views_calibrated must be true or false."})


def _run(
    data: bytes,
    settings: Settings,
    backend: str | None,
    *,
    include_metrics: bool,
    view: str = "frontal",
    views_calibrated: bool = False,
) -> AnalysisResult:
    if view not in VIEW_NAMES:
        raise HTTPException(status_code=400, detail={"error": "unknown_view", "message": "The capture view is not frontal, left_45, or right_45."})
    active = _backend_settings(settings, backend)
    try:
        return analyze_bytes(
            data,
            active,
            include_metrics=include_metrics,
            view=view,
            views_calibrated=views_calibrated,
        )
    except ScanLabError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail={"error": "payload_too_large", "message": "The upload exceeds the size limit."})
        chunks.append(chunk)
    return b"".join(chunks)


def _manifest_with_memory_names(payload: dict) -> dict:
    """Repeatability paths in the API are upload filenames, never filesystem paths."""

    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise HTTPException(status_code=400, detail={"error": "invalid_manifest", "message": "targets must be a list."})
    rewritten = []
    for target in targets:
        if not isinstance(target, dict):
            raise HTTPException(status_code=400, detail={"error": "invalid_manifest", "message": "Each target must be an object."})
        paths = target.get("paths")
        if not isinstance(paths, list):
            raise HTTPException(status_code=400, detail={"error": "invalid_manifest", "message": "Each target needs paths."})
        safe = []
        for raw in paths:
            if not isinstance(raw, str) or "/" in raw or "\\" in raw or raw.startswith("."):
                raise HTTPException(status_code=400, detail={"error": "invalid_filename", "message": "Manifest paths must be bare upload filenames."})
            safe.append(raw)
        rewritten.append({**target, "paths": safe})
    return {**payload, "targets": rewritten}


def _repeat_from_memory(payload: dict, files: dict[str, bytes], settings: Settings) -> RepeatabilityReport:
    """Adapt the file manifest runner to uploads by staging nothing on disk.

    The file runner is reused through an in-memory path map so the API cannot
    read arbitrary local paths.
    """

    from scanlab.cv.repeatability import report_from_groups
    from scanlab.cv.versions import METRIC_NAMES

    targets = payload["targets"]
    collected: dict[str, dict[str, list[float]]] = {name: {} for name in METRIC_NAMES}
    versions: dict[str, set[str]] = {name: set() for name in METRIC_NAMES}
    dropped: list[dict[str, str]] = []
    for target in targets:
        target_id = str(target.get("target_id") or "")
        paths = target.get("paths") or []
        if not target_id or len(paths) < 2:
            raise ScanLabError("invalid_manifest", "Each target needs an id and at least two images.")
        analyses = []
        failed_reason = None
        for name in paths:
            data = files.get(name)
            if data is None:
                failed_reason = "missing_image"
                break
            try:
                analysis = analyze_bytes(data, settings, include_metrics=True)
            except ScanLabError as exc:
                if exc.code == "landmark_backend_unavailable":
                    raise
                failed_reason = exc.code
                break
            if not analysis.accepted:
                failed_reason = analysis.rejection_reasons[0] if analysis.rejection_reasons else "rejected"
                break
            analyses.append(analysis)
        if failed_reason:
            dropped.append({"target_id": target_id, "reason": failed_reason})
            continue
        for metric_name in METRIC_NAMES:
            values = []
            for analysis in analyses:
                match = next(item for item in analysis.metrics if item.name == metric_name)
                values.append(match.value)
                versions[metric_name].add(match.version)
            collected[metric_name][target_id] = values
    grouped = {}
    extras = []
    for name in METRIC_NAMES:
        groups = [collected[name][key] for key in sorted(collected[name])]
        version = next(iter(versions[name])) if len(versions[name]) == 1 else None
        if len(versions[name]) > 1 or not groups:
            from scanlab.cv.repeatability import _blank_metric

            reason = "version_mismatch" if len(versions[name]) > 1 else "too_few_targets"
            extras.append(_blank_metric(name, version, len(groups), 0, reason))
            continue
        grouped[name] = (version, groups)
    independent = payload.get("independent_captures")
    if independent is not None and not isinstance(independent, bool):
        raise ScanLabError("invalid_manifest", "independent_captures must be a boolean.")
    report = report_from_groups(
        grouped,
        dataset_class=str(payload.get("dataset_class") or "synthetic"),
        independent_captures=independent if isinstance(independent, bool) else None,
        numeric_source="deterministic_cv",
        dropped_targets=dropped,
    )
    if extras:
        from scanlab.cv.repeatability import _sample_warnings, combine_gates

        report.metrics.extend(extras)
        overall, reasons = combine_gates([item.repeatability_gate for item in report.metrics])
        report.repeatability_gate = overall  # type: ignore[assignment]
        report.gate_reasons = reasons
        report.warnings = _sample_warnings(report.metrics, report.independent_captures)
    return report


app = create_app()
