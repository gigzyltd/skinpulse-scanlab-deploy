"""Private consented-dataset ingest.

The manifest stays on disk the operator already controls. Rows without
affirmative consent are rejected before any image file is opened. Output is
metrics and a repeatability report, never image bytes. Optical scores use
frontal captures unless views_calibrated is set.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from scanlab.cv.pipeline import Settings, analyze_bytes
from scanlab.cv.repeatability import report_from_groups, within_target_summary
from scanlab.cv.scoring import skin_score_from_proxies
from scanlab.cv.support import ScanLabError, round_half_up
from scanlab.cv.versions import (
    FACE_DISTANCE_RATIO_MAX,
    METRIC_NAMES,
    PIPELINE_VERSION,
    SCHEMA_PRIVATE_DATASET,
)
from scanlab.cv.views import SCORED_VIEW_NAMES, VIEW_NAMES

REQUIRED_COLUMNS = (
    "file",
    "subject_id",
    "session_id",
    "view",
    "device",
    "lighting",
    "consent",
    "labels",
)
OPTIONAL_COLUMNS = ("distance_cm",)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LABEL_RE = re.compile(r"^[a-z0-9_./+-]{1,64}$")
_TRUE_CONSENT = {"1", "true", "yes", "y"}
_OUTPUT_PREFIX = "services/scanlab/.cache/"
_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".bmp",
    ".dng",
    ".arw",
    ".cr2",
    ".nef",
    ".orf",
    ".rw2",
    ".raf",
    ".raw",
    ".jp2",
    ".avif",
}


@dataclass(frozen=True)
class ManifestRow:
    file: str
    subject_id: str
    session_id: str
    view: str
    device: str
    lighting: str
    consent: bool
    labels: tuple[str, ...]
    distance_cm: float | None
    line_number: int


def parse_manifest(text: str) -> list[ManifestRow]:
    """Parse a manifest. A missing consent flag rejects the file before use."""

    if not text.strip():
        raise ScanLabError("invalid_manifest", "The manifest is empty.")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ScanLabError("invalid_manifest", "The manifest needs a header row.")
    fields = [(name or "").strip() for name in reader.fieldnames]
    if any(not name for name in fields) or len(fields) != len(set(fields)):
        raise ScanLabError("invalid_manifest", "The manifest header has a blank or duplicate column.")
    missing = [name for name in REQUIRED_COLUMNS if name not in fields]
    unknown = [name for name in fields if name not in REQUIRED_COLUMNS and name not in OPTIONAL_COLUMNS]
    if missing or unknown:
        raise ScanLabError(
            "invalid_manifest",
            "The manifest columns must be the pseudonymous schema. Unexpected or missing columns were rejected.",
        )
    raw_rows = list(reader)
    if not raw_rows:
        raise ScanLabError("invalid_manifest", "The manifest has no data rows.")
    if len(raw_rows) > 100_000:
        raise ScanLabError("invalid_manifest", "The manifest exceeds the row limit.")
    withheld = [
        index + 2
        for index, raw in enumerate(raw_rows)
        if (raw.get("consent") or "").strip().lower() not in _TRUE_CONSENT
    ]
    if withheld:
        listed = ", ".join(str(number) for number in withheld[:20])
        raise ScanLabError(
            "consent_required",
            f"Manifest lines without consent were rejected: {listed}. No image was opened.",
        )
    rows: list[ManifestRow] = []
    for index, raw in enumerate(raw_rows):
        line_number = index + 2
        cleaned = {(key or "").strip(): (value or "").strip() for key, value in raw.items()}
        view = _view_name(cleaned["view"], line_number)
        distance = _distance(cleaned.get("distance_cm", ""), line_number)
        rows.append(
            ManifestRow(
                file=_relative_file(cleaned["file"], line_number),
                subject_id=_identifier(cleaned["subject_id"], "subject_id", line_number),
                session_id=_identifier(cleaned["session_id"], "session_id", line_number),
                view=view,
                device=_short_label(cleaned["device"], "device", line_number),
                lighting=_short_label(cleaned["lighting"], "lighting", line_number),
                consent=True,
                labels=_labels(cleaned["labels"], line_number),
                distance_cm=distance,
                line_number=line_number,
            )
        )
    return rows


def assert_output_allowed(path: Path) -> None:
    """Keep reports out of the git tree unless they sit in the ignored cache."""

    resolved = path.resolve()
    repo = _repo_root(Path(__file__).resolve())
    if repo is None:
        return
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError:
        return
    if relative == _OUTPUT_PREFIX.rstrip("/") or relative.startswith(_OUTPUT_PREFIX):
        return
    raise ScanLabError(
        "output_not_gitignored",
        "Write dataset output under services/scanlab/.cache/ or outside the repository.",
    )


def ingest_dataset(
    manifest_path: Path,
    root: Path,
    output: Path,
    settings: Settings,
    *,
    views_calibrated: bool = False,
) -> dict:
    """Read a private folder and write metrics plus a repeatability report."""

    rows = parse_manifest(manifest_path.read_text(encoding="utf-8-sig"))
    base = root.resolve()
    assert_output_allowed(output)
    records = []
    opened = 0
    for row in rows:
        record, did_open = _ingest_row(row, base, settings, views_calibrated=views_calibrated)
        records.append(record)
        opened += did_open
    frontal = _repeatability(records, view="frontal", views_calibrated=views_calibrated)
    oblique = {}
    if views_calibrated:
        for view in ("left_45", "right_45"):
            oblique[view] = _repeatability(records, view=view, views_calibrated=True)
    payload = {
        "schema_version": SCHEMA_PRIVATE_DATASET,
        "pipeline_version": PIPELINE_VERSION,
        "rows": len(records),
        "images_opened": opened,
        "consent_rejected": 0,
        "ingest_rejections": sum(1 for record in records if record["ingest_error"]),
        "views_calibrated": views_calibrated,
        "scoring_views": ["frontal"] if not views_calibrated else list(SCORED_VIEW_NAMES),
        "withheld_from_scoring": sum(1 for record in records if record["scoring_policy"] == "withheld_until_calibrated"),
        "clinical_go": False,
        "output_contains_images": False,
        "per_image_metrics": "per_image_metrics.jsonl",
        "repeatability_report": "repeatability_report.json",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "per_image_metrics.jsonl", records)
    _write_json(output / "repeatability_report.json", frontal)
    written_oblique = []
    for view, report in oblique.items():
        name = f"repeatability_{view}.json"
        _write_json(output / name, report)
        written_oblique.append(name)
    payload["oblique_repeatability_reports"] = written_oblique
    _write_json(output / "ingest_summary.json", payload)
    _reject_image_outputs(output)
    return payload


_FATAL_ROW_ERRORS = {"landmark_backend_unavailable", "model_checksum_mismatch"}


def _ingest_row(row: ManifestRow, root: Path, settings: Settings, *, views_calibrated: bool) -> tuple[dict, int]:
    """Analyse one consented row. A row failure is recorded; it does not abort the file.

    Consent is already enforced. A missing landmark backend is environmental and
    still aborts, because every later row would fail the same way.
    """

    if row.view not in VIEW_NAMES:
        return _rejected_record(row, "unknown_view"), 0
    path = _resolve_inside(root, row.file, row.line_number)
    if not path.is_file():
        return _rejected_record(row, "missing_image"), 0
    try:
        analysis = analyze_bytes(
            path.read_bytes(),
            settings,
            include_metrics=True,
            view=row.view,
            views_calibrated=views_calibrated,
        )
    except ScanLabError as exc:
        if exc.code in _FATAL_ROW_ERRORS:
            raise
        return _rejected_record(row, exc.code), 1
    return _record(row, analysis), 1


def _repeatability(records: list[dict], *, view: str, views_calibrated: bool) -> dict:
    grouped_rows: dict[str, list[dict]] = {}
    dropped: list[dict[str, str]] = []
    for record in records:
        if record["view"] != view:
            continue
        target_id = f"{record['subject_id']}|{record['session_id']}|{view}"
        grouped_rows.setdefault(target_id, []).append(record)
    usable: dict[str, list[dict]] = {}
    for target_id, group in grouped_rows.items():
        ordered = sorted(group, key=lambda item: item["file"])
        if len(ordered) < 2:
            dropped.append({"target_id": target_id, "reason": "too_few_replicates"})
            continue
        failed = next((item for item in ordered if not item["metrics_included"]), None)
        if failed is not None:
            reason = failed["rejection_reasons"][0] if failed["rejection_reasons"] else "not_scored"
            dropped.append({"target_id": target_id, "reason": reason})
            continue
        usable[target_id] = ordered
    grouped: dict[str, tuple[str | None, list[list[float]]]] = {}
    for name in METRIC_NAMES:
        groups = []
        versions: set[str] = set()
        for target_id in sorted(usable):
            values = []
            for record in usable[target_id]:
                metric = next(item for item in record["metrics"] if item["name"] == name)
                values.append(float(metric["value"]))
                versions.add(str(metric["version"]))
            groups.append(values)
        version = next(iter(versions)) if len(versions) == 1 else None
        if len(versions) > 1:
            groups = []
            version = None
        grouped[name] = (version, groups)
    report = report_from_groups(
        grouped,
        dataset_class="owned",
        independent_captures=None,
        numeric_source="deterministic_cv",
        dropped_targets=dropped,
    )
    per_target, distance_warnings = _per_target(grouped_rows, usable)
    body = report.model_dump(mode="json")
    if distance_warnings:
        body["warnings"] = list(body["warnings"]) + distance_warnings
    return {
        "schema_version": SCHEMA_PRIVATE_DATASET,
        "scored_view": view,
        "views_calibrated": views_calibrated,
        "clinical_go": False,
        "per_target": per_target,
        "report": body,
    }


def _per_target(
    grouped_rows: dict[str, list[dict]],
    usable: dict[str, list[dict]],
) -> tuple[list[dict], list[str]]:
    """One row per subject|session|view, plus a distance warning when face area jumps."""

    breakdown = []
    warnings = []
    for target_id in sorted(grouped_rows):
        ordered = sorted(grouped_rows[target_id], key=lambda item: item["file"])
        areas = [
            float(item["face_area_fraction"])
            for item in ordered
            if isinstance(item.get("face_area_fraction"), (int, float))
        ]
        ratio = None
        mismatch = False
        if len(areas) >= 2 and min(areas) > 0.0:
            ratio = round_half_up(max(areas) / min(areas), 4)
            mismatch = ratio > FACE_DISTANCE_RATIO_MAX
        if mismatch:
            warnings.append(f"face_distance_mismatch:{target_id}")
        metric_stats: dict[str, dict] = {}
        if target_id in usable:
            for name in METRIC_NAMES:
                values = []
                for record in usable[target_id]:
                    metric = next(item for item in record["metrics"] if item["name"] == name)
                    values.append(float(metric["value"]))
                metric_stats[name] = {"values": values, **within_target_summary(values)}
        breakdown.append(
            {
                "target_id": target_id,
                "replicates": len(ordered),
                "scored": target_id in usable,
                "face_area_fractions": areas,
                "face_area_ratio": ratio,
                "distance_mismatch": mismatch,
                "metrics": metric_stats,
            }
        )
    return breakdown, warnings


def _face_area(analysis) -> float | None:
    for check in analysis.quality_checks:
        if check.name == "face_size":
            return float(check.components.get("area_fraction", check.value))
    return None


def _rejected_record(row: ManifestRow, reason: str) -> dict:
    body = _record_shell(row)
    body.update(
        {
            "image_sha256": None,
            "source_width": None,
            "source_height": None,
            "width": None,
            "height": None,
            "orientation": None,
            "downscaled": None,
            "accepted": False,
            "metrics_included": False,
            "scoring_policy": "not_scored",
            "rejection_reasons": [reason],
            "warnings": [],
            "quality_checks": [],
            "metrics": [],
            "face_area_fraction": None,
            "skin_score": None,
            "ingest_error": reason,
            "pipeline_version": PIPELINE_VERSION,
        }
    )
    return body


def _record(row: ManifestRow, analysis) -> dict:
    preparation = analysis.capture_preparation
    body = _record_shell(row)
    body.update(
        {
            "image_sha256": analysis.image_sha256,
            "source_width": None if preparation is None else preparation.source_width,
            "source_height": None if preparation is None else preparation.source_height,
            "width": analysis.width,
            "height": analysis.height,
            "orientation": None if preparation is None else preparation.orientation,
            "downscaled": None if preparation is None else preparation.downscaled,
            "accepted": analysis.accepted,
            "metrics_included": analysis.metrics_included,
            "scoring_policy": analysis.scoring_policy,
            "rejection_reasons": list(analysis.rejection_reasons),
            "warnings": list(analysis.warnings),
            "quality_checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "value": check.value,
                    "confidence": check.confidence,
                    "version": check.version,
                }
                for check in analysis.quality_checks
            ],
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "confidence": metric.confidence,
                    "version": metric.version,
                    "clinical_validity": metric.clinical_validity,
                }
                for metric in analysis.metrics
            ],
            "face_area_fraction": _face_area(analysis),
            "skin_score": skin_score_from_proxies(analysis.metrics) if analysis.metrics_included else None,
            "ingest_error": None,
            "pipeline_version": analysis.pipeline_version,
        }
    )
    return body


def _record_shell(row: ManifestRow) -> dict:
    return {
        "file": row.file,
        "subject_id": row.subject_id,
        "session_id": row.session_id,
        "view": row.view,
        "device": row.device,
        "lighting": row.lighting,
        "consent": True,
        "labels": list(row.labels),
        "distance_cm": row.distance_cm,
        "line_number": row.line_number,
        "exif_stripped": True,
        "gps_stripped": True,
        "clinical_go": False,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _reject_image_outputs(output: Path) -> None:
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            path.unlink()
            raise ScanLabError("image_output_forbidden", "Dataset output must not contain image files.")


def _repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_inside(root: Path, raw: str, line_number: int) -> Path:
    if "://" in raw:
        raise ScanLabError("unsupported_image_reference", "Only local image files are allowed.")
    full = (root / raw).resolve()
    base = root.resolve()
    if full != base and base not in full.parents:
        raise ScanLabError("path_escapes_root", "An image path escapes the manifest root.")
    return full


def _relative_file(value: str, line_number: int) -> str:
    text = value.strip().replace("\\", "/")
    if not text or text.startswith("/") or "://" in text or text.startswith("~"):
        raise ScanLabError("invalid_manifest", "Image paths must be relative to the private folder.")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ScanLabError("path_escapes_root", "An image path escapes the manifest root.")
    if not parts:
        raise ScanLabError("invalid_manifest", "An image path is empty.")
    return "/".join(parts)


def _identifier(value: str, field: str, line_number: int) -> str:
    if not _ID_RE.fullmatch(value):
        raise ScanLabError(
            "invalid_manifest",
            f"{field} must be a pseudonymous id on line {line_number}.",
        )
    return value


def _short_label(value: str, field: str, line_number: int) -> str:
    if not value or len(value) > 64 or any(char in value for char in "\n\r\t\\/@") or ".." in value:
        raise ScanLabError("invalid_manifest", f"{field} on line {line_number} is not a short capture label.")
    return value


def _view_name(value: str, line_number: int) -> str:
    token = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{1,32}", token):
        raise ScanLabError("invalid_manifest", f"view on line {line_number} is not a view name.")
    return token


def _labels(value: str, line_number: int) -> tuple[str, ...]:
    if not value:
        return ()
    labels = []
    for part in re.split(r"[;|]", value):
        token = part.strip().lower().replace(" ", "_")
        if not token:
            continue
        if not _LABEL_RE.fullmatch(token):
            raise ScanLabError("invalid_label", f"A label on line {line_number} is not a short code.")
        labels.append(token)
    return tuple(labels)


def _distance(value: str, line_number: int) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ScanLabError("invalid_manifest", f"distance_cm on line {line_number} is not a number.") from exc
    if number <= 0.0 or number > 500.0:
        raise ScanLabError("invalid_manifest", f"distance_cm on line {line_number} is out of range.")
    return number
