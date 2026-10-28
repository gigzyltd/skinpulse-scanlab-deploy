"""Repeatability harness.

The gate is ICC(2,1) absolute agreement >= 0.90, median pairwise |delta| <= 2,
90th percentile of |delta| <= 3, and pooled within-target CV <= 5%. A GO here is
only that numeric threshold result. product_validity stays not_established and
clinical_go stays false, including for synthetic and fixture inputs.

ICC(2,1), McGraw & Wong / Shrout & Fleiss, targets as rows and replicates as
columns:

    (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from scanlab.cv.models import GateThresholds, MetricRepeatability, RepeatabilityReport
from scanlab.cv.pipeline import Settings, analyze_bytes
from scanlab.cv.support import ScanLabError, round_half_up
from scanlab.cv.versions import (
    CV_PERCENT_MAX,
    CV_VERSION,
    DELTA_VERSION,
    ICC_MIN,
    ICC_VERSION,
    LIMITATIONS,
    MEDIAN_ABS_DELTA_MAX,
    METRIC_NAMES,
    P90_ABS_DELTA_MAX,
    REPEATABILITY_STATEMENT,
    SCHEMA_REPEATABILITY,
    SMALL_SAMPLE_REPLICATES,
    SMALL_SAMPLE_TARGETS,
)

_DATASET_CLASSES = {"synthetic", "fixture", "owned", "unknown"}


@dataclass
class IccStats:
    icc_2_1: float | None
    icc_1_1: float | None
    icc_3_1: float | None


def percentile_type7(values: list[float], percent: float) -> float:
    """Hyndman-Fan type 7 percentile, the common linear definition."""

    if not values:
        raise ValueError("empty")
    ordered = sorted(values)
    count = len(ordered)
    if count == 1:
        return ordered[0]
    rank = 1.0 + (count - 1) * (percent / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower - 1]
    weight = rank - lower
    return ordered[lower - 1] * (1.0 - weight) + ordered[upper - 1] * weight


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def anova_icc(groups: list[list[float]]) -> IccStats:
    """Balanced ICC(2,1), ICC(1,1) and ICC(3,1). Undefined when variance is zero."""

    n = len(groups)
    k = len(groups[0])
    flat = [value for group in groups for value in group]
    grand = sum(flat) / (n * k)
    target_means = [sum(group) / k for group in groups]
    replicate_means = [sum(groups[row][col] for row in range(n)) / n for col in range(k)]
    ss_rows = k * sum((value - grand) ** 2 for value in target_means)
    ss_cols = n * sum((value - grand) ** 2 for value in replicate_means)
    ss_total = sum((value - grand) ** 2 for value in flat)
    ss_error = ss_total - ss_rows - ss_cols
    if abs(ss_error) < 1e-9:
        ss_error = 0.0
    if ss_error < 0.0:
        ss_error = 0.0
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_error) / (n * (k - 1))
    if abs(msr) < 1e-12 and abs(mse) < 1e-12:
        return IccStats(None, None, None)

    def divide(numerator: float, denominator: float) -> float | None:
        if abs(denominator) < 1e-12:
            return None
        return numerator / denominator

    icc_21 = divide(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n)
    icc_31 = divide(msr - mse, msr + (k - 1) * mse)
    icc_11 = divide(msr - msw, msr + (k - 1) * msw)
    return IccStats(icc_21, icc_11, icc_31)


def pooled_cv_percent(groups: list[list[float]]) -> float | None:
    variances = []
    flat = []
    for group in groups:
        mean = sum(group) / len(group)
        variances.append(sum((value - mean) ** 2 for value in group) / (len(group) - 1))
        flat.extend(group)
    grand = sum(flat) / len(flat)
    if abs(grand) < 1e-12:
        return None
    pooled = sum(variances) / len(variances)
    return 100.0 * math.sqrt(pooled) / abs(grand)


def within_target_summary(values: list[float]) -> dict[str, float | None]:
    """Median, p90, and sample CV for one target. None when the figure is undefined."""

    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return {"median_abs_delta": None, "p90_abs_delta": None, "cv_percent": None}
    deltas = pairwise_abs_deltas([values])
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        cv_percent = None
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        cv_percent = _r6(100.0 * math.sqrt(variance) / abs(mean))
    return {
        "median_abs_delta": _r6(median(deltas)),
        "p90_abs_delta": _r6(percentile_type7(deltas, 90)),
        "cv_percent": cv_percent,
    }


def pairwise_abs_deltas(groups: list[list[float]]) -> list[float]:
    deltas = []
    for group in groups:
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                deltas.append(abs(group[left] - group[right]))
    return deltas


def decide_gate(
    icc: float | None,
    median_abs_delta: float | None,
    p90_abs_delta: float | None,
    cv_percent: float | None,
) -> tuple[str, list[str]]:
    """Apply the phase-0 thresholds. None is inconclusive, not a passing zero."""

    reasons: list[str] = []
    inconclusive = False
    no_go = False
    if icc is None:
        reasons.append("icc_undefined")
        inconclusive = True
    elif icc < ICC_MIN:
        reasons.append("icc_below_min")
        no_go = True
    if median_abs_delta is None:
        reasons.append("delta_undefined")
        inconclusive = True
    elif median_abs_delta > MEDIAN_ABS_DELTA_MAX:
        reasons.append("median_abs_delta_above_max")
        no_go = True
    if p90_abs_delta is None:
        reasons.append("p90_undefined")
        inconclusive = True
    elif p90_abs_delta > P90_ABS_DELTA_MAX:
        reasons.append("p90_abs_delta_above_max")
        no_go = True
    if cv_percent is None:
        reasons.append("cv_undefined")
        inconclusive = True
    elif cv_percent > CV_PERCENT_MAX:
        reasons.append("cv_above_max")
        no_go = True
    if no_go:
        return "NO-GO", reasons
    if inconclusive:
        return "INCONCLUSIVE", reasons
    return "GO", []


def combine_gates(results: list[str]) -> tuple[str, list[str]]:
    if not results:
        return "INCONCLUSIVE", ["no_metrics"]
    if any(result == "NO-GO" for result in results):
        return "NO-GO", ["metric_no_go"]
    if any(result == "INCONCLUSIVE" for result in results):
        return "INCONCLUSIVE", ["metric_inconclusive"]
    return "GO", []


def _r6(value: float | None) -> float | None:
    if value is None:
        return None
    return round_half_up(value, 6)


def _blank_metric(name: str, version: str | None, targets: int, replicates: int, reason: str) -> MetricRepeatability:
    return MetricRepeatability(
        metric=name,
        metric_version=version,
        targets=targets,
        replicates=replicates,
        icc=None,
        icc_version=ICC_VERSION,
        icc_1_1=None,
        icc_3_1=None,
        median_abs_delta=None,
        p90_abs_delta=None,
        cv_percent=None,
        cv_version=CV_VERSION,
        delta_version=DELTA_VERSION,
        repeatability_gate="INCONCLUSIVE",
        gate_reasons=[reason],
    )


def summarise_metric(name: str, groups: list[list[float]], version: str | None) -> MetricRepeatability:
    n = len(groups)
    if n < 2:
        return _blank_metric(name, version, n, 0, "too_few_targets")
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        return _blank_metric(name, version, n, 0, "unbalanced_replicates")
    k = lengths.pop()
    if k < 2:
        return _blank_metric(name, version, n, k, "too_few_replicates")
    if any(not math.isfinite(value) for group in groups for value in group):
        return _blank_metric(name, version, n, k, "non_finite_value")
    stats = anova_icc(groups)
    deltas = pairwise_abs_deltas(groups)
    icc = _r6(stats.icc_2_1)
    median_delta = _r6(median(deltas))
    p90 = _r6(percentile_type7(deltas, 90))
    cv_rounded = _r6(pooled_cv_percent(groups))
    gate, reasons = decide_gate(icc, median_delta, p90, cv_rounded)
    return MetricRepeatability(
        metric=name,
        metric_version=version,
        targets=n,
        replicates=k,
        icc=icc,
        icc_version=ICC_VERSION,
        icc_1_1=_r6(stats.icc_1_1),
        icc_3_1=_r6(stats.icc_3_1),
        median_abs_delta=median_delta,
        p90_abs_delta=p90,
        cv_percent=cv_rounded,
        cv_version=CV_VERSION,
        delta_version=DELTA_VERSION,
        repeatability_gate=gate,  # type: ignore[arg-type]
        gate_reasons=reasons,
    )


def _sample_warnings(metrics: list[MetricRepeatability], independent_captures: bool | None) -> list[str]:
    warnings = []
    if independent_captures is not True:
        warnings.append("independent_captures_not_declared")
    if any(
        metric.targets < SMALL_SAMPLE_TARGETS or metric.replicates < SMALL_SAMPLE_REPLICATES
        for metric in metrics
    ):
        warnings.append("small_sample")
    return warnings


def report_from_groups(
    grouped: dict[str, tuple[str | None, list[list[float]]]],
    *,
    dataset_class: str,
    independent_captures: bool | None,
    numeric_source: str,
    dropped_targets: list[dict[str, str]] | None = None,
) -> RepeatabilityReport:
    if dataset_class not in _DATASET_CLASSES:
        raise ScanLabError("unknown_dataset_class", "dataset_class is not recognised.")
    metrics = [
        summarise_metric(name, groups, version)
        for name, (version, groups) in grouped.items()
    ]
    overall, reasons = combine_gates([metric.repeatability_gate for metric in metrics])
    return RepeatabilityReport(
        schema_version=SCHEMA_REPEATABILITY,
        statement=REPEATABILITY_STATEMENT,
        dataset_class=dataset_class,
        independent_captures=independent_captures,
        numeric_source=numeric_source,  # type: ignore[arg-type]
        thresholds=GateThresholds(),
        metrics=metrics,
        repeatability_gate=overall,  # type: ignore[arg-type]
        gate_reasons=reasons,
        warnings=_sample_warnings(metrics, independent_captures),
        dropped_targets=dropped_targets or [],
        limitations=list(LIMITATIONS),
    )


def report_from_measurements(payload: dict) -> RepeatabilityReport:
    dataset_class = str(payload.get("dataset_class") or "unknown")
    independent = payload.get("independent_captures")
    if independent is not None and not isinstance(independent, bool):
        raise ScanLabError("invalid_manifest", "independent_captures must be a boolean.")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise ScanLabError("invalid_manifest", "metrics must be a non-empty object.")
    grouped: dict[str, tuple[str | None, list[list[float]]]] = {}
    for name in sorted(raw_metrics):
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            raise ScanLabError("invalid_manifest", "A metric name is invalid.")
        entry = raw_metrics[name]
        if not isinstance(entry, dict):
            raise ScanLabError("invalid_manifest", "Each metric entry must be an object.")
        version = entry.get("version")
        if version is not None and not isinstance(version, str):
            raise ScanLabError("invalid_manifest", "A metric version must be a string.")
        targets = entry.get("targets")
        if not isinstance(targets, dict) or not targets:
            raise ScanLabError("invalid_manifest", "Each metric needs a targets object.")
        groups = []
        for target_id in sorted(targets):
            values = targets[target_id]
            if not isinstance(values, list) or not values:
                raise ScanLabError("invalid_manifest", "Replicates must be a non-empty list.")
            groups.append([float(value) for value in values])
        grouped[name] = (version if isinstance(version, str) else None, groups)
    return report_from_groups(
        grouped,
        dataset_class=dataset_class,
        independent_captures=independent if isinstance(independent, bool) else None,
        numeric_source="supplied_measurements",
    )


def _resolve(root: Path | None, raw: str) -> Path:
    if "://" in raw:
        raise ScanLabError("unsupported_image_reference", "Only local image files are allowed.")
    path = Path(raw)
    if root is not None:
        full = (root / path).resolve()
        base = root.resolve()
        if full != base and base not in full.parents:
            raise ScanLabError("path_escapes_root", "An image path escapes the manifest root.")
        return full
    return path.resolve()


def report_from_image_manifest(payload: dict, settings: Settings, *, root: Path | None = None) -> RepeatabilityReport:
    dataset_class = str(payload.get("dataset_class") or "synthetic")
    independent = payload.get("independent_captures")
    if independent is not None and not isinstance(independent, bool):
        raise ScanLabError("invalid_manifest", "independent_captures must be a boolean.")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ScanLabError("invalid_manifest", "targets must be a non-empty list.")
    collected: dict[str, dict[str, list[float]]] = {name: {} for name in METRIC_NAMES}
    versions: dict[str, set[str]] = {name: set() for name in METRIC_NAMES}
    dropped: list[dict[str, str]] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ScanLabError("invalid_manifest", "Each target must be an object.")
        target_id = str(target.get("target_id") or "")
        paths = target.get("paths")
        if not target_id or not isinstance(paths, list) or len(paths) < 2:
            raise ScanLabError("invalid_manifest", "Each target needs an id and at least two paths.")
        analyses = []
        failed = False
        for raw_path in paths:
            if not isinstance(raw_path, str):
                raise ScanLabError("invalid_manifest", "Image paths must be strings.")
            file_path = _resolve(root, raw_path)
            if not file_path.is_file():
                dropped.append({"target_id": target_id, "reason": "missing_image"})
                failed = True
                break
            data = file_path.read_bytes()
            try:
                analysis = analyze_bytes(data, settings, include_metrics=True)
            except ScanLabError as exc:
                if exc.code == "landmark_backend_unavailable":
                    raise
                dropped.append({"target_id": target_id, "reason": exc.code})
                failed = True
                break
            if not analysis.accepted or not analysis.metrics_included:
                reason = analysis.rejection_reasons[0] if analysis.rejection_reasons else "rejected"
                dropped.append({"target_id": target_id, "reason": reason})
                failed = True
                break
            analyses.append(analysis)
        if failed:
            continue
        for name in METRIC_NAMES:
            values = []
            for analysis in analyses:
                match = next(metric for metric in analysis.metrics if metric.name == name)
                values.append(match.value)
                versions[name].add(match.version)
            collected[name][target_id] = values
    grouped: dict[str, tuple[str | None, list[list[float]]]] = {}
    for name in METRIC_NAMES:
        target_map = collected[name]
        if len(versions[name]) > 1:
            grouped[name] = (None, [])
            continue
        version = next(iter(versions[name])) if versions[name] else None
        groups = [target_map[key] for key in sorted(target_map)]
        grouped[name] = (version, groups)
    # Empty groups become too_few inside summarise only when the list is empty.
    # Filter names with no successful targets into an explicit inconclusive metric.
    final: dict[str, tuple[str | None, list[list[float]]]] = {}
    extras: list[MetricRepeatability] = []
    for name, (version, groups) in grouped.items():
        if not groups:
            extras.append(
                MetricRepeatability(
                    metric=name,
                    metric_version=version,
                    targets=0,
                    replicates=0,
                    icc=None,
                    icc_version=ICC_VERSION,
                    icc_1_1=None,
                    icc_3_1=None,
                    median_abs_delta=None,
                    p90_abs_delta=None,
                    cv_percent=None,
                    cv_version=CV_VERSION,
                    delta_version=DELTA_VERSION,
                    repeatability_gate="INCONCLUSIVE",
                    gate_reasons=["version_mismatch"] if len(versions[name]) > 1 else ["too_few_targets"],
                )
            )
            continue
        final[name] = (version, groups)
    report = report_from_groups(
        final,
        dataset_class=dataset_class,
        independent_captures=independent if isinstance(independent, bool) else None,
        numeric_source="deterministic_cv",
        dropped_targets=dropped,
    )
    if extras:
        report.metrics.extend(extras)
        overall, reasons = combine_gates([metric.repeatability_gate for metric in report.metrics])
        report.repeatability_gate = overall  # type: ignore[assignment]
        report.gate_reasons = reasons
        report.warnings = _sample_warnings(report.metrics, report.independent_captures)
    return report


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScanLabError("invalid_manifest", "The manifest is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ScanLabError("invalid_manifest", "The manifest must be a JSON object.")
    return payload
