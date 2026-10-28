"""Versioned, weighted Skin Score.

Version 1.0.0 uses equal weights because no validated weight set exists.
The noise threshold is a proposed laboratory gate of 5 points. It is not a
measured minimal detectable change, and the score is not a clinical result.

Inputs are the eight sub-metrics on a 0–100 scale where a higher number is
the better appearance direction already used by the caller. Missing metrics
are omitted and the remaining weights are renormalized. Confidence is coverage
times the mean of the confidences that were actually supplied. When a provider
supplies no confidences, the confidence factor is 0.5 so a full vector is not
reported as certain.

A change whose absolute difference is below the version's noise threshold is
reported as ``No confirmed change``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping

from scanlab.providers.types import ProviderAnalysis

NO_CONFIRMED_CHANGE = "No confirmed change"
ACTIVE_VERSION = "1.0.0"

METRICS: tuple[str, ...] = (
    "redness",
    "texture",
    "pores",
    "pigmentation",
    "acne",
    "fine_lines",
    "dark_circles",
    "dryness_oiliness",
)

LABELS = {
    "redness": "redness",
    "texture": "texture",
    "pores": "pores",
    "pigmentation": "pigmentation",
    "acne": "acne",
    "fine_lines": "fine lines",
    "dark_circles": "dark circles",
    "dryness_oiliness": "dryness/oiliness",
}

_EQUAL = Decimal("0.125")
_CONFIDENCE_WITHOUT_PROVIDER = Decimal("0.5")


def _version_record() -> dict:
    return {
        "weights": {metric: _EQUAL for metric in METRICS},
        "noise_threshold": Decimal("5"),
        "threshold_kind": "proposed_lab_gate",
        "confidence_method": "coverage_times_mean_supplied_confidence",
        "notes": (
            "Equal weights. The 5-point gate is not a measured MDC. "
            "Not clinically validated."
        ),
    }


VERSIONS = {ACTIVE_VERSION: _version_record()}


class ScoreInputError(ValueError):
    pass


@dataclass(frozen=True)
class SkinScore:
    version: str
    overall: float | None
    confidence: float
    noise_threshold: float
    threshold_kind: str
    weights_used: dict[str, float]
    used: dict[str, float]
    missing: tuple[str, ...]
    confidence_method: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "overall": self.overall,
            "confidence": self.confidence,
            "noise_threshold": self.noise_threshold,
            "threshold_kind": self.threshold_kind,
            "weights_used": self.weights_used,
            "used": self.used,
            "missing": list(self.missing),
            "confidence_method": self.confidence_method,
        }


@dataclass(frozen=True)
class ChangeReport:
    version: str | None
    label: str
    confirmed: bool
    comparable: bool
    delta: float | None
    noise_threshold: float | None
    threshold_kind: str | None
    direction: str | None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "label": self.label,
            "confirmed": self.confirmed,
            "comparable": self.comparable,
            "delta": self.delta,
            "noise_threshold": self.noise_threshold,
            "threshold_kind": self.threshold_kind,
            "direction": self.direction,
        }


def score(
    metrics: Mapping[str, float | None],
    confidences: Mapping[str, float | None] | None = None,
    *,
    version: str = ACTIVE_VERSION,
) -> SkinScore:
    record = VERSIONS.get(version)
    if record is None:
        raise ScoreInputError(f"unknown score version {version}")
    unknown = [key for key in metrics if key not in record["weights"]]
    if unknown:
        raise ScoreInputError("unknown metric " + ", ".join(sorted(unknown)))
    supplied = confidences or {}
    unknown_c = [key for key in supplied if key not in record["weights"]]
    if unknown_c:
        raise ScoreInputError("unknown confidence " + ", ".join(sorted(unknown_c)))

    used: dict[str, Decimal] = {}
    used_confidence: list[Decimal] = []
    for metric in METRICS:
        raw_value = metrics.get(metric)
        if raw_value is None:
            continue
        value = _unit_interval_score(raw_value, metric)
        used[metric] = value
        raw_confidence = supplied.get(metric)
        if raw_confidence is not None:
            used_confidence.append(_confidence(raw_confidence, metric))

    missing = tuple(metric for metric in METRICS if metric not in used)
    coverage = Decimal(len(used)) / Decimal(len(METRICS))
    if used_confidence:
        factor = sum(used_confidence, Decimal(0)) / Decimal(len(used_confidence))
    elif used:
        factor = _CONFIDENCE_WITHOUT_PROVIDER
    else:
        factor = Decimal(0)
    confidence = _round4(coverage * factor)

    if not used:
        return SkinScore(
            version=version,
            overall=None,
            confidence=float(confidence),
            noise_threshold=float(record["noise_threshold"]),
            threshold_kind=record["threshold_kind"],
            weights_used={},
            used={},
            missing=missing,
            confidence_method=record["confidence_method"],
        )

    weight_sum = sum((record["weights"][metric] for metric in used), Decimal(0))
    total = sum((record["weights"][metric] * used[metric] for metric in used), Decimal(0))
    overall = _round2(total / weight_sum)
    weights_used = {
        metric: float(_round4(record["weights"][metric] / weight_sum)) for metric in used
    }
    return SkinScore(
        version=version,
        overall=float(overall),
        confidence=float(confidence),
        noise_threshold=float(record["noise_threshold"]),
        threshold_kind=record["threshold_kind"],
        weights_used=weights_used,
        used={metric: float(used[metric]) for metric in used},
        missing=missing,
        confidence_method=record["confidence_method"],
    )


def numeric_inputs(analysis: ProviderAnalysis) -> tuple[dict[str, float], dict[str, float]]:
    """Copy only higher-better 0–100 metrics. Presence and categories stay out."""
    values: dict[str, float] = {}
    confidences: dict[str, float] = {}
    for metric in analysis.metrics.values():
        if metric.scale != "higher_better_1_to_100" or metric.value is None:
            continue
        if metric.metric not in METRICS:
            continue
        values[metric.metric] = float(metric.value)
        if metric.confidence is not None:
            confidences[metric.metric] = float(metric.confidence)
    return values, confidences


def compare_scores(before: SkinScore, after: SkinScore) -> ChangeReport:
    if (
        before.version != after.version
        or before.version not in VERSIONS
        or before.overall is None
        or after.overall is None
    ):
        return ChangeReport(
            version=before.version if before.version == after.version else None,
            label="Not comparable",
            confirmed=False,
            comparable=False,
            delta=None,
            noise_threshold=None,
            threshold_kind=None,
            direction=None,
        )
    record = VERSIONS[before.version]
    delta = _round2(Decimal(str(after.overall)) - Decimal(str(before.overall)))
    threshold = record["noise_threshold"]
    if abs(delta) < threshold:
        return ChangeReport(
            version=before.version,
            label=NO_CONFIRMED_CHANGE,
            confirmed=False,
            comparable=True,
            delta=float(delta),
            noise_threshold=float(threshold),
            threshold_kind=record["threshold_kind"],
            direction=None,
        )
    return ChangeReport(
        version=before.version,
        label="Changed",
        confirmed=True,
        comparable=True,
        delta=float(delta),
        noise_threshold=float(threshold),
        threshold_kind=record["threshold_kind"],
        direction="higher_score" if delta > 0 else "lower_score",
    )


def _unit_interval_score(value: float, metric: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except Exception as exc:
        raise ScoreInputError(f"{metric} is not a number") from exc
    if number < 0 or number > 100:
        raise ScoreInputError(f"{metric} must be from 0 to 100")
    return number


def _confidence(value: float, metric: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except Exception as exc:
        raise ScoreInputError(f"{metric} confidence is not a number") from exc
    if number < 0 or number > 1:
        raise ScoreInputError(f"{metric} confidence must be from 0 to 1")
    return number


def _round2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round4(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
