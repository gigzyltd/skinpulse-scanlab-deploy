"""Compare providers on consistency, latency, and cost per scan.

The harness reads analyses the adapters already produced. It does not call
a network and it does not decide that a provider is accurate. Scales that
differ, including YouCam's 1–100 score and AILabTools presence flags, are
marked not comparable. AILabTools Skin Analyze costs 15 credits on success.
Gemini stays unpriced and score-free.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Sequence

from scanlab.providers.ailabtools import CREDITS_PER_SUCCESS
from scanlab.providers.types import ProviderAnalysis

HARNESS_VERSION = "1.0.0"


def compare_providers(
    analyses: Sequence[ProviderAnalysis],
    *,
    consistency_tolerance: float = 0.0,
) -> dict:
    if consistency_tolerance < 0:
        raise ValueError("consistency tolerance cannot be negative")
    grouped: dict[tuple[str, str], list[ProviderAnalysis]] = defaultdict(list)
    for analysis in analyses:
        image_id = analysis.image_id or "unspecified"
        grouped[(image_id, analysis.provider)].append(analysis)

    providers: dict[str, dict] = {}
    for (image_id, provider), rows in sorted(grouped.items()):
        key = f"{provider}:{image_id}"
        providers[key] = _summarize(rows, consistency_tolerance)

    return {
        "harness_version": HARNESS_VERSION,
        "validation": "not_a_clinical_comparison",
        "ailabtools_credits_per_successful_analysis": CREDITS_PER_SUCCESS,
        "consistency_tolerance": consistency_tolerance,
        "providers": providers,
        "cross_provider": _cross(analyses),
    }


def _summarize(rows: Sequence[ProviderAnalysis], tolerance: float) -> dict:
    successes = [row for row in rows if row.success]
    latencies = [row.latency_ms for row in rows]
    costs = [row.cost_amount for row in successes if row.cost_amount is not None]
    unit = _one([row.cost_unit for row in rows])
    emits = _one([row.emits_scores for row in rows])
    metric_report, consistent = _consistency(successes, tolerance)
    return {
        "provider": rows[0].provider,
        "image_id": rows[0].image_id,
        "scans": len(rows),
        "successful_scans": len(successes),
        "emits_scores": emits,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "cost_per_successful_scan": (sum(costs) / len(costs)) if costs else None,
        "cost_unit": unit,
        "consistency": {
            "tolerance": tolerance,
            "metrics": metric_report,
            "consistent": consistent,
        },
    }


def _consistency(rows: Sequence[ProviderAnalysis], tolerance: float) -> tuple[dict, bool | None]:
    names: set[str] = set()
    for row in rows:
        names.update(row.metrics)
    if len(rows) < 2:
        return {}, None
    report: dict[str, dict] = {}
    flags: list[bool] = []
    for name in sorted(names):
        observations = [row.metrics[name] for row in rows if name in row.metrics]
        if len(observations) < 2:
            report[name] = {"status": "insufficient_repeats", "consistent": None}
            continue
        scale = _one([item.scale for item in observations])
        if scale == "higher_better_1_to_100" or (
            scale == "presence_0_absent_1_present" and all(item.value is not None for item in observations)
        ):
            values = [float(item.value) for item in observations if item.value is not None]
            center = statistics.median(values)
            deviation = max(abs(value - center) for value in values)
            ok = deviation <= tolerance
            report[name] = {
                "status": "numeric",
                "scale": scale,
                "values": values,
                "max_abs_deviation": deviation,
                "consistent": ok,
            }
            flags.append(ok)
            continue
        if scale == "category":
            categories = [item.category for item in observations]
            ok = len(set(categories)) == 1
            report[name] = {
                "status": "category",
                "scale": scale,
                "categories": categories,
                "consistent": ok,
            }
            flags.append(ok)
            continue
        report[name] = {"status": "not_comparable", "scale": scale, "consistent": None}
    overall = all(flags) if flags else None
    return report, overall


def _cross(analyses: Sequence[ProviderAnalysis]) -> dict:
    by_metric: dict[str, dict[str, str]] = defaultdict(dict)
    for analysis in analyses:
        if not analysis.success or not analysis.emits_scores:
            continue
        for name, metric in analysis.metrics.items():
            by_metric[name][analysis.provider] = metric.scale
    report = {}
    for name, scales in sorted(by_metric.items()):
        unique = set(scales.values())
        providers = sorted(scales)
        if len(providers) < 2:
            status = "single_provider"
        elif len(unique) == 1 and "higher_better_1_to_100" in unique:
            status = "same_numeric_scale"
        elif len(unique) == 1:
            status = "same_non_score_scale"
        else:
            status = "not_comparable"
        report[name] = {"status": status, "providers": scales}
    gemini_rows = [item for item in analyses if item.provider == "gemini"]
    if gemini_rows:
        breached = any(item.emits_scores or item.metrics for item in gemini_rows)
        report["gemini"] = {
            "status": "contract_breach" if breached else "explanation_only",
            "providers": {},
        }
    return report


def _one(values: list):
    unique = set(values)
    if len(unique) == 1:
        return values[0]
    return None
