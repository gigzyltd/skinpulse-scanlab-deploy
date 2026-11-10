"""Lighting-sensitivity harness for the five optical proxies and Skin Score.

Synthetic protocol drawings only. This does not read a photograph, write a
crop, or claim real-person or device repeatability. Deltas reuse the
repeatability definitions: within-target pairwise absolute difference, median,
and the Hyndman-Fan type 7 90th percentile.

Each condition changes one capture factor. Exposure, colour cast, side light
and noise are never applied together. Negative controls must withhold metrics
or, for a real texture or colour change, must still move that metric.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from scanlab.cv.geometry import Regions, build_regions
from scanlab.cv.pipeline import Settings, analyze_rgb, build_backend
from scanlab.cv.repeatability import median, pairwise_abs_deltas, percentile_type7
from scanlab.cv.scoring import OPTICAL_PROXY_INVERT_MAPPING, skin_score_from_proxies
from scanlab.cv.support import round_half_up
from scanlab.cv.synthetic import render_synthetic_face
from scanlab.cv.versions import (
    COLOUR_IDENTITY,
    COLOUR_NEUTRAL_VERSION,
    DELTA_VERSION,
    EXPOSURE_MEAN_MAX,
    EXPOSURE_MEAN_MIN,
    ILLUMINATION_VERSION,
    LIGHTING_SIDE_COMFORT,
    LIGHTING_SIDE_FAIL,
    LIGHTING_VERSION,
    METRIC_NAMES,
    METRIC_VERSIONS,
    NORMALISATION_VERSION,
    PIPELINE_VERSION,
)
from scanlab.score.engine import ACTIVE_VERSION

SCHEMA_SENSITIVITY = "scanlab-sensitivity-v1"
_SCORE_NAMES = (*METRIC_NAMES, "aggregate")
_SOFT_CEILING = 99.99

# Five protocol drawings. Parameters are the drawing, not a person.
_TARGETS = (
    {"id": "plain"},
    {"id": "redder", "redness_add": 16.0},
    {"id": "texture", "texture_amp": 5.0},
    {"id": "pores", "pore_spacing": 10.0},
    {"id": "pigment_shine", "pigment_strength": 22.0, "shine_count": 3},
)

# One factor per condition. The first level is the neutral drawing of that factor.
_EXPOSURE_LEVELS = (0.7, 0.85, 1.0, 1.15, 1.3)
_CAST_LEVELS = (
    (1.0, 1.0, 1.0),
    (1.03, 1.0, 0.97),
    (1.08, 1.0, 0.92),
    (0.94, 1.0, 1.06),
    (0.96, 1.04, 0.96),
)
_SIDE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
_NOISE_LEVELS = (0.0, 2.0, 4.0, 8.0, 12.0)
_NOISE_SEED = 34000

_DISCRIMINATION = (
    ("redness", {"redness_add": 28.0}),
    ("texture", {"texture_amp": 8.0}),
    ("pores", {"pore_spacing": 7.0}),
    ("pigmentation", {"pigment_strength": 40.0}),
    ("shine_oiliness", {"shine_count": 6}),
)


def sensitivity_report(
    settings: Settings,
    *,
    source_commit: str | None = None,
    module_transplant: dict | None = None,
) -> dict:
    """Run the synthetic sensitivity cases and return aggregate counts only."""

    if module_transplant is None:
        from scanlab.cv.baseline_transplant import candidate_transplant_record

        module_transplant = candidate_transplant_record()
    conditions = {
        "exposure": _condition_exposure(settings),
        "colour_cast": _condition_cast(settings),
        "side_light": _condition_side(settings),
        "noise": _condition_noise(settings),
    }
    report = {
        "schema_version": SCHEMA_SENSITIVITY,
        "dataset_class": "synthetic",
        "independent_captures": False,
        "numeric_source": "deterministic_cv",
        "llm_used": False,
        "clinical_go": False,
        "product_validity": "not_established",
        "real_person_repeatability": "UNTESTED",
        "device_repeatability": "UNTESTED",
        "hypothesis_under_5_points": "not_a_pass_gate",
        "source_commit": source_commit,
        "settings": {
            "landmark_backend": settings.landmark_backend,
            "max_edge_px": settings.max_edge_px,
            "max_bytes": settings.max_bytes,
        },
        "pipeline_version": PIPELINE_VERSION,
        "normalisation_version": NORMALISATION_VERSION,
        "illumination_version": ILLUMINATION_VERSION,
        "module_transplant": module_transplant,
        "colour_neutral_version": COLOUR_NEUTRAL_VERSION,
        "colour_identity": COLOUR_IDENTITY,
        "metric_versions": dict(METRIC_VERSIONS),
        "score_version": ACTIVE_VERSION,
        "score_mapping": OPTICAL_PROXY_INVERT_MAPPING,
        "delta_version": DELTA_VERSION,
        "percentile": "hyndman-fan-type-7",
        "quality_gate": {
            "lighting_version": LIGHTING_VERSION,
            "lighting_side_comfort": LIGHTING_SIDE_COMFORT,
            "lighting_side_fail": LIGHTING_SIDE_FAIL,
            "exposure_mean_min": EXPOSURE_MEAN_MIN,
            "exposure_mean_max": EXPOSURE_MEAN_MAX,
        },
        "targets": [_public_target(item) for item in _TARGETS],
        "conditions": conditions,
        "negative_controls": _negative_controls(settings),
        "discrimination": _discrimination(settings),
        "limitations": [
            "Synthetic protocol drawings measured with the geometric landmark backend.",
            "Not a consented recapture, not a phone result, and not a clinical measurement.",
            "The under-5-point figure is a hypothesis. These deltas do not establish it.",
            "A saturated proxy can show a small delta because it is pinned, not because it is stable.",
        ],
    }
    return report


def _public_target(spec: dict) -> dict:
    public = {"id": spec["id"], "gray_card": True}
    for key, value in spec.items():
        if key != "id":
            public[key] = value
    return public


def _render(spec: dict, **overrides) -> np.ndarray:
    params = {key: value for key, value in spec.items() if key != "id"}
    params["gray_card"] = True
    params.update(overrides)
    return render_synthetic_face(**params)


def _analyze(rgb: np.ndarray, settings: Settings):
    # Fixed label, not a content hash of a person or a stored photo.
    return analyze_rgb(rgb, settings, image_sha256="synthetic-protocol")


def _with_noise(rgb: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    if sigma <= 0.0:
        return rgb
    rng = np.random.default_rng(seed)
    noisy = rgb.astype(np.float64) + rng.normal(0.0, float(sigma), size=rgb.shape)
    return np.ascontiguousarray(np.clip(np.rint(noisy), 0, 255).astype(np.uint8))


def _condition_exposure(settings: Settings) -> dict:
    samples = []
    for spec in _TARGETS:
        for level in _EXPOSURE_LEVELS:
            samples.append(_sample(settings, spec, _analyze(_render(spec, exposure=level), settings), level))
    return _summarise("exposure", [ _r4(level) for level in _EXPOSURE_LEVELS ], samples)


def _condition_cast(settings: Settings) -> dict:
    samples = []
    for spec in _TARGETS:
        for level in _CAST_LEVELS:
            samples.append(_sample(settings, spec, _analyze(_render(spec, colour_cast=level), settings), level))
    levels = [[_r4(channel) for channel in cast] for cast in _CAST_LEVELS]
    return _summarise("colour_cast", levels, samples)


def _condition_side(settings: Settings) -> dict:
    samples = []
    for spec in _TARGETS:
        for level in _SIDE_LEVELS:
            samples.append(_sample(settings, spec, _analyze(_render(spec, gradient=level), settings), level))
    return _summarise("side_light", [_r4(level) for level in _SIDE_LEVELS], samples)


def _condition_noise(settings: Settings) -> dict:
    samples = []
    for target_index, spec in enumerate(_TARGETS):
        clean = _render(spec)
        for level_index, level in enumerate(_NOISE_LEVELS):
            seed = _NOISE_SEED + target_index * 100 + level_index
            image = _with_noise(clean, level, seed)
            samples.append(_sample(settings, spec, _analyze(image, settings), level))
    return _summarise("noise", [_r4(level) for level in _NOISE_LEVELS], samples)


def _sample(settings: Settings, spec: dict, result, level) -> dict:
    del settings
    values = {name: None for name in _SCORE_NAMES}
    saturated = {name: False for name in METRIC_NAMES}
    if result.metrics_included:
        for metric in result.metrics:
            values[metric.name] = metric.value
            saturated[metric.name] = metric.value in (0.0, 100.0) or metric.value >= _SOFT_CEILING
        score = skin_score_from_proxies(result.metrics)
        values["aggregate"] = None if score is None else score["overall"]
    colour_mode = None if result.normalisation is None else result.normalisation.colour_mode
    return {
        "target_id": spec["id"],
        "level": level,
        "accepted": bool(result.accepted),
        "metrics_included": bool(result.metrics_included),
        "rejection_reasons": list(result.rejection_reasons),
        "colour_mode": colour_mode,
        "values": values,
        "saturated": saturated,
    }


def _summarise(name: str, levels: list, samples: list[dict]) -> dict:
    scored = [sample for sample in samples if sample["metrics_included"]]
    rejected = [sample for sample in samples if not sample["accepted"]]
    reasons: dict[str, int] = {}
    for sample in rejected:
        for reason in sample["rejection_reasons"] or ["rejected"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    present = sum(1 for sample in scored if sample["colour_mode"] == COLOUR_NEUTRAL_VERSION)
    absent = sum(1 for sample in scored if sample["colour_mode"] == COLOUR_IDENTITY)
    groups = {metric: [] for metric in _SCORE_NAMES}
    saturated_counts = {metric: 0 for metric in METRIC_NAMES}
    scored_counts = {metric: 0 for metric in METRIC_NAMES}
    by_target: dict[str, list[dict]] = {}
    for sample in samples:
        by_target.setdefault(sample["target_id"], []).append(sample)
    for target_samples in by_target.values():
        accepted = [sample for sample in target_samples if sample["metrics_included"]]
        for metric in _SCORE_NAMES:
            series = [sample["values"][metric] for sample in accepted if sample["values"][metric] is not None]
            if series:
                groups[metric].append(series)
        for sample in accepted:
            for metric in METRIC_NAMES:
                scored_counts[metric] += 1
                if sample["saturated"][metric]:
                    saturated_counts[metric] += 1
    metrics = {metric: _delta_summary(groups[metric]) for metric in _SCORE_NAMES}
    warnings = []
    for metric, count in saturated_counts.items():
        total = scored_counts[metric]
        if total and count * 2 >= total:
            warnings.append(f"saturation_dominates:{metric}")
    return {
        "condition": name,
        "factor": name,
        "levels": levels,
        "targets": len(by_target),
        "samples": len(samples),
        "accepted": len(samples) - len(rejected),
        "rejected": len(rejected),
        "rejection_rate": _rate(len(rejected), len(samples)),
        "withheld_metrics": sum(1 for sample in samples if not sample["metrics_included"]),
        "rejection_reasons": reasons,
        "neutral_reference_present": present,
        "neutral_reference_absent": absent,
        "neutral_reference": _neutral_label(present, absent),
        "saturated_counts": saturated_counts,
        "scored_counts": scored_counts,
        "metrics": metrics,
        "warnings": warnings,
    }


def _delta_summary(groups: list[list[float]]) -> dict:
    deltas = pairwise_abs_deltas(groups)
    if not deltas:
        return {"n_pairs": 0, "median_abs_delta": None, "p90_abs_delta": None, "max_abs_delta": None}
    return {
        "n_pairs": len(deltas),
        "median_abs_delta": _r4(median(deltas)),
        "p90_abs_delta": _r4(percentile_type7(deltas, 90)),
        "max_abs_delta": _r4(max(deltas)),
    }


def _neutral_label(present: int, absent: int) -> str:
    if present == 0 and absent == 0:
        return "no_scored_capture"
    if absent == 0:
        return "present_on_all_scored"
    if present == 0:
        return "absent_on_all_scored"
    return "mixed"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return _r4(numerator / denominator)


def _r4(value: float) -> float:
    return round_half_up(float(value), 4)


def _negative_controls(settings: Settings) -> dict:
    plain = _TARGETS[0]
    controls = {
        "extreme_clipping_dark": _control_result(
            _analyze(_render(plain, exposure=0.18), settings),
            "exposure 0.18 on the plain protocol drawing",
        ),
        "extreme_clipping_bright": _control_result(
            _analyze(_render(plain, exposure=1.5), settings),
            "exposure 1.5 clips channels in the drawing and is over the exposure mean gate",
        ),
        "insufficient_face": _control_result(
            _analyze(
                render_synthetic_face(width=640, height=800, face_rx=36, face_ry=48, gray_card=True),
                settings,
            ),
            "face below the face-size minimum",
        ),
        "severe_blur": _control_result(
            _analyze(_render(plain, blur_sigma=2.0), settings),
            "blur sigma 2 on the plain protocol drawing",
        ),
        "insufficient_roi": _insufficient_roi(settings),
    }
    return controls


def _control_result(result, note: str) -> dict:
    return {
        "accepted": bool(result.accepted),
        "metrics_included": bool(result.metrics_included),
        "metric_count": len(result.metrics),
        "rejection_reasons": list(result.rejection_reasons),
        "note": note,
    }


def cleared_roi_regions(regions: Regions) -> Regions:
    """Empty the masks the ROI, exposure, and lighting checks read."""

    empty = np.zeros_like(regions.skin)
    return replace(
        regions,
        skin=empty,
        cheeks=empty,
        forehead=empty,
        nose=empty,
        tzone=empty,
        image_left_cheek=empty,
        image_right_cheek=empty,
    )


def _insufficient_roi(settings: Settings) -> dict:
    """Inject cleared masks into analyze_rgb and return that result.

    The geometric protocol does not produce a detected face that fails the ROI
    minimum before the face-size gate. The cleared masks are passed as
    regions_override. Quality checks and the metric gate stay the production
    path. accepted, metrics_included, and metric_count are copied from the
    AnalysisResult. They are not constructed here.
    """

    image = render_synthetic_face(gray_card=True)
    backend = build_backend(settings)
    try:
        detection = backend.detect(image)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    if detection.anchors is None:
        result = _analyze(image, settings)
        payload = _control_result(
            result,
            "The protocol face was not detected, so cleared regions were not injected.",
        )
        payload["evidence"] = "analyze_rgb"
        payload["region_injection"] = False
        payload["roi_coverage_failed"] = "roi_coverage" in result.rejection_reasons
        return payload
    cleared = cleared_roi_regions(build_regions(detection.anchors))
    result = analyze_rgb(
        image,
        settings,
        image_sha256="synthetic-protocol",
        regions_override=cleared,
    )
    payload = _control_result(
        result,
        (
            "Detected protocol face. Cheek, T-zone, forehead, nose, skin, and both "
            "cheek masks were cleared and passed to analyze_rgb as regions_override. "
            "accepted, metrics_included, and metric_count are that call's result."
        ),
    )
    payload["evidence"] = "analyze_rgb"
    payload["region_injection"] = True
    payload["roi_coverage_failed"] = "roi_coverage" in result.rejection_reasons
    return payload


def _discrimination(settings: Settings) -> dict:
    reference_image = render_synthetic_face(gray_card=True)
    reference = _analyze(reference_image, settings)
    reference_values = _metric_map(reference)
    cases = {}
    for name, overrides in _DISCRIMINATION:
        altered = _analyze(render_synthetic_face(gray_card=True, **overrides), settings)
        altered_values = _metric_map(altered)
        before = reference_values.get(name)
        after = altered_values.get(name)
        delta = None if before is None or after is None else _r4(abs(after - before))
        cases[name] = {
            "reference_accepted": bool(reference.accepted and reference.metrics_included),
            "altered_accepted": bool(altered.accepted and altered.metrics_included),
            "reference_value": before,
            "altered_value": after,
            "abs_delta": delta,
            "moved": delta is not None and delta > 0.0,
        }
    return cases


def _metric_map(result) -> dict[str, float | None]:
    if not result.metrics_included:
        return {name: None for name in METRIC_NAMES}
    return {metric.name: metric.value for metric in result.metrics}
