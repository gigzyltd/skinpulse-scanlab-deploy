"""Offline lighting and pose fixtures for the current Scan Lab pipeline.

Synthetic protocol drawings only. This module does not change a formula, a
gate, a version string, or a deployed server. It measures the pipeline that
is already imported, then the vendored scanlab-cv-0.2.1 normalisation on the
same drawings, and restores the process.

The 0.2.1 column is a comparison. It is not a claim that either column is
white-balance corrected, angle-normalised, or a phone result.

Run from the repository root:

    PYTHONPATH=services python3 -m scanlab.scripts.lighting_pose_fixtures
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager

import numpy as np

from scanlab.cv.baseline_transplant import BASELINE_COMMIT, apply_baseline_transplant
from scanlab.cv.geometric_backend import skin_mask
from scanlab.cv.pipeline import Settings, analyze_rgb
from scanlab.cv.scoring import skin_score_from_proxies
from scanlab.cv.support import round_half_up
from scanlab.cv.synthetic import render_synthetic_face
from scanlab.cv.versions import (
    COLOUR_IDENTITY,
    COLOUR_NEUTRAL_VERSION,
    ILLUMINATION_VERSION,
    LIGHTING_SIDE_COMFORT,
    LIGHTING_SIDE_FAIL,
    LIGHTING_VERSION,
    METRIC_NAMES,
    NORMALISATION_VERSION,
    PIPELINE_VERSION,
    POSE_ABS_MAX_DEG,
    POSE_VERSION,
    REDNESS_VERSION,
)

SCHEMA = "scanlab-lighting-pose-fixtures-v1"
_SETTINGS = Settings(landmark_backend="geometric")
_IMAGE_LABEL = "synthetic-protocol"
_DECISION_COMMENT = "6603192517"

# One factor per series. Levels are drawing parameters, not a calibration.
_EXPOSURE = (0.18, 0.26, 0.45, 0.70, 1.0, 1.15, 1.30, 1.50)
_WARM_WHOLE = (
    (1.0, 1.0, 1.0),
    (1.03, 1.0, 0.97),
    (1.06, 1.0, 0.90),
    (1.15, 1.0, 0.75),
)
_WARM_MARKS = (
    (1.15, 1.0, 0.75),
    (1.20, 1.0, 0.68),
)
_COOL_WHOLE = (
    (0.97, 1.0, 1.03),
    (0.94, 1.0, 1.06),
    (0.90, 1.0, 1.12),
    (0.85, 1.0, 1.18),
)
_YAW_SHIFT_PX = (0.0, 16.0, 24.0, 32.0)
_PITCH_RATIO = (0.55, 0.45, 0.72, 0.78)
_TRUE_REDNESS_ADD = 28.0
_TEXTURE_AMP = 8.0


def fixture_report() -> dict:
    """Measure signed score deltas. Does not edit the pipeline."""

    cases = _cases()
    reference = render_synthetic_face(gray_card=True)
    candidate_ref = _reading(reference)
    candidate = [_reading(case["image"]) for case in cases]
    with _baseline_bound() as transplant:
        baseline_ref = _reading(reference)
        baseline = [_reading(case["image"]) for case in cases]
        baseline_record = {
            "used": True,
            "role": "comparison_only",
            "source_commit": BASELINE_COMMIT,
            "pipeline_version": transplant["pipeline_version"],
            "normalisation_version": transplant["normalisation_version"],
            "illumination_version": transplant["illumination_version"],
            "restored_after_measurement": True,
        }
    rows = [
        _row(case, candidate[index], baseline[index], candidate_ref, baseline_ref)
        for index, case in enumerate(cases)
    ]
    redness_control = _control_pair(redness_add=_TRUE_REDNESS_ADD)
    texture_control = _control_pair(texture_amp=_TEXTURE_AMP)
    report = {
        "schema_version": SCHEMA,
        "decision_comment": _DECISION_COMMENT,
        "pipeline_version": PIPELINE_VERSION,
        "normalisation_version": NORMALISATION_VERSION,
        "illumination_version": ILLUMINATION_VERSION,
        "redness_version": REDNESS_VERSION,
        "pose_version": POSE_VERSION,
        "lighting_version": LIGHTING_VERSION,
        "landmark_backend": "geometric",
        "dataset_class": "synthetic",
        "independent_captures": False,
        "numeric_source": "deterministic_cv",
        "llm_used": False,
        "network_calls": False,
        "images_written": False,
        "photos_used": False,
        "formula_changed": False,
        "gates_changed": False,
        "gate_widened": False,
        "white_balance_claim": False,
        "angle_claim": False,
        "deployed": False,
        "colour_neutral_version": COLOUR_NEUTRAL_VERSION,
        "colour_identity": COLOUR_IDENTITY,
        "gates": {
            "lighting_side_comfort": LIGHTING_SIDE_COMFORT,
            "lighting_side_fail": LIGHTING_SIDE_FAIL,
            "pose_abs_max_deg": POSE_ABS_MAX_DEG,
        },
        "baseline": baseline_record,
        "reference": {
            "candidate": _public_reading(candidate_ref),
            "baseline": _public_reading(baseline_ref),
        },
        "rows": rows,
        "controls": {
            "true_redness": redness_control,
            "texture": texture_control,
        },
        "summary": _summary(rows, redness_control, texture_control),
        "limitations": [
            "Synthetic protocol drawings and the geometric landmark backend.",
            "Not a phone capture, not MediaPipe, and not a clinical measurement.",
            "No formula, gate, version, or deployment change is applied.",
            "A neutral highlight reference is not a validated white balance.",
            "Yaw and pitch here move landmarks on a flat drawing. They are not a rotated head.",
            "In-gate score movement is not evidence that angles are normalised.",
            "The owner dark-versus-lamp scores are not remeasured. Those photos are not in this fixture.",
            "scanlab-cv-0.2.1 is the vendored normalisation bound for this comparison only.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        sys.stderr.write("usage: python3 -m scanlab.scripts.lighting_pose_fixtures\n")
        return 2
    json.dump(fixture_report(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cases() -> list[dict]:
    cases = []
    for level in _EXPOSURE:
        cases.append(_case("exposure", level, render_synthetic_face(gray_card=True, exposure=level)))
    for level in _WARM_WHOLE:
        cases.append(
            _case("warm_whole", level, render_synthetic_face(gray_card=True, colour_cast=level))
        )
    for level in _WARM_MARKS:
        cases.append(_case("warm_marks", level, _marks_held(level)))
    for level in _COOL_WHOLE:
        cases.append(
            _case("cool_whole", level, render_synthetic_face(gray_card=True, colour_cast=level))
        )
    for level in _YAW_SHIFT_PX:
        cases.append(_case("yaw", level, render_synthetic_face(gray_card=True, yaw_shift_px=level)))
    for level in _PITCH_RATIO:
        cases.append(_case("pitch", level, render_synthetic_face(gray_card=True, pitch_ratio=level)))
    return cases


def _case(series: str, level, image: np.ndarray) -> dict:
    return {"series": series, "level": _public_level(level), "image": image}


def _marks_held(level: tuple[float, float, float]) -> np.ndarray:
    """Tint skin and the gray card. Leave protocol inks so this detector still fires."""

    base = render_synthetic_face(gray_card=True).astype(np.float64)
    gains = np.array(level, dtype=np.float64).reshape(1, 1, 3)
    tinted = np.clip(np.rint(base * gains), 0, 255)
    skin = skin_mask(base.astype(np.uint8))
    card = np.zeros(skin.shape, dtype=bool)
    card[8:28, 8:28] = True
    out = base.copy()
    out[skin | card] = tinted[skin | card]
    return np.ascontiguousarray(out.astype(np.uint8))


def _reading(image: np.ndarray) -> dict:
    result = analyze_rgb(image, _SETTINGS, image_sha256=_IMAGE_LABEL)
    values = {name: None for name in METRIC_NAMES}
    for metric in result.metrics:
        values[metric.name] = metric.value
    score = None
    if result.metrics_included:
        scored = skin_score_from_proxies(result.metrics)
        score = None if scored is None else scored["overall"]
    pose = next((check for check in result.quality_checks if check.name == "pose"), None)
    exposure = next((check for check in result.quality_checks if check.name == "exposure"), None)
    mode = None if result.normalisation is None else result.normalisation.colour_mode
    return {
        "pipeline_version": result.pipeline_version,
        "accepted": bool(result.accepted),
        "metrics_included": bool(result.metrics_included),
        "rejection_reasons": list(result.rejection_reasons),
        "colour_mode": mode,
        "metrics": values,
        "aggregate": score,
        "pose": None
        if pose is None
        else {
            "passed": bool(pose.passed),
            "yaw_deg": pose.components.get("yaw_deg"),
            "pitch_deg": pose.components.get("pitch_deg"),
            "roll_deg": pose.components.get("roll_deg"),
        },
        "exposure": None
        if exposure is None
        else {
            "passed": bool(exposure.passed),
            "mean_luma": exposure.components.get("mean_luma"),
        },
    }


def _public_reading(reading: dict) -> dict:
    return {
        "pipeline_version": reading["pipeline_version"],
        "accepted": reading["accepted"],
        "metrics_included": reading["metrics_included"],
        "rejection_reasons": list(reading["rejection_reasons"]),
        "colour_mode": reading["colour_mode"],
        "metrics": dict(reading["metrics"]),
        "aggregate": reading["aggregate"],
        "pose": reading["pose"],
        "exposure": reading["exposure"],
    }


def _row(case: dict, candidate: dict, baseline: dict, candidate_ref: dict, baseline_ref: dict) -> dict:
    return {
        "series": case["series"],
        "level": case["level"],
        "candidate": _public_reading(candidate),
        "baseline": _public_reading(baseline),
        "candidate_delta": _delta(candidate, candidate_ref),
        "baseline_delta": _delta(baseline, baseline_ref),
    }


def _delta(reading: dict, reference: dict) -> dict | None:
    if not reading["metrics_included"] or not reference["metrics_included"]:
        return None
    out = {
        name: _r4(float(reading["metrics"][name]) - float(reference["metrics"][name]))
        for name in METRIC_NAMES
    }
    out["aggregate"] = _r4(float(reading["aggregate"]) - float(reference["aggregate"]))
    return out


def _control_pair(**overrides) -> dict:
    reference = render_synthetic_face(gray_card=True)
    altered = render_synthetic_face(gray_card=True, **overrides)
    before = _reading(reference)
    after = _reading(altered)
    delta = _delta(after, before)
    return {
        "overrides": {key: _public_level(value) for key, value in overrides.items()},
        "candidate_delta": delta,
        "altered_accepted": bool(after["accepted"] and after["metrics_included"]),
        "reference_accepted": bool(before["accepted"] and before["metrics_included"]),
    }


def _summary(rows: list[dict], redness_control: dict, texture_control: dict) -> dict:
    def pick(series: str, level) -> dict:
        wanted = _public_level(level)
        matches = [row for row in rows if row["series"] == series and row["level"] == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"missing fixture row {series} {wanted}")
        return matches[0]

    def signed(row: dict, column: str, metric: str):
        delta = row[column]
        if delta is None:
            return None
        return delta[metric]

    yaw_in_gate = [
        row
        for row in rows
        if row["series"] == "yaw" and row["candidate"]["metrics_included"] and row["level"] != 0.0
    ]
    pitch_in_gate = [
        row
        for row in rows
        if row["series"] == "pitch" and row["candidate"]["metrics_included"] and row["level"] != 0.55
    ]
    return {
        "dark_edge_exposure": 0.26,
        "dark_edge_candidate_redness_delta": signed(pick("exposure", 0.26), "candidate_delta", "redness"),
        "dark_edge_candidate_aggregate_delta": signed(pick("exposure", 0.26), "candidate_delta", "aggregate"),
        "dark_edge_baseline_redness_delta": signed(pick("exposure", 0.26), "baseline_delta", "redness"),
        "dark_edge_baseline_aggregate_delta": signed(pick("exposure", 0.26), "baseline_delta", "aggregate"),
        "mid_dark_candidate_aggregate_delta": signed(pick("exposure", 0.45), "candidate_delta", "aggregate"),
        "mid_dark_baseline_aggregate_delta": signed(pick("exposure", 0.45), "baseline_delta", "aggregate"),
        "bright_candidate_redness_delta": signed(pick("exposure", 1.30), "candidate_delta", "redness"),
        "bright_candidate_aggregate_delta": signed(pick("exposure", 1.30), "candidate_delta", "aggregate"),
        "bright_baseline_aggregate_delta": signed(pick("exposure", 1.30), "baseline_delta", "aggregate"),
        "warm_crossed_candidate_redness_delta": signed(pick("warm_whole", (1.06, 1.0, 0.90)), "candidate_delta", "redness"),
        "warm_crossed_candidate_aggregate_delta": signed(
            pick("warm_whole", (1.06, 1.0, 0.90)), "candidate_delta", "aggregate"
        ),
        "warm_lamp_candidate_redness_delta": signed(pick("warm_marks", (1.15, 1.0, 0.75)), "candidate_delta", "redness"),
        "warm_lamp_candidate_aggregate_delta": signed(
            pick("warm_marks", (1.15, 1.0, 0.75)), "candidate_delta", "aggregate"
        ),
        "cool_candidate_redness_delta": signed(pick("cool_whole", (0.94, 1.0, 1.06)), "candidate_delta", "redness"),
        "cool_candidate_aggregate_delta": signed(pick("cool_whole", (0.94, 1.0, 1.06)), "candidate_delta", "aggregate"),
        "cool_strong_candidate_redness_delta": signed(pick("cool_whole", (0.90, 1.0, 1.12)), "candidate_delta", "redness"),
        "yaw_in_gate_max_abs_aggregate_delta": _max_abs(signed(row, "candidate_delta", "aggregate") for row in yaw_in_gate),
        "pitch_in_gate_max_abs_aggregate_delta": _max_abs(
            signed(row, "candidate_delta", "aggregate") for row in pitch_in_gate
        ),
        "true_redness_candidate_delta": redness_control["candidate_delta"]["redness"],
        "texture_candidate_delta": texture_control["candidate_delta"]["texture"],
    }


def _max_abs(values) -> float | None:
    numbers = [abs(float(value)) for value in values if value is not None]
    if not numbers:
        return None
    return _r4(max(numbers))


@contextmanager
def _baseline_bound():
    """Bind the vendored 0.2.1 normalisation, then put the process back."""

    import scanlab.cv.normalise as normalise
    import scanlab.cv.pipeline as pipeline
    import scanlab.cv.sensitivity as sensitivity

    saved_normalise = pipeline.normalise
    saved_pipeline = (
        pipeline.PIPELINE_VERSION,
        pipeline.NORMALISATION_VERSION,
        pipeline.ILLUMINATION_VERSION,
    )
    saved_sensitivity = (
        sensitivity.PIPELINE_VERSION,
        sensitivity.NORMALISATION_VERSION,
        sensitivity.ILLUMINATION_VERSION,
    )
    record = apply_baseline_transplant()
    try:
        yield record
    finally:
        pipeline.normalise = saved_normalise
        (
            pipeline.PIPELINE_VERSION,
            pipeline.NORMALISATION_VERSION,
            pipeline.ILLUMINATION_VERSION,
        ) = saved_pipeline
        (
            sensitivity.PIPELINE_VERSION,
            sensitivity.NORMALISATION_VERSION,
            sensitivity.ILLUMINATION_VERSION,
        ) = saved_sensitivity
        if pipeline.normalise is not normalise.normalise:
            pipeline.normalise = normalise.normalise
        if pipeline.PIPELINE_VERSION != PIPELINE_VERSION:
            pipeline.PIPELINE_VERSION = PIPELINE_VERSION
            pipeline.NORMALISATION_VERSION = NORMALISATION_VERSION
            pipeline.ILLUMINATION_VERSION = ILLUMINATION_VERSION


def _public_level(level):
    if isinstance(level, tuple):
        return [_r4(channel) for channel in level]
    return _r4(float(level))


def _r4(value: float) -> float:
    return round_half_up(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
