"""Offline warm-colour redness regression for scanlab-cv-0.2.2.

Synthetic protocol drawings only. Nothing here reads a photograph, writes an
image, or contacts a network. The pipeline, gates, and Skin Score are called
as they are on this revision. This script does not change them.

Two cast series:

- ``whole_frame`` multiplies the finished drawing, including the gray card.
  A strong warm multiply also tints the dark mouth, and the geometric
  protocol detector can then return ``landmarks_incomplete``. That rejection
  is this detector, not MediaPipe.
- ``marks_held`` applies the same multiply only to skin and the gray card,
  and leaves eyes, mouth, and background at the protocol inks. The shared
  white-balance path can then be measured after the detector still finds the
  drawing.

Run from the repository root:

    PYTHONPATH=services python3 -m scanlab.scripts.warm_colour_redness
"""

from __future__ import annotations

import json
import sys

import numpy as np

from scanlab.cv.colorimetry import rec709_luma
from scanlab.cv.geometric_backend import skin_mask
from scanlab.cv.pipeline import Settings, analyze_rgb
from scanlab.cv.scoring import skin_score_from_proxies
from scanlab.cv.support import round_half_up
from scanlab.cv.synthetic import render_synthetic_face
from scanlab.cv.versions import (
    COLOUR_IDENTITY,
    COLOUR_NEUTRAL_VERSION,
    EXPOSURE_CLIP_LUMA_HIGH,
    EXPOSURE_CLIP_LUMA_LOW,
    METRIC_NAMES,
    PIPELINE_VERSION,
)

SCHEMA = "scanlab-warm-colour-redness-v1"
_SETTINGS = Settings(landmark_backend="geometric")
_IMAGE_LABEL = "synthetic-protocol"
_NEUTRAL_SPREAD_MAX = 12.0
_NEUTRAL_LUMA_MIN = 40.0
_NOISE_SEED = 34000

# RGB gains, warm direction only: red up, blue down. Not a Kelvin calibration.
_WARM_LEVELS = (
    (1.0, 1.0, 1.0),
    (1.03, 1.0, 0.97),
    (1.06, 1.0, 0.90),
    (1.10, 1.0, 0.82),
    (1.15, 1.0, 0.75),
    (1.20, 1.0, 0.68),
)
_EXPOSURE_LEVELS = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
_NOISE_LEVELS = (0.0, 2.0, 4.0, 8.0, 12.0)
_LAMP = (1.15, 1.0, 0.75)
_TRUE_REDNESS_ADD = 28.0


def regression_report() -> dict:
    """Measure exposure, warm cast, and noise separately on pipeline 0.2.2."""

    whole = [_cast_reading("whole_frame", level, _whole_frame(level)) for level in _WARM_LEVELS]
    held = [_cast_reading("marks_held", level, _marks_held(level)) for level in _WARM_LEVELS]
    exposure = [_factor_reading("exposure", level, _exposure(level)) for level in _EXPOSURE_LEVELS]
    noise = [_factor_reading("noise", level, _noise(level)) for level in _NOISE_LEVELS]
    true_colour = _true_colour()
    overlap = _spread_overlap()
    return {
        "schema_version": SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
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
        "neutral_rule": {
            "spread_max": _NEUTRAL_SPREAD_MAX,
            "luma_min": _NEUTRAL_LUMA_MIN,
            "minimum_pixels": 40,
            "colour_neutral_version": COLOUR_NEUTRAL_VERSION,
            "colour_identity": COLOUR_IDENTITY,
        },
        "warm_levels": [_public_level(level) for level in _WARM_LEVELS],
        "whole_frame": whole,
        "marks_held": held,
        "exposure": exposure,
        "noise": noise,
        "true_colour": true_colour,
        "spread_overlap": overlap,
        "correction": {
            "applied": False,
            "safe_to_apply": False,
            "reason": "spread_populations_overlap",
        },
        "limitations": [
            "Synthetic protocol drawings and the geometric landmark backend.",
            "Not a phone capture, not MediaPipe, and not a clinical measurement.",
            "Restoring eyes, mouth, and background isolates the white-balance path from this detector.",
            "No owner photograph is read or transferred.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        sys.stderr.write("usage: python3 -m scanlab.scripts.warm_colour_redness\n")
        return 2
    json.dump(regression_report(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _whole_frame(level: tuple[float, float, float]) -> np.ndarray:
    return render_synthetic_face(gray_card=True, colour_cast=tuple(level))


def _exposure(level: float) -> np.ndarray:
    return render_synthetic_face(gray_card=True, exposure=float(level))


def _noise(level: float) -> np.ndarray:
    image = render_synthetic_face(gray_card=True)
    if level <= 0.0:
        return image
    rng = np.random.default_rng(_NOISE_SEED)
    noisy = image.astype(np.float64) + rng.normal(0.0, float(level), size=image.shape)
    return np.ascontiguousarray(np.clip(np.rint(noisy), 0, 255).astype(np.uint8))


def _marks_held(level: tuple[float, float, float], *, redness_add: float = 0.0) -> np.ndarray:
    """Tint skin and the gray card. Leave protocol inks so the detector still fires."""

    base = render_synthetic_face(gray_card=True, redness_add=redness_add).astype(np.float64)
    gains = np.array(level, dtype=np.float64).reshape(1, 1, 3)
    tinted = np.clip(np.rint(base * gains), 0, 255)
    skin = skin_mask(base.astype(np.uint8))
    card = _card_mask(skin.shape)
    out = base.copy()
    out[skin | card] = tinted[skin | card]
    return np.ascontiguousarray(out.astype(np.uint8))


def _card_mask(shape: tuple[int, int]) -> np.ndarray:
    card = np.zeros(shape, dtype=bool)
    card[8:28, 8:28] = True
    return card


def _cast_reading(series: str, level, image: np.ndarray) -> dict:
    reading = _reading(image)
    reading["series"] = series
    reading["level"] = _public_level(level)
    reading["card_spread"] = _card_spread(image)
    reading["neutral_pixels"] = _neutral_count(image)
    return reading


def _factor_reading(factor: str, level: float, image: np.ndarray) -> dict:
    reading = _reading(image)
    reading["factor"] = factor
    reading["level"] = _r4(level)
    reading["card_spread"] = _card_spread(image)
    reading["neutral_pixels"] = _neutral_count(image)
    return reading


def _reading(image: np.ndarray) -> dict:
    result = analyze_rgb(image, _SETTINGS, image_sha256=_IMAGE_LABEL)
    values = {name: None for name in METRIC_NAMES}
    for metric in result.metrics:
        values[metric.name] = metric.value
    score = None
    if result.metrics_included:
        scored = skin_score_from_proxies(result.metrics)
        score = None if scored is None else scored["overall"]
    exposure = next((check for check in result.quality_checks if check.name == "exposure"), None)
    regions = []
    for region in result.regions or []:
        regions.append({"id": region.id, "status": region.status, "value": region.value})
    mode = None if result.normalisation is None else result.normalisation.colour_mode
    return {
        "accepted": bool(result.accepted),
        "metrics_included": bool(result.metrics_included),
        "rejection_reasons": list(result.rejection_reasons),
        "colour_mode": mode,
        "warnings": list(result.warnings),
        "metrics": values,
        "aggregate": score,
        "exposure": None
        if exposure is None
        else {
            "passed": bool(exposure.passed),
            "mean_luma": exposure.components.get("mean_luma"),
            "clip_low": exposure.components.get("clip_low"),
            "clip_high": exposure.components.get("clip_high"),
        },
        "skin_channel_clip_fraction": _skin_clip_fraction(image),
        "regions": regions,
    }


def _true_colour() -> dict:
    neutral_ref = _reading(render_synthetic_face(gray_card=True))
    neutral_red = _reading(render_synthetic_face(gray_card=True, redness_add=_TRUE_REDNESS_ADD))
    warm_ref = _reading(_marks_held(_LAMP))
    warm_red = _reading(_marks_held(_LAMP, redness_add=_TRUE_REDNESS_ADD))
    neutral_delta = _subtract(_metric(neutral_red, "redness"), _metric(neutral_ref, "redness"))
    warm_delta = _subtract(_metric(warm_red, "redness"), _metric(warm_ref, "redness"))
    return {
        "redness_add": _TRUE_REDNESS_ADD,
        "warm_cast": _public_level(_LAMP),
        "neutral": {
            "reference_redness": _metric(neutral_ref, "redness"),
            "altered_redness": _metric(neutral_red, "redness"),
            "abs_delta": None if neutral_delta is None else _r4(abs(neutral_delta)),
            "colour_mode": neutral_ref["colour_mode"],
        },
        "warm_marks_held": {
            "reference_redness": _metric(warm_ref, "redness"),
            "altered_redness": _metric(warm_red, "redness"),
            "abs_delta": None if warm_delta is None else _r4(abs(warm_delta)),
            "colour_mode": warm_ref["colour_mode"],
        },
        "retained": bool(
            neutral_delta is not None
            and warm_delta is not None
            and neutral_delta > 5.0
            and warm_delta > 5.0
        ),
    }


def _spread_overlap() -> dict:
    identity = render_synthetic_face(gray_card=True)
    warm = _marks_held(_LAMP)
    identity_skin = _spreads(identity)[skin_mask(identity)]
    warm_card = _spreads(warm)[_card_mask(warm.shape[:2])]
    skin_min = float(identity_skin.min()) if identity_skin.size else None
    card_median = float(np.median(warm_card)) if warm_card.size else None
    separates = False
    if skin_min is not None and card_median is not None:
        separates = skin_min > card_median
    return {
        "warm_cast": _public_level(_LAMP),
        "identity_skin_min_spread": None if skin_min is None else _r4(skin_min),
        "identity_skin_median_spread": None if identity_skin.size == 0 else _r4(float(np.median(identity_skin))),
        "warm_card_median_spread": None if card_median is None else _r4(card_median),
        "threshold_can_separate": separates,
    }


def _spreads(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float64)
    return values.max(axis=2) - values.min(axis=2)


def _card_spread(image: np.ndarray) -> float:
    card = image[8:28, 8:28].astype(np.float64)
    return _r4(float((card.max(axis=2) - card.min(axis=2)).mean()))


def _neutral_count(image: np.ndarray) -> int:
    values = image.astype(np.float64)
    spread = values.max(axis=2) - values.min(axis=2)
    luma = rec709_luma(values)
    return int(np.count_nonzero((spread <= _NEUTRAL_SPREAD_MAX) & (luma >= _NEUTRAL_LUMA_MIN)))


def _skin_clip_fraction(image: np.ndarray) -> float:
    skin = skin_mask(image)
    count = int(np.count_nonzero(skin))
    if count == 0:
        return 0.0
    channels = image.astype(np.float64)
    clipped = np.any(
        (channels <= EXPOSURE_CLIP_LUMA_LOW) | (channels >= EXPOSURE_CLIP_LUMA_HIGH),
        axis=2,
    )
    return _r4(float(np.count_nonzero(clipped & skin)) / float(count))


def _metric(reading: dict, name: str):
    return reading["metrics"][name]


def _subtract(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _public_level(level) -> list[float]:
    return [_r4(channel) for channel in level]


def _r4(value: float) -> float:
    return round_half_up(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
