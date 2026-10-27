"""Map Scan Lab optical proxies onto Skin Score 1.0.0.

The CV proxies rise with the attribute. Skin Score 1.0.0 expects a higher
number to mean a better appearance, so each proxy is inverted (100 - value)
and shine is stored as dryness/oiliness. Acne, fine lines, and dark circles
are not invented. This matches the mobile client's optical-proxy mapping and
is lab display math, not a validated conversion.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from scanlab.cv.models import SubMetric
from scanlab.score.engine import ACTIVE_VERSION, score
from scanlab.score.engine import ScoreInputError

OPTICAL_PROXY_INVERT_MAPPING = "optical-proxy-invert/1.0.0"

_PROXY_TO_SCORE = {
    "redness": "redness",
    "texture": "texture",
    "pores": "pores",
    "pigmentation": "pigmentation",
    "shine_oiliness": "dryness_oiliness",
}


def invert_optical_proxy(value: float) -> float:
    number = Decimal(str(value))
    if not number.is_finite() or number < 0 or number > 100:
        raise ScoreInputError("optical proxy out of range")
    inverted = (Decimal(100) - number).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return float(inverted)


def skin_score_from_proxies(metrics: list[SubMetric]) -> dict | None:
    values: dict[str, float] = {}
    confidences: dict[str, float] = {}
    for metric in metrics:
        target = _PROXY_TO_SCORE.get(metric.name)
        if target is None or target in values:
            continue
        values[target] = invert_optical_proxy(metric.value)
        confidences[target] = metric.confidence
    if not values:
        return None
    body = score(values, confidences, version=ACTIVE_VERSION).to_dict()
    body["input_mapping"] = OPTICAL_PROXY_INVERT_MAPPING
    body["clinical_validity"] = "not_established"
    return body
