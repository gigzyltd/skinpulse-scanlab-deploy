"""Attach Skin Score 1.0.0 to a CV ingest.

The numbers come from the HTTP API adapter (`optical-proxy-invert/1.0.0`):
proxies rise with the attribute, Skin Score wants higher = better, and
shine_oiliness becomes dryness_oiliness. Acne, fine lines, and dark circles
are not invented.

Ingest adds `calibration: uncalibrated`. That label is not a change to the
API mapping. clinical_validity stays not_established. This is not a clinical
result and not a GO.
"""

from __future__ import annotations

from scanlab.cv.models import SubMetric
from scanlab.server.scoring import OPTICAL_PROXY_INVERT_MAPPING
from scanlab.server.scoring import invert_optical_proxy
from scanlab.server.scoring import skin_score_from_proxies as _api_skin_score

CALIBRATION = "uncalibrated"

__all__ = [
    "CALIBRATION",
    "OPTICAL_PROXY_INVERT_MAPPING",
    "invert_optical_proxy",
    "skin_score_from_proxies",
]


def skin_score_from_proxies(metrics: list[SubMetric]) -> dict | None:
    body = _api_skin_score(metrics)
    if body is None:
        return None
    labelled = dict(body)
    labelled["calibration"] = CALIBRATION
    return labelled
