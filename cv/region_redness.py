"""Image-left and image-right cheek redness on the existing ROI masks.

The number uses the same illumination-normalised CIE a* mapping as the overall
redness proxy. It is not an input to that proxy or to Skin Score. A cheek that
is too small, channel-clipped, or outside the capture gate has a status and no
value. Hidden skin is not reconstructed.

Clipping is ``region-channel-clip-v1``. A pixel counts when any colour channel
on the source image, or on the illumination-normalised image used for CIE a*,
is outside the existing exposure rails. Mean luma is not enough: a red channel
at the rail can still have a mid-range Rec.709 luma. The fraction is counted
on the cheek mask. Above the existing exposure clip maximum the cheek is
``saturated`` and the number is not reconstructed.

Specular highlights are excluded from the CIE a* mean and are not reconstructed.
They stay inside the cheek mask, so a clipped highlight counts toward that
fraction. An isolated set at or below the maximum does not withhold the cheek.
"""

from __future__ import annotations

import numpy as np

from scanlab.cv.geometry import Regions
from scanlab.cv.metrics import _score, _specular, a_star_mean
from scanlab.cv.models import RegionProvenance, RegionRedness
from scanlab.cv.normalise import Normalised
from scanlab.cv.support import round_half_up
from scanlab.cv.versions import (
    EXPOSURE_CLIP_LUMA_HIGH,
    EXPOSURE_CLIP_LUMA_LOW,
    EXPOSURE_CLIP_MAX,
    ILLUMINATION_VERSION,
    NORMALISATION_VERSION,
    PIPELINE_VERSION,
    REDNESS_A_LOW,
    REDNESS_A_SPAN,
    REDNESS_VERSION,
    REGION_CHANNEL_CLIP_VERSION,
    REGION_REDNESS_MIN_PIXELS,
    REGION_REDNESS_PROVENANCE,
)

_REGION_IDS = ("image_left_cheek", "image_right_cheek")

_DESCRIPTIONS = {
    "image_left_cheek": (
        "CIE a* on the image-left cheek ROI after the same colour and illumination "
        "normalisation as the redness proxy, mapped to 0-100. Image-left is the left "
        "side of the picture. Optical proxy only, not a clinical measurement. "
        "Pixels outside this mask are not reconstructed."
    ),
    "image_right_cheek": (
        "CIE a* on the image-right cheek ROI after the same colour and illumination "
        "normalisation as the redness proxy, mapped to 0-100. Image-right is the right "
        "side of the picture. Optical proxy only, not a clinical measurement. "
        "Pixels outside this mask are not reconstructed."
    ),
}


def cheek_redness_regions(
    rgb: np.ndarray,
    regions: Regions,
    *,
    normalised: Normalised | None,
    out_of_gate: bool,
) -> list[RegionRedness]:
    """Measure each cheek mask. ``out_of_gate`` withholds every value."""

    provenance = RegionProvenance(
        method=REGION_REDNESS_PROVENANCE,
        pipeline_version=PIPELINE_VERSION,
        metric_version=REDNESS_VERSION,
        normalisation_version=NORMALISATION_VERSION,
        illumination=ILLUMINATION_VERSION,
    )
    specular = None
    if normalised is not None and not out_of_gate:
        specular = _specular(normalised.illuminated)
    return [
        _one_cheek(
            region_id,
            getattr(regions, region_id),
            rgb=rgb,
            normalised=normalised,
            specular=specular,
            out_of_gate=out_of_gate,
            provenance=provenance,
        )
        for region_id in _REGION_IDS
    ]


def _channel_clipped(rgb: np.ndarray) -> np.ndarray:
    """True where any colour channel is outside the exposure rails.

    Rec.709 luma is a convex combination of the channels, so a luma outside
    the same rails implies some channel is outside them. The converse is not
    true: one channel at the rail can leave luma in range.
    """

    channels = np.asarray(rgb, dtype=np.float64)
    return np.any(
        (channels <= EXPOSURE_CLIP_LUMA_LOW) | (channels >= EXPOSURE_CLIP_LUMA_HIGH),
        axis=2,
    )


def _clip_fraction(rgb: np.ndarray, mask: np.ndarray) -> float:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(_channel_clipped(rgb) & mask)) / float(count)


def _fraction(clipped: np.ndarray, pixels: int) -> float:
    if pixels == 0:
        return 0.0
    return float(np.count_nonzero(clipped)) / float(pixels)


def _one_cheek(
    region_id: str,
    mask: np.ndarray,
    *,
    rgb: np.ndarray,
    normalised: Normalised | None,
    specular: np.ndarray | None,
    out_of_gate: bool,
    provenance: RegionProvenance,
) -> RegionRedness:
    pixels = int(np.count_nonzero(mask))
    description = _DESCRIPTIONS[region_id]
    if pixels < REGION_REDNESS_MIN_PIXELS:
        return _withheld(region_id, "unavailable", pixels, description, provenance)
    source_clipped = _channel_clipped(rgb) & mask
    source_fraction = _fraction(source_clipped, pixels)
    if source_fraction > EXPOSURE_CLIP_MAX:
        return _saturated(region_id, pixels, description, provenance, source_fraction)
    if out_of_gate or normalised is None or specular is None:
        return _withheld(region_id, "uncertain", pixels, description, provenance)
    # Disjoint source and output clips add. A pixel clipped in either image
    # counts once. The output image is the one CIE a* reads.
    combined = source_clipped | (_channel_clipped(normalised.illuminated) & mask)
    combined_fraction = _fraction(combined, pixels)
    if combined_fraction > EXPOSURE_CLIP_MAX:
        return _saturated(region_id, pixels, description, provenance, combined_fraction)
    # Specular highlights are excluded from the mean and are not reconstructed.
    # They were already counted in the clip fraction above. An isolated set at
    # or below EXPOSURE_CLIP_MAX does not withhold the cheek.
    usable = mask & ~specular
    usable_pixels = int(np.count_nonzero(usable))
    if usable_pixels < REGION_REDNESS_MIN_PIXELS:
        return _withheld(region_id, "unavailable", pixels, description, provenance)
    a_mean, _ = a_star_mean(normalised.illuminated, usable)
    value = round_half_up(_score(a_mean, REDNESS_A_SPAN, REDNESS_A_LOW), 4)
    return RegionRedness(
        id=region_id,  # type: ignore[arg-type]
        status="measured",
        value=value,
        pixels=pixels,
        description=description,
        provenance=provenance,
    )


def _saturated(
    region_id: str,
    pixels: int,
    description: str,
    provenance: RegionProvenance,
    fraction: float,
) -> RegionRedness:
    reported = round_half_up(fraction, 4)
    reason = (
        f"{description} {REGION_CHANNEL_CLIP_VERSION}: clip fraction {reported:.4f} "
        f"exceeds {EXPOSURE_CLIP_MAX:.2f}. The cheek number is not reconstructed."
    )
    return _withheld(region_id, "saturated", pixels, reason, provenance)


def _withheld(
    region_id: str,
    status: str,
    pixels: int,
    description: str,
    provenance: RegionProvenance,
) -> RegionRedness:
    return RegionRedness(
        id=region_id,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        pixels=pixels,
        description=description,
        provenance=provenance,
    )
