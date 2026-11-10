# SP-050 warm-colour redness regression (offline)

Revision under test: pipeline `scanlab-cv-0.2.2`, geometric landmark backend, neutral rule `highlight-neutral-v1` (channel spread ≤ 12, luma ≥ 40, at least 40 pixels). No formula, gate, API, or deploy change. Synthetic drawings only. No owner photograph was read or transferred.

Reproduce:

```bash
PYTHONPATH=services python3 -m scanlab.scripts.warm_colour_redness
```

`whole_frame` tints the finished drawing. `marks_held` tints skin and the gray card only, and leaves eyes, mouth, and background as protocol ink so this geometric detector still finds the drawing. A strong whole-frame tint turns the mouth into skin colour and returns `landmarks_incomplete`. That rejection is this detector, not MediaPipe. The shared white-balance path is the `marks_held` series.

## What moved

Identity drawing: redness **41.7989**, texture 3.6014, pores 0, pigmentation 0.8582, shine 0, aggregate **90.75**. Neutral reference present (400 pixels). Exposure mean luma 167.9876, luma clip 0. Both cheeks measured.

| Series | Gain (R, G, B) | Mode | Redness | Texture | Pores | Pigment | Shine | Aggregate | Card spread | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| whole frame | 1.03, 1, 0.97 | neutral reference | 41.6775 | 3.6008 | 0 | 0.8589 | 0 | 90.77 | 11 | accepted, clip 0 |
| whole frame | 1.06, 1, 0.90 | reference absent | 52.2414 | 3.6238 | 0 | 0.8547 | 0 | 88.66 | 28 | accepted, clip 0 |
| whole frame | 1.10 and warmer | — | withheld | — | — | — | — | withheld | ≥ 51 | `landmarks_incomplete` |
| marks held | 1.06, 1, 0.90 | reference absent | 53.5265 | 3.6847 | 0 | 0.857 | 0 | 88.39 | 29 | accepted, clip 0 |
| marks held | 1.15, 1, 0.75 | reference absent | 72.3254 | 3.8064 | 0 | 0.8528 | 0 | 84.60 | 72 | accepted, skin channel clip 0 |
| marks held | 1.20, 1, 0.68 | reference absent | 81.5527 | 3.8612 | 0 | 0.8467 | 0 | 82.75 | 94 | accepted; both cheeks `saturated`; skin channel clip 0.9873; luma clip still 0 |

The first whole-frame step that loses the gray card raises redness by **10.4425** and drops the aggregate by **2.09**. With the detector held, gain 1.15, 1, 0.75 raises redness by **30.5265** (to 72.3254) and drops the aggregate by **6.15**. Pores and shine stay pinned at 0, so the aggregate moves by about one fifth of the redness change. The exposure gate stays passed: mean luma stays near 170 and luma clip stays 0.

Exposure, same drawing, no cast: 0.7 → redness 42.0034 / aggregate 90.71; 1.0 → 41.7989 / 90.75; 1.15 → 42.8213 / 90.54. In-band exposure is not this jump. Exposure 1.3 is still accepted (mean luma 213.1372, luma clip 0) while skin channel clip is 0.9941, both cheeks are `saturated`, and headline redness is still published at **18.6826**, so the aggregate rises to **95.42**. Exposure 1.5 is rejected `exposure` and withholds metrics.

Noise (seed 34000), no cast: sigma 12 keeps redness at 44.0635 and moves texture 3.6014 → 62.6755, pores 0 → 36.878, pigmentation 0.8582 → 37.275. Aggregate falls to 63.82 because those proxies move, not because of the warm cast.

True colour (`redness_add` 28): neutral reference, redness 41.7989 → 76.5, absolute delta **34.7011**. Same add under marks-held gain 1.15, 1, 0.75: 72.3254 → 83.416, absolute delta **11.0906**. The real redness change is still visible and is smaller once the lamp is already in the score.

## Why no correction is safe here

Daylight skin on this drawing has minimum channel spread **65** (median 78). The gray card under gain 1.15, 1, 0.75 has median spread **72**. Any spread limit that still accepts that card also accepts daylight skin pixels. White balance would then be taken from skin and would remove the true-colour delta above. A limit that stays below 65 leaves the lamp card (spread 72) unused, which is the current failure. No single threshold separates the two populations.

Using skin as the gray reference is the same unsafe step. The luma anchor is one gain on every channel on purpose; making it per-channel would be a second white balance. Widening the #74 lighting gate does not touch chromatic cast. Withholding redness whenever the neutral reference is absent would also withhold a true redness reading on a capture that simply has no gray card. Cheek saturation already withholds the per-cheek number at gain 1.20 and at exposure 1.3, but the headline redness is still emitted, and the lamp step at gain 1.15 moves redness to 72.3254 with channel clip 0, so applying that cheek rule to the headline would not remove this instability.

No formula or gate change is applied in this slice.

| | |
| --- | --- |
| IMPLEMENTED | Offline synthetic regression, targeted tests, this report |
| VERIFIED | Script tests on in-memory drawings against `scanlab-cv-0.2.2` |
| UNTESTED | Real faces, phones, MediaPipe on a warm lamp, clinical meaning |
