# SP-034 stability diagnosis — four accepted scans

Decision: GitHub #73 [CTO EXECUTE item A](https://github.com/gigzyltd/skinpulse/issues/73#issuecomment-6622314310), on the field note in [comment 6622102853](https://github.com/gigzyltd/skinpulse/issues/73#issuecomment-6622102853).

This file is the report. No formula, gate, default, API, or deployment change is in it. No new grid was added. No photograph, crop, EXIF field, or identifying hash was read or written. Real-person repeatability is **UNTESTED**. The tables below are synthetic drawings on the geometric backend, plus arithmetic on the published scan figures and the published gate constants.

## How this was measured

Two commands that already exist. Both were run on this tree. The sensitivity numbers match the locked table in `services/scanlab/cv/tests/test_sensitivity.py`. The lighting and pose numbers match `services/scanlab/scripts/tests/test_lighting_pose_fixtures.py`.

```bash
PYTHONPATH=services python3 -m scanlab.cv.cli sensitivity --backend geometric --normalisation baseline --commit f87975571a1ea57a964ba6968f799973cbe2b659
PYTHONPATH=services python3 -m scanlab.cv.cli sensitivity --backend geometric --normalisation candidate
PYTHONPATH=services python3 -m scanlab.scripts.lighting_pose_fixtures
```

| Item | Hosted column | Repository column |
| --- | --- | --- |
| What it is | Vendored `normalise.py` from `f87975571a1ea57a964ba6968f799973cbe2b659`, bound over `scanlab.cv.pipeline.normalise` | `normalise.py` imported with this process. No module transplant |
| Pipeline | `scanlab-cv-0.2.1` | `scanlab-cv-0.2.2` |
| Normalisation | `colour-illum-v1` | `colour-illum-v2` |
| Illumination | `masked-gaussian-v1` | `masked-gaussian-luma-anchor-v2` |
| Metrics, gates, drawings, geometric backend | This tree. They are not transplanted | Same modules |
| Landmark backend | `geometric` | `geometric` |

The hosted column is that normalisation, not a replay of the phone bytes and not a copy of the running server image. The phone figures were not remeasured. They are the figures already published in the field note.

## The four scans, as published numbers

Bot-reported accepted results on hosted `scanlab-cv-0.2.1`, 4 November. This run did not open the contribution store.

| Scan | Overall |
| --- | ---: |
| 1 | 67.5 |
| 2 | 76.4 |
| 3 | 63.8 |
| 4 | 65.9 |

Scans 2–4 are the 96-second window. The largest overall gap in that window is 76.4 − 63.8 = **12.6**. Scan 4 is 2.1 above scan 3. The field note gives redness 67 / 45 / 66, texture 71.5 / 57.9 / 41.2, pores 72.6 / 58.3 / 67.1 for those three scans in that order, face-area fraction 0.156 → 0.094, scan 4 pitch −10°, confidence 0.53 → 0.28, every gate passed, and dryness/oiliness in 93–100. Pigmentation, yaw, roll, exposure, lighting, blur, and the six per-check confidences were not in that note. There is no per-scan alignment of face area with overall.

Those redness and texture figures are the higher-better Skin Score inputs, the same direction as overall. They are not the raw optical proxies. Overall is the equal-weight mean of the inverted proxies (`optical-proxy-invert/1.0.0`), so the raw proxies on a five-metric score sum to `5 × (100 − overall)`. Scan 2 sums to 118. Redness 67 and texture 71.5 already sum to 138.5, which cannot fit. As higher-better inputs they do fit: the five inverted values on scan 2 sum to 382, the three published values sum to 211.1, and the two unpublished inputs (pigmentation and dryness/oiliness) sum to 170.9.

| Step | Overall | Redness | Texture | Pores | Other two, combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 → 3 | −12.6 | −22 | −13.6 | −14.3 | −13.1 |
| 3 → 4 | +2.1 | +21 | −16.7 | +8.8 | −2.6 |

The three published metrics account for 9.98 of the 12.6-point drop. The other 2.62 points are the unpublished pair. Dryness/oiliness sits in a 7-point band (93–100), so that remainder is not dryness alone. Pigmentation was not published.

Redness and pores fall and then recover. Texture falls on both steps (71.5 → 41.2). One exposure or cast shift that reversed between scan 3 and scan 4 would reverse the redness leg and the texture leg together. Texture did not reverse. The texture leg is a separate movement from the redness leg.

## Face size

`face-area-fraction-v1` fails below **0.08** and reaches confidence 1 at **0.18**. Confidence between those lines is `(fraction − 0.08) / 0.10`.

| Fraction | Gate | Face-size confidence |
| --- | --- | ---: |
| 0.156 | pass | 0.76 |
| 0.094 | pass | 0.14 |
| 0.08 | fail line | 0.00 |
| 0.18 | comfort | 1.00 |

Both published fractions pass. 0.094 is 0.014 above the fail line. Their ratio is 0.156 / 0.094 = **1.6596**, above `FACE_DISTANCE_RATIO_MAX` 1.5. The repeatability report already emits `face_distance_mismatch` at that ratio. The warning does not change a score or a gate.

The sensitivity harness has one face-size case: the drawing at fraction **0.0106** (min side 73 px), rejected `face_size`, 0 metrics. The identity drawing used by every other case is fraction **0.2899**, min side 239 px, face-size confidence 1. That is above the comfort line. The phone fractions are below it. The harness does not contain an accepted series from 0.156 to 0.094, and this task does not add one.

Texture and pores downsample a cheek only when face width exceeds 320 px. The identity drawing is 239 px wide, so the grids are in the unscaled regime. The published note has no `min_side_px` and no frame size, so this run cannot say which regime the phone captures were in.

## Pose

The lighting-pose fixture moves the nose landmark on a flat ellipse. It does not rotate a head. In-gate aggregate movement on that drawing is a few hundredths on both versions. The frontal fail line stays ±15°.

| Drawing | Measured pose | 0.2.1 aggregate delta | 0.2.2 aggregate delta | 0.2.2 texture delta |
| --- | --- | ---: | ---: | ---: |
| Yaw shift 16 px | yaw 7.9034° | +0.01 | +0.01 | −0.0297 |
| Yaw shift 24 px | yaw 11.8552° | −0.08 | −0.06 | +0.3073 |
| Yaw shift 32 px | yaw 15.8069° | withheld | withheld | rejected `pose` |
| Pitch ratio 0.45 | pitch −8.1765° | +0.03 | +0.02 | −0.058 |
| Pitch ratio 0.72 | pitch 13.2277° | +0.01 | +0.01 | +0.1103 |
| Pitch ratio 0.78 | pitch 16.0294° | withheld | withheld | rejected `pose` |

−8.1765° is the existing level nearest the published −10°. Pose confidence at 10°, when 10° is the worst of yaw, pitch, and roll, is `(15 − 10) / 10 = 0.5`. The capture still passes. The same fixture's largest in-gate aggregate delta is 0.08 on 0.2.1 and 0.06 on 0.2.2. A 12.6-point overall swing and a 30.3-point texture swing are outside that result. The out-of-gate rows withhold every metric. The phone scans published scores, so they did not take the reject path.

Version does not change these pose deltas in any way that matters to a 12.6-point swing. The 0.2.1 and 0.2.2 columns differ by at most 0.02 on the aggregate.

## Exposure and cast

The sensitivity harness changes one factor at a time. Deltas are within-target pairwise absolute differences, Hyndman–Fan type 7. A saturated reading stays in the pool. `saturation_dominates` means at least half of the scored readings are at 0, 100, or 99.99.

### Exposure — 25 scored, 0 rejected, neutral reference present on all scored

| Metric | 0.2.1 median | 0.2.1 p90 | 0.2.1 max | 0.2.2 median | 0.2.2 p90 | 0.2.2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| redness | 17.9849 | 28.2812 | 43.0612 | 1.9164 | 24.0091 | 44.9876 |
| texture | 2.9643 | 6.2396 | 9.6214 | 0.0691 | 1.4841 | 1.6187 |
| pores | 0.0 | 2.0204 | 10.1311 | 0.0 | 0.0 | 0.0 |
| pigmentation | 0.6529 | 2.89 | 6.2177 | 0.0411 | 0.6655 | 2.6307 |
| shine_oiliness | 0.0 | 0.0 | 7.3035 | 0.0 | 0.0 | 7.3035 |
| aggregate | 3.12 | 6.704 | 8.85 | 1.4 | 5.139 | 9.08 |

Pores 19/25 (0.2.1) and 20/25 (0.2.2) are saturated. Shine is 21/25 on both. `saturation_dominates` is set for both. The pore and shine medians of 0 are pins.

The five-level grid starts at exposure 0.7. The separate fixture includes the darker accepted step 0.26 (mean luma 43.457, fail line 40) and the bright accepted step 1.3 (mean luma 213.1372, a channel already clipped).

| Step | 0.2.1 redness delta | 0.2.1 aggregate | 0.2.2 redness delta | 0.2.2 aggregate |
| --- | ---: | ---: | ---: | ---: |
| 0.26 dark edge | −50.0884 | +11.13 | −14.2653 | +3.31 |
| 0.45 | −35.3442 | +8.0 | −0.5709 | +0.09 |
| 1.3 bright, still accepted | −18.0932 | +3.14 | −23.1163 | +4.67 |

Exposure 0.18 and 1.5 are rejected `exposure` on both versions and publish no metrics. The largest 0.2.1 exposure aggregate move in these tables is 11.13, on the dark edge. That is still short of 12.6. The 0.2.2 column cuts that dark-edge aggregate move to 3.31 and cuts the exposure texture max from 9.6214 to 1.6187. The exposure redness p90 stays at 24.0091, and the max stays at 44.9876. The bright-step redness move is raw redness falling, which raises the displayed redness score. The phone's displayed redness fell by 22 and then recovered. That sign is the warm-cast direction below, not the dark-edge direction.

### Colour cast — 25 scored, 0 rejected, neutral reference mixed (present 10, absent 15)

| Metric | 0.2.1 median | 0.2.1 p90 | 0.2.1 max | 0.2.2 median | 0.2.2 p90 | 0.2.2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| redness | 22.1584 | 42.0916 | 54.3314 | 17.8484 | 33.9581 | 45.4373 |
| texture | 0.1548 | 0.4367 | 0.6035 | 0.0186 | 0.0609 | 0.0922 |
| pores | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pigmentation | 0.0245 | 0.2771 | 5.259 | 0.0059 | 0.0524 | 0.1309 |
| shine_oiliness | 0.0 | 0.0 | 7.3035 | 0.0 | 0.0 | 7.3035 |
| aggregate | 4.46 | 8.461 | 10.89 | 3.56 | 6.654 | 9.09 |

Pores are saturated 20/25 and shine 21/25 on both versions. Cast texture stays under 1 point. A 30-point texture swing is outside cast. A 22-point displayed redness swing fits inside the cast redness range on both versions.

On the fixture, a whole-frame gain of 1.03, 1, 0.97 keeps the neutral reference and moves 0.2.2 redness by −0.1214. Gain 1.06, 1, 0.90 loses it: 0.2.2 redness +10.4425, aggregate −2.09. Marks-held gain 1.15, 1, 0.75 raises 0.2.2 redness by 30.5265 and lowers the aggregate by 6.15 (0.2.1 aggregate −7.7). The stronger marks-held gain lowers the 0.2.2 aggregate by 8.0. Cool gain 0.94, 1, 1.06 lowers 0.2.2 redness by 13.3757 and raises the aggregate by 2.67 (0.2.1 aggregate +3.27). The redness leg of the phone series can sit inside this cast range. The aggregate effect of those cast steps is 6 to 8 points on 0.2.2 and at most 10.89 in the pooled 0.2.1 grid. Cast does not cover the 12.6 overall gap or the texture leg. The 0.2.2 anchor reduces the cast redness median from 22.1584 to 17.8484 and leaves the tail in place, because a cast past the spread-12 neutral test still skips colour-gain correction.

### Side light — 15 scored, 10 rejected `pose`, 10 withheld

| Metric | 0.2.1 median | 0.2.1 p90 | 0.2.1 max | 0.2.2 median | 0.2.2 p90 | 0.2.2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| redness | 0.1444 | 0.2277 | 0.2336 | 0.1155 | 0.3048 | 0.3651 |
| texture | 1.9038 | 10.6582 | 16.2274 | 1.3978 | 7.9286 | 12.1988 |
| pores | 0.0 | 1.6931 | 2.8204 | 0.0 | 0.4794 | 0.75 |
| pigmentation | 0.8884 | 2.9833 | 4.6227 | 0.7597 | 2.4905 | 3.89 |
| shine_oiliness | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aggregate | 0.47 | 3.05 | 4.69 | 0.35 | 2.194 | 3.39 |

Gradients 0.3 and 0.4 fail the existing pose check. Accepted side light does not move redness. Its texture max, 16.2274 on 0.2.1 and 12.1988 on 0.2.2, is below the phone's 30.3-point texture span. Pores on the scored rows are saturated 11/15. The phone pore figures, 58.3 to 72.6 on the higher-better scale, are mid-scale. They are not this pin.

### Noise — 25 scored, 0 rejected

| Metric | 0.2.1 median | 0.2.1 p90 | 0.2.1 max | 0.2.2 median | 0.2.2 p90 | 0.2.2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| redness | 1.3929 | 3.7778 | 5.0521 | 1.249 | 3.4534 | 4.7289 |
| texture | 26.9244 | 65.3723 | 70.7944 | 22.5041 | 54.0615 | 58.8818 |
| pores | 10.6076 | 69.9384 | 83.2535 | 3.5092 | 38.443 | 58.0982 |
| pigmentation | 17.653 | 37.6239 | 44.1802 | 14.6041 | 31.194 | 36.6382 |
| shine_oiliness | 0.0 | 0.0578 | 0.7667 | 0.0 | 0.0578 | 0.7667 |
| aggregate | 12.78 | 34.197 | 38.2 | 9.16 | 24.771 | 29.85 |

Noise is the only single factor in this harness whose aggregate range covers 12.6 (0.2.1 median 12.78, 0.2.2 p90 24.771) and whose texture range covers 30 points. The field note does not record a noise measurement. Covering the span is not an identification of the phone captures. 0.2.2 lowers the noise texture median from 26.9244 to 22.5041 and leaves the p90 at 54.0615.

## What each factor can and cannot cover

Absolute size of the published swings: overall 12.6, displayed redness 22, texture 30.3, pores 14.3.

| Factor | Overall | Redness 22 | Texture 30.3 | Pores 14.3 |
| --- | --- | --- | --- | --- |
| Face size 0.156 and 0.094 | Both pass. Score effect is not in the harness | Unmeasured | Unmeasured | Unmeasured |
| Pose, nearest −8.2° and in-gate yaw | ≤ 0.08 either version | ≤ 0.03 | ≤ 0.44 | 0, pinned |
| Exposure, 0.2.1 | max 8.85; dark edge 11.13 | p90 28.3, and the large fixture steps lower raw redness | max 9.6 | max 10.1, mostly pinned |
| Exposure, 0.2.2 | max 9.08; dark edge 3.31 | p90 24.0; bright step lowers raw redness by 23.1 | max 1.6 | max 0, pinned |
| Cast, 0.2.1 / 0.2.2 | max 10.89 / 9.09 | fits, including the warm-cast sign | max 0.60 / 0.09 | max 0, pinned |
| Accepted side light | max 4.69 / 3.39 | max 0.23 / 0.37 | max 16.2 / 12.2 | max 2.8 / 0.75, mostly pinned |
| Noise | covers, on 0.2.1 already at the median | max about 5, does not cover 22 | covers | covers |

The redness recovery and the texture decline are different shapes. Cast can cover the redness leg and cannot cover texture. Exposure on 0.2.1 can cover a redness move of this size and cannot cover a 30-point texture move. Pose on the existing drawing covers neither. Face size is unmeasured. No single measured factor covers every leg, and the harness never applies two factors at once.

## Rejection, missing values, and saturation

All four phone scans were accepted, so the 12.6-point gap is a gap between published scores. It is not a rejected capture and not a withheld metric. The harness withholds metrics for exposure 0.18 and 1.5 (`exposure`), the tiny face (`face_size`), blur sigma 2 (`blur`), cleared ROI (`exposure`, `lighting_uniformity`, `roi_coverage`), side-light gradients 0.3 and 0.4 (`pose`), yaw 15.8069° (`pose`), and pitch 16.0294° (`pose`). Those controls still withhold on both versions. The phone series did not land on them.

Dryness/oiliness at 93–100 is the inverted shine proxy: raw shine 0–7. Shine's largest pairwise move in the exposure and cast grids is 7.3035, on both versions, and 21 of 25 exposure readings are saturated. A metric pinned at the floor can move by about one of those steps and still look saturated on the displayed scale. Divided by five, 7.3035 points of shine is 1.46 points of overall. It does not produce the 12.6. The 0.2.2 anchor does not move this pin: shine is still measured on the pre-illumination image. The phone pore figures are mid-scale, so the pore leg is not the synthetic pore pin (19–20 of 25 on the exposure grid).

## Confidence

Two different formulas are in the tree.

- Each metric uses the server blend: half the weakest check plus half the mean of the six checks (`quality_confidence` in `services/scanlab/cv/quality.py`). A failed gate forces 0.
- The phone's capture label uses the minimum of the check confidences (`captureQualityConfidence` in `mobile/src/analysis/skinScore.ts`). At or above 0.5 the label is "Clear capture". Below 0.5 it is "Retake recommended". The gates can all pass while that label says retake.

Face fraction 0.156 with the other five checks at 1 gives a blend of 0.86 and a minimum of 0.76. Fraction 0.094 with the others at 1 gives a blend of 0.4983 and a minimum of 0.14. Pitch 10° as the worst angle, others at 1, gives a blend of 0.7083 and a minimum of 0.5. Fraction 0.094 and pitch 10° together, others at 1, give a blend of 0.4567 and a minimum of 0.14.

None of those is 0.53, and none is 0.28. Reaching either published confidence needs at least one more check below 1, or the two figures are not those two captures. A client minimum of 0.28 also cannot be the same capture as fraction 0.094, because that fraction's own face-size confidence is 0.14 and the minimum would be 0.14 or lower. The note does not say which confidence field was copied, and it does not give the six check confidences. The drop is not explained by face size and pitch alone.

## Controls that still move

A deliberate change on the identity drawing stays accepted on both versions.

| Metric | 0.2.1 value → changed | 0.2.1 abs | 0.2.2 value → changed | 0.2.2 abs |
| --- | --- | ---: | --- | ---: |
| redness | 51.9344 → 95.5215 | 43.5871 | 41.7989 → 76.5 | 34.7011 |
| texture | 5.1635 → 19.0663 | 13.9028 | 3.6014 → 13.6144 | 10.013 |
| pores | 0.0 → 87.1909 | 87.1909 | 0.0 → 87.153 | 87.153 |
| pigmentation | 1.0318 → 10.7182 | 9.6864 | 0.8582 → 8.915 | 8.0568 |
| shine_oiliness | 0.0 → 14.6071 | 14.6071 | 0.0 → 14.6071 | 14.6071 |

The 0.2.2 anchor reduces exposure-driven texture movement and keeps this texture control. A further flatten would have to be checked against this control. This report does not change the anchor.

## Proposal

No correction. The four captures do not establish a new face-size line, a new pose line, or a root cause.

The smallest change the measurements would support is no change to the gate, the formula, the default, or the deployed 0.2.1 server. Specifically:

- 0.094 passes the published 0.08 line, and −10° passes the published ±15° line. The pose fixture's in-gate aggregate movement is at most 0.08. Tightening either gate is not a measured fix for a 12.6-point swing.
- Repository 0.2.2 reduces the dark-edge aggregate move (11.13 → 3.31) and the exposure texture max (9.6214 → 1.6187). It leaves exposure redness p90 at 24.0091, cast redness median at 17.8484, the shine pin, and a noise aggregate p90 of 24.771. The phone bytes were not run on it. Deploying it is not a demonstrated fix for this series, and this decision does not authorise a deploy.
- The distance warning at ratio 1.5 already marks 0.156 against 0.094. It does not change the score. Turning that warning into a reject would be a new threshold. These four captures do not justify one.

## Evidence gap

What would be required before a threshold or a formula change, and what this run does not have:

- Per-scan exposure mean, clip fraction, lighting ratio, blur, yaw, roll, `min_side_px`, and the six check confidences.
- Pigmentation, which carries part of the 12.6.
- An accepted face-size series. The harness has a reject below 0.08 and a comfort-sized identity drawing at 0.2899. Building that series would be a new grid.
- The same bytes scored on 0.2.1 and 0.2.2. That would be a photo transfer. It was not done.
- One-factor recaptures. The 96-second window changed framing and, on the published numbers, changed redness and texture in different shapes. The harness does not combine factors, and the field note is not a one-factor record.

## Status

| | |
| --- | --- |
| IMPLEMENTED | This report. Existing sensitivity harness and existing lighting-pose fixture, both versions, on the synthetic drawings. Arithmetic on the published scan figures and the published gate constants |
| VERIFIED | Those two commands on this tree. Sensitivity deltas match the locked test table. Pose, exposure-edge, and cast fixture rows match the locked fixture tests |
| BLOCKED | Nothing in this slice. No gate, formula, default, API, or deploy change was proposed as a patch |
| UNTESTED | The four phone images on either version. MediaPipe. A same-bytes rerun. Real-person repeatability. Clinical meaning |
