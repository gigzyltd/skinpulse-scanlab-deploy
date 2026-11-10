# SP-034 lighting and pose fixtures (offline)

Decision: GitHub #73 comment 6603192517. Synthetic protocol drawings only. Pipeline `scanlab-cv-0.2.2` is measured as it is. The vendored `scanlab-cv-0.2.1` normalisation is bound for a comparison column and then restored.

```
formula_changed: false
gates_changed: false
gate_widened: false
white_balance_claim: false
angle_claim: false
deployed: false
```

No formula, threshold, gate, or version string was edited. The lighting gate stays 0.15 / 0.30. The frontal pose gate stays ±15°. #74 is not in this change.

Reproduce:

```bash
PYTHONPATH=services python3 -m scanlab.scripts.lighting_pose_fixtures
```

Deltas are signed: condition minus the identity drawing (gray card, exposure 1, frontal landmarks). Identity candidate redness is 41.7989 and the aggregate is 90.75. Identity baseline redness is 51.9344 and the aggregate is 88.37. Skin Score is `optical-proxy-invert/1.0.0` (higher aggregate means the inverted proxies look better). Pores and shine stay at 0 on the plain drawing, so a 0 delta on those two is a pin.

## Exposure

Mean luma is the exposure check. 0.26 is the darkest accepted step on this drawing (mean luma 43.457, fail line 40). 0.45 is inside the band and below the comfort line of 80 (mean luma 75.4433). 1.3 is still accepted (mean luma 213.1372) while a colour channel has clipped. 0.18 and 1.5 are rejected `exposure` and publish no metrics.

| Exposure | Candidate redness | Candidate aggregate | Baseline aggregate |
| --- | ---: | ---: | ---: |
| 0.26 dark edge | -14.2653 | 3.31 | 11.13 |
| 0.45 mid dark | -0.5709 | 0.09 | 8.0 |
| 0.7 | 0.2045 | -0.04 | 4.28 |
| 1.15 | 1.0224 | -0.21 | -2.32 |
| 1.3 bright, still accepted | -23.1163 | 4.67 | 3.14 |

On this drawing the 0.2.1 aggregate rises by 11.13 at the dark edge and by 8.0 at exposure 0.45. The 0.2.2 aggregate rise at those same points is 3.31 and 0.09. The dark-edge redness delta on 0.2.2 is still -14.2653, and pigmentation on that step falls by the whole identity residual (delta -0.8582). The bright accepted step lowers candidate redness by 23.1163 and raises the aggregate by 4.67. Clipped channels are not rebuilt. The exposure gate was not tightened and was not widened.

## Warm and cool

Whole-frame gain 1.03, 1, 0.97 keeps `highlight-neutral-v1` and moves candidate redness by -0.1214. Gain 1.06, 1, 0.90 loses that reference (`identity-no-neutral-reference`), raises candidate redness by 10.4425, and the aggregate delta is -2.09. The same whole-frame gain at 1.15, 1, 0.75 returns `landmarks_incomplete` from this geometric detector and publishes no score.

Marks-held gain 1.15, 1, 0.75 tints skin and the gray card only, so the detector still runs. Candidate redness rises by 30.5265 and the aggregate delta is -6.15. Baseline aggregate on that step falls by 7.7. Marks-held gain 1.20, 1, 0.68 raises candidate redness by 39.7538 and lowers the aggregate by 8.0.

Cool whole-frame gain 0.97, 1, 1.03 keeps the neutral reference (redness delta 0.2634). Gain 0.94, 1, 1.06 drops it: candidate redness -13.3757, aggregate 2.67, baseline aggregate 3.27. Gain 0.90, 1, 1.12 moves candidate redness by -21.0773. Gain 0.85, 1, 1.18 returns `face_not_found` from this detector.

A cast past the spread-12 neutral test is still scored as redness. This fixture does not add a white-balance correction. SP-050 already measured that a wider spread threshold also accepts daylight skin, and that using skin as the gray reference removes a real redness change. Those statements stay the reason this slice does not ship a synthetic-only white balance.

## Yaw and pitch

The drawing stays a flat ellipse. Yaw and pitch move the nose landmark. They do not rotate a head. In-gate aggregate movement on that drawing is small. That is not an angle-normalisation result.

| Pose parameter | Measured degrees | Candidate result | Aggregate delta |
| --- | ---: | --- | ---: |
| yaw shift 24 px | yaw 11.8552 | accepted | -0.06 |
| yaw shift 32 px | yaw 15.8069 | rejected `pose` | withheld |
| pitch ratio 0.72 | pitch 13.2277 | accepted | 0.01 |
| pitch ratio 0.78 | pitch 16.0294 | rejected `pose` | withheld |

The largest accepted yaw aggregate delta in this set is 0.06. The largest accepted pitch aggregate delta is 0.02. Texture at yaw 11.8552 still moves by 0.3073. Pigmentation at pitch 13.2277 moves by -0.1364. The pose gate stays ±15°. Out-of-gate captures are refused with reason `pose` and the degree is on the pose check. No score is published for them.

## Controls

A redness add of 28 on the identity drawing moves candidate redness by 34.7011 and the aggregate by -6.99. A texture amplitude of 8 moves candidate texture by 10.013. Both stay accepted. The pipeline still responds to a deliberate colour change and a deliberate texture change.

## Limitations

| | |
| --- | --- |
| IMPLEMENTED | Offline synthetic fixture, 0.2.1 comparison column, this report |
| VERIFIED | Script tests on in-memory drawings against `scanlab-cv-0.2.2`, with the vendored 0.2.1 normalisation restored afterward |
| UNTESTED | Real faces, phones, MediaPipe, the owner dark-versus-lamp pair (78 and 57), clinical meaning |

The owner figures are not in this fixture and are not claimed as reproduced. A version bump was not made, because a formula or threshold change is what that bump records, and this slice changes neither.
