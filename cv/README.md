# Scan Lab CV (phase 0)

Deterministic local computer vision for SkinPulse Scan Lab. It scores capture quality and five optical proxies, and it applies a repeatability threshold gate. An LLM is not on this path. The service does not call a cloud API, store a photo, or claim a clinical or product result.

A repeatability **GO** means only that the supplied repeats met the numeric thresholds below. `clinical_go` is always false and `product_validity` is always `not_established`. Identical bytes are not a capture study. Real faces, devices, and consented repeats are not in this package.

## What it measures

Quality gate (`quality-gate` checks, each with a value, a confidence, and a version):

| Check | Definition | Hard fail |
| --- | --- | --- |
| Blur | Population variance of a 3×3 Laplacian on the face mask | `< 40` |
| Exposure | Mean Rec.709 luma of skin, plus clipped-pixel fraction | mean outside 40–215, or clip fraction `> 0.10` |
| Face size | Face-mask area / image area, and minimum side | fraction `< 0.08` or side `< 80 px` |
| Pose | 2D landmark-ratio proxy, degrees | any of yaw, pitch, roll above 15° |
| Lighting | Population CV of 3×3 skin-cell median luma | CV `> 0.15` or range `> 45` |
| ROI coverage | Cheek, T-zone, and skin pixel counts | below the versioned minimums |

These cutoffs are versioned lab settings so obvious synthetic failures separate from a sharp protocol drawing. They are not a validated camera standard.

Sub-metrics, each `{value, confidence, version}` on a 0–100 optical-proxy scale:

- **redness** — mean CIE a* on the cheeks
- **texture** — Laplacian energy of illumination-normalised cheek luma
- **pores** — black-hat fraction on the cheeks
- **pigmentation** — dark residual against a local luminance baseline
- **shine_oiliness** — bright, low-chroma fraction of the T-zone after colour normalisation

Shine/oiliness is a specular proxy, not sebum. None of these is hydration, skin age, or a diagnosis. Values rise with the attribute. The separate Skin Score under `services/scanlab/score` treats a higher number as a better appearance, so these proxies are not passed into it. This pipeline does not call YouCam, AILabTools, or Gemini.

Colour normalisation uses only bright low-chroma pixels (a gray card or a neutral highlight). Skin is not the white balance reference. If no neutral reference exists, gains stay at identity and confidence on redness is reduced. Illumination is a masked Gaussian removed before texture, pores, and pigmentation. Shine is measured before that removal.

## Repeatability gate

For each metric, on a balanced set of targets and replicates:

- ICC(2,1) absolute agreement `>= 0.90`
- median within-target pairwise `|delta|` `<= 2`
- 90th percentile of those `|delta|` values `<= 3` (Hyndman–Fan type 7)
- pooled within-target CV `<= 5%` (`100 * sqrt(mean of sample variances) / |grand mean|`)

The gate compares values rounded half-up to 6 decimal places. ICC(1,1) and ICC(3,1) are reported and are not the gate. An undefined ICC (for example every value identical) is **INCONCLUSIVE**, not a GO. One NO-GO metric makes the report NO-GO.

## Landmarks

- `mediapipe` (default) — MediaPipe Face Landmarker. Requires the Python extra and a **local** model file. This process does not download the model. If either is missing the call returns `landmark_backend_unavailable` and no metrics.
- `geometric` — classical OpenCV detector for the procedural drawing protocol used by the tests. It is not a general photograph face detector.

Manual model setup, when you want the photographic backend offline:

```bash
# Save the Face Landmarker task file yourself, then point at the local path.
export SCANLAB_FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
export SCANLAB_LANDMARK_BACKEND=mediapipe
```

The float16 task file published with MediaPipe is:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

## Run

From the repository root, with Python 3.11+. This tree is the `scanlab.cv` subpackage of `services/scanlab` (the same import root as the provider adapters). No paid package is required. The MediaPipe extra is optional and is not needed for the synthetic tests.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r services/scanlab/cv/requirements.txt
cd services/scanlab/cv
pytest
PYTHONPATH=../.. python -m scanlab.cv.cli --help
```

`requirements.txt` lists the versions this tree was executed with. `pytest` adds `services/` to its path. Use `PYTHONPATH=services` from the repository root for the module entry point.

Environment variable names only:

| Name | Meaning |
| --- | --- |
| `SCANLAB_LANDMARK_BACKEND` | `mediapipe` (default) or `geometric` |
| `SCANLAB_FACE_LANDMARKER_MODEL` | Local path to a Face Landmarker `.task` file |
| `SCANLAB_MAX_EDGE_PX` | Reject images whose longer side exceeds this (default 1600) |
| `SCANLAB_MAX_BYTES` | Reject payloads larger than this (default 8000000) |

```bash
python -m scanlab.cv.cli quality drawing.png --backend geometric
python -m scanlab.cv.cli analyze drawing.png --backend geometric --overlay overlay.png
python -m scanlab.cv.cli repeatability --measurements repeats.json
python -m scanlab.cv.cli repeatability --images manifest.json --root /path/to/images --backend geometric
python -m scanlab.cv.cli serve --host 127.0.0.1 --port 8080
```

`analyze` exits 0 when the capture is accepted, 2 when the quality gate rejects it, and 1 on an input or backend error. The JSON is the result either way. `serve` binds to loopback unless you pass another host.

HTTP:

- `GET /health`
- `POST /v1/quality` multipart field `file`, optional form field `backend`
- `POST /v1/analyze` same body
- `POST /v1/repeatability` form field `manifest` plus uploads. Paths in the manifest are bare upload filenames, not disk paths.

## Tests

`pytest` uses procedural drawings only. It does not read `design/reference` or any real user photo. A GO in those tests shows the threshold arithmetic on controlled repeats. It is not evidence that a phone capture is repeatable.
