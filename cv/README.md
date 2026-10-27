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
| Pose | 2D landmark-ratio proxy, degrees. Frontal gate only | any of yaw, pitch, roll above 15° |
| Lighting | Population CV of 3×3 skin-cell median luma | CV `> 0.15` or range `> 45` |
| ROI coverage | Cheek, T-zone, and skin pixel counts | below the versioned minimums |

These cutoffs are versioned lab settings so obvious synthetic failures separate from a sharp protocol drawing. They are not a validated camera standard.

The frontal pose gate is unchanged: yaw, pitch, and roll must each be within ±15°. A turned face is not accepted by leaving that gate alone.

## Capture resolution

Before the quality gate, bytes are decoded, EXIF orientation is applied, and the longer edge is downscaled to `SCANLAB_MAX_EDGE_PX` (default 1600). The working image is pixels only: EXIF and GPS are not kept. Images are not upscaled. A source whose oriented edge exceeds 12000 px is rejected (`image_too_large`) so a decode bomb is not resized into the gate. Phone-sized frames, including a 1737×3088 capture, become 900×1600 and are then gated.

## Multi-view captures

Views are explicit. The default is `frontal`.

| View | Yaw window | Pitch | Roll | ROI | Scores |
| --- | --- | --- | --- | --- | --- |
| `frontal` | ±15° | ±15° | ±15° | both cheeks, T-zone, skin | five optical proxies |
| `left_45` | −60° to −30° (target −45°) | ±15° | ±15° | near (image-left) cheek required; far cheek is not a gate | withheld |
| `right_45` | +30° to +60° (target +45°) | ±15° | ±15° | near (image-right) cheek required; far cheek is not a gate | withheld |

Yaw is the pose-proxy-v1 nose offset. Positive yaw means the nose lies toward image-right. The near cheek is the cheek on that same image side. Pitch and roll on the oblique views use their own ±15° constants; they do not edit the frontal check.

Optical scores use the frontal view only. Pass `--views-calibrated` (or the API form field) to score an oblique view as well, using the near cheek for cheek metrics. That flag records an explicit choice. It is not evidence that the oblique proxies have been calibrated, and `clinical_go` stays false. Repeatability for a private dataset follows the same rule: frontal groups only, unless the flag is set, in which case each view is reported separately and is not pooled with the frontal numbers.

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

- `mediapipe` (default) — MediaPipe Face Landmarker. Requires the Python extra and the pinned local task file. This process does not download the model. A file that does not match the pin is rejected with `model_checksum_mismatch`. If MediaPipe or the file is missing the call returns `landmark_backend_unavailable` and no metrics.
- `geometric` — classical OpenCV detector for the procedural drawing protocol used by the tests. It is not a general photograph face detector.

Pinned photographic model (`face_landmarker/float16/1`):

```bash
python3 services/scanlab/scripts/download_face_landmarker.py
python3 services/scanlab/scripts/download_face_landmarker.py --check
```

The script writes `services/scanlab/.cache/models/face_landmarker.task` only after the sha256 matches. That directory is gitignored. When `SCANLAB_FACE_LANDMARKER_MODEL` is unset, the pipeline uses the cache if and only if the pin matches. Tests that need the model skip when the cache is absent, so CI does not download it. The MediaPipe wheel is the `mediapipe` extra (`pip install 'mediapipe>=0.10.14'`) and needs `libEGL.so.1` on the machine (`apt install libegl1` on Debian/Ubuntu).

```bash
# Optional override. The file must still match the pin.
export SCANLAB_FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
export SCANLAB_LANDMARK_BACKEND=mediapipe
PYTHONPATH=services python3 -m scanlab.cv.cli quality /path/to/photo.jpg
```

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
| `SCANLAB_FACE_LANDMARKER_MODEL` | Local path to the pinned Face Landmarker `.task` file. Empty uses the checksum-matched cache |
| `SCANLAB_MODEL_CACHE_DIR` | Directory for that cache (default `services/scanlab/.cache/models`) |
| `SCANLAB_MAX_EDGE_PX` | Downscale the longer edge to this before the quality gate (default 1600) |
| `SCANLAB_MAX_BYTES` | Reject payloads larger than this (default 8000000) |

```bash
python -m scanlab.cv.cli quality drawing.png --backend geometric
python -m scanlab.cv.cli quality photo.jpg --view left_45
python -m scanlab.cv.cli dataset ingest --manifest manifest.csv --root /private/photos --output /tmp/scanlab-dataset --backend geometric
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

`pytest` uses procedural drawings only. It does not read `design/reference` or any real user photo. A GO in those tests shows the threshold arithmetic on controlled repeats. It is not evidence that a phone capture is repeatable. MediaPipe tests skip when the pinned task file is not in the local cache.

## Private dataset ingest

Photographs never belong in git. Put a local folder and a manifest outside the tree, or under `data/` (gitignored except this schema's example). The manifest columns are:

`file,subject_id,session_id,view,device,lighting,consent,labels`

Optional: `distance_cm`. Any other column is rejected, so a name or email column cannot be ingested by accident. `subject_id` and `session_id` are pseudonymous (`S001`, `S001-20261027`); do not put a legal name in them. `view` is `frontal`, `left_45`, or `right_45`. `consent` must be `true`, `yes`, `y`, or `1` on every row. A row without consent rejects the whole ingest before any image is opened.

```bash
PYTHONPATH=services python3 -m scanlab.cv.cli dataset ingest \
  --manifest /private/manifest.csv \
  --root /private/photos \
  --output services/scanlab/.cache/dataset-output \
  --backend mediapipe
```

Output (metrics only) is `per_image_metrics.jsonl`, `repeatability_report.json`, and `ingest_summary.json`. The repeatability report is the existing gate: ICC(2,1), median and p90 pairwise `|delta|`, and pooled CV, with `clinical_go` false. Targets are `subject|session|view`. Scores in the default report are frontal. Oblique rows are quality-gated and listed, and their scores are withheld unless `--views-calibrated` is passed. The output directory must be outside the repository or under `services/scanlab/.cache/`.

`data/README.md` and `docs/scanlab/purchased-dataset-label-eval.md` describe the schema and how future purchased labels map onto these metrics. No purchased dataset is assumed to exist.
