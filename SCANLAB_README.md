# Scan Lab

Python adapters and the versioned Skin Score. This package does not call a
provider during tests, and it does not deploy cloud resources.

## HTTP API

`python -m scanlab.server` serves `GET /healthz`, `POST /v1/quality`, and `POST /v1/analyze` for the mobile client. Photos stay in memory. Skin Score 1.0.0 is computed from the deterministic CV proxies. YouCam, AILabTools, and Gemini stay off unless `SCANLAB_ENABLE_YOUCAM`, `SCANLAB_ENABLE_AILABTOOLS`, or `SCANLAB_ENABLE_GEMINI` is set to `1`. Gemini does not receive the photo unless `SCANLAB_GEMINI_ALLOW_IMAGE=1` as well. Deploy notes: `docs/ops/scanlab-api-deploy.md`. The phase-0 `python -m scanlab.cv.cli serve` process is the local lab API and does not apply this auth boundary.

```bash
python3 -m pip install -r services/scanlab/cv/requirements.txt
PYTHONPATH=services python3 -m pytest services/scanlab/server/tests -q
```

## Run the tests

```bash
python3 -m unittest discover -s services/scanlab/tests -v
```

The standard library is enough. No package install and no API key are required.
The deterministic CV pipeline in `cv/` is separate: `cd services/scanlab/cv && pytest` after installing `cv/requirements.txt`. It does not call these providers.
Tests replay the JSON fixtures in `tests/fixtures/` through a scripted
transport. The default transport raises if something tries to open a socket.

## Environment names

Copy `services/scanlab/.env.example`. Values stay in the process environment
or a secret store. Do not commit them.

| Name | Use |
| --- | --- |
| `YOUCAM_API_KEY` | Bearer token for YouCam API v2 |
| `YOUCAM_SECRET_KEY` | Required with the API key. Webhook checks use it only when it is a `whsec_` signing secret |
| `AILABTOOLS_API_KEY` | `ailabapi-api-key` header |
| `GEMINI_API_KEY` | `x-goog-api-key` header |

A live call also needs `allow_network=True` on the client. That spends YouCam
units or AILabTools credits. Do not turn it on without Sergei's approval.
Gemini has no dollar price in this lab.

## Providers

- **YouCam / Perfect Corp** skin analysis v2.1 (`format=json`), file upload on `/s2s/v2.0/file`. Nine HD concerns. A successful scan costs **20 units** (the published 9–12 HD band). Failures cost nothing. Metrics use `raw_score`, not `ui_score`. `skin_age` and the provider `all` score are not SkinPulse metrics. Mask URLs are not stored.
- **AILabTools** `POST /api/portrait/analysis/skin-analysis`. **15 credits** per successful analysis. Presence and skin type are kept as presence or category. Redness and texture are absent on this API. They are not filled in.
- **Gemini** `gemini-3.8-flash` `generateContent` with an inline image. Explanation only. The adapter has no metric fields. A reply that contains a digit is withheld.

## Skin Score 1.0.0

Inputs: redness, texture, pores, pigmentation, acne, fine lines, dark circles, dryness/oiliness. Each input is 0–100, higher means the better appearance direction already chosen by the caller. Weights are equal because no validated weights exist. Missing inputs are left out and the remaining weights are renormalized.

Confidence is coverage times the mean of the confidences the provider actually supplied. With values but no confidences, the factor is 0.5. This is not a statistical confidence interval.

The noise threshold is **5 points**, a proposed laboratory gate. It is not a measured minimal detectable change. A smaller absolute change is reported as `No confirmed change`.

Do not treat the score, a provider `ui_score`, or a Gemini sentence as a clinical measurement.
