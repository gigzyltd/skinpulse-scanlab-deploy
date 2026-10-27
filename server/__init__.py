"""Scan Lab HTTP API.

Photos are handled in memory for one request and are not written or logged.
Deterministic CV metrics and Skin Score 1.0.0 are the default. YouCam,
AILabTools, and Gemini stay off unless an environment flag turns one on.
"""

API_VERSION = "0.1.0"
