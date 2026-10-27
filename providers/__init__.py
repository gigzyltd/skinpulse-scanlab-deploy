"""Skin-analysis provider adapters.

YouCam and AILabTools return provider measurements. Gemini explains stored
output and never emits a score.
"""

from scanlab.providers.ailabtools import AILabToolsClient
from scanlab.providers.gemini import GeminiExplainer
from scanlab.providers.youcam import YouCamClient

__all__ = ["AILabToolsClient", "GeminiExplainer", "YouCamClient"]
