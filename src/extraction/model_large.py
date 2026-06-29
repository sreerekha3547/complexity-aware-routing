"""Large (capable) extraction tier: Claude Opus 4.8."""
from __future__ import annotations

from src.extraction.base_extractor import BaseExtractor


class LargeModel(BaseExtractor):
    tier = "large"
    model_id = "claude-opus-4-8"
    price_in = 5.0
    price_out = 25.0


def extract(document, **kwargs):
    return LargeModel().extract(document, **kwargs)
