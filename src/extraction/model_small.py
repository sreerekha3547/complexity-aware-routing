"""Small (cheap) extraction tier: Claude Haiku 4.5."""
from __future__ import annotations

from src.extraction.base_extractor import BaseExtractor


class SmallModel(BaseExtractor):
    tier = "small"
    model_id = "claude-haiku-4-5-20251001"
    price_in = 1.0
    price_out = 5.0


def extract(document, **kwargs):
    return SmallModel().extract(document, **kwargs)
