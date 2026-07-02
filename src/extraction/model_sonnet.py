"""Second-pair large tier: Claude Sonnet 4.6.

Used only for the cross-pair generalization experiment (exp_12). The small tier
stays Haiku 4.5 (already cached under tier "small"), so a Haiku-vs-Sonnet pair
reuses the Haiku cache and only pays for Sonnet extractions. Cached under a
separate tier dir ("large_sonnet") so it never clobbers the Opus cache.

Sonnet 4.6 pricing: $3 / $15 per MTok (vs Opus 4.8 $5 / $25). The cost ratio
against Haiku ($1/$5) is 3x, a different ratio than the Opus pair's 5x.
"""
from __future__ import annotations

from src.extraction.base_extractor import BaseExtractor


class SonnetModel(BaseExtractor):
    tier = "large_sonnet"
    model_id = "claude-sonnet-4-6"
    price_in = 3.0
    price_out = 15.0


def extract(document, **kwargs):
    return SonnetModel().extract(document, **kwargs)
