"""Layout features: line density, section structure, column spread, aspect ratio.

Delegates to the complexity signal extractor so the same Document-level
computation is not duplicated. Values are raw (unnormalized); use
complexity_score.normalize_signals for corpus-relative comparison.
"""
from __future__ import annotations

from src.data.base_loader import Document
from src.features.complexity_score import extract_signals


def extract_layout_features(document: Document) -> dict:
    """Extract layout complexity signals from a Document.

    Returns:
        line_density      -- labelled groups per 1000 px of image height
        section_count     -- distinct top-level label sections
                             (e.g. menu, sub_total, total)
        label_diversity   -- distinct label categories present in the doc
        item_density      -- groups per word (table density proxy)
        aspect_ratio      -- image height / width  (tall = long receipt)
        crowded_line_frac -- fraction of groups containing more than 3 words
        x_spread_ratio    -- word x-range / image width (column-span proxy)
    """
    sig = extract_signals(document)

    xs = [c for w in document.words for c in (w.bbox[0], w.bbox[2])]
    x_spread = (max(xs) - min(xs)) / document.width if xs and document.width > 0 else 0.0

    return {
        "line_density"      : sig.line_density,
        "section_count"     : sig.section_count,
        "label_diversity"   : sig.label_diversity,
        "item_density"      : sig.item_density,
        "aspect_ratio"      : sig.aspect_ratio,
        "crowded_line_frac" : sig.crowded_line_frac,
        "x_spread_ratio"    : round(x_spread, 4),
    }
