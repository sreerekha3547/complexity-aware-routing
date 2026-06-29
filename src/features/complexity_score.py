"""Dataset-agnostic complexity signals + composite score.

Operates on the common ``Document`` object (see ``src.data``), so the same
signals apply to CORD, SROIE, and FUNSD. Ported from notebooks/cord_eda.py §11.

Two stages:
  1. ``extract_signals(doc)`` -> raw per-document signals (unnormalized).
  2. ``compute_complexity_score(signals, weights)`` -> single 0..1 score, but
     only after signals are min-max normalized across a corpus (see
     ``normalize_signals``). A score is meaningless for a single document in
     isolation, since normalization is corpus-relative.

Higher score = harder document = route to the more capable (expensive) model.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from src.data.base_loader import Document


# default weights (sum = 1.0); equal-weight baseline across 10 signals.
# Will be tuned once oracle labels are available.
DEFAULT_WEIGHTS: dict[str, float] = {
    "short_token_ratio": 0.10,
    "inv_chars_per_word": 0.10,
    "word_height_cv"   : 0.10,
    "crowded_line_frac": 0.10,
    "line_density"     : 0.10,
    "section_count"    : 0.10,
    "label_diversity"  : 0.10,
    "label_entropy"    : 0.10,
    "item_density"     : 0.10,
    "aspect_ratio"     : 0.10,
}


@dataclass
class Signals:
    """Raw, unnormalized complexity signals for one document."""

    short_token_ratio: float    # OCR: fraction of <=2-char alpha tokens
    inv_chars_per_word: float   # OCR: 1 / avg chars per word (high = fragmented)
    word_height_cv: float       # OCR: bbox height coeff. of variation (font/skew)
    crowded_line_frac: float    # OCR: fraction of groups with >3 words
    line_density: float         # layout: groups per 1000px of image height
    section_count: float        # layout: distinct top-level label sections
    label_diversity: float      # layout: distinct label categories (count)
    label_entropy: float        # layout: Shannon entropy of label distribution
    item_density: float         # layout: groups per word
    aspect_ratio: float         # layout: height / width

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _infer_groups_from_layout(words: list) -> dict[int, list]:
    """Cluster words into pseudo-lines by y-coordinate.

    Used as a fallback for datasets that have no group_id (e.g. SROIE), so
    that line_density and item_density are meaningful rather than always 1.
    Gap threshold is 80 % of the median word height, which is robust across
    different scan resolutions.
    """
    if not words:
        return {}
    heights = [w.height for w in words if w.height > 0]
    gap = statistics.median(heights) * 0.8 if heights else 10.0
    sorted_words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    groups: dict[int, list] = {}
    gid = 0
    band_y = sorted_words[0].bbox[1]
    for w in sorted_words:
        if w.bbox[1] - band_y > gap:
            gid += 1
            band_y = w.bbox[1]
        groups.setdefault(gid, []).append(w)
    return groups


def extract_signals(doc: Document) -> Signals:
    """Compute raw complexity signals from a normalized Document."""
    words = doc.words
    n_words = max(len(words), 1)
    groups = doc.groups()
    if not groups:
        # SROIE has no group_ids; fall back to y-band line clustering
        groups = _infer_groups_from_layout(words)
    n_groups = max(len(groups), 1)

    # OCR signals -------------------------------------------------------------
    short_tokens = sum(1 for w in words if len(w.text) <= 2 and w.text.isalpha())
    char_total = sum(len(w.text) for w in words)
    avg_chars = char_total / n_words
    inv_chars = 1.0 / avg_chars if avg_chars > 0 else 0.0

    heights = [w.height for w in words if w.height > 0]
    if len(heights) > 1:
        m = statistics.mean(heights)
        height_cv = (statistics.pstdev(heights) / m) if m > 0 else 0.0
    else:
        height_cv = 0.0

    # "crowded" = a group (line/item/entity) with more than 3 words
    crowded = sum(1 for g in groups.values() if len(g) > 3)
    crowded_frac = crowded / n_groups

    # Layout signals ----------------------------------------------------------
    line_density = (n_groups / (doc.height / 1000)) if doc.height > 0 else 0.0
    sections = {(lbl.split(".")[0]) for lbl in doc.labels} if doc.labels else set()
    section_count = float(len(sections))
    label_diversity = float(len(set(doc.labels)))

    # Shannon entropy of label distribution — more informative than raw count.
    # Uniform distribution across many labels → high entropy (complex doc).
    # All words share one label → near-zero entropy (simple, uniform doc).
    if doc.labels:
        counts = Counter(doc.labels)
        total_labels = len(doc.labels)
        label_entropy = -sum(
            (c / total_labels) * math.log2(c / total_labels)
            for c in counts.values()
        )
    else:
        label_entropy = 0.0

    item_density = n_groups / n_words
    aspect = doc.aspect_ratio

    return Signals(
        short_token_ratio=short_tokens / n_words,
        inv_chars_per_word=inv_chars,
        word_height_cv=height_cv,
        crowded_line_frac=crowded_frac,
        line_density=line_density,
        section_count=section_count,
        label_diversity=label_diversity,
        label_entropy=label_entropy,
        item_density=item_density,
        aspect_ratio=aspect,
    )


def normalize_signals(signals: list[Signals]) -> list[dict[str, float]]:
    """Min-max normalize each signal across a corpus to 0..1.

    Normalization is corpus-relative, so pass the full set of documents you
    intend to route over (ideally fit on the training split).
    """
    if not signals:
        return []
    keys = list(DEFAULT_WEIGHTS.keys())
    cols = {k: [getattr(s, k) for s in signals] for k in keys}
    bounds = {k: (min(v), max(v)) for k, v in cols.items()}

    out: list[dict[str, float]] = []
    for s in signals:
        row = {}
        for k in keys:
            lo, hi = bounds[k]
            row[k] = 0.0 if hi == lo else (getattr(s, k) - lo) / (hi - lo)
        out.append(row)
    return out


def compute_complexity_score(
    normalized: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of normalized signals -> single 0..1 complexity score."""
    w = weights or DEFAULT_WEIGHTS
    return sum(normalized.get(k, 0.0) * wt for k, wt in w.items())


def score_corpus(
    docs: list[Document],
    weights: dict[str, float] | None = None,
) -> list[float]:
    """Convenience: raw docs -> per-document composite complexity scores."""
    signals = [extract_signals(d) for d in docs]
    normalized = normalize_signals(signals)
    return [compute_complexity_score(n, weights) for n in normalized]
