"""Field-level extraction metrics + error-type buckets.

Extraction quality is scored as field-level precision / recall / F1 over a
**multiset of (label, value) pairs**. This handles repeated fields naturally
(a receipt with three ``menu.nm`` values is three items in the multiset) and
works across datasets with different schemas.

Values are normalized before comparison (lowercase, collapse whitespace, drop
punctuation except decimal points) — "normalized exact match", which is the
defensible default for numeric receipt fields.

The error buckets answer the user's question about *why* F1 drops:
  - missed_fields    : a label present in ground truth, absent from prediction (recall failure)
  - wrong_values     : label present in both, but the value(s) differ          (key-value mapping error)
  - spurious_fields  : a label predicted that isn't in ground truth            (precision failure)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.data.base_loader import Document

Pair = tuple[str, str]  # (label, normalized_value)


# -- normalization --------------------------------------------------------------
_PUNCT = re.compile(r"[^\w.]+")


def normalize_value(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop punctuation except '.'."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# -- ground-truth / prediction -> (label, value) pairs --------------------------
def document_to_pairs(doc: Document) -> list[Pair]:
    """Build the ground-truth (label, value) multiset from a Document.

    - SROIE: doc-level ``fields`` dict (4 fields, one value each).
    - CORD / FUNSD: words grouped by (group_id, label), tokens joined in
      reading order to form each field value.
    """
    if doc.fields:
        return [(k, normalize_value(v)) for k, v in doc.fields.items() if v]

    # group words by (group_id, label); join tokens left-to-right, top-to-bottom
    buckets: dict[tuple, list] = {}
    for w in doc.words:
        if w.label is None:
            continue
        key = (w.group_id, w.label)
        buckets.setdefault(key, []).append(w)

    pairs: list[Pair] = []
    for (_, label), words in buckets.items():
        words_sorted = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
        value = normalize_value(" ".join(w.text for w in words_sorted))
        if value:
            pairs.append((label, value))
    return pairs


def prediction_to_pairs(pred: dict | list) -> list[Pair]:
    """Normalize a model prediction into a (label, value) multiset.

    Accepts either ``{"fields": [{"label","value"}, ...]}``, a flat list of
    such dicts, or a ``{label: value | [values]}`` dict.
    """
    items = pred.get("fields", pred) if isinstance(pred, dict) else pred
    pairs: list[Pair] = []
    if isinstance(items, dict):
        for label, val in items.items():
            vals = val if isinstance(val, list) else [val]
            pairs.extend((label, normalize_value(str(v))) for v in vals if str(v).strip())
    else:
        for item in items:
            label = item.get("label")
            value = item.get("value")
            if label and value is not None and str(value).strip():
                pairs.append((label, normalize_value(str(value))))
    return pairs


# -- scoring --------------------------------------------------------------------
@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    tp: int
    n_pred: int
    n_gold: int
    missed_fields: int       # labels in gold, absent from pred (recall failure)
    wrong_values: int        # labels in both, value sets differ (mapping error)
    spurious_fields: int     # labels in pred, absent from gold (precision failure)

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "n_pred": self.n_pred,
            "n_gold": self.n_gold,
            "missed_fields": self.missed_fields,
            "wrong_values": self.wrong_values,
            "spurious_fields": self.spurious_fields,
        }


def score_pairs(pred: list[Pair], gold: list[Pair]) -> Score:
    """Multiset precision / recall / F1 over (label, value) pairs, with buckets."""
    pred_ms, gold_ms = Counter(pred), Counter(gold)

    tp = sum((pred_ms & gold_ms).values())  # multiset intersection
    n_pred, n_gold = sum(pred_ms.values()), sum(gold_ms.values())

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # label-level buckets
    pred_labels = {l for l, _ in pred}
    gold_by_label: dict[str, set] = {}
    pred_by_label: dict[str, set] = {}
    for l, v in gold:
        gold_by_label.setdefault(l, set()).add(v)
    for l, v in pred:
        pred_by_label.setdefault(l, set()).add(v)

    missed = sum(1 for l in gold_by_label if l not in pred_labels)
    wrong = sum(
        1 for l, gv in gold_by_label.items()
        if l in pred_by_label and not (gv & pred_by_label[l])
    )
    spurious = sum(1 for l in pred_by_label if l not in gold_by_label)

    return Score(precision, recall, f1, tp, n_pred, n_gold, missed, wrong, spurious)


def evaluate(prediction: dict | list, doc: Document) -> Score:
    """Score a model prediction against a Document's ground truth."""
    return score_pairs(prediction_to_pairs(prediction), document_to_pairs(doc))
