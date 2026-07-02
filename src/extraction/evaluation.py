"""Field-level extraction metrics + error-type buckets.

Extraction quality is scored as field-level precision / recall / F1 over a
multiset of ``(label, value)`` pairs. This handles repeated fields naturally
(a receipt with three ``menu.nm`` values is three items in the multiset) and
works across datasets with different schemas.

Value matching is **field-type canonical** by default:
  - money / numeric / code fields : equal if both parse to the same number
    after stripping currency tokens and thousands separators. Handles
    ``"RM 110.00" == "110.00" == "110"`` and ``"18,000" == "18000"``.
  - date fields (label contains "date") : equal if every gold date-number is
    present in the prediction (handles ``"23/2/2018 20:04:08"`` as a superset
    of ``"23/2/2018"``). A genuinely wrong day/month/year fails.
  - text fields (company, address, names) : normalized exact match.

The stricter, exact-match scorer is still available via ``canonical=False`` for
the strict-vs-canonical transparency comparison (both are reported in EXP-02).
Rationale: exact-match manufactures penalties when a model formats a
semantically correct value differently (currency prefix, appended timestamp,
thousands separator), which contaminates the F1-gap routing signal.

The error buckets answer *why* F1 drops:
  - missed_fields    : a label present in ground truth, absent from prediction (recall failure)
  - wrong_values     : label present in both, but no value matches            (key-value mapping error)
  - spurious_fields  : a label predicted that isn't in ground truth            (precision failure)
"""
from __future__ import annotations

import re
from collections import defaultdict
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


# -- field-type canonical matching ----------------------------------------------
_CURRENCY = {"rm", "usd", "myr", "sgd", "rp", "idr", "$"}


def _as_number(s: str):
    """Return float if the (normalized) string is a pure number after stripping
    currency tokens and inter-digit spaces (thousands separators), else None."""
    toks = [t for t in s.split() if t not in _CURRENCY]
    joined = "".join(toks)  # collapse "1 234.00" -> "1234.00"
    if not joined:
        return None
    try:
        return float(joined)
    except ValueError:
        return None


def _date_nums(s: str) -> set[str]:
    return set(re.findall(r"\d+", s))


def canonical_match(label: str, gold_v: str, pred_v: str) -> bool:
    """Field-type-aware equivalence on already-normalized values."""
    if gold_v == pred_v:
        return True
    if "date" in label.lower():
        gn, pn = _date_nums(gold_v), _date_nums(pred_v)
        return bool(gn) and gn <= pn  # every gold date-number present in pred
    ng, npv = _as_number(gold_v), _as_number(pred_v)
    if ng is not None and npv is not None:
        return abs(ng - npv) < 1e-6
    return False


def _match(label: str, gold_v: str, pred_v: str, canonical: bool) -> bool:
    return canonical_match(label, gold_v, pred_v) if canonical else gold_v == pred_v


# -- ground-truth / prediction -> (label, value) pairs --------------------------
def document_to_pairs(doc: Document) -> list[Pair]:
    """Build the ground-truth (label, value) multiset from a Document.

    - SROIE / VRDU: doc-level ``fields`` dict (one value each).
    - CORD / FUNSD: words grouped by (group_id, label), tokens joined in
      reading order to form each field value.
    """
    if doc.fields:
        return [(k, normalize_value(v)) for k, v in doc.fields.items() if v]

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
    wrong_values: int        # labels in both, no value matches (mapping error)
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


def score_pairs(pred: list[Pair], gold: list[Pair], canonical: bool = True) -> Score:
    """Multiset precision / recall / F1 over (label, value) pairs, with buckets.

    Matching is field-type canonical by default; pass ``canonical=False`` for
    strict normalized-exact-match (the pre-fix behavior).
    """
    gold_by: dict[str, list] = defaultdict(list)
    pred_by: dict[str, list] = defaultdict(list)
    for l, v in gold:
        gold_by[l].append(v)
    for l, v in pred:
        pred_by[l].append(v)

    tp = 0
    for label, gvs in gold_by.items():
        pvs = pred_by.get(label, [])
        used = [False] * len(pvs)
        for gv in gvs:
            for i, pv in enumerate(pvs):
                if not used[i] and _match(label, gv, pv, canonical):
                    used[i] = True
                    tp += 1
                    break

    n_pred, n_gold = len(pred), len(gold)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # label-level buckets
    missed = sum(1 for l in gold_by if l not in pred_by)
    wrong = sum(
        1 for l, gvs in gold_by.items()
        if l in pred_by and not any(_match(l, gv, pv, canonical)
                                    for gv in gvs for pv in pred_by[l])
    )
    spurious = sum(1 for l in pred_by if l not in gold_by)

    return Score(precision, recall, f1, tp, n_pred, n_gold, missed, wrong, spurious)


def evaluate(prediction: dict | list, doc: Document, canonical: bool = True) -> Score:
    """Score a model prediction against a Document's ground truth."""
    return score_pairs(prediction_to_pairs(prediction), document_to_pairs(doc),
                       canonical=canonical)
