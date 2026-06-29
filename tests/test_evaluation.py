"""Unit tests for the field-level F1 scorer (no API calls)."""
from src.data.base_loader import Document, Word
from src.extraction.evaluation import (
    document_to_pairs,
    evaluate,
    normalize_value,
    prediction_to_pairs,
    score_pairs,
)


def test_normalize_value():
    assert normalize_value("  Rp 75,000 ") == "rp 75 000"
    assert normalize_value("Nasi Campur!") == "nasi campur"
    assert normalize_value("9.00") == "9.00"  # decimal preserved


def test_perfect_match():
    gold = [("total", "9.00"), ("menu.nm", "bread")]
    s = score_pairs(list(gold), list(gold))
    assert s.f1 == 1.0 and s.precision == 1.0 and s.recall == 1.0


def test_missed_field_is_recall_failure():
    gold = [("total", "9.00"), ("date", "25/12/2018")]
    pred = [("total", "9.00")]
    s = score_pairs(pred, gold)
    assert s.recall == 0.5 and s.precision == 1.0
    assert s.missed_fields == 1 and s.spurious_fields == 0


def test_wrong_value_is_mapping_error():
    gold = [("total", "9.00")]
    pred = [("total", "6.30")]
    s = score_pairs(pred, gold)
    assert s.tp == 0 and s.wrong_values == 1 and s.missed_fields == 0


def test_spurious_field_is_precision_failure():
    gold = [("total", "9.00")]
    pred = [("total", "9.00"), ("tax", "1.00")]
    s = score_pairs(pred, gold)
    assert s.precision == 0.5 and s.recall == 1.0 and s.spurious_fields == 1


def test_multiset_handles_repeats():
    gold = [("menu.nm", "tea"), ("menu.nm", "tea")]
    pred = [("menu.nm", "tea")]  # only one of two
    s = score_pairs(pred, gold)
    assert s.tp == 1 and s.recall == 0.5


def test_sroie_fields_to_pairs():
    doc = Document(doc_id="x", dataset="sroie", split="test",
                   fields={"company": "ACME", "total": "9.00"})
    pairs = dict(document_to_pairs(doc))
    assert pairs["company"] == "acme" and pairs["total"] == "9.00"


def test_cord_grouped_pairs_and_end_to_end():
    doc = Document(
        doc_id="r", dataset="cord", split="test",
        words=[
            Word("Ice", (0, 0, 30, 20), label="menu.nm", group_id=1),
            Word("Tea", (32, 0, 60, 20), label="menu.nm", group_id=1),
            Word("18,000", (80, 0, 130, 20), label="menu.price", group_id=1),
        ],
        width=200, height=100,
    )
    gold = dict(document_to_pairs(doc))
    assert gold["menu.nm"] == "ice tea"

    pred = {"fields": [{"label": "menu.nm", "value": "Ice Tea"},
                       {"label": "menu.price", "value": "18000"}]}
    # menu.nm matches; menu.price "18000" != "18 000" -> wrong value
    s = evaluate(pred, doc)
    assert s.tp == 1 and s.wrong_values == 1


def test_prediction_dict_form():
    pairs = prediction_to_pairs({"total": "9.00", "items": ["tea", "bread"]})
    assert ("total", "9.00") in pairs and ("items", "tea") in pairs
