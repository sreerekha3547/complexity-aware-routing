"""Unit tests for feature extractors."""
import pytest

from src.data.base_loader import Document, Word
from src.features.complexity_score import (
    compute_complexity_score,
    extract_signals,
    normalize_signals,
    score_corpus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_doc(doc_id: str = "simple") -> Document:
    """Short receipt — low complexity."""
    return Document(
        doc_id=doc_id, dataset="cord", split="test",
        words=[
            Word(text="COFFEE", bbox=(0, 0, 60, 20),  label="menu.nm",           group_id=1),
            Word(text="5,000",  bbox=(80, 0, 120, 20), label="menu.price",        group_id=1),
            Word(text="TOTAL",  bbox=(0, 40, 60, 60),  label="total.total_price", group_id=2, is_key=True),
            Word(text="5,000",  bbox=(80, 40, 120, 60),label="total.total_price", group_id=2),
        ],
        width=200, height=300,
    )


def _make_complex_doc(doc_id: str = "complex") -> Document:
    """Dense receipt with many items and structural sections — high complexity."""
    words = []
    for i in range(10):
        y, gid = i * 30, i + 1
        words += [
            Word(text="x",         bbox=(0, y, 10, y+20),   label="menu.cnt",   group_id=gid),
            Word(text=f"ITEM{i}",   bbox=(15, y, 75, y+20),  label="menu.nm",    group_id=gid),
            Word(text="10,000",    bbox=(80, y, 130, y+20),  label="menu.price", group_id=gid),
        ]
    y = 330
    words += [
        Word(text="SUBTOTAL", bbox=(0, y,    70, y+20),  label="sub_total.subtotal_price", group_id=11, is_key=True),
        Word(text="100,000",  bbox=(80, y,   130, y+20), label="sub_total.subtotal_price", group_id=11),
        Word(text="TAX",      bbox=(0, y+30, 40, y+50),  label="sub_total.tax_price",      group_id=12, is_key=True),
        Word(text="10,000",   bbox=(80, y+30,130, y+50), label="sub_total.tax_price",      group_id=12),
        Word(text="TOTAL",    bbox=(0, y+60, 50, y+80),  label="total.total_price",        group_id=13, is_key=True),
        Word(text="110,000",  bbox=(80, y+60,130, y+80), label="total.total_price",        group_id=13),
    ]
    return Document(
        doc_id=doc_id, dataset="cord", split="test",
        words=words, width=200, height=700,
    )


def _make_sroie_doc(doc_id: str = "sroie0") -> Document:
    """SROIE-style doc: words have no group_id, no labels."""
    return Document(
        doc_id=doc_id, dataset="sroie", split="test",
        words=[
            Word(text="SUPERMART",  bbox=(0, 0,  90, 20)),
            Word(text="RECEIPT",    bbox=(0, 30, 70, 50)),
            Word(text="BREAD",      bbox=(0, 60, 55, 80)),
            Word(text="3.50",       bbox=(80, 60, 120, 80)),
            Word(text="MILK",       bbox=(0, 90, 40, 110)),
            Word(text="2.80",       bbox=(80, 90, 120, 110)),
            Word(text="TOTAL",      bbox=(0, 130, 50, 150)),
            Word(text="6.30",       bbox=(80, 130, 120, 150)),
        ],
        width=200, height=400,
    )


# ---------------------------------------------------------------------------
# extract_signals
# ---------------------------------------------------------------------------

def test_extract_signals_value_ranges():
    sig = extract_signals(_make_simple_doc())
    assert 0.0 <= sig.short_token_ratio <= 1.0
    assert sig.inv_chars_per_word >= 0.0
    assert sig.word_height_cv >= 0.0
    assert 0.0 <= sig.crowded_line_frac <= 1.0
    assert sig.section_count >= 1.0
    assert sig.line_density > 0.0
    assert sig.aspect_ratio == pytest.approx(300 / 200)


def test_extract_signals_complex_higher_than_simple():
    sig_s = extract_signals(_make_simple_doc())
    sig_c = extract_signals(_make_complex_doc())
    # a denser, more structured doc should have higher section_count and label_diversity
    assert sig_c.section_count >= sig_s.section_count
    assert sig_c.label_diversity >= sig_s.label_diversity


def test_extract_signals_sroie_no_group_id_fallback():
    """SROIE docs have no group_id; y-band fallback must produce sane values."""
    sig = extract_signals(_make_sroie_doc())
    assert sig.line_density > 0.0, "line_density must be > 0 even without group_ids"
    assert sig.section_count == 0.0, "no labels => no sections"
    assert sig.crowded_line_frac >= 0.0


# ---------------------------------------------------------------------------
# normalize_signals
# ---------------------------------------------------------------------------

def test_normalize_single_doc_all_zeros():
    # With one doc hi == lo for every signal, so all values are 0.
    normed = normalize_signals([extract_signals(_make_simple_doc())])
    assert all(v == 0.0 for v in normed[0].values())


def test_normalize_multiple_docs_bounds():
    docs = [_make_simple_doc(), _make_complex_doc()]
    normed = normalize_signals([extract_signals(d) for d in docs])
    for row in normed:
        for v in row.values():
            assert 0.0 <= v <= 1.0


def test_normalize_preserves_ordering():
    """The more complex doc should score higher after normalization."""
    docs = [_make_simple_doc(), _make_complex_doc()]
    sigs = [extract_signals(d) for d in docs]
    normed = normalize_signals(sigs)
    score_s = compute_complexity_score(normed[0])
    score_c = compute_complexity_score(normed[1])
    assert score_c > score_s


# ---------------------------------------------------------------------------
# compute_complexity_score
# ---------------------------------------------------------------------------

def test_compute_complexity_score_range():
    sigs = [extract_signals(_make_simple_doc()), extract_signals(_make_complex_doc())]
    normed = normalize_signals(sigs)
    for row in normed:
        score = compute_complexity_score(row)
        assert 0.0 <= score <= 1.0


def test_compute_complexity_score_custom_weights():
    sigs = [extract_signals(_make_simple_doc()), extract_signals(_make_complex_doc())]
    normed = normalize_signals(sigs)
    weights = {"line_density": 1.0}  # only one signal
    for row in normed:
        score = compute_complexity_score(row, weights=weights)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_corpus
# ---------------------------------------------------------------------------

def test_score_corpus_length_and_bounds():
    docs = [_make_simple_doc(), _make_complex_doc(), _make_sroie_doc()]
    scores = score_corpus(docs)
    assert len(scores) == len(docs)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_score_corpus_empty():
    assert score_corpus([]) == []
