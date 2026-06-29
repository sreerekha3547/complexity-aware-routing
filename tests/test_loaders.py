"""Smoke tests: each loader normalizes real data into Documents, and the
complexity signals run on all three datasets without error."""
import pytest

from src.data import get_loader
from src.features.complexity_score import extract_signals, score_corpus


@pytest.mark.parametrize(
    "name,split,expected_min",
    [
        ("cord", "test", 100),
        ("sroie", "test", 50),
        ("funsd", "test", 40),
    ],
)
def test_loader_smoke(name, split, expected_min):
    loader = get_loader(name)
    docs = loader.load_split(split)
    assert len(docs) >= expected_min

    d = docs[0]
    assert d.dataset == name
    assert d.n_words > 0
    # every word has a 4-tuple bbox
    assert all(len(w.bbox) == 4 for w in d.words)


@pytest.mark.parametrize("name", ["cord", "sroie", "funsd"])
def test_signals_run(name):
    docs = get_loader(name).load_split("test")
    sig = extract_signals(docs[0])
    assert 0.0 <= sig.short_token_ratio <= 1.0
    assert sig.aspect_ratio >= 0.0

    scores = score_corpus(docs)
    assert len(scores) == len(docs)
    assert all(0.0 <= s <= 1.0 for s in scores)
