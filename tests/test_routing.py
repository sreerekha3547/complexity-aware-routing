"""Unit tests for routing strategies."""
from src.routing.baselines import always_small, always_large, random_router


def test_always_small():
    assert always_small({}) == "small"


def test_always_large():
    assert always_large({}) == "large"


def test_random_router_valid_output():
    result = random_router({})
    assert result in ("small", "large")
