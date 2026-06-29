"""Baseline routing strategies: always-small, always-large, random."""


def always_small(features: dict) -> str:
    return "small"


def always_large(features: dict) -> str:
    return "large"


def random_router(features: dict) -> str:
    import random
    return random.choice(["small", "large"])
