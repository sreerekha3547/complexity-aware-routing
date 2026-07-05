"""Dataset adapters: normalize CORD / SROIE / FUNSD into a common Document."""
from __future__ import annotations

from pathlib import Path

from .base_loader import BaseLoader, Document, Word
from .cord_loader import CordLoader
from .funsd_loader import FunsdLoader
from .sroie_loader import SroieLoader
from .vrdu_loader import VrduLoader

# default dataset roots relative to project root (data/<dir>)
_DEFAULT_ROOTS = {
    "cord": "data/CORD",
    "sroie": "data/SROIE2019",
    "funsd": "data/FUNSD",
    "vrdu": "data/vrdu/registration-form",
}

_REGISTRY: dict[str, type[BaseLoader]] = {
    "cord": CordLoader,
    "sroie": SroieLoader,
    "funsd": FunsdLoader,
    "vrdu": VrduLoader,
}


def get_loader(name: str, root: str | Path | None = None) -> BaseLoader:
    """Instantiate a loader by name, defaulting to the standard data/ path."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. options: {list(_REGISTRY)}")
    return _REGISTRY[key](root or _DEFAULT_ROOTS[key])


__all__ = [
    "BaseLoader",
    "Document",
    "Word",
    "CordLoader",
    "SroieLoader",
    "FunsdLoader",
    "VrduLoader",
    "get_loader",
]
