"""Dataset adapters: normalize CORD / SROIE / FUNSD into a common Document."""
from __future__ import annotations

from pathlib import Path

from .base_loader import BaseLoader, Document, Word
from .cord_loader import CordLoader
from .deepform_loader import DeepformLoader
from .docile_loader import DocileLoader
from .funsd_loader import FunsdLoader
from .poie_loader import PoieLoader
from .sroie_loader import SroieLoader
from .vrdu_loader import VrduLoader

# default dataset roots relative to project root (data/<dir>)
_DEFAULT_ROOTS = {
    "cord": "data/CORD",
    "sroie": "data/SROIE2019",
    "funsd": "data/FUNSD",
    "vrdu": "data/vrdu/registration-form",
    "docile": "data/docile",
    "poie": "data/poie/nfv5/nfv5_3125",
    "deepform": "data/deepform",
}

_REGISTRY: dict[str, type[BaseLoader]] = {
    "cord": CordLoader,
    "sroie": SroieLoader,
    "funsd": FunsdLoader,
    "vrdu": VrduLoader,
    "docile": DocileLoader,
    "poie": PoieLoader,
    "deepform": DeepformLoader,
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
    "DocileLoader",
    "PoieLoader",
    "DeepformLoader",
    "get_loader",
]
