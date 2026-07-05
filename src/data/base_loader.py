"""Common document representation shared across all datasets.

Every dataset loader normalizes its native format into a single `Document`
object so that feature extractors and routers never touch raw dataset JSON.

Bounding boxes are normalized to axis-aligned ``(x0, y0, x1, y1)`` in pixel
coordinates, regardless of whether the source stored quads, 8-point polygons,
or corner pairs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class Word:
    """A single OCR token with its position and optional field label."""

    text: str
    bbox: BBox
    label: str | None = None      # field/category, e.g. "menu.nm", "total", "question"
    group_id: int | None = None   # links words belonging to the same line/item/entity
    is_key: bool = False          # token is a field key (e.g. "TOTAL:") rather than a value

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Document:
    """A single document, normalized across datasets."""

    doc_id: str
    dataset: str                       # "cord" | "sroie" | "funsd"
    split: str                         # "train" | "dev" | "test"
    words: list[Word] = field(default_factory=list)
    image_path: Path | None = None
    width: int = 0                     # image width in px (0 if unknown)
    height: int = 0                    # image height in px (0 if unknown)
    fields: dict[str, str] = field(default_factory=dict)  # doc-level key-value GT (SROIE)
    raw: dict = field(default_factory=dict)               # original annotation, for debugging

    # -- convenience accessors --------------------------------------------------
    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def labels(self) -> list[str]:
        return [w.label for w in self.words if w.label is not None]

    @property
    def aspect_ratio(self) -> float:
        return self.height / self.width if self.width else 0.0

    def groups(self) -> dict[int, list[Word]]:
        """Words bucketed by group_id (items/lines/entities). Ungrouped words skipped."""
        out: dict[int, list[Word]] = {}
        for w in self.words:
            if w.group_id is not None:
                out.setdefault(w.group_id, []).append(w)
        return out


class BaseLoader(ABC):
    """Abstract dataset loader.

    Subclasses implement :meth:`iter_documents` for one split. The shared
    :meth:`load_split` / :meth:`load_all` wrappers give every dataset a
    uniform call surface.
    """

    name: str = "base"
    #: maps the dataset's native split folder names to canonical split names
    splits: dict[str, str] = {}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

    @abstractmethod
    def iter_documents(self, split: str) -> Iterator[Document]:
        """Yield normalized ``Document`` objects for one canonical split."""
        raise NotImplementedError

    def load_split(self, split: str) -> list[Document]:
        return list(self.iter_documents(split))

    def load_all(self) -> dict[str, list[Document]]:
        return {s: self.load_split(s) for s in self.splits.values()}


# -- shared geometry helpers ----------------------------------------------------
def quad_to_bbox(quad: dict) -> BBox:
    """CORD ``{x1..x4, y1..y4}`` quad -> axis-aligned bbox."""
    xs = [quad[k] for k in ("x1", "x2", "x3", "x4") if k in quad]
    ys = [quad[k] for k in ("y1", "y2", "y3", "y4") if k in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def poly8_to_bbox(coords: list[float]) -> BBox:
    """SROIE 8-value polygon ``[x1,y1,...,x4,y4]`` -> axis-aligned bbox."""
    xs = coords[0::2]
    ys = coords[1::2]
    return (min(xs), min(ys), max(xs), max(ys))
