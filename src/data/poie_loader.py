"""POIE loader (photographed Nutrition Facts labels) -- a variable-capture-quality genre.

POIE (Kuang et al., ICDAR 2023) is real-world product nutrition labels captured
by camera: severe distortion, folds, perspective, noisy backgrounds -- i.e. the
variable-capture-quality profile our diagnostic says pre-inference routing needs
(unlike clean digital invoices). Key-value extraction of 21 nutrition fields.

Native format (data/poie/nfv5/nfv5_3125/)
-----------------------------------------
train.txt / test.txt   One JSON per line:
    { file_name: "image_files/X.jpg", height, width,
      annotations: [ {polygon: [x1,y1,x2,y2,...], text, entity:[...]}, ... ],
      entity_dict: { "<CLASS>": "<value>", ... } }   # the KV ground truth
class_list.json        the 21 entity classes.

We use `entity_dict` (class -> value) as the extraction target and the annotation
text boxes as doc.words; OCR-quality features come from a Tesseract pass on the
photographed jpg (as for the other datasets).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base_loader import BaseLoader, Document, Word

_SPLIT_FILES = {"train": "train.txt", "test": "test.txt"}


def _poly_bbox(poly: list[float]) -> tuple[float, float, float, float]:
    xs, ys = poly[0::2], poly[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


class PoieLoader(BaseLoader):
    name = "poie"
    splits = {"train": "train", "test": "test"}

    def iter_documents(self, split: str, keep: set[str] | None = None) -> Iterator[Document]:
        with open(self.root / _SPLIT_FILES[split], encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_id = Path(rec["file_name"]).stem
                if keep is not None and doc_id not in keep:
                    continue

                words: list[Word] = []
                for a in rec.get("annotations", []):
                    text = str(a.get("text", "")).strip()
                    poly = a.get("polygon")
                    if not text or not poly or len(poly) < 4:
                        continue
                    words.append(Word(text=text, bbox=_poly_bbox(poly)))

                fields = {k: str(v).strip() for k, v in rec.get("entity_dict", {}).items()
                          if str(v).strip()}

                img = self.root / rec["file_name"]
                yield Document(
                    doc_id=doc_id,
                    dataset=self.name,
                    split=split,
                    words=words,
                    image_path=img if img.exists() else None,
                    width=int(rec.get("width", 0)),
                    height=int(rec.get("height", 0)),
                    fields=fields,
                    raw={},
                )
