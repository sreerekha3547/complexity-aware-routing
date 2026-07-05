"""FUNSD loader.

Native format: ``data/FUNSD/{training_data,testing_data}/`` with
  - ``annotations/<id>.json`` -- ``form[]`` entities, each with a box, label
    (other/header/question/answer), word-level boxes, and ``linking`` pairs.
  - ``images/<id>.png``

Each form entity becomes a group (``group_id`` = entity id); its label propagates
to every word in the entity. FUNSD boxes are already axis-aligned
``[x0, y0, x1, y1]``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from PIL import Image

from .base_loader import BaseLoader, Document, Word


class FunsdLoader(BaseLoader):
    name = "funsd"
    # map canonical split -> native folder
    splits = {"train": "train", "test": "test"}
    _native_dirs = {"train": "training_data", "test": "testing_data"}

    def iter_documents(self, split: str) -> Iterator[Document]:
        native = self._native_dirs[split]
        ann_dir = self.root / native / "annotations"
        img_dir = self.root / native / "images"

        for fp in sorted(ann_dir.glob("*.json")):
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)

            words: list[Word] = []
            for entity in doc.get("form", []):
                label = entity.get("label")
                gid = entity.get("id")
                for w in entity.get("words", []):
                    text = w.get("text", "")
                    box = w.get("box")
                    if not text or not box:
                        continue
                    words.append(
                        Word(
                            text=text,
                            bbox=(box[0], box[1], box[2], box[3]),
                            label=label,
                            group_id=gid,
                        )
                    )

            img_path, w, h = self._resolve_image(img_dir, fp.stem)
            yield Document(
                doc_id=fp.stem,
                dataset=self.name,
                split=split,
                words=words,
                image_path=img_path,
                width=w,
                height=h,
                raw=doc,
            )

    @staticmethod
    def _resolve_image(img_dir: Path, stem: str) -> tuple[Path | None, int, int]:
        for ext in (".png", ".jpg", ".jpeg"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                try:
                    with Image.open(p) as im:
                        return p, im.width, im.height
                except OSError:
                    return p, 0, 0
        return None, 0, 0
