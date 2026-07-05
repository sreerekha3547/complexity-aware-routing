"""SROIE 2019 loader.

Native format: ``data/SROIE2019/{train,test}/`` with three parallel folders:
  - ``box/<id>.txt``      -- one line per OCR token: ``x1,y1,...,x4,y4,text``
  - ``entities/<id>.txt`` -- doc-level JSON with company/date/address/total
  - ``img/<id>.jpg``      -- the receipt image

SROIE has no per-word field labels (only doc-level key-values), so ``Word.label``
is left ``None``; the four target fields live in ``Document.fields``.

There is no official dev split, so we expose ``train`` and ``test`` only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from PIL import Image

from .base_loader import BaseLoader, Document, Word, poly8_to_bbox


class SroieLoader(BaseLoader):
    name = "sroie"
    splits = {"train": "train", "test": "test"}

    def iter_documents(self, split: str) -> Iterator[Document]:
        box_dir = self.root / split / "box"
        ent_dir = self.root / split / "entities"
        img_dir = self.root / split / "img"

        for fp in sorted(box_dir.glob("*.txt")):
            words: list[Word] = []
            for raw_line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not raw_line.strip():
                    continue
                parts = raw_line.split(",", 8)  # first 8 are coords, rest is text (may contain commas)
                if len(parts) < 9:
                    continue
                coords = [float(p) for p in parts[:8]]
                text = parts[8]
                if not text:
                    continue
                words.append(Word(text=text, bbox=poly8_to_bbox(coords)))

            fields: dict[str, str] = {}
            ent_fp = ent_dir / fp.name
            if ent_fp.exists():
                try:
                    fields = json.loads(ent_fp.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    fields = {}

            img_path, w, h = self._resolve_image(img_dir, fp.stem)
            yield Document(
                doc_id=fp.stem,
                dataset=self.name,
                split=split,
                words=words,
                image_path=img_path,
                width=w,
                height=h,
                fields=fields,
            )

    @staticmethod
    def _resolve_image(img_dir: Path, stem: str) -> tuple[Path | None, int, int]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                try:
                    with Image.open(p) as im:
                        return p, im.width, im.height
                except OSError:
                    return p, 0, 0
        return None, 0, 0
