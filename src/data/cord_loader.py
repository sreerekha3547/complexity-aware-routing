"""CORD loader.

Native format: ``data/CORD/{train,dev,test}/json/*.json`` with matching images
under ``.../image/``. Each ``valid_line`` carries a category, a group_id linking
the qty/name/price of one item, and word-level quads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base_loader import BaseLoader, Document, Word, quad_to_bbox


class CordLoader(BaseLoader):
    name = "cord"
    splits = {"train": "train", "dev": "dev", "test": "test"}

    def iter_documents(self, split: str) -> Iterator[Document]:
        json_dir = self.root / split / "json"
        img_dir = self.root / split / "image"
        for fp in sorted(json_dir.glob("*.json")):
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)

            sz = doc.get("meta", {}).get("image_size", {})
            words: list[Word] = []
            for line in doc.get("valid_line", []):
                cat = line.get("category")
                gid = line.get("group_id")
                for w in line.get("words", []):
                    text = w.get("text", "")
                    if not text:
                        continue
                    words.append(
                        Word(
                            text=text,
                            bbox=quad_to_bbox(w.get("quad", {})),
                            label=cat,
                            group_id=gid,
                            is_key=bool(w.get("is_key", 0)),
                        )
                    )

            img_path = img_dir / f"{fp.stem}.png"
            yield Document(
                doc_id=fp.stem,
                dataset=self.name,
                split=split,
                words=words,
                image_path=img_path if img_path.exists() else None,
                width=sz.get("width", 0),
                height=sz.get("height", 0),
                raw=doc,
            )
