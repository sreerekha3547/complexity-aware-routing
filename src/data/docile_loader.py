"""DocILE loader (invoices/orders) -- a third document genre for within-genre routing.

Native format (data/docile/, from `download_dataset.sh TOKEN labeled-trainval`)
------------------------------------------------------------------------------
{split}.json                One JSON list of doc-ids per split (train.json, val.json).
annotations/{docid}.json    { "metadata": {"page_count": N, ...},
                              "field_extractions": [ {fieldtype, text, page, bbox}, ... ],
                              "li_extractions": [...] }   # line items -- skipped here
ocr/{docid}.json            doctr export: {"pages": [ {"blocks":[{"lines":[{"words":[
                              {"value": str, "geometry": [[x0,y0],[x1,y1]]}]}]}], ... ]}
pdfs/{docid}.pdf            source PDF; first page rendered to PNG for image features.

We use the flat KILE key-value fields (`field_extractions`, fieldtype -> value) for the
same extraction-F1 protocol as CORD/SROIE/VRDU; line items are ignored. Bboxes are
normalized [0,1]; we scale to the rendered page pixels so the geometric layout
features (line/item density, crowding) are on the same footing as the other datasets.

We map DocILE's own `train` -> our train split and `val` -> our held-out `test`.

NOTE: coordinate normalization, split filenames, and multi-page stacking are validated
on the real download; adjust `_SPLIT_FILES` / bbox handling if the smoke test flags them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base_loader import BaseLoader, Document, Word

_IMAGE_CACHE = Path("data/processed/docile_images")
_SPLIT_FILES = {"train": "train.json", "test": "val.json"}


class DocileLoader(BaseLoader):
    name = "docile"
    splits = {"train": "train", "test": "test"}

    def iter_documents(self, split: str, keep: set[str] | None = None) -> Iterator[Document]:
        split_fp = self.root / _SPLIT_FILES[split]
        with open(split_fp, encoding="utf-8") as f:
            docids = json.load(f)

        for docid in docids:
            if keep is not None and docid not in keep:
                continue  # skip before the expensive PDF render (subset sampling)
            ann_fp = self.root / "annotations" / f"{docid}.json"
            ocr_fp = self.root / "ocr" / f"{docid}.json"
            if not ann_fp.exists() or not ocr_fp.exists():
                continue
            ann = json.loads(ann_fp.read_text(encoding="utf-8"))
            ocr = json.loads(ocr_fp.read_text(encoding="utf-8"))

            # Render first PDF page for image-quality features (cached).
            pdf_path = self.root / "pdfs" / f"{docid}.pdf"
            img_cache = _IMAGE_CACHE / split / f"{docid}.png"
            if not img_cache.exists() and pdf_path.exists():
                _render_pdf_page(pdf_path, img_cache)
            image_path = img_cache if img_cache.exists() else None

            # OCR words -> Word objects. doctr geometry is normalized [0,1];
            # scale by the REAL rendered page pixel size (so aspect_ratio and
            # densities are genuine) and stack pages vertically so
            # reconstruct_text() reading order (sort by y,x) is preserved.
            PAGE_W, PAGE_H = 1000.0, 1400.0  # fallback if no image
            if image_path is not None:
                from PIL import Image
                with Image.open(image_path) as im:
                    PAGE_W, PAGE_H = float(im.width), float(im.height)
            words: list[Word] = []
            y_off = 0.0
            pages = ocr.get("pages", [])
            for page in pages:
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for w in line.get("words", []):
                            text = str(w.get("value", "")).strip()
                            geo = w.get("geometry")
                            if not text or not geo or len(geo) < 2:
                                continue
                            (x0n, y0n), (x1n, y1n) = geo[0], geo[1]
                            words.append(Word(text=text, bbox=(
                                x0n * PAGE_W, y0n * PAGE_H + y_off,
                                x1n * PAGE_W, y1n * PAGE_H + y_off)))
                y_off += PAGE_H

            # Ground-truth key-value fields: fieldtype -> unique values.
            # KILE repeats the SAME value at multiple page locations; dedupe so
            # the extraction-F1 target is the value, not the number of mentions.
            fields: dict[str, list[str]] = {}
            for fe in ann.get("field_extractions", []):
                ft = fe.get("fieldtype")
                val = (fe.get("text") or "").strip()
                if ft and val:
                    seen = fields.setdefault(ft, [])
                    if val not in seen:
                        seen.append(val)
            fields_flat = {k: " | ".join(vs) for k, vs in fields.items()}

            n_pages = ann.get("metadata", {}).get("page_count", len(pages) or 1)
            yield Document(
                doc_id=docid,
                dataset=self.name,
                split=split,
                words=words,
                image_path=image_path,
                width=int(PAGE_W),
                height=int(PAGE_H * max(n_pages, 1)),
                fields=fields_flat,
                raw={"n_pages": n_pages},
            )


def _render_pdf_page(pdf_path: Path, out_path: Path, scale: float = 2.0) -> None:
    """Render first page of a PDF to a PNG using PyMuPDF."""
    import fitz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()
