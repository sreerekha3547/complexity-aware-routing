"""VRDU registration-form loader (Amendment template, SD_0 split).

Native format
-------------
data/vrdu/registration-form/main/dataset.jsonl
    One JSON line per document. Each record has:
      - filename: PDF filename used as the document ID
      - ocr.pages[]: list of pages, each with dimension + tokens[]
          token.bbox: [page_idx, x1_norm, y1_norm, x2_norm, y2_norm]
      - annotations: [[label, [[text, bbox, segs], ...]], ...]

data/vrdu/registration-form/few_shot-splits/
    FARA-lv1-single_Amendment-train_200-test_300-valid_100-SD_0.json
    Lists of filenames for train/test/valid.

PDF -> image conversion
-----------------------
PyMuPDF (fitz) renders the first PDF page to a PNG at 2x scale and caches it
under data/processed/vrdu_images/{split}/{doc_id}.png. Subsequent loads use
the cache. This enables the same image-based feature pipeline (OCR confidence,
blur score, contrast, skew) used for CORD and SROIE.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base_loader import BaseLoader, Document, Word

_IMAGE_CACHE = Path("data/processed/vrdu_images")

# Canonical split file -- 200 train / 300 test / 100 valid, seed 0.
_SPLIT_FILE = (
    "few_shot-splits/"
    "FARA-lv1-single_Amendment-train_200-test_300-valid_100-SD_0.json"
)


class VrduLoader(BaseLoader):
    name = "vrdu"
    splits = {"train": "train", "test": "test", "valid": "valid"}

    def iter_documents(self, split: str) -> Iterator[Document]:
        # Resolve which filenames belong to this split
        split_fp = self.root / _SPLIT_FILE
        with open(split_fp, encoding="utf-8") as f:
            split_data = json.load(f)
        filenames_in_split: set[str] = set(split_data[split])

        # Stream the main jsonl, yielding docs that belong to this split
        jsonl_fp = self.root / "main" / "dataset.jsonl"
        with open(jsonl_fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["filename"] not in filenames_in_split:
                    continue

                doc_id = Path(rec["filename"]).stem
                pdf_path = self.root / "main" / "pdfs" / rec["filename"]

                # Render first PDF page to PNG (cached after first run)
                img_cache_path = _IMAGE_CACHE / split / f"{doc_id}.png"
                if not img_cache_path.exists() and pdf_path.exists():
                    _render_pdf_page(pdf_path, img_cache_path)
                image_path = img_cache_path if img_cache_path.exists() else None

                pages = rec.get("ocr", {}).get("pages", [])

                # Convert OCR tokens -> Word objects across all pages.
                # y_offset stacks pages vertically so reading order is preserved
                # in reconstruct_text() which sorts by (y, x).
                words: list[Word] = []
                y_offset = 0.0
                for page in pages:
                    dim = page.get("dimension", {})
                    pw = float(dim.get("width", 1))
                    ph = float(dim.get("height", 1))

                    for tok in page.get("tokens", []):
                        text = tok.get("text", "").strip()
                        bbox_raw = tok.get("bbox", [])
                        if not text or len(bbox_raw) < 5:
                            continue
                        # bbox: [page_idx, x1_norm, y1_norm, x2_norm, y2_norm]
                        _, x1n, y1n, x2n, y2n = bbox_raw
                        x0, y0 = x1n * pw, y1n * ph + y_offset
                        x1, y1 = x2n * pw, y2n * ph + y_offset
                        words.append(Word(text=text, bbox=(x0, y0, x1, y1)))

                    y_offset += ph

                # Ground truth: entity annotations -> doc.fields dict.
                # Each annotation is [label_str, [[text, bbox, segs], ...]].
                # Multiple instances of the same label are joined with " | ".
                fields: dict[str, list[str]] = {}
                for ann in rec.get("annotations", []):
                    if len(ann) < 2:
                        continue
                    label = ann[0]
                    instances = ann[1] if isinstance(ann[1], list) else []
                    for inst in instances:
                        text_val = inst[0].strip() if inst else ""
                        if text_val:
                            fields.setdefault(label, []).append(text_val)

                fields_flat = {k: " | ".join(vs) for k, vs in fields.items()}

                dim0 = pages[0].get("dimension", {}) if pages else {}
                yield Document(
                    doc_id=doc_id,
                    dataset=self.name,
                    split=split,
                    words=words,
                    image_path=image_path,
                    width=int(dim0.get("width", 0)),
                    height=int(dim0.get("height", 0)),
                    fields=fields_flat,
                    raw=rec,
                )


def _render_pdf_page(pdf_path: Path, out_path: Path, scale: float = 2.0) -> None:
    """Render first page of a PDF to a PNG using PyMuPDF (no Poppler needed)."""
    import fitz  # PyMuPDF -- already installed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        mat  = fitz.Matrix(scale, scale)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()
