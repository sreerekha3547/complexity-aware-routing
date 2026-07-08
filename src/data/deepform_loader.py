"""DeepForm loader (degraded/scanned political-ad disclosure forms) -- a
variable-capture-quality NON-receipt genre.

DeepForm (Stray & Svetlichnaya) is FCC TV/cable political ad-buy forms: old,
faxed, photocopied, hand-annotated scans -- extreme capture degradation -- with
buried key fields. That is exactly the dual profile our diagnostic says
pre-inference routing needs: (1) hard enough that the cheap model drops fields
(headroom), AND (2) scan-quality variance so pre-inference features can predict
WHICH docs are hard -- the two conditions clean digital invoices / near-ceiling
nutrition labels each failed.

Source: the DUE benchmark re-host. Layout under data/deepform/:
    {train,dev,test}/document.jsonl           KV annotations, one JSON/line:
        { "name": <docid>, "split": ...,
          "annotations": [ {"key": <field>, "values": [ {"value": <str>} ]}, ...] }
    {train,dev,test}/documents_content.jsonl  OCR (tool: djvu), one JSON/line:
        { "name": <docid>,
          "contents": [ { "tokens_layer": {
                "tokens":    [<str>, ...],                 # all pages
                "positions": [[x0,y0,x1,y1], ...],         # pixel coords
                "structures": {
                    "pages": {"structure_value": [[tok_start,tok_end],...],
                               "positions": [[0,0,PW,PH], ...]},  # per page
                    "lines": {...} } } } ] }
    page_pngs/<docid>_page0.png               rendered first page (from the tar).

DeepForm docs are multi-page (median 3, up to 100+), but the 5 target fields live
on the page-0 order-worksheet and full-doc token counts explode (up to 68k). So we
use the PAGE-0 slice of tokens with the page-0 image -- consistent single-page
treatment with the other datasets, and cheap. Token positions are pixel coords at
the OCR resolution; we scale to the rendered PNG via the page-0 page box.

We map DUE `dev` -> our held-out `test` (DeepForm `test` gold is withheld/partial).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from PIL import Image

from .base_loader import BaseLoader, Document, Word

# DeepForm scans are trusted local files; some are huge merged renders (100+
# pages) that trip PIL's decompression-bomb guard and would exhaust memory in
# Tesseract/OpenCV. Lift the guard and downscale anything oversized.
Image.MAX_IMAGE_PIXELS = None
_MAX_PIXELS = 25_000_000                       # ~25 Mpx: ample for OCR features
_DOWNSCALE_CACHE = Path("data/processed/deepform_images")

_SPLIT_DIRS = {"train": "train", "test": "dev"}
_IMAGE_DIR = "page_pngs"


def _prepare_image(img: Path) -> Path:
    """Return a path to `img`, downscaled+cached if it exceeds _MAX_PIXELS."""
    with Image.open(img) as im:
        w, h = im.size
        if w * h <= _MAX_PIXELS:
            return img
        cached = _DOWNSCALE_CACHE / img.name
        if not cached.exists():
            scale = (_MAX_PIXELS / (w * h)) ** 0.5
            cached.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").resize((max(1, int(w * scale)),
                                      max(1, int(h * scale)))).save(cached)
        return cached

# DeepForm's five target fields (with tolerant key aliases).
_KEY_ALIASES = {
    "advertiser"     : "advertiser",
    "contract_num"   : "contract_num",
    "contract_number": "contract_num",
    "flight_from"    : "flight_from",
    "flight_to"      : "flight_to",
    "gross_amount"   : "gross_amount",
}


class DeepformLoader(BaseLoader):
    name = "deepform"
    splits = {"train": "train", "test": "test"}

    def _content_index(self, split_dir: Path) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        with open(split_dir / "documents_content.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    idx[rec["name"]] = rec
        return idx

    def iter_documents(self, split: str, keep: set[str] | None = None) -> Iterator[Document]:
        split_dir = self.root / _SPLIT_DIRS[split]
        content = self._content_index(split_dir)

        with open(split_dir / "document.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ann = json.loads(line)
                docid = ann["name"]
                if keep is not None and docid not in keep:
                    continue
                crec = content.get(docid)
                if crec is None or not crec.get("contents"):
                    continue
                tl = crec["contents"][0].get("tokens_layer", {}) or {}
                tokens = tl.get("tokens", []) or []
                positions = tl.get("positions", []) or []
                pages = (tl.get("structures", {}) or {}).get("pages", {}) or {}
                page_val = pages.get("structure_value", []) or []
                page_pos = pages.get("positions", []) or []

                # page-0 token slice + page-0 page box (pixel coords).
                p0_end = int(page_val[0][1]) if page_val else len(tokens)
                pw = ph = 1.0
                if page_pos:
                    x0, y0, x1, y1 = page_pos[0]
                    pw, ph = float(x1 - x0) or 1.0, float(y1 - y0) or 1.0

                img = self.root / _IMAGE_DIR / f"{docid}_page0.png"
                image_path = _prepare_image(img) if img.exists() else None
                img_w, img_h = pw, ph
                if image_path is not None:
                    with Image.open(image_path) as im:
                        img_w, img_h = float(im.width), float(im.height)
                sx, sy = img_w / pw, img_h / ph

                words: list[Word] = []
                for i in range(min(p0_end, len(tokens), len(positions))):
                    text = str(tokens[i]).strip()
                    if not text:
                        continue
                    bx0, by0, bx1, by1 = positions[i]
                    words.append(Word(text=text, bbox=(bx0 * sx, by0 * sy, bx1 * sx, by1 * sy)))

                fields: dict[str, str] = {}
                for a in ann.get("annotations", []):
                    key = _KEY_ALIASES.get(str(a.get("key", "")).strip())
                    vals = a.get("values") or []
                    if key and vals:
                        v = str(vals[0].get("value", "")).strip()
                        if v:
                            fields[key] = v

                yield Document(
                    doc_id=docid,
                    dataset=self.name,
                    split=split,
                    words=words,
                    image_path=image_path,
                    width=int(img_w),
                    height=int(img_h),
                    fields=fields,
                    raw={"n_pages": len(page_val)},
                )
