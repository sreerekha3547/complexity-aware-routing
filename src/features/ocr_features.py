"""Easy per-document features.

Stage 1 features (this module):
  - token_count           : number of OCR tokens (from annotations)
  - page_count            : number of page images (1 for CORD/SROIE/FUNSD)
  - avg_ocr_confidence    : mean Tesseract per-word confidence over the image
  - low_conf_ratio        : fraction of tokens below the confidence threshold
  - ocr_var               : variance of per-token confidence
  - ocr_stage             : which preprocessing stage produced a usable result
                            (0 = native, 1 = 2x upscale, -1 = all stages failed)

OCR confidence is NOT present in any dataset's annotations, so it is generated
by running Tesseract over the document image and used as a routing signal.
Results are cached per document under data/processed/ocr_cache/<dataset>/<id>.json.

Preprocessing policy
--------------------
We try two stages, stopping at the first usable result:

  Stage 0 – native image, PSM 11 (sparse text)
  Stage 1 – 2x grayscale upscale (recovers low-DPI / small-format scans)

Low-contrast images (grey thermal receipts), blurry images, or photos on dark
backgrounds intentionally return low or zero confidence — this IS the routing
signal. Those documents should be routed to an expensive model. Do not add
further preprocessing stages to paper over genuine OCR difficulty.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

import pytesseract
from PIL import Image

from src.data.base_loader import Document

# -- locate the Tesseract binary on Windows ------------------------------------
_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]
for _c in _CANDIDATES:
    if Path(_c).exists():
        pytesseract.pytesseract.tesseract_cmd = _c
        break

_CACHE_ROOT = Path("data/processed/ocr_cache")

# PSM 11 = "sparse text": finds text in no particular order, robust to
# irregular receipt layouts where PSM 3 (auto) often returns zero tokens.
_DEFAULT_PSM = 11

# Minimum bar to consider a Tesseract pass usable — guards against a single
# noise token (conf=0) stopping the pipeline prematurely.
_MIN_USEFUL_TOKENS = 3
_MIN_USEFUL_CONF = 5.0

# Two-stage preprocessing pipeline. Stage 0 is a no-op (native image).
# Stage 1 recovers genuinely low-resolution scans via upscaling.
# Anything harder than low-res (grey text, dark background) is a routing signal,
# not a preprocessing problem — leave those at low/zero confidence.
_PREPROCESS_STAGES: list[Callable[[Image.Image], Image.Image]] = [
    lambda im: im,
    lambda im: im.convert("L").resize((im.width * 2, im.height * 2)),
]


# -- helpers -------------------------------------------------------------------

def _usable_result(confs: list[float]) -> bool:
    """True when the result is good enough to stop trying further stages."""
    if not confs:
        return False
    return len(confs) >= _MIN_USEFUL_TOKENS and statistics.mean(confs) >= _MIN_USEFUL_CONF


def _run_tesseract(im: Image.Image, psm: int) -> list[float]:
    """Return per-word Tesseract confidences (0..100); drops noise tokens (conf<0)."""
    data = pytesseract.image_to_data(
        im, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    return [
        float(c)
        for c, txt in zip(data["conf"], data["text"])
        if str(txt).strip() and float(c) >= 0
    ]


def _ocr_confidences(
    doc: Document, *, use_cache: bool = True, psm: int = _DEFAULT_PSM
) -> tuple[list[float], int]:
    """Per-word Tesseract confidences and the stage index that produced them.

    Returns (confs, stage) where stage is -1 if all stages failed.
    Results are cached to disk (stage is NOT cached — only the confidence list).
    """
    cache_fp = _CACHE_ROOT / doc.dataset / doc.split / f"{doc.doc_id}.json"
    if use_cache and cache_fp.exists():
        cached = json.loads(cache_fp.read_text())
        # Stage is not stored in cache; use -1 sentinel when loading from cache
        # so downstream code knows it came from cache, not a fresh run.
        stage = 0 if _usable_result(cached) else -1
        return cached, stage

    if doc.image_path is None or not Path(doc.image_path).exists():
        return [], -1

    confs: list[float] = []
    stage = -1
    with Image.open(doc.image_path) as im:
        for i, preprocess in enumerate(_PREPROCESS_STAGES):
            confs = _run_tesseract(preprocess(im), psm)
            if _usable_result(confs):
                stage = i
                break

    if use_cache:
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        cache_fp.write_text(json.dumps(confs))
    return confs, stage


# -- annotation-only features --------------------------------------------------

def token_count(doc: Document) -> int:
    """Number of OCR tokens in the document (from ground-truth annotations)."""
    return doc.n_words


def page_count(doc: Document) -> int:
    """Number of page images. CORD/SROIE/FUNSD are single-page (always 1)."""
    return 1 if doc.image_path is not None else 0


# -- OCR-derived features ------------------------------------------------------

LOW_CONF_THRESHOLD = 60.0


def avg_ocr_confidence(doc: Document, *, use_cache: bool = True) -> float:
    """Mean Tesseract word confidence (0..100). Returns 0.0 if no text found."""
    confs, _ = _ocr_confidences(doc, use_cache=use_cache)
    return statistics.mean(confs) if confs else 0.0


def low_confidence_ratio(
    doc: Document, *, threshold: float = LOW_CONF_THRESHOLD, use_cache: bool = True
) -> float:
    """Fraction of tokens with confidence below ``threshold`` (0..1)."""
    confs, _ = _ocr_confidences(doc, use_cache=use_cache)
    if not confs:
        return 0.0
    return sum(1 for c in confs if c < threshold) / len(confs)


def ocr_std(doc: Document, *, use_cache: bool = True) -> float:
    """Standard deviation of per-token confidence. Returns 0.0 if fewer than two tokens."""
    confs, _ = _ocr_confidences(doc, use_cache=use_cache)
    if len(confs) < 2:
        return 0.0
    return statistics.pstdev(confs)


# -- bundle --------------------------------------------------------------------

def extract_ocr_features(doc: Document, *, use_cache: bool = True) -> dict:
    """Stage-1 easy features for one document.

    All OCR-derived stats share a single cached Tesseract pass per document.
    ``ocr_stage`` records which preprocessing stage succeeded (-1 = all failed).
    """
    confs, stage = _ocr_confidences(doc, use_cache=use_cache)
    avg = statistics.mean(confs) if confs else 0.0
    low_ratio = (
        sum(1 for c in confs if c < LOW_CONF_THRESHOLD) / len(confs)
    ) if confs else 0.0
    std = statistics.pstdev(confs) if len(confs) >= 2 else 0.0
    return {
        "doc_id"        : doc.doc_id,
        "dataset"       : doc.dataset,
        "split"         : doc.split,
        "tokens"        : token_count(doc),
        "pages"         : page_count(doc),
        "ocr_conf"      : round(avg, 2),
        "low_conf_ratio": round(low_ratio, 3),
        "ocr_std"       : round(std, 1),
        "ocr_stage"     : stage,
    }
