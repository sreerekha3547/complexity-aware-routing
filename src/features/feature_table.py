"""Build a per-document feature table across one or more datasets.

Columns produced
----------------
OCR features    : doc_id, dataset, split, tokens, pages,
                  ocr_conf, low_conf_ratio, ocr_std, ocr_stage
Quality features: blur_score, image_contrast, skew_angle,
                  estimated_dpi, is_grayscale
Layout features : x_spread_ratio
Complexity      : short_token_ratio, inv_chars_per_word, word_height_cv,
                  crowded_line_frac, line_density, section_count,
                  label_diversity, label_entropy, item_density, aspect_ratio,
                  complexity_score  (corpus-normalised 0..1)

Output is a pandas DataFrame; optionally written to results/tables/.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data import get_loader
from src.data.base_loader import Document
from src.features.complexity_score import (
    compute_complexity_score,
    extract_signals,
    normalize_signals,
)
from src.features.layout_features import extract_layout_features
from src.features.ocr_features import extract_ocr_features
from src.features.quality_features import extract_quality_features

_TABLE_DIR = Path("results/tables")


def _extract_doc(doc: Document) -> tuple[Document, dict, dict, dict]:
    """Extract all per-document features in one call (safe to run in a thread).

    Returns (doc, ocr_row, quality_row, layout_row). Complexity signals are
    extracted separately since normalization requires the full corpus first.
    OCR is the bottleneck — Tesseract spawns a subprocess per doc, so multiple
    threads run genuinely in parallel despite the GIL.
    """
    ocr     = extract_ocr_features(doc)
    quality = extract_quality_features(doc)
    layout  = extract_layout_features(doc)
    return doc, ocr, quality, layout


def build_feature_table(
    datasets: list[str],
    split: str = "test",
    limit: int | None = None,
    save_as: str | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Compute all features for each document and return as a DataFrame.

    Args:
        datasets: dataset names, e.g. ["cord", "sroie", "funsd"].
        split:    canonical split to load.
        limit:    cap docs per dataset (useful for quick runs); None = all.
        save_as:  optional filename under results/tables/ to write CSV.
        workers:  number of parallel OCR threads. OCR is the bottleneck
                  (Tesseract subprocess per doc). 4-8 is a good default;
                  cached docs are instant regardless of worker count.

    Notes:
        Complexity scores are normalised across the combined corpus of all
        requested datasets so scores are directly comparable between datasets.
    """
    all_docs = []
    for name in datasets:
        docs = get_loader(name).load_split(split)
        if limit is not None:
            docs = docs[:limit]
        all_docs.extend(docs)

    # --- Phase 1: per-doc extraction (parallelisable) ----------------------
    # Complexity signals are cheap and needed for corpus normalization below,
    # so extract them here too. The heavy work is OCR inside _extract_doc.
    signals_list = [None] * len(all_docs)
    per_doc: dict[str, tuple] = {}  # doc_id+split -> (ocr, quality, layout)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_extract_doc, doc): i
                for i, doc in enumerate(all_docs)
            }
            done = 0
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                doc, ocr, quality, layout = fut.result()
                key = f"{doc.doc_id}|{doc.split}"
                per_doc[key] = (ocr, quality, layout)
                signals_list[i] = extract_signals(doc)
                done += 1
                if done % 50 == 0 or done == len(all_docs):
                    print(f"    {done}/{len(all_docs)} docs processed", flush=True)
    else:
        for i, doc in enumerate(all_docs):
            _, ocr, quality, layout = _extract_doc(doc)
            key = f"{doc.doc_id}|{doc.split}"
            per_doc[key] = (ocr, quality, layout)
            signals_list[i] = extract_signals(doc)

    # --- Phase 2: corpus-wide normalization (sequential, fast) -------------
    normalized = normalize_signals(signals_list)

    # --- Phase 3: assemble rows --------------------------------------------
    rows: list[dict] = []
    for doc, sig, norm in zip(all_docs, signals_list, normalized):
        key = f"{doc.doc_id}|{doc.split}"
        ocr, quality, layout = per_doc[key]

        row: dict = {}
        row.update(ocr)
        row.update(quality)
        row["x_spread_ratio"] = layout["x_spread_ratio"]
        row.update(sig.as_dict())
        row["complexity_score"] = round(compute_complexity_score(norm), 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_as:
        _TABLE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_TABLE_DIR / save_as, index=False)

    return df
