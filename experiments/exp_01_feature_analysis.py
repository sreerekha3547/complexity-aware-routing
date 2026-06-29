"""EXP-01: Stage-1 feature extraction + validation.

Builds the first per-document feature table (tokens, pages, ocr_conf) across
CORD / SROIE / FUNSD, validates the outputs (no failed OCR, sane ranges), and
writes the table to results/tables/.

Run from project root:
    python experiments/exp_01_feature_analysis.py            # 25 docs/dataset
    python experiments/exp_01_feature_analysis.py --full     # all test docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make the project root importable when run as a script (python experiments/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features.feature_table import build_feature_table, _TABLE_DIR

DATASETS = ["cord", "sroie", "funsd"]


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation warnings (empty == all checks passed)."""
    warnings: list[str] = []

    failed = df[df["ocr_conf"] == 0.0]
    if len(failed):
        warnings.append(
            f"{len(failed)} docs with ocr_conf==0 (OCR found no text): "
            f"{failed['doc_id'].tolist()[:5]}"
        )

    if (df["tokens"] <= 0).any():
        warnings.append(f"{(df['tokens'] <= 0).sum()} docs with zero tokens")

    out_of_range = df[(df["ocr_conf"] < 0) | (df["ocr_conf"] > 100)]
    if len(out_of_range):
        warnings.append(f"{len(out_of_range)} docs with ocr_conf outside [0,100]")

    if (df["pages"] != 1).any():
        warnings.append(f"{(df['pages'] != 1).sum()} docs with pages != 1")

    bad_ratio = df[(df["low_conf_ratio"] < 0) | (df["low_conf_ratio"] > 1)]
    if len(bad_ratio):
        warnings.append(f"{len(bad_ratio)} docs with low_conf_ratio outside [0,1]")

    if (df["ocr_std"] < 0).any():
        warnings.append(f"{(df['ocr_std'] < 0).sum()} docs with negative ocr_std")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="process all docs (no limit)")
    parser.add_argument("--split", default="test",
                        help="split to process, or 'all' to build train+dev+test together")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel OCR threads (default 4)")
    args = parser.parse_args()

    limit = None if args.full else 25

    if args.split == "all":
        # Build one combined table across all available splits for all datasets.
        # Normalization is corpus-wide across the combined corpus.
        splits_per_dataset = {
            "cord" : ["train", "dev", "test"],
            "sroie": ["train", "test"],
            "funsd": ["train", "test"],
            "vrdu" : ["train", "test"],
        }
        frames = []
        for dataset, splits in splits_per_dataset.items():
            for split in splits:
                try:
                    sub = build_feature_table([dataset], split=split, limit=limit,
                                              workers=args.workers)
                    frames.append(sub)
                    print(f"  {dataset}/{split}: {len(sub)} rows")
                except Exception as e:
                    print(f"  {dataset}/{split}: skipped ({e})")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(_TABLE_DIR / "feature_table.csv", index=False)
        print(f"\n  -> {df.shape[0]} rows written to results/tables/feature_table.csv")
    else:
        print(f"Building stage-1 feature table  (split={args.split}, "
              f"limit={'all' if limit is None else limit}/dataset)...")
        df = build_feature_table(
            DATASETS, split=args.split, limit=limit, save_as="feature_table.csv",
            workers=args.workers,
        )
        print(f"  -> {df.shape[0]} rows written to results/tables/feature_table.csv\n")

    # -- per-dataset summary ---------------------------------------------------
    print("Per-dataset summary:")
    summary = df.groupby("dataset")[
        ["tokens", "ocr_conf", "low_conf_ratio", "ocr_std", "image_contrast", "complexity_score"]
    ].mean().round(2)
    print(summary, "\n")

    # -- preview (Doc | OCR Conf | Tokens | Pages | low-conf | var) ------------
    print("Sample rows (first 4 per dataset):")
    preview = df.groupby("dataset").head(4)[
        ["doc_id", "dataset", "ocr_conf", "ocr_std", "ocr_stage",
         "image_contrast", "skew_angle", "complexity_score"]
    ]
    print(preview.to_string(index=False), "\n")

    # -- validation ------------------------------------------------------------
    warnings = validate(df)
    if warnings:
        print("VALIDATION WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
        return 1

    print("VALIDATION: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
