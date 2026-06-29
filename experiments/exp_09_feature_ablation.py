"""EXP-09: Feature-family ablation + permutation importance (reviewers R10, R5).

(A) Train the router on each feature family alone (and all together) to show
    which family carries the routing signal -- answers RQ1 directly.
(B) Report permutation importance for the actual deployed model (calibrated RF),
    not just LR coefficients, so feature attribution matches the model used.

Outputs
-------
results/tables/exp09_feature_ablation.csv
results/tables/exp09_permutation_importance.csv

Run
---
    python experiments/exp_09_feature_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from experiments.exp_03_routing_model import (
    FEATURE_COLS, apply_tau, load_raw_data,
)

TABLE_DIR = Path("results/tables")
TAU = 0.02

FAMILIES = {
    "OCR quality":  ["ocr_conf", "ocr_std", "ocr_stage",
                     "short_token_ratio", "inv_chars_per_word"],
    "Image quality": ["blur_score", "image_contrast", "word_height_cv"],
    "Layout":       ["crowded_line_frac", "line_density",
                     "section_count", "aspect_ratio"],
    "Content":      ["label_entropy", "label_diversity",
                     "item_density", "tokens"],
}


def build_rf() -> Pipeline:
    return Pipeline([("clf", CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42),
        method="isotonic", cv=3,
    ))])


def auc_for(train_df, test_df, cols) -> float:
    rf = build_rf().fit(train_df[cols].fillna(0).values, train_df["label"].values)
    p = rf.predict_proba(test_df[cols].fillna(0).values)[:, 1]
    return roc_auc_score(test_df["label"].values, p)


def main() -> int:
    print("EXP-09: Feature-family ablation + permutation importance\n")
    train_raw, test_raw, _ = load_raw_data()
    train_df = apply_tau(train_raw, TAU)
    test_df = apply_tau(test_raw, TAU)  # pooled CORD+SROIE test

    # --- (A) family ablation ---------------------------------------------
    print("(A) Pooled held-out AUC by feature family:")
    rows = []
    for fam, cols in FAMILIES.items():
        a = auc_for(train_df, test_df, cols)
        rows.append({"feature_set": fam, "n_features": len(cols),
                     "pooled_auc": round(a, 3)})
        print(f"  {fam:14} ({len(cols)})  AUC={a:.3f}")
    a_all = auc_for(train_df, test_df, FEATURE_COLS)
    rows.append({"feature_set": "All", "n_features": len(FEATURE_COLS),
                 "pooled_auc": round(a_all, 3)})
    print(f"  {'All':14} ({len(FEATURE_COLS)})  AUC={a_all:.3f}")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "exp09_feature_ablation.csv", index=False)

    # --- (B) permutation importance of the deployed model ----------------
    print("\n(B) Permutation importance (calibrated RF, pooled test, AUC drop):")
    rf = build_rf().fit(train_df[FEATURE_COLS].fillna(0).values,
                        train_df["label"].values)
    r = permutation_importance(
        rf, test_df[FEATURE_COLS].fillna(0).values, test_df["label"].values,
        scoring="roc_auc", n_repeats=30, random_state=42,
    )
    imp = sorted(zip(FEATURE_COLS, r.importances_mean, r.importances_std),
                 key=lambda x: x[1], reverse=True)
    prows = []
    for feat, m, s in imp:
        prows.append({"feature": feat, "auc_drop_mean": round(m, 4),
                      "auc_drop_std": round(s, 4)})
        print(f"  {feat:20} {m:+.4f} +/- {s:.4f}")
    pd.DataFrame(prows).to_csv(
        TABLE_DIR / "exp09_permutation_importance.csv", index=False)

    print(f"\n  -> {TABLE_DIR}/exp09_feature_ablation.csv")
    print(f"  -> {TABLE_DIR}/exp09_permutation_importance.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
