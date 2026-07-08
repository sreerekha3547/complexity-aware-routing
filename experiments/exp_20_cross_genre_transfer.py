"""EXP-20: Cross-genre transfer of the router (free -- uses cached oracle labels).

Does a router trained on one genre predict difficulty on ANOTHER genre, or must it
be refit per genre? We assemble (13 clean features + gap) for every genre we have
oracle labels for, then fill a train-genre x test-genre AUC matrix:

  - diagonal  (same genre): 5-fold CV AUC -- the within-genre ceiling.
  - off-diag  (transfer)  : train on ALL of genre A, score ALL of genre B.

Genres: receipts (CORD+SROIE), deepform (degraded ad-buy forms), vrdu
(registration forms). No API calls -- everything is read from results/tables/.

    python experiments/exp_20_cross_genre_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.exp_03_routing_model import FEATURE_COLS

TAU = 0.02
TABLE_DIR = Path("results/tables")


# Datasets whose oracle csv already carries features + gap (self-contained).
_SELF_CONTAINED = {
    "deepform": "oracle_labels_deepform.csv",
    "docile"  : "oracle_labels_docile.csv",
    "poie"    : "poie_pilot.csv",
}


def _feat_gap_for(ds: str) -> pd.DataFrame:
    """Assemble (doc_id + 13 features + gap) for one dataset from cached tables."""
    if ds in _SELF_CONTAINED:
        df = pd.read_csv(TABLE_DIR / _SELF_CONTAINED[ds])
        return df[["doc_id"] + FEATURE_COLS + ["gap"]].copy()
    feat = pd.read_csv(TABLE_DIR / "feature_table.csv")
    feat = feat[feat.dataset == ds][["doc_id"] + FEATURE_COLS]
    gap = pd.concat([pd.read_csv(TABLE_DIR / f"oracle_labels_{ds}_{sp}.csv")
                     for sp in ("train", "test")], ignore_index=True)[["doc_id", "gap"]]
    return feat.merge(gap, on="doc_id", how="inner")


def _xy(df: pd.DataFrame):
    return df[FEATURE_COLS].fillna(0).values, (df.gap > TAU).astype(int).values


def main() -> int:
    # genre -> assembled frame
    groups = {
        "receipts" : pd.concat([_feat_gap_for("cord"), _feat_gap_for("sroie")], ignore_index=True),
        "deepform" : _feat_gap_for("deepform"),
        "invoices" : _feat_gap_for("docile"),
        "nutrition": _feat_gap_for("poie"),
        "vrdu"     : _feat_gap_for("vrdu"),
    }
    order = ["receipts", "deepform", "invoices", "nutrition", "vrdu"]

    print("Assembled (n, large-required%):")
    for g in order:
        X, y = _xy(groups[g])
        print(f"  {g:<10} n={len(y):<5} large-required={100*y.mean():.0f}%")
    print()

    def rf():
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

    # AUC matrix: rows = train genre, cols = test genre
    auc = pd.DataFrame(index=order, columns=order, dtype=float)
    for tr in order:
        Xtr, ytr = _xy(groups[tr])
        for te in order:
            Xte, yte = _xy(groups[te])
            if len(np.unique(yte)) < 2:
                auc.loc[tr, te] = np.nan
                continue
            if tr == te:
                cv = StratifiedKFold(5, shuffle=True, random_state=42)
                p = cross_val_predict(rf(), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
                auc.loc[tr, te] = roc_auc_score(ytr, p)
            else:
                m = rf().fit(Xtr, ytr)
                auc.loc[tr, te] = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    auc.to_csv(TABLE_DIR / "exp20_cross_genre_auc.csv")
    print("AUC matrix  (rows=TRAIN genre, cols=TEST genre; diagonal=within-genre CV)\n")
    print("            " + "".join(f"{c:>12}" for c in order))
    for tr in order:
        print(f"  {tr:<10}" + "".join(
            f"{auc.loc[tr,c]:>12.3f}" if pd.notna(auc.loc[tr,c]) else f"{'--':>12}"
            for c in order))
    print()

    diag = np.nanmean([auc.loc[g, g] for g in order])
    off = np.nanmean([auc.loc[a, b] for a in order for b in order if a != b])
    print(f"  mean within-genre (diagonal) : {diag:.3f}")
    print(f"  mean cross-genre  (off-diag) : {off:.3f}")
    print(f"  transfer drop                : {diag-off:+.3f}")
    print("\n  -> results/tables/exp20_cross_genre_auc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
